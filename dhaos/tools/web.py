"""Outils web : recherche (DuckDuckGo, SearXNG, Brave) et lecture de pages.

Fonctions publiques réutilisables (``dhaos.kb.ingest`` s'en sert pour
ingérer des URLs) :

- ``web_search(settings, query, max_results=None, *, client=None)`` →
  ``list[SearchResult]`` selon ``settings.web.provider`` ;
- ``fetch_page(settings, url, *, max_chars=None, client=None)`` →
  ``PageContent`` (HTML nettoyé, texte brut, JSON, Markdown ou PDF).

Sécurité : les URLs viennent du modèle et sont **non fiables**. Seuls les
schémas ``http``/``https`` sont acceptés ; l'hôte est résolu et refusé s'il
pointe vers une adresse interne ou non globale (boucle locale, réseaux privés,
lien-local, CGNAT ``100.64.0.0/10``, ``0.0.0.0``, ``localhost``…), y compris à
chaque redirection (garde anti-SSRF). La connexion est **épinglée** sur
l'adresse IP validée (URL de connexion par IP, en-tête ``Host`` et SNI portant
le nom d'hôte) afin qu'une seconde résolution DNS divergente (rebinding) ne
puisse pas contourner la garde. La lecture est plafonnée à ``MAX_FETCH_BYTES``
octets **avant et après** décompression (gzip/deflate décodés par tranches
bornées), l'ensemble du téléchargement est soumis à une échéance globale
(``web.timeout × FETCH_TIME_BUDGET_FACTOR``), puis le texte est tronqué à
``web.fetch_max_chars`` caractères.
"""
from __future__ import annotations

import contextlib
import ipaddress
import re
import socket
import time
import zlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from ..config import Settings
from ..utils import is_probably_binary, truncate
from .base import Tool, ToolContext, ToolError, ToolResult

# Nombre maximal d'octets lus pour une page (5 Mo).
MAX_FETCH_BYTES = 5_000_000
# Nombre maximal de redirections suivies (chaque saut repasse par la garde).
MAX_REDIRECTS = 10
# Échéance globale d'un téléchargement (redirections et lecture comprises),
# en multiples de ``web.timeout`` qui, lui, ne borne que chaque opération.
FETCH_TIME_BUDGET_FACTOR = 3
# Taille des tranches lues sur le réseau (octets compressés).
_RAW_CHUNK = 65_536
# Encodages de contenu décodés ici, par tranches bornées.
_SUPPORTED_ENCODINGS = frozenset({"", "identity", "gzip", "x-gzip", "deflate"})
# Plafond absolu du nombre de résultats de recherche.
MAX_SEARCH_RESULTS = 50

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_TEXT_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/csv",
        "text/xml",
        "application/json",
        "application/xml",
        "application/javascript",
        "text/javascript",
        "application/x-yaml",
        "application/yaml",
        "text/yaml",
    }
)
_PDF_TYPES = frozenset({"application/pdf", "application/x-pdf"})
_STRIP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe", "svg")
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


@dataclass
class SearchResult:
    """Un résultat de recherche web."""

    title: str
    url: str
    snippet: str


@dataclass
class PageContent:
    """Contenu textuel d'une page récupérée."""

    url: str  # URL finale (après redirections)
    title: str
    text: str
    content_type: str
    status: int


# ---------------------------------------------------------------- utilitaires
def _own_client(settings: Settings) -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=settings.web.timeout,
        headers={"User-Agent": settings.web.user_agent},
    )


def _request_headers(settings: Settings, **extra: str) -> dict[str, str]:
    headers = {"User-Agent": settings.web.user_agent}
    headers.update(extra)
    return headers


def _clamp_results(settings: Settings, max_results: int | None) -> int:
    n = settings.web.max_results if max_results is None else int(max_results)
    return max(1, min(n, MAX_SEARCH_RESULTS))


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _is_internal_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # IPv6 encapsulant une IPv4 (::ffff:10.0.0.1) : on juge l'adresse IPv4.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        return _is_internal_ip(mapped)
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified:
        return True
    if ip.is_multicast or ip.is_reserved:
        return True
    # Tout ce qui n'est pas routable publiquement : CGNAT 100.64.0.0/10
    # (RFC 6598, Tailscale…), 192.0.0.0/24, 198.18.0.0/15, 240.0.0.0/4…
    if not ip.is_global:
        return True
    return False


def resolve_host(host: str) -> list[str]:
    """Résout un nom d'hôte en adresses IP (chaînes) via ``socket.getaddrinfo``."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError) as e:
        raise ToolError(f"hôte introuvable : {host} ({e})") from e
    addresses: list[str] = []
    for info in infos:
        sockaddr = info[4] if len(info) > 4 else None
        if sockaddr:
            addresses.append(str(sockaddr[0]))
    if not addresses:
        raise ToolError(f"hôte introuvable : {host}")
    return addresses


def check_url(url: str) -> str:
    """Valide une URL fournie par le modèle et renvoie sa forme normalisée.

    Lève ``ToolError`` si le schéma n'est pas http/https, si l'hôte manque ou
    s'il désigne une adresse interne (garde anti-SSRF).
    """
    return _validate_url(url)[0]


def _validate_url(url: str) -> tuple[str, str, list[str]]:
    """Comme ``check_url`` mais renvoie ``(url, hôte, adresses IP validées)``
    afin que la connexion puisse être épinglée sur une adresse vérifiée."""
    if not isinstance(url, str) or not url.strip():
        raise ToolError("URL vide")
    url = url.strip()
    if re.search(r"[\s\x00-\x1f]", url):
        raise ToolError("URL invalide : caractères de contrôle ou espaces")
    try:
        parts = urlsplit(url)
    except ValueError as e:
        raise ToolError(f"URL invalide : {e}") from e
    scheme = (parts.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ToolError(f"schéma non autorisé : seuls http et https sont acceptés ({url})")
    try:
        host = parts.hostname
    except ValueError as e:
        raise ToolError(f"URL invalide : {e}") from e
    if not host:
        raise ToolError(f"URL sans hôte : {url}")
    host = host.strip("[]").lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ToolError(f"adresse interne refusée : {host}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        candidates = resolve_host(host)
    else:
        candidates = [str(ip)]
    addresses: list[str] = []
    for candidate in candidates:
        try:
            resolved = ipaddress.ip_address(candidate.split("%", 1)[0])
        except ValueError:
            raise ToolError(f"adresse interne refusée : {host} ({candidate})") from None
        if _is_internal_ip(resolved):
            raise ToolError(f"adresse interne refusée : {host} ({resolved})")
        if str(resolved) not in addresses:
            addresses.append(str(resolved))
    return url, host, addresses


def _pinned_url(url: str, ip: str) -> str:
    """URL de connexion où l'hôte est remplacé par l'adresse IP validée
    (port et identifiants conservés, IPv6 entre crochets, fragment retiré)."""
    parts = urlsplit(url)
    literal = f"[{ip}]" if ":" in ip else ip
    netloc = literal
    if parts.port is not None:
        netloc += f":{parts.port}"
    if parts.username is not None:
        creds = parts.username if parts.password is None else f"{parts.username}:{parts.password}"
        netloc = f"{creds}@{netloc}"
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))


def _host_header(url: str) -> str:
    """Valeur de l'en-tête ``Host`` pour l'URL nominale (``hôte[:port]``)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{parts.port}" if parts.port is not None else host


def _normalize_lines(text: str) -> str:
    """Nettoie chaque ligne et réduit les lignes vides consécutives à une seule."""
    lines: list[str] = []
    blank = False
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            if lines and not blank:
                lines.append("")
            blank = True
            continue
        lines.append(line)
        blank = False
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def html_to_text(html: str | bytes) -> tuple[str, str]:
    """Extrait ``(titre, texte)`` d'un document HTML (sans scripts, styles,
    navigation, entêtes, pieds de page, formulaires…)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title is not None:
        title = _clean_str(soup.title.get_text())
    for tag in soup.find_all(_STRIP_TAGS + ("title",)):
        tag.decompose()
    text = soup.get_text("\n")
    return title, _normalize_lines(text)


def pdf_to_text(data: bytes) -> str:
    """Texte des pages d'un PDF, concaténées (pypdf)."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(data))
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 — page illisible : on continue
                pages.append("")
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"PDF illisible : {type(e).__name__}: {e}") from e
    return _normalize_lines("\n\n".join(pages))


def _split_content_type(value: str) -> tuple[str, str | None]:
    mime, _, rest = value.partition(";")
    charset = None
    for param in rest.split(";"):
        key, _, val = param.strip().partition("=")
        if key.strip().lower() == "charset" and val:
            charset = val.strip().strip('"').strip("'")
    return mime.strip().lower(), charset


def _decode(data: bytes, charset: str | None) -> str:
    for enc in (charset, "utf-8"):
        if not enc:
            continue
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _sniff_type(data: bytes) -> str:
    head = data[:2048]
    if head.startswith(b"%PDF"):
        return "application/pdf"
    lowered = head.lstrip().lower()
    if lowered.startswith(b"<!doctype html") or b"<html" in lowered or b"<head" in lowered or b"<body" in lowered:
        return "text/html"
    if not is_probably_binary(head):
        return "text/plain"
    return "application/octet-stream"


def _content_encoding(response: httpx.Response) -> str:
    return response.headers.get("content-encoding", "").strip().lower()


class _BoundedDecoder:
    """Décodeur gzip/deflate incrémental dont la sortie est bornée à chaque
    appel (``max_length``) : un fragment réseau très compressible ne peut pas
    faire allouer plus que le plafond restant."""

    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self._raw_deflate_tried = False
        if encoding in ("gzip", "x-gzip"):
            self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        else:
            self._obj = zlib.decompressobj()
        self.pending = b""  # entrée compressée non encore consommée

    def decompress(self, data: bytes, max_length: int) -> bytes:
        data = self.pending + data
        try:
            out = self._obj.decompress(data, max_length)
        except zlib.error:
            # deflate « brut » (sans en-tête zlib) : on retente une fois.
            if self.encoding != "deflate" or self._raw_deflate_tried:
                raise
            self._raw_deflate_tried = True
            self._obj = zlib.decompressobj(-zlib.MAX_WBITS)
            out = self._obj.decompress(data, max_length)
        self.pending = self._obj.unconsumed_tail
        return out


def _raw_chunks(response: httpx.Response):
    """Octets **bruts** (non décodés) de la réponse, tranche réseau par
    tranche réseau (au plus 64 Ko chez httpcore), sans regroupement — sinon
    l'échéance globale ne serait vérifiée qu'une fois 64 Ko accumulés. Une
    réponse dont le corps est déjà chargé en mémoire
    (``httpx.Response(content=...)``, cas des doubles de test) est servie
    depuis ce contenu par tranches de ``_RAW_CHUNK`` ; httpx l'a alors déjà
    décodé, voir ``_preloaded``."""
    if _preloaded(response):
        return response.iter_bytes(_RAW_CHUNK)
    return response.iter_raw()


def _preloaded(response: httpx.Response) -> bool:
    """Vrai si le corps a été chargé (et décodé) par httpx à la construction."""
    return bool(response.is_stream_consumed)


def _check_deadline(deadline: float | None, budget: float) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise ToolError(f"récupération impossible : délai global dépassé ({budget:g}s)")


def _read_body(response: httpx.Response, limit: int, *, deadline: float | None = None,
               budget: float = 0.0) -> tuple[bytes, bool]:
    """Lit au plus ``limit`` octets (compressés **et** décompressés) ; renvoie
    ``(données, tronqué)``. Vérifie l'échéance globale à chaque tranche."""
    encoding = "" if _preloaded(response) else _content_encoding(response)
    if encoding not in _SUPPORTED_ENCODINGS:
        raise ToolError(f"encodage de contenu non pris en charge : {encoding}")
    decoder = _BoundedDecoder(encoding) if encoding in ("gzip", "x-gzip", "deflate") else None
    buf = bytearray()
    raw_total = 0
    truncated = False
    chunks = iter(_raw_chunks(response))
    for chunk in chunks:
        _check_deadline(deadline, budget)
        if not chunk:
            continue
        raw_total += len(chunk)
        remaining = limit - len(buf)
        if decoder is not None:
            try:
                # On demande un octet de plus que le reste pour détecter le débordement.
                data = decoder.decompress(chunk, remaining + 1)
            except zlib.error as e:
                raise ToolError(f"récupération impossible : contenu {encoding} corrompu ({e})") from e
        else:
            data = chunk
        if len(data) <= remaining and raw_total <= limit and not (decoder is not None and decoder.pending):
            buf += data
            continue
        buf += data[:remaining]
        # Plafond atteint (sortie, entrée compressée ou reliquat non consommé) :
        # tronqué si des données débordent ou s'il en reste à lire.
        truncated = (
            len(data) > remaining
            or bool(decoder is not None and decoder.pending)
            or raw_total > limit
            or any(c for c in chunks)
        )
        break
    return bytes(buf), truncated


# ------------------------------------------------------------------ recherche
def _search_duckduckgo(query: str, n: int) -> list[SearchResult]:
    try:
        from ddgs import DDGS  # import paresseux : dépendance optionnelle au chargement

        raw = DDGS().text(query, max_results=n)
    except Exception as e:  # noqa: BLE001 — réseau, quota, parseur…
        raise ToolError(f"recherche indisponible : {type(e).__name__}: {e}") from e
    results: list[SearchResult] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        url = _clean_str(item.get("href") or item.get("url"))
        if not url:
            continue
        results.append(SearchResult(_clean_str(item.get("title")), url, _clean_str(item.get("body"))))
        if len(results) >= n:
            break
    return results


def _get_json(client: httpx.Client, url: str, *, params: dict[str, Any], headers: dict[str, str],
              timeout: float) -> Any:
    try:
        response = client.get(url, params=params, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        raise ToolError(f"recherche indisponible : {type(e).__name__}: {e}") from e
    if response.status_code >= 400:
        raise ToolError(f"recherche indisponible : HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as e:
        raise ToolError("recherche indisponible : réponse JSON invalide") from e


def _search_searxng(settings: Settings, query: str, n: int, client: httpx.Client) -> list[SearchResult]:
    base = (settings.web.searxng_url or "").strip()
    if not base:
        raise ToolError("recherche indisponible : web.searxng_url n'est pas configuré")
    language = (settings.agent.language or "fr").strip() or "fr"
    payload = _get_json(
        client,
        base.rstrip("/") + "/search",
        params={"q": query, "format": "json", "language": language},
        headers=_request_headers(settings, Accept="application/json"),
        timeout=settings.web.timeout,
    )
    items = payload.get("results") if isinstance(payload, dict) else None
    results: list[SearchResult] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        url = _clean_str(item.get("url"))
        if not url:
            continue
        results.append(SearchResult(_clean_str(item.get("title")), url, _clean_str(item.get("content"))))
        if len(results) >= n:
            break
    return results


def _search_brave(settings: Settings, query: str, n: int, client: httpx.Client) -> list[SearchResult]:
    key = (settings.web.brave_api_key or "").strip()
    if not key:
        raise ToolError("recherche indisponible : web.brave_api_key n'est pas configuré")
    payload = _get_json(
        client,
        BRAVE_SEARCH_URL,
        params={"q": query, "count": n},
        headers=_request_headers(settings, Accept="application/json", **{"X-Subscription-Token": key}),
        timeout=settings.web.timeout,
    )
    web = payload.get("web") if isinstance(payload, dict) else None
    items = web.get("results") if isinstance(web, dict) else None
    results: list[SearchResult] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        url = _clean_str(item.get("url"))
        if not url:
            continue
        results.append(SearchResult(_clean_str(item.get("title")), url, _clean_str(item.get("description"))))
        if len(results) >= n:
            break
    return results


def web_search(
    settings: Settings,
    query: str,
    max_results: int | None = None,
    *,
    client: httpx.Client | None = None,
) -> list[SearchResult]:
    """Recherche web selon ``settings.web.provider`` (duckduckgo, searxng, brave).

    ``client`` : client httpx injectable (tests) ; sinon un client temporaire
    est créé avec le délai et le User-Agent configurés.
    """
    if not isinstance(query, str) or not query.strip():
        raise ToolError("requête de recherche vide")
    query = " ".join(query.split())
    n = _clamp_results(settings, max_results)
    provider = settings.web.provider
    if provider == "duckduckgo":
        return _search_duckduckgo(query, n)
    if provider not in ("searxng", "brave"):
        raise ToolError(f"fournisseur de recherche inconnu : {provider}")
    if client is not None:
        return _search_searxng(settings, query, n, client) if provider == "searxng" else _search_brave(settings, query, n, client)
    with _own_client(settings) as own:
        return _search_searxng(settings, query, n, own) if provider == "searxng" else _search_brave(settings, query, n, own)


# ---------------------------------------------------------------- lecture
def _fetch_raw(settings: Settings, url: str, client: httpx.Client) -> tuple[httpx.Response, bytes, bool, str]:
    """GET avec redirections manuelles (garde anti-SSRF à chaque saut),
    connexion épinglée sur l'adresse validée, lecture plafonnée et échéance
    globale. Renvoie ``(réponse finale, corps, tronqué, URL nominale finale)``."""
    per_op = float(settings.web.timeout)
    budget = per_op * FETCH_TIME_BUDGET_FACTOR
    deadline = time.monotonic() + budget
    current, host, addresses = _validate_url(url)
    base_headers = _request_headers(
        settings,
        Accept="text/html,application/xhtml+xml,text/plain,application/pdf,*/*;q=0.5",
        **{"Accept-Encoding": "gzip, deflate"},
    )
    for _ in range(MAX_REDIRECTS + 1):
        _check_deadline(deadline, budget)
        headers = dict(base_headers, Host=_host_header(current))
        extensions = {"sni_hostname": host}
        try:
            for i, ip in enumerate(addresses):
                op_timeout = min(per_op, max(0.01, deadline - time.monotonic()))
                request = client.build_request("GET", _pinned_url(current, ip), headers=headers,
                                               extensions=extensions, timeout=op_timeout)
                try:
                    response = client.send(request, stream=True, follow_redirects=False)
                except httpx.ConnectError:
                    # Adresse injoignable : on essaie la suivante (toutes sont validées).
                    if i == len(addresses) - 1:
                        raise
                    continue
                break
            with contextlib.closing(response):
                if response.status_code in _REDIRECT_CODES and response.headers.get("location"):
                    target = urljoin(current, response.headers["location"])
                    current, host, addresses = _validate_url(target)
                    continue
                if response.status_code >= 400:
                    raise ToolError(f"HTTP {response.status_code} pour {current}")
                body, truncated = _read_body(response, MAX_FETCH_BYTES, deadline=deadline, budget=budget)
                return response, body, truncated, current
        except ToolError:
            raise
        except httpx.HTTPError as e:
            raise ToolError(f"récupération impossible : {type(e).__name__}: {e}") from e
    raise ToolError(f"trop de redirections pour {url}")


def fetch_page(
    settings: Settings,
    url: str,
    *,
    max_chars: int | None = None,
    client: httpx.Client | None = None,
) -> PageContent:
    """Récupère une page http/https et renvoie son contenu textuel.

    HTML → texte nettoyé (titre = ``<title>``) ; texte brut / JSON / Markdown →
    tel quel ; PDF → texte des pages (pypdf). Autre type ⇒ ``ToolError``.
    Le texte est tronqué à ``max_chars`` (défaut ``web.fetch_max_chars``).
    """
    if max_chars is None:
        max_chars = settings.web.fetch_max_chars
    max_chars = max(1, int(max_chars))

    if client is not None:
        response, body, truncated_bytes, final_url = _fetch_raw(settings, url, client)
    else:
        with _own_client(settings) as own:
            response, body, truncated_bytes, final_url = _fetch_raw(settings, url, own)

    mime, charset = _split_content_type(response.headers.get("content-type", ""))
    if not mime or mime == "application/octet-stream":
        mime = _sniff_type(body)

    title = ""
    if mime in _HTML_TYPES:
        title, text = html_to_text(_decode(body, charset) if charset else body)
    elif mime in _PDF_TYPES:
        text = pdf_to_text(body)
    elif mime in _TEXT_TYPES or mime.startswith("text/"):
        text = _normalize_lines(_decode(body, charset))
    else:
        raise ToolError(f"type non pris en charge : {mime} ({final_url})")

    text = truncate(text, max_chars)
    if truncated_bytes:
        # Après la troncature en caractères, pour que le marqueur survive.
        text += f"\n… [lecture arrêtée à {MAX_FETCH_BYTES} octets]"
    return PageContent(
        url=final_url,
        title=title,
        text=text,
        content_type=mime,
        status=response.status_code,
    )


# ------------------------------------------------------------------- outils
def format_search_results(results: list[SearchResult], query: str) -> str:
    if not results:
        return f"aucun résultat pour : {query}"
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        head = f"{i}. {r.title or '(sans titre)'} — {r.url}"
        lines.append(f"{head}\n  {r.snippet}" if r.snippet else head)
    return "\n".join(lines)


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Recherche sur le web (DuckDuckGo, SearXNG ou Brave selon la configuration) "
        "et renvoie une liste numérotée de résultats : titre, URL et extrait. "
        "Utilisez ensuite fetch_url pour lire une page en entier."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 1000, "description": "Termes de recherche."},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SEARCH_RESULTS,
                "description": "Nombre maximal de résultats (défaut : configuration web.max_results).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args.get("query", ""))
        max_results = args.get("max_results")
        results = web_search(ctx.settings, query, max_results)
        return ToolResult(
            format_search_results(results, query),
            data=[{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results],
        )


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "Télécharge une page web (http/https uniquement, jamais d'adresse interne) "
        "et renvoie son texte : HTML nettoyé, texte brut, JSON, Markdown ou PDF. "
        "La sortie est tronquée à max_chars caractères."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "minLength": 1, "maxLength": 4000, "description": "URL http(s) à lire."},
            "max_chars": {
                "type": "integer",
                "minimum": 1,
                "description": "Nombre maximal de caractères renvoyés (défaut : web.fetch_max_chars).",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = str(args.get("url", ""))
        max_chars = args.get("max_chars")
        page = fetch_page(ctx.settings, url, max_chars=max_chars)
        content = f"Titre : {page.title or '(sans titre)'}\nURL : {page.url}\n\n{page.text}"
        return ToolResult(
            content,
            data={"url": page.url, "title": page.title, "content_type": page.content_type, "status": page.status},
        )


def tools(settings: Settings) -> list[Tool]:
    return [WebSearchTool(), FetchUrlTool()]


__all__ = [
    "FETCH_TIME_BUDGET_FACTOR",
    "MAX_FETCH_BYTES",
    "MAX_REDIRECTS",
    "MAX_SEARCH_RESULTS",
    "FetchUrlTool",
    "PageContent",
    "SearchResult",
    "WebSearchTool",
    "check_url",
    "fetch_page",
    "format_search_results",
    "html_to_text",
    "pdf_to_text",
    "resolve_host",
    "tools",
    "web_search",
]
