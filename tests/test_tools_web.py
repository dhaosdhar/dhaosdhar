"""Tests des outils web : recherche (3 fournisseurs) et lecture de pages,
sans réseau (httpx.MockTransport, DDGS factice, DNS simulé)."""
from __future__ import annotations

import socket
from typing import Any, Callable

import httpx
import pytest

from dhaos.config import Settings
from dhaos.tools import web
from dhaos.tools.base import ToolContext, ToolError, ToolRegistry
from dhaos.tools.web import PageContent, SearchResult, fetch_page, web_search

from .fakes import tool_call

PUBLIC_IP = "93.184.216.34"


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Résolution DNS simulée : tout nom d'hôte → une adresse publique
    (modifiable via la fonction renvoyée)."""
    state = {"ip": PUBLIC_IP}

    def fake_getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (state["ip"], 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    def set_ip(ip: str) -> None:
        state["ip"] = ip

    return set_ip


@pytest.fixture
def registry(settings: Settings) -> ToolRegistry:
    return ToolRegistry(web.tools(settings))


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def make_pdf(text: str) -> bytes:
    """PDF minimal (une page, une ligne de texte) lisible par pypdf."""
    content = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


HTML_PAGE = """<!DOCTYPE html>
<html><head><title>  Page de   test </title>
<style>body { color: red }</style>
<script>console.log("JS_SECRET");</script>
</head>
<body>
<nav>Menu NAV_TEXT</nav>
<header>HEADER_TEXT</header>
<main>
  <h1>Bonjour</h1>


  <p>Premier   paragraphe.</p>

  <p>Second paragraphe.</p>
  <form><input name="x"/>FORM_TEXT</form>
  <iframe src="x">IFRAME_TEXT</iframe>
  <svg><text>SVG_TEXT</text></svg>
  <noscript>NOSCRIPT_TEXT</noscript>
</main>
<aside>ASIDE_TEXT</aside>
<footer>FOOTER_TEXT</footer>
</body></html>"""


# ------------------------------------------------------------ web_search
class TestSearxng:
    def test_results(self, settings: Settings) -> None:
        settings.web.provider = "searxng"
        settings.web.searxng_url = "http://searx.example/"
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["params"] = dict(request.url.params)
            seen["ua"] = request.headers.get("user-agent")
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "Un", "url": "https://a.example/1", "content": "extrait  un"},
                        {"title": "Deux", "url": "https://a.example/2", "content": "extrait deux"},
                        {"title": "Sans URL", "content": "ignoré"},
                        "pas un dict",
                    ]
                },
            )

        with make_client(handler) as client:
            results = web_search(settings, "  python   asyncio ", client=client)

        assert seen["url"].startswith("http://searx.example/search?")
        assert seen["params"]["q"] == "python asyncio"
        assert seen["params"]["format"] == "json"
        assert seen["params"]["language"] == settings.agent.language
        assert seen["ua"] == settings.web.user_agent
        assert results == [
            SearchResult("Un", "https://a.example/1", "extrait un"),
            SearchResult("Deux", "https://a.example/2", "extrait deux"),
        ]

    def test_max_results_clamped(self, settings: Settings) -> None:
        settings.web.provider = "searxng"
        settings.web.searxng_url = "http://searx.example"
        payload = {"results": [{"title": f"t{i}", "url": f"https://a.example/{i}", "content": ""} for i in range(20)]}
        with make_client(lambda r: httpx.Response(200, json=payload)) as client:
            assert len(web_search(settings, "q", 3, client=client)) == 3
            assert len(web_search(settings, "q", client=client)) == settings.web.max_results
            assert len(web_search(settings, "q", 0, client=client)) == 1

    def test_missing_url(self, settings: Settings) -> None:
        settings.web.provider = "searxng"
        settings.web.searxng_url = None
        with pytest.raises(ToolError, match="searxng_url"):
            web_search(settings, "q", client=make_client(lambda r: httpx.Response(200, json={})))

    def test_http_error(self, settings: Settings) -> None:
        settings.web.provider = "searxng"
        settings.web.searxng_url = "http://searx.example"
        with make_client(lambda r: httpx.Response(503, text="down")) as client:
            with pytest.raises(ToolError, match="recherche indisponible : HTTP 503"):
                web_search(settings, "q", client=client)

    def test_invalid_json(self, settings: Settings) -> None:
        settings.web.provider = "searxng"
        settings.web.searxng_url = "http://searx.example"
        with make_client(lambda r: httpx.Response(200, text="<html>")) as client:
            with pytest.raises(ToolError, match="JSON invalide"):
                web_search(settings, "q", client=client)

    def test_transport_error(self, settings: Settings) -> None:
        settings.web.provider = "searxng"
        settings.web.searxng_url = "http://searx.example"

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refus", request=request)

        with make_client(handler) as client:
            with pytest.raises(ToolError, match="recherche indisponible"):
                web_search(settings, "q", client=client)


class TestBrave:
    def test_results(self, settings: Settings) -> None:
        settings.web.provider = "brave"
        settings.web.brave_api_key = "brave-key"
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url).split("?")[0]
            seen["params"] = dict(request.url.params)
            seen["token"] = request.headers.get("x-subscription-token")
            return httpx.Response(
                200,
                json={"web": {"results": [{"title": "B1", "url": "https://b.example/1", "description": "desc"}]}},
            )

        with make_client(handler) as client:
            results = web_search(settings, "rust", 5, client=client)
        assert seen["url"] == web.BRAVE_SEARCH_URL
        assert seen["params"] == {"q": "rust", "count": "5"}
        assert seen["token"] == "brave-key"
        assert results == [SearchResult("B1", "https://b.example/1", "desc")]

    def test_missing_key(self, settings: Settings) -> None:
        settings.web.provider = "brave"
        settings.web.brave_api_key = None
        with pytest.raises(ToolError, match="brave_api_key"):
            web_search(settings, "q", client=make_client(lambda r: httpx.Response(200, json={})))

    def test_unexpected_payload(self, settings: Settings) -> None:
        settings.web.provider = "brave"
        settings.web.brave_api_key = "k"
        with make_client(lambda r: httpx.Response(200, json={"web": "?"})) as client:
            assert web_search(settings, "q", client=client) == []


class TestDuckDuckGo:
    def test_results(self, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        import ddgs

        calls: list[tuple[str, int]] = []

        class FakeDDGS:
            def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
                calls.append((query, kwargs["max_results"]))
                return [
                    {"title": "D1", "href": "https://d.example/1", "body": "corps 1"},
                    {"title": "D2", "href": "https://d.example/2", "body": "corps 2"},
                ]

        monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
        settings.web.provider = "duckduckgo"
        results = web_search(settings, "python", 2)
        assert calls == [("python", 2)]
        assert results == [
            SearchResult("D1", "https://d.example/1", "corps 1"),
            SearchResult("D2", "https://d.example/2", "corps 2"),
        ]

    def test_failure_is_tool_error(self, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        import ddgs

        class BrokenDDGS:
            def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
                raise RuntimeError("ratelimit")

        monkeypatch.setattr(ddgs, "DDGS", BrokenDDGS)
        settings.web.provider = "duckduckgo"
        with pytest.raises(ToolError, match="recherche indisponible : RuntimeError: ratelimit"):
            web_search(settings, "python")


def test_search_rejects_empty_query(settings: Settings) -> None:
    with pytest.raises(ToolError, match="vide"):
        web_search(settings, "   ")


def test_search_unknown_provider(settings: Settings) -> None:
    settings.web.provider = "bing"  # type: ignore[assignment] — entrée corrompue
    with pytest.raises(ToolError, match="inconnu"):
        web_search(settings, "q")


# ------------------------------------------------------------- fetch_page
class TestFetchPage:
    def test_html(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["ua"] = request.headers.get("user-agent")
            return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=HTML_PAGE)

        with make_client(handler) as client:
            page = fetch_page(settings, "https://www.example.com/doc", client=client)

        assert isinstance(page, PageContent)
        assert seen["ua"] == settings.web.user_agent
        assert page.url == "https://www.example.com/doc"
        assert page.title == "Page de test"
        assert page.content_type == "text/html"
        assert page.status == 200
        for forbidden in ("JS_SECRET", "NAV_TEXT", "HEADER_TEXT", "FORM_TEXT", "IFRAME_TEXT",
                          "SVG_TEXT", "NOSCRIPT_TEXT", "ASIDE_TEXT", "FOOTER_TEXT", "color: red"):
            assert forbidden not in page.text
        assert page.text == "Bonjour\n\nPremier paragraphe.\n\nSecond paragraphe."

    def test_plain_text_and_json(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/t.txt":
                return httpx.Response(200, headers={"content-type": "text/plain"}, content="ligne 1\r\n\r\n\r\nligne 2  \n".encode())
            if request.url.path == "/d.json":
                return httpx.Response(200, headers={"content-type": "application/json"}, text='{"a": 1}')
            return httpx.Response(200, headers={"content-type": "text/markdown; charset=latin-1"},
                                  content="# Titre\n\nÉté".encode("latin-1"))

        with make_client(handler) as client:
            txt = fetch_page(settings, "http://files.example/t.txt", client=client)
            js = fetch_page(settings, "http://files.example/d.json", client=client)
            md = fetch_page(settings, "http://files.example/r.md", client=client)
        assert txt.text == "ligne 1\n\nligne 2"
        assert txt.title == ""
        assert txt.content_type == "text/plain"
        assert js.text == '{"a": 1}'
        assert js.content_type == "application/json"
        assert md.text == "# Titre\n\nÉté"

    def test_pdf(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        data = make_pdf("Bonjour PDF")
        with make_client(lambda r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=data)) as client:
            page = fetch_page(settings, "https://docs.example/a.pdf", client=client)
        assert page.content_type == "application/pdf"
        assert "Bonjour PDF" in page.text

    def test_corrupt_pdf(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        with make_client(lambda r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 garbage")) as client:
            with pytest.raises(ToolError, match="PDF illisible"):
                fetch_page(settings, "https://docs.example/a.pdf", client=client)

    def test_sniff_without_content_type(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/h":
                return httpx.Response(200, content=b"<html><title>T</title><body>corps</body></html>")
            if request.url.path == "/p":
                return httpx.Response(200, content=make_pdf("Sniffed"))
            return httpx.Response(200, content=b"juste du texte")

        with make_client(handler) as client:
            assert fetch_page(settings, "https://x.example/h", client=client).title == "T"
            assert "Sniffed" in fetch_page(settings, "https://x.example/p", client=client).text
            assert fetch_page(settings, "https://x.example/t", client=client).text == "juste du texte"

    def test_unsupported_type(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        with make_client(lambda r: httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG")) as client:
            with pytest.raises(ToolError, match="type non pris en charge"):
                fetch_page(settings, "https://img.example/a.png", client=client)

    def test_binary_without_type(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        with make_client(lambda r: httpx.Response(200, content=b"\x00\x01\x02\xff" * 100)) as client:
            with pytest.raises(ToolError, match="type non pris en charge"):
                fetch_page(settings, "https://bin.example/blob", client=client)

    def test_http_error_status(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        with make_client(lambda r: httpx.Response(404, text="nope")) as client:
            with pytest.raises(ToolError, match="HTTP 404"):
                fetch_page(settings, "https://www.example.com/missing", client=client)

    def test_transport_error(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("lent", request=request)

        with make_client(handler) as client:
            with pytest.raises(ToolError, match="récupération impossible"):
                fetch_page(settings, "https://slow.example/", client=client)

    def test_redirect_followed(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        hops: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            hops.append(str(request.url))
            if request.url.path == "/old":
                return httpx.Response(302, headers={"location": "/new"})
            if request.url.path == "/new":
                return httpx.Response(301, headers={"location": "https://other.example/final"})
            return httpx.Response(200, headers={"content-type": "text/plain"}, text="arrivé")

        with make_client(handler) as client:
            page = fetch_page(settings, "https://www.example.com/old", client=client)
        assert hops == ["https://www.example.com/old", "https://www.example.com/new", "https://other.example/final"]
        assert page.url == "https://other.example/final"
        assert page.text == "arrivé"

    def test_redirect_to_internal_refused(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "127.0.0.1":
                return httpx.Response(200, headers={"content-type": "text/plain"}, text="SECRET")
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8080/admin"})

        with make_client(handler) as client:
            with pytest.raises(ToolError, match="adresse interne refusée"):
                fetch_page(settings, "https://www.example.com/", client=client)

    def test_too_many_redirects(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        with make_client(lambda r: httpx.Response(302, headers={"location": "/loop"})) as client:
            with pytest.raises(ToolError, match="trop de redirections"):
                fetch_page(settings, "https://www.example.com/loop", client=client)

    def test_size_limit(self, settings: Settings, public_dns: Callable[[str], None],
                        monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web, "MAX_FETCH_BYTES", 100)
        body = b"a" * 1000
        with make_client(lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, content=body)) as client:
            page = fetch_page(settings, "https://big.example/", client=client)
        assert page.text.startswith("a" * 100)
        assert "a" * 101 not in page.text
        assert "lecture arrêtée à 100 octets" in page.text

    def test_exact_size_not_flagged(self, settings: Settings, public_dns: Callable[[str], None],
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web, "MAX_FETCH_BYTES", 100)
        with make_client(lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, content=b"b" * 100)) as client:
            page = fetch_page(settings, "https://big.example/", client=client)
        assert page.text == "b" * 100

    def test_max_chars_truncation(self, settings: Settings, public_dns: Callable[[str], None]) -> None:
        settings.web.fetch_max_chars = 10
        with make_client(lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, text="x" * 50)) as client:
            default = fetch_page(settings, "https://www.example.com/", client=client)
            explicit = fetch_page(settings, "https://www.example.com/", max_chars=5, client=client)
        assert default.text.startswith("x" * 10) and "tronqué : 40 caractères omis" in default.text
        assert explicit.text.startswith("xxxxx\n") and "45 caractères omis" in explicit.text

    def test_own_client_uses_settings(self, settings: Settings, public_dns: Callable[[str], None],
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        """Sans client injecté, fetch_page construit un httpx.Client configuré."""
        captured: dict[str, Any] = {}
        real_client = httpx.Client

        def fake_client(**kwargs: Any) -> httpx.Client:
            captured.update(kwargs)
            return real_client(transport=httpx.MockTransport(
                lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, text="ok")))

        monkeypatch.setattr(web.httpx, "Client", fake_client)
        page = fetch_page(settings, "https://www.example.com/")
        assert page.text == "ok"
        assert captured["follow_redirects"] is True
        assert captured["timeout"] == settings.web.timeout
        assert captured["headers"]["User-Agent"] == settings.web.user_agent


class TestUrlGuard:
    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "example.com", "javascript:alert(1)", "", "   "])
    def test_bad_scheme(self, settings: Settings, url: str) -> None:
        with pytest.raises(ToolError):
            fetch_page(settings, url, client=make_client(lambda r: httpx.Response(200, text="x")))

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/",
            "http://LOCALHOST:8080/x",
            "http://foo.localhost/",
            "http://127.0.0.1/",
            "http://127.1.2.3/",
            "http://0.0.0.0/",
            "http://10.0.0.1/",
            "http://172.16.5.5/",
            "http://172.31.255.255/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://[::ffff:10.0.0.1]/",
            "http://[fe80::1]/",
        ],
    )
    def test_internal_literal_refused(self, settings: Settings, url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_dns(*args: Any, **kwargs: Any) -> list[Any]:
            raise AssertionError("aucune résolution DNS attendue pour une adresse littérale")

        monkeypatch.setattr(socket, "getaddrinfo", no_dns)
        with pytest.raises(ToolError, match="adresse interne refusée"):
            fetch_page(settings, url, client=make_client(lambda r: httpx.Response(200, text="x")))

    @pytest.mark.parametrize("ip", ["127.0.0.1", "10.1.2.3", "172.20.0.1", "192.168.0.10", "169.254.1.1", "0.0.0.0", "::1", "fd00::1"])
    def test_hostname_resolving_to_internal_refused(self, settings: Settings, public_dns: Callable[[str], None], ip: str) -> None:
        public_dns(ip)
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, text="x")

        with pytest.raises(ToolError, match="adresse interne refusée"):
            fetch_page(settings, "http://internal.corp.example/", client=make_client(handler))
        assert called is False

    def test_dns_failure(self, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing(*args: Any, **kwargs: Any) -> list[Any]:
            raise socket.gaierror(-2, "Name or service not known")

        monkeypatch.setattr(socket, "getaddrinfo", failing)
        with pytest.raises(ToolError, match="hôte introuvable"):
            fetch_page(settings, "https://nope.invalid/", client=make_client(lambda r: httpx.Response(200, text="x")))

    def test_public_hostname_allowed(self, public_dns: Callable[[str], None]) -> None:
        assert web.check_url(" https://www.example.com/a?b=1 ") == "https://www.example.com/a?b=1"


# ---------------------------------------------------------------- outils
class TestTools:
    def test_registry_names(self, registry: ToolRegistry) -> None:
        assert registry.names == ["web_search", "fetch_url"]
        for spec in registry.specs():
            assert spec.parameters["type"] == "object"

    def test_web_search_tool_output(self, settings: Settings, ctx: ToolContext, registry: ToolRegistry,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            web, "web_search",
            lambda s, q, n=None, *, client=None: [
                SearchResult("Titre A", "https://a.example/", "extrait A"),
                SearchResult("", "https://b.example/", ""),
            ],
        )
        result = registry.execute(tool_call("web_search", query="test", max_results=2), ctx)
        assert result.is_error is False
        assert result.content == "1. Titre A — https://a.example/\n  extrait A\n2. (sans titre) — https://b.example/"
        assert result.data[0]["url"] == "https://a.example/"

    def test_web_search_tool_no_results(self, ctx: ToolContext, registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web, "web_search", lambda s, q, n=None, *, client=None: [])
        result = registry.execute(tool_call("web_search", query="rien"), ctx)
        assert result.content == "aucun résultat pour : rien"

    def test_web_search_tool_missing_key_is_clean_error(self, settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
        settings.web.provider = "brave"
        settings.web.brave_api_key = None
        result = registry.execute(tool_call("web_search", query="x"), ctx)
        assert result.is_error is True
        assert "brave_api_key" in result.content

    def test_web_search_tool_invalid_args(self, ctx: ToolContext, registry: ToolRegistry) -> None:
        assert registry.execute(tool_call("web_search"), ctx).is_error is True
        assert registry.execute(tool_call("web_search", query="x", max_results=0), ctx).is_error is True
        assert registry.execute(tool_call("web_search", query="x", extra=1), ctx).is_error is True

    def test_web_search_tool_provider_failure(self, settings: Settings, ctx: ToolContext, registry: ToolRegistry,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        import ddgs

        class BrokenDDGS:
            def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
                raise ConnectionError("hors ligne")

        monkeypatch.setattr(ddgs, "DDGS", BrokenDDGS)
        settings.web.provider = "duckduckgo"
        result = registry.execute(tool_call("web_search", query="x"), ctx)
        assert result.is_error is True
        assert result.content.startswith("recherche indisponible")

    def test_fetch_url_tool_output(self, ctx: ToolContext, registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            web, "fetch_page",
            lambda s, url, *, max_chars=None, client=None: PageContent(url, "Doc", f"corps ({max_chars})", "text/html", 200),
        )
        result = registry.execute(tool_call("fetch_url", url="https://www.example.com/", max_chars=42), ctx)
        assert result.is_error is False
        assert result.content == "Titre : Doc\nURL : https://www.example.com/\n\ncorps (42)"
        assert result.data == {"url": "https://www.example.com/", "title": "Doc", "content_type": "text/html", "status": 200}

    def test_fetch_url_tool_untitled(self, ctx: ToolContext, registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web, "fetch_page", lambda s, url, *, max_chars=None, client=None: PageContent(url, "", "t", "text/plain", 200))
        result = registry.execute(tool_call("fetch_url", url="https://www.example.com/"), ctx)
        assert result.content.startswith("Titre : (sans titre)\n")

    def test_fetch_url_tool_refuses_file_scheme(self, ctx: ToolContext, registry: ToolRegistry) -> None:
        result = registry.execute(tool_call("fetch_url", url="file:///etc/passwd"), ctx)
        assert result.is_error is True
        assert "http" in result.content

    def test_fetch_url_tool_refuses_localhost(self, ctx: ToolContext, registry: ToolRegistry) -> None:
        result = registry.execute(tool_call("fetch_url", url="http://localhost:11434/api/tags"), ctx)
        assert result.is_error is True
        assert "adresse interne refusée" in result.content

    def test_fetch_url_tool_invalid_args(self, ctx: ToolContext, registry: ToolRegistry) -> None:
        assert registry.execute(tool_call("fetch_url"), ctx).is_error is True
        assert registry.execute(tool_call("fetch_url", url="https://www.example.com/", max_chars=-1), ctx).is_error is True

    def test_fetch_url_tool_output_truncated_by_registry(self, settings: Settings, ctx: ToolContext, registry: ToolRegistry,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
        settings.tools.max_output_chars = 60
        monkeypatch.setattr(web, "fetch_page", lambda s, url, *, max_chars=None, client=None: PageContent(url, "T", "y" * 500, "text/plain", 200))
        result = registry.execute(tool_call("fetch_url", url="https://www.example.com/"), ctx)
        assert len(result.content) < 200 and "tronqué" in result.content

    def test_unexpected_exception_never_escapes(self, ctx: ToolContext, registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> PageContent:
            raise ValueError("inattendu")

        monkeypatch.setattr(web, "fetch_page", boom)
        result = registry.execute(tool_call("fetch_url", url="https://www.example.com/"), ctx)
        assert result.is_error is True
        assert result.content == "ValueError: inattendu"
