"""Extraction du texte des sources à ingérer : fichiers, dossiers, URLs.

- ``extract(path, settings)`` : ``(titre, texte)`` ou ``None`` si le fichier
  est trop gros, binaire ou sans texte. PDF via pypdf, HTML via BeautifulSoup,
  le reste lu en UTF-8 (``errors="replace"``) après détection binaire.
- ``iter_files(root, recursive, settings)`` : parcours sans suivre les liens
  symboliques, motifs ``kb.ignore_patterns`` appliqués à chaque composant.
- ``extract_url(url, settings)`` : via ``dhaos.tools.web.fetch_page`` (import
  paresseux, module d'un autre propriétaire ; ``fetch_page`` de ce module est
  l'indirection à monkeypatcher dans les tests).
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..config import Settings
from ..utils import is_probably_binary

SAMPLE_BYTES = 8192
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi", ".pyx", ".ipynb", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".rs",
    ".java", ".kt", ".kts", ".scala", ".groovy", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh",
    ".hxx", ".cs", ".rb", ".php", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".sql",
    ".yaml", ".yml", ".toml", ".json", ".jsonc", ".json5", ".html", ".htm", ".xhtml", ".css",
    ".scss", ".sass", ".less", ".xml", ".xsl", ".svg", ".lua", ".pl", ".pm", ".r", ".swift", ".m",
    ".mm", ".dart", ".ex", ".exs", ".erl", ".hs", ".ml", ".mli", ".clj", ".cljs", ".vue", ".svelte",
    ".tf", ".hcl", ".mk", ".cmake", ".gradle", ".proto", ".graphql", ".gql", ".ini", ".cfg", ".conf",
    ".properties", ".dockerfile", ".nix", ".zig", ".v", ".vhd", ".asm", ".s", ".el", ".lisp", ".scm",
    ".ninja", ".bzl", ".vim",
})
CODE_FILENAMES: frozenset[str] = frozenset({
    "Dockerfile", "Makefile", "GNUmakefile", "CMakeLists.txt", "Jenkinsfile", "Vagrantfile",
    "Rakefile", "Gemfile", "Justfile", "justfile", "BUILD", "WORKSPACE", "meson.build",
})
# Formats convertis en prose lors de l'extraction : découpés comme du texte.
_PROSE_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".html", ".htm", ".xhtml"})


# ----------------------------------------------------------------- détection
def is_url(source: str | Path) -> bool:
    return bool(_URL_RE.match(str(source).strip()))


def document_kind(path: str | Path) -> str:
    """``"code"`` ou ``"text"`` selon l'extension (ou le nom) du fichier."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in _PROSE_EXTENSIONS:
        return "text"
    if suffix in CODE_EXTENSIONS or p.name in CODE_FILENAMES:
        return "code"
    return "text"


def is_ignored(path: str | Path, patterns: Iterable[str]) -> bool:
    """``True`` si un composant du chemin correspond à un motif (fnmatch)."""
    pats = [p for p in patterns if p]
    if not pats:
        return False
    for part in Path(path).parts:
        for pat in pats:
            if fnmatch.fnmatchcase(part, pat):
                return True
    return False


def iter_files(root: str | Path, recursive: bool, settings: Settings) -> Iterator[Path]:
    """Fichiers réguliers sous ``root`` (triés), sans suivre les liens symboliques,
    en écartant les motifs ``kb.ignore_patterns`` (relatifs à ``root``)."""
    root_path = Path(root)
    patterns = list(settings.kb.ignore_patterns)
    if not root_path.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=True, followlinks=False):
        current = Path(dirpath)
        kept: list[str] = []
        for d in sorted(dirnames):
            full = current / d
            if full.is_symlink() or is_ignored(d, patterns):
                continue
            kept.append(d)
        dirnames[:] = kept
        for name in sorted(filenames):
            full = current / name
            if full.is_symlink() or is_ignored(name, patterns):
                continue
            if full.is_file():
                yield full
        if not recursive:
            break


# ---------------------------------------------------------------- extraction
def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(raw: bytes | str) -> tuple[str, str]:
    """``(titre, texte)`` d'un document HTML (scripts/styles écartés)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title is not None else ""
    return " ".join(title.split()), _normalize_text(soup.get_text("\n"))


def pdf_to_text(path: Path) -> tuple[str, str]:
    """``(titre, texte)`` d'un PDF via pypdf (titre des métadonnées si présent)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001
            pass
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 — page illisible : on continue
            continue
    title = ""
    try:
        meta = reader.metadata
        if meta is not None and meta.title:
            title = " ".join(str(meta.title).split())
    except Exception:  # noqa: BLE001
        title = ""
    return title, _normalize_text("\n\n".join(pages))


def extract(path: str | Path, settings: Settings) -> tuple[str, str] | None:
    """``(titre, texte)`` d'un fichier, ou ``None`` s'il n'y a rien à indexer
    (trop gros, binaire, texte vide). Lève ``OSError`` si illisible."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"fichier introuvable : {p}")
    size = p.stat().st_size
    if size > int(settings.kb.max_file_bytes):
        return None
    suffix = p.suffix.lower()
    title = p.name
    if suffix == ".pdf":
        meta_title, text = pdf_to_text(p)
        title = meta_title or title
    elif suffix in (".html", ".htm", ".xhtml"):
        raw = p.read_bytes()
        if is_probably_binary(raw[:SAMPLE_BYTES]):
            return None
        html_title, text = html_to_text(raw)
        title = html_title or title
    else:
        with open(p, "rb") as f:
            sample = f.read(SAMPLE_BYTES)
            if is_probably_binary(sample):
                return None
            data = sample + f.read()
        text = data.decode("utf-8", errors="replace")
    if not text.strip():
        return None
    return title, text


# ----------------------------------------------------------------------- URL
def fetch_page(settings: Settings, url: str, *, max_chars: int | None = None, client: Any = None) -> Any:
    """Indirection paresseuse vers ``dhaos.tools.web.fetch_page`` (monkeypatchable)."""
    from ..tools.web import fetch_page as _impl

    return _impl(settings, url, max_chars=max_chars, client=client)


def extract_url(url: str, settings: Settings) -> tuple[str, str]:
    """``(titre, texte)`` d'une page http/https. Lève ``ValueError`` si l'URL
    est invalide ; les erreurs réseau remontent telles quelles."""
    url = str(url).strip()
    if not is_url(url):
        raise ValueError(f"URL non prise en charge (http/https attendu) : {url}")
    page = fetch_page(settings, url, max_chars=int(settings.kb.max_file_bytes))
    title = " ".join(str(getattr(page, "title", "") or "").split()) or url
    text = str(getattr(page, "text", "") or "")
    return title, text


__all__ = [
    "CODE_EXTENSIONS",
    "CODE_FILENAMES",
    "document_kind",
    "extract",
    "extract_url",
    "fetch_page",
    "html_to_text",
    "is_ignored",
    "is_url",
    "iter_files",
    "pdf_to_text",
]
