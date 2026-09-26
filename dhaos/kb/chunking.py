"""Découpage de textes et de code en chunks de taille bornée.

- ``kind="text"`` : unités = paragraphes (séparés par des lignes vides),
  regroupés jusqu'à ``chunk_chars`` ; un paragraphe trop long est scindé sur
  les phrases/lignes puis en dur.
- ``kind="code"`` : unités = blocs délimités par des lignes vides ou des lignes
  ouvrant une définition (``def``, ``class``, ``function``, ``fn``, ``pub fn``,
  ``func``, ``impl``, ``export``, ``#include``, ``@``) ; un bloc trop long est
  scindé sur les lignes puis en dur.

Les unités sont des tranches exactes du texte d'origine : la concaténation des
chunks produits sans chevauchement couvre tout le texte (au blanc près). Chaque
chunk est strippé, jamais vide, et commence par les ``overlap`` derniers
caractères du chunk précédent (coupés sur un blanc).
"""
from __future__ import annotations

import re

_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n\s*")
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+|\n")
_LINE_RE = re.compile(r"\n")
_CODE_BOUNDARY_RE = re.compile(
    r"^\s*(?:@|#include\b|(?:async\s+)?def\b|class\b|function\b|fn\b|pub\s+fn\b|func\b|impl\b|export\b)"
)


def chunk_text(text: str, *, chunk_chars: int, overlap: int, kind: str = "text") -> list[str]:
    """Découpe ``text`` en chunks d'au plus ``chunk_chars`` caractères."""
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunk_chars = max(1, int(chunk_chars))
    overlap = max(0, min(int(overlap), chunk_chars // 2))
    kind = "code" if kind == "code" else "text"

    raw_units = _code_units(text) if kind == "code" else _text_units(text)
    units: list[str] = []
    for unit in raw_units:
        units.extend(_split_long(unit, chunk_chars, kind))
    return _group(units, chunk_chars, overlap)


# ------------------------------------------------------------------- unités
def _text_units(text: str) -> list[str]:
    """Paragraphes, séparateur (lignes vides) rattaché au paragraphe précédent."""
    units: list[str] = []
    pos = 0
    for m in _PARAGRAPH_RE.finditer(text):
        units.append(text[pos : m.end()])
        pos = m.end()
    if pos < len(text):
        units.append(text[pos:])
    return [u for u in units if u.strip()]


def _code_units(text: str) -> list[str]:
    """Blocs de code : coupés sur les lignes vides et avant les lignes de définition."""
    units: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if not line.strip():
            current.append(line)
            units.append("".join(current))
            current = []
        elif current and _CODE_BOUNDARY_RE.match(line):
            units.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        units.append("".join(current))
    return [u for u in units if u.strip()]


def _split_keep(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Scinde après chaque occurrence du motif, en gardant le séparateur."""
    pieces: list[str] = []
    pos = 0
    for m in pattern.finditer(text):
        if m.end() == 0:
            continue
        pieces.append(text[pos : m.end()])
        pos = m.end()
    if pos < len(text):
        pieces.append(text[pos:])
    return pieces


def _split_long(unit: str, chunk_chars: int, kind: str) -> list[str]:
    """Réduit une unité trop longue en morceaux d'au plus ``chunk_chars``."""
    if len(unit) <= chunk_chars:
        return [unit]
    pattern = _LINE_RE if kind == "code" else _SENTENCE_RE
    out: list[str] = []
    for piece in _split_keep(unit, pattern):
        if len(piece) <= chunk_chars:
            out.append(piece)
        else:
            out.extend(piece[i : i + chunk_chars] for i in range(0, len(piece), chunk_chars))
    return [p for p in out if p.strip()]


# ------------------------------------------------------------- regroupement
def _tail(text: str, overlap: int) -> str:
    """Les ``overlap`` derniers caractères de ``text``, coupés après un blanc."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    m = re.search(r"\s", tail)
    if m is not None and m.end() < len(tail):
        tail = tail[m.end() :]
    return tail


def _group(units: list[str], chunk_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    prefix = ""
    cur_len = 0
    for unit in units:
        if current and cur_len + len(unit) > chunk_chars:
            full = prefix + "".join(current)
            chunks.append(full)
            prefix = _tail(full, overlap)
            if len(prefix) + len(unit) > chunk_chars:
                prefix = prefix[len(prefix) + len(unit) - chunk_chars :]
            current = []
            cur_len = len(prefix)
        current.append(unit)
        cur_len += len(unit)
    if current:
        chunks.append(prefix + "".join(current))
    return [c.strip() for c in chunks if c.strip()]


__all__ = ["chunk_text"]
