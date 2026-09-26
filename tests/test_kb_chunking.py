"""Tests du découpage en chunks (texte et code)."""
from __future__ import annotations

import re

import pytest

from dhaos.kb.chunking import chunk_text


def _no_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _paragraphs(n: int, size: int = 80) -> str:
    return "\n\n".join(f"Paragraphe {i} " + ("mot " * (size // 4)).strip() + "." for i in range(n))


@pytest.mark.parametrize("kind", ["text", "code"])
def test_empty_or_blank_gives_no_chunk(kind: str) -> None:
    assert chunk_text("", chunk_chars=100, overlap=10, kind=kind) == []
    assert chunk_text("   \n\n \t", chunk_chars=100, overlap=10, kind=kind) == []


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("  bonjour\n", chunk_chars=100, overlap=10) == ["bonjour"]


@pytest.mark.parametrize("kind", ["text", "code"])
def test_coverage_without_overlap(kind: str) -> None:
    text = _paragraphs(12, 150)
    chunks = chunk_text(text, chunk_chars=300, overlap=0, kind=kind)
    assert len(chunks) > 1
    assert _no_ws("".join(chunks)) == _no_ws(text)


def test_chunks_are_bounded_stripped_and_non_empty() -> None:
    text = _paragraphs(20, 120)
    chunks = chunk_text(text, chunk_chars=250, overlap=40)
    assert chunks
    for c in chunks:
        assert c == c.strip()
        assert c
        assert len(c) <= 250


def test_paragraphs_are_grouped_up_to_chunk_chars() -> None:
    text = "\n\n".join(f"p{i}" for i in range(10))  # 10 paragraphes de 2 caractères
    chunks = chunk_text(text, chunk_chars=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].startswith("p0") and chunks[0].endswith("p9")


def test_overlap_repeats_end_of_previous_chunk() -> None:
    text = _paragraphs(10, 100)
    chunks = chunk_text(text, chunk_chars=260, overlap=60)
    assert len(chunks) >= 3
    overlaps = 0
    for prev, nxt in zip(chunks, chunks[1:]):
        k = max((i for i in range(1, min(len(prev), len(nxt)) + 1) if prev.endswith(nxt[:i])), default=0)
        if k > 0:
            overlaps += 1
    assert overlaps >= len(chunks) - 2
    # toutes les lignes d'origine se retrouvent dans un chunk
    for line in text.split("\n"):
        if line.strip():
            assert any(line.strip() in c for c in chunks)


def test_long_paragraph_is_split_on_sentences_then_hard() -> None:
    sentences = " ".join(f"Phrase numéro {i} qui se termine ici." for i in range(30))
    chunks = chunk_text(sentences, chunk_chars=120, overlap=0)
    assert len(chunks) > 3
    assert all(len(c) <= 120 for c in chunks)
    assert _no_ws("".join(chunks)) == _no_ws(sentences)
    # coupe sur les phrases : les chunks se terminent sur un point (sauf le dernier éventuellement)
    assert sum(c.endswith(".") for c in chunks) >= len(chunks) - 1

    giant = "x" * 1000
    hard = chunk_text(giant, chunk_chars=300, overlap=0)
    assert [len(c) for c in hard] == [300, 300, 300, 100]


def test_code_boundaries_start_new_chunks() -> None:
    code = "\n".join(
        [
            "import os",
            "",
            "def first():",
            "    return 1",
            "",
            "@decorator",
            "def second():",
            "    return 2",
            "class Third:",
            "    def method(self):",
            "        return 3",
            "",
            "export function fourth() {}",
            "#include <stdio.h>",
            "pub fn fifth() {}",
        ]
    )
    chunks = chunk_text(code, chunk_chars=40, overlap=0, kind="code")
    assert _no_ws("".join(chunks)) == _no_ws(code)
    starts = [c.splitlines()[0] for c in chunks]
    # chaque chunk commence à une frontière (ligne de définition / après ligne vide)
    for s in starts:
        assert re.match(r"^\s*(import|def|@|class|export|#include|pub fn)", s), s
    assert any(s.startswith("class Third") for s in starts)
    assert any(s.startswith("def method") for s in starts)  # chunk strippé : indentation retirée


def test_code_long_line_hard_split() -> None:
    code = "x = '" + "a" * 500 + "'\n"
    chunks = chunk_text(code, chunk_chars=100, overlap=0, kind="code")
    assert all(len(c) <= 100 for c in chunks)
    assert _no_ws("".join(chunks)) == _no_ws(code)


def test_windows_newlines_and_tiny_chunk_chars() -> None:
    text = "un\r\n\r\ndeux\r\n\r\ntrois"
    assert chunk_text(text, chunk_chars=4, overlap=0) == ["un", "deux", "troi", "s"]
    chunks = chunk_text(text, chunk_chars=4, overlap=10)  # chevauchement borné à chunk_chars // 2
    assert chunks[:3] == ["un", "deux", "troi"]
    assert all(c and c == c.strip() and len(c) <= 4 for c in chunks)
    assert "s" in chunks[-1]


def test_overlap_is_clamped_below_chunk_size() -> None:
    text = _paragraphs(8, 60)
    chunks = chunk_text(text, chunk_chars=100, overlap=10_000)
    assert chunks
    assert all(len(c) <= 100 for c in chunks)
    for line in text.split("\n"):
        if line.strip():
            assert any(line.strip() in c for c in chunks)
