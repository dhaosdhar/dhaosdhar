"""Tests des outils de bases de savoir (kb_list, kb_search, kb_add_note)
contre un double ``FakeKB`` respectant le contrat de ``KnowledgeManager``."""
from __future__ import annotations

from typing import Any

import pytest

from dhaos.config import Settings
from dhaos.kb.manager import BaseInfo, Hit, KnowledgeError
from dhaos.tools import knowledge
from dhaos.tools.base import ToolContext, ToolRegistry

from .fakes import tool_call


class FakeKB:
    """Double minimal de KnowledgeManager : bases en mémoire, recherche par
    sous-chaîne, KnowledgeError pour toute base inconnue."""

    def __init__(self) -> None:
        self.bases: dict[str, BaseInfo] = {}
        self.notes: dict[str, list[tuple[int, str, str | None]]] = {}
        self.search_calls: list[dict[str, Any]] = []
        self._next_id = 1

    def _norm(self, name: str) -> str:
        return name.strip().lower()

    def _require(self, name: str) -> BaseInfo:
        info = self.bases.get(self._norm(name))
        if info is None:
            raise KnowledgeError(f"base inconnue : {name}")
        return info

    def create_base(self, name: str, description: str = "") -> BaseInfo:
        key = self._norm(name)
        if not key:
            raise KnowledgeError("nom de base invalide")
        if key in self.bases:
            raise KnowledgeError(f"la base existe déjà : {name}")
        info = BaseInfo(name=key, description=description, embedder="hash:512", created_at="2026-01-01")
        self.bases[key] = info
        self.notes[key] = []
        return info

    def list_bases(self) -> list[BaseInfo]:
        return list(self.bases.values())

    def get_base(self, name: str) -> BaseInfo | None:
        return self.bases.get(self._norm(name))

    def add_text(self, base: str, text: str, *, source: str | None = None, title: str | None = None) -> int:
        info = self._require(base)
        doc_id = self._next_id
        self._next_id += 1
        self.notes[info.name].append((doc_id, text, title))
        info.n_docs += 1
        info.n_chunks += max(1, len(text) // 100)
        return doc_id

    def search(self, query: str, *, bases: list[str] | None = None, top_k: int | None = None,
               mode: str = "hybrid") -> list[Hit]:
        self.search_calls.append({"query": query, "bases": bases, "top_k": top_k, "mode": mode})
        targets = [self._require(b).name for b in bases] if bases else list(self.bases)
        hits: list[Hit] = []
        for name in targets:
            for doc_id, text, title in self.notes[name]:
                if query.lower() in text.lower():
                    hits.append(Hit(base=name, source=f"note:{title or doc_id}", title=title or "",
                                    text=text, score=0.5 + 0.01 * doc_id, chunk_ord=0, doc_id=doc_id))
        hits.sort(key=lambda h: -h.score)
        return hits[: top_k or 8]


@pytest.fixture
def kb() -> FakeKB:
    return FakeKB()


@pytest.fixture
def kb_ctx(ctx: ToolContext, kb: FakeKB) -> ToolContext:
    ctx.kb = kb
    return ctx


@pytest.fixture
def registry(settings: Settings) -> ToolRegistry:
    return ToolRegistry(knowledge.tools(settings))


def test_registry_names(registry: ToolRegistry) -> None:
    assert registry.names == ["kb_list", "kb_search", "kb_add_note"]
    for spec in registry.specs():
        assert spec.parameters["type"] == "object"
        assert spec.description


# ------------------------------------------------------------------ kb_list
def test_kb_list_empty(kb_ctx: ToolContext, registry: ToolRegistry) -> None:
    result = registry.execute(tool_call("kb_list"), kb_ctx)
    assert result.is_error is False
    assert result.content == "aucune base de savoir"


def test_kb_list_format(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    kb.create_base("developpeur", "Bonnes pratiques  Python")
    kb.add_text("developpeur", "x" * 250)
    kb.create_base("infra")
    result = registry.execute(tool_call("kb_list"), kb_ctx)
    assert result.content == "developpeur — Bonnes pratiques Python (1 docs, 2 chunks)\ninfra (0 docs, 0 chunks)"
    assert result.data[0]["name"] == "developpeur"


def test_kb_list_rejects_extra_args(kb_ctx: ToolContext, registry: ToolRegistry) -> None:
    assert registry.execute(tool_call("kb_list", foo=1), kb_ctx).is_error is True


# ---------------------------------------------------------------- kb_search
def test_kb_search_hits(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    kb.create_base("dev")
    kb.add_text("dev", "Utiliser pytest pour les tests.", title="tests")
    kb.add_text("dev", "Rien à voir.")
    kb.create_base("ops")
    kb.add_text("ops", "pytest tourne aussi en CI.")
    result = registry.execute(tool_call("kb_search", query="pytest"), kb_ctx)
    assert result.is_error is False
    assert kb.search_calls == [{"query": "pytest", "bases": None, "top_k": None, "mode": "hybrid"}]
    assert result.content == (
        "[1] ops/note:3 (score 0.53)\npytest tourne aussi en CI.\n\n"
        "[2] dev/note:tests (score 0.51)\nUtiliser pytest pour les tests."
    )
    assert [d["base"] for d in result.data] == ["ops", "dev"]


def test_kb_search_with_bases_and_top_k(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    kb.create_base("dev")
    kb.add_text("dev", "alpha un")
    kb.add_text("dev", "alpha deux")
    result = registry.execute(tool_call("kb_search", query="alpha", bases=["dev"], top_k=1), kb_ctx)
    assert result.is_error is False
    assert kb.search_calls[-1] == {"query": "alpha", "bases": ["dev"], "top_k": 1, "mode": "hybrid"}
    assert result.content.startswith("[1] dev/") and "[2]" not in result.content


def test_kb_search_empty_bases_means_all(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    kb.create_base("dev")
    registry.execute(tool_call("kb_search", query="x", bases=[]), kb_ctx)
    assert kb.search_calls[-1]["bases"] is None


def test_kb_search_no_result(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    kb.create_base("dev")
    result = registry.execute(tool_call("kb_search", query="introuvable"), kb_ctx)
    assert result.is_error is False
    assert result.content == "aucun résultat"


def test_kb_search_unknown_base(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    kb.create_base("dev")
    result = registry.execute(tool_call("kb_search", query="x", bases=["dev", "fantome"]), kb_ctx)
    assert result.is_error is True
    assert result.content == "base inconnue : fantome"


def test_kb_search_invalid_args(kb_ctx: ToolContext, registry: ToolRegistry) -> None:
    assert registry.execute(tool_call("kb_search"), kb_ctx).is_error is True
    assert registry.execute(tool_call("kb_search", query=""), kb_ctx).is_error is True
    assert registry.execute(tool_call("kb_search", query="x", bases="dev"), kb_ctx).is_error is True
    assert registry.execute(tool_call("kb_search", query="x", top_k=0), kb_ctx).is_error is True
    assert registry.execute(tool_call("kb_search", query="x", top_k=1000), kb_ctx).is_error is True
    assert registry.execute(tool_call("kb_search", query="   "), kb_ctx).is_error is True


def test_kb_search_without_manager(ctx: ToolContext, registry: ToolRegistry) -> None:
    assert ctx.kb is None
    for call in (tool_call("kb_list"), tool_call("kb_search", query="x"), tool_call("kb_add_note", base="b", text="t")):
        result = registry.execute(call, ctx)
        assert result.is_error is True
        assert result.content == "bases de savoir indisponibles"


# -------------------------------------------------------------- kb_add_note
def test_kb_add_note_creates_base(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    result = registry.execute(tool_call("kb_add_note", base="developpeur", text="Toujours typer.", title="Règle"), kb_ctx)
    assert result.is_error is False
    assert result.content == "note ajoutée à developpeur (doc 1) (base créée)"
    assert result.data == {"base": "developpeur", "doc_id": 1, "created": True, "title": "Règle"}
    assert kb.get_base("developpeur") is not None
    assert kb.notes["developpeur"] == [(1, "Toujours typer.", "Règle")]


def test_kb_add_note_existing_base(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    kb.create_base("dev", "desc")
    result = registry.execute(tool_call("kb_add_note", base="dev", text="note sans titre"), kb_ctx)
    assert result.content == "note ajoutée à dev (doc 1)"
    assert kb.notes["dev"] == [(1, "note sans titre", None)]
    assert len(kb.bases) == 1


def test_kb_add_note_then_search(kb_ctx: ToolContext, registry: ToolRegistry) -> None:
    registry.execute(tool_call("kb_add_note", base="dev", text="Le linter est ruff."), kb_ctx)
    result = registry.execute(tool_call("kb_search", query="ruff", bases=["dev"]), kb_ctx)
    assert "Le linter est ruff." in result.content


def test_kb_add_note_invalid_args(kb_ctx: ToolContext, registry: ToolRegistry) -> None:
    assert registry.execute(tool_call("kb_add_note", base="dev"), kb_ctx).is_error is True
    assert registry.execute(tool_call("kb_add_note", text="t"), kb_ctx).is_error is True
    assert registry.execute(tool_call("kb_add_note", base="", text="t"), kb_ctx).is_error is True
    assert registry.execute(tool_call("kb_add_note", base="dev", text="   "), kb_ctx).is_error is True
    assert registry.execute(tool_call("kb_add_note", base="dev", text="t", title=3), kb_ctx).is_error is True


def test_kb_add_note_manager_error(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    def failing_create(name: str, description: str = "") -> BaseInfo:
        raise KnowledgeError("nom de base invalide : Bad Name!")

    kb.create_base = failing_create  # type: ignore[method-assign]
    result = registry.execute(tool_call("kb_add_note", base="Bad Name!", text="t"), kb_ctx)
    assert result.is_error is True
    assert result.content == "nom de base invalide : Bad Name!"


def test_unexpected_exception_never_escapes(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    def boom(*args: Any, **kwargs: Any) -> list[Hit]:
        raise RuntimeError("sqlite verrouillé")

    kb.search = boom  # type: ignore[method-assign]
    result = registry.execute(tool_call("kb_search", query="x"), kb_ctx)
    assert result.is_error is True
    assert result.content == "RuntimeError: sqlite verrouillé"
