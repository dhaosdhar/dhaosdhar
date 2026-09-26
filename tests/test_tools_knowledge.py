"""Tests des outils de bases de savoir (kb_list, kb_search, kb_add_note)
contre un double ``FakeKB`` respectant le contrat de ``KnowledgeManager``."""
from __future__ import annotations

from typing import Any

import pytest

from dhaos.config import Settings
from dhaos.kb.manager import BaseInfo, Hit, KnowledgeError, KnowledgeManager, NoteResult
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
        return self.add_note(base, text, source=source, title=title).doc_id

    def add_note(self, base: str, text: str, *, source: str | None = None, title: str | None = None) -> NoteResult:
        """Même titre explicite ⇒ remplacement (comme le vrai gestionnaire)."""
        info = self._require(base)
        if title:
            for i, (doc_id, old_text, old_title) in enumerate(self.notes[info.name]):
                if old_title == title:
                    self.notes[info.name][i] = (doc_id, text, title)
                    return NoteResult(doc_id=doc_id, source=f"note:{title}", title=title, replaced=old_text != text)
        doc_id = self._next_id
        self._next_id += 1
        self.notes[info.name].append((doc_id, text, title))
        info.n_docs += 1
        info.n_chunks += max(1, len(text) // 100)
        return NoteResult(doc_id=doc_id, source=source or f"note:{title or doc_id}", title=title or "", replaced=False)

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
    assert result.data == {"base": "developpeur", "doc_id": 1, "created": True, "title": "Règle", "replaced": False}
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


def test_kb_add_note_same_title_reports_replacement(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    """Régression : un titre réutilisé remplaçait silencieusement la note
    précédente (« note ajoutée », même doc_id, aucun signalement)."""
    first = registry.execute(tool_call("kb_add_note", base="dev", text="Toujours typer.", title="Règle"), kb_ctx)
    second = registry.execute(tool_call("kb_add_note", base="dev", text="Toujours documenter.", title="Règle"), kb_ctx)
    assert second.is_error is False
    assert second.content == "note remplacée dans dev (doc 1)"
    assert "ajoutée" not in second.content
    assert second.data["replaced"] is True and second.data["doc_id"] == first.data["doc_id"]
    assert first.data["replaced"] is False
    # même texte à nouveau : ni doublon ni annonce de remplacement
    third = registry.execute(tool_call("kb_add_note", base="dev", text="Toujours documenter.", title="Règle"), kb_ctx)
    assert third.content == "note ajoutée à dev (doc 1)" and third.data["replaced"] is False
    assert kb.notes["dev"] == [(1, "Toujours documenter.", "Règle")]


def test_kb_add_note_is_journaled(kb_ctx: ToolContext, registry: ToolRegistry) -> None:
    registry.execute(tool_call("kb_add_note", base="dev", text="a", title="T"), kb_ctx)
    registry.execute(tool_call("kb_add_note", base="dev", text="b", title="T"), kb_ctx)
    entries = [e for e in kb_ctx.journal.tail(10) if e.get("kind") == "kb_note"]
    assert [(e["base"], e["doc_id"], e["replaced"], e["created"]) for e in entries] == [
        ("dev", 1, False, True),
        ("dev", 1, True, False),
    ]


def test_kb_add_note_falls_back_on_add_text(kb_ctx: ToolContext, kb: FakeKB, registry: ToolRegistry) -> None:
    """Un gestionnaire minimal n'exposant que ``add_text`` reste accepté."""
    class MinimalKB:
        def __init__(self, inner: FakeKB) -> None:
            self.inner = inner

        def get_base(self, name: str) -> BaseInfo | None:
            return self.inner.get_base(name)

        def create_base(self, name: str, description: str = "") -> BaseInfo:
            return self.inner.create_base(name, description)

        def add_text(self, base: str, text: str, *, source: str | None = None, title: str | None = None) -> int:
            return self.inner.add_text(base, text, source=source, title=title)

    kb_ctx.kb = MinimalKB(kb)
    result = registry.execute(tool_call("kb_add_note", base="dev", text="x", title="T"), kb_ctx)
    assert result.content == "note ajoutée à dev (doc 1) (base créée)" and result.data["replaced"] is False


# ------------------------------------------- kb_add_note contre le vrai gestionnaire
@pytest.fixture
def real_kb(settings: Settings) -> Any:
    manager = KnowledgeManager(settings)
    yield manager
    manager.close()


def test_kb_add_note_untitled_same_first_line_keeps_both(ctx: ToolContext, real_kb: KnowledgeManager,
                                                          registry: ToolRegistry) -> None:
    """Régression : deux notes sans titre commençant par la même ligne
    s'écrasaient (source dérivée ``note:<première ligne>``) ; la première
    disparaissait de la recherche sans aucun signalement."""
    ctx.kb = real_kb
    first = registry.execute(tool_call("kb_add_note", base="projet", text="Décision\nLa base passe sur PostgreSQL."), ctx)
    second = registry.execute(tool_call("kb_add_note", base="projet", text="Décision\nLe cache passe sur Redis."), ctx)
    assert first.is_error is False and second.is_error is False
    assert first.data["doc_id"] != second.data["doc_id"]
    assert first.data["replaced"] is False and second.data["replaced"] is False
    docs = real_kb.list_documents("projet")
    assert len(docs) == 2 and all(d.title == "Décision" for d in docs)
    assert len({d.source for d in docs}) == 2
    hits = registry.execute(tool_call("kb_search", query="PostgreSQL", bases=["projet"]), ctx)
    assert "PostgreSQL" in hits.content
    # une note strictement identique reste dédupliquée
    again = registry.execute(tool_call("kb_add_note", base="projet", text="Décision\nLe cache passe sur Redis."), ctx)
    assert again.data["doc_id"] == second.data["doc_id"] and again.data["replaced"] is False
    assert len(real_kb.list_documents("projet")) == 2


def test_kb_add_note_explicit_title_replacement_real_manager(ctx: ToolContext, real_kb: KnowledgeManager,
                                                             registry: ToolRegistry) -> None:
    ctx.kb = real_kb
    first = registry.execute(tool_call("kb_add_note", base="dev", text="Utiliser pytest.", title="Tests"), ctx)
    second = registry.execute(tool_call("kb_add_note", base="dev", text="Utiliser pytest et ruff.", title="Tests"), ctx)
    assert first.content == "note ajoutée à dev (doc 1) (base créée)"
    assert second.content == "note remplacée dans dev (doc 1)" and second.data["replaced"] is True
    assert [d.source for d in real_kb.list_documents("dev")] == ["note:Tests"]
    third = registry.execute(tool_call("kb_add_note", base="dev", text="Utiliser pytest et ruff.", title="Tests"), ctx)
    assert third.data["replaced"] is False and "ajoutée" in third.content


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
