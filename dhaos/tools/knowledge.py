"""Outils de bases de savoir : ``kb_list``, ``kb_search``, ``kb_add_note``.

Ils s'appuient sur ``ctx.kb`` (``dhaos.kb.manager.KnowledgeManager``) et sont
programmés contre son contrat : ``list_bases()``, ``get_base(name)``,
``create_base(name)``, ``add_note(base, text, title=...)`` (→ ``NoteResult``,
repli sur ``add_text`` → ``int`` si absent), ``search(query, bases=...,
top_k=...)``. Les erreurs métier (``KnowledgeError``) sont renvoyées au
modèle comme résultats en erreur, jamais propagées. Sans gestionnaire
(``ctx.kb is None``), les outils répondent « bases de savoir indisponibles ».

``kb_add_note`` signale explicitement le **remplacement** d'une note
existante de même titre (message « note remplacée », ``data["replaced"]``) et
journalise chaque ajout (``ctx.journal``, genre ``kb_note``) : les titres
viennent du modèle et un titre réutilisé écraserait sinon silencieusement du
savoir accumulé.
"""
from __future__ import annotations

from typing import Any

from ..config import Settings
from ..kb.manager import KnowledgeError
from .base import Tool, ToolContext, ToolError, ToolResult

# Plafonds défensifs : les arguments viennent du modèle.
MAX_TOP_K = 50
MAX_BASES_PER_QUERY = 32
MAX_NOTE_CHARS = 1_000_000


def _require_kb(ctx: ToolContext) -> Any:
    if ctx.kb is None:
        raise ToolError("bases de savoir indisponibles")
    return ctx.kb


def _clean_name(value: Any, what: str) -> str:
    name = " ".join(str(value or "").split())
    if not name:
        raise ToolError(f"{what} vide")
    return name


def format_base_line(info: Any) -> str:
    """``nom — description (n docs, n chunks)`` (description omise si vide)."""
    name = getattr(info, "name", "")
    description = " ".join(str(getattr(info, "description", "") or "").split())
    n_docs = int(getattr(info, "n_docs", 0) or 0)
    n_chunks = int(getattr(info, "n_chunks", 0) or 0)
    head = f"{name} — {description}" if description else str(name)
    return f"{head} ({n_docs} docs, {n_chunks} chunks)"


def format_hits(hits: list[Any], snippet_chars: int | None = None) -> str:
    """Passages numérotés ; chaque texte est tronqué à ``snippet_chars`` (les
    modèles locaux lisent lentement : 8 chunks entiers = plusieurs minutes)."""
    if not hits:
        return "aucun résultat"
    blocks: list[str] = []
    truncated = 0
    for i, hit in enumerate(hits, start=1):
        base = getattr(hit, "base", "") or "?"
        source = getattr(hit, "source", "") or getattr(hit, "title", "") or "?"
        score = float(getattr(hit, "score", 0.0) or 0.0)
        text = str(getattr(hit, "text", "") or "").strip()
        if snippet_chars and len(text) > snippet_chars:
            text = text[:snippet_chars].rstrip() + " […]"
            truncated += 1
        blocks.append(f"[{i}] {base}/{source} (score {score:.2f})\n{text}")
    out = "\n\n".join(blocks)
    if truncated:
        out += f"\n\n({truncated} passage(s) tronqué(s) à {snippet_chars} caractères ; read_file sur la source pour le texte complet)"
    return out


class KBListTool(Tool):
    name = "kb_list"
    description = (
        'Liste les bases de savoir locales (nom, description, taille).'
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        kb = _require_kb(ctx)
        try:
            bases = list(kb.list_bases())
        except KnowledgeError as e:
            raise ToolError(str(e)) from e
        if not bases:
            return ToolResult("aucune base de savoir", data=[])
        return ToolResult(
            "\n".join(format_base_line(b) for b in bases),
            data=[
                {
                    "name": getattr(b, "name", ""),
                    "description": getattr(b, "description", ""),
                    "n_docs": getattr(b, "n_docs", 0),
                    "n_chunks": getattr(b, "n_chunks", 0),
                }
                for b in bases
            ],
        )


class KBSearchTool(Tool):
    name = "kb_search"
    description = (
        'Recherche hybride dans les bases de savoir locales (toutes si bases absent) ; renvoie les passages pertinents avec base, source et score.'
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 4000, "description": "Question ou mots-clés."},
            "bases": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "maxItems": MAX_BASES_PER_QUERY,
                "description": "Noms des bases à interroger (défaut : toutes).",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TOP_K,
                "description": "Nombre maximal de passages (défaut : kb.tool_top_k).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        kb = _require_kb(ctx)
        query = _clean_name(args.get("query"), "requête")
        raw_bases = args.get("bases") or []
        bases = [b for b in (" ".join(str(x).split()) for x in raw_bases) if b] or None
        top_k = args.get("top_k")
        if top_k is None:
            top_k = int(getattr(ctx.settings.kb, "tool_top_k", 5) or 5)
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        snippet = int(getattr(ctx.settings.kb, "tool_snippet_chars", 700) or 0) or None
        try:
            hits = list(kb.search(query, bases=bases, top_k=top_k))
        except KnowledgeError as e:
            raise ToolError(str(e)) from e
        return ToolResult(
            format_hits(hits, snippet),
            data=[
                {
                    "base": getattr(h, "base", ""),
                    "source": getattr(h, "source", ""),
                    "title": getattr(h, "title", ""),
                    "score": getattr(h, "score", 0.0),
                    "doc_id": getattr(h, "doc_id", None),
                    "chunk_ord": getattr(h, "chunk_ord", None),
                }
                for h in hits
            ],
        )


class KBAddNoteTool(Tool):
    name = "kb_add_note"
    description = (
        'Ajoute une note à une base de savoir (créée si absente) : conclusions, procédures, décisions à retenir.'
    )
    parameters = {
        "type": "object",
        "properties": {
            "base": {"type": "string", "minLength": 1, "maxLength": 200, "description": "Nom de la base."},
            "text": {"type": "string", "minLength": 1, "maxLength": MAX_NOTE_CHARS, "description": "Contenu de la note."},
            "title": {"type": "string", "maxLength": 500, "description": "Titre de la note (optionnel)."},
        },
        "required": ["base", "text"],
        "additionalProperties": False,
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        kb = _require_kb(ctx)
        base = _clean_name(args.get("base"), "nom de base")
        text = str(args.get("text") or "")
        if not text.strip():
            raise ToolError("note vide")
        title_raw = args.get("title")
        title = " ".join(str(title_raw).split()) if title_raw else None
        try:
            created = False
            if kb.get_base(base) is None:
                kb.create_base(base)
                created = True
            doc_id, replaced = _add_note(kb, base, text, title or None)
        except KnowledgeError as e:
            raise ToolError(str(e)) from e
        journal = getattr(ctx, "journal", None)
        if journal is not None:
            journal.record("kb_note", base=base, doc_id=doc_id, title=title, replaced=replaced, created=created)
        suffix = " (base créée)" if created else ""
        verb = "remplacée dans" if replaced else "ajoutée à"
        return ToolResult(
            f"note {verb} {base} (doc {doc_id}){suffix}",
            data={"base": base, "doc_id": doc_id, "created": created, "title": title, "replaced": replaced},
        )


def _add_note(kb: Any, base: str, text: str, title: str | None) -> tuple[int, bool]:
    """Ajoute la note via ``add_note`` (→ ``(doc_id, remplacée)``) ; un
    gestionnaire minimal n'exposant que ``add_text`` renvoie ``(doc_id, False)``."""
    add_note = getattr(kb, "add_note", None)
    if add_note is None:
        return int(kb.add_text(base, text, title=title)), False
    result = add_note(base, text, title=title)
    return int(getattr(result, "doc_id")), bool(getattr(result, "replaced", False))


def tools(settings: Settings) -> list[Tool]:
    return [KBListTool(), KBSearchTool(), KBAddNoteTool()]


__all__ = [
    "KBAddNoteTool",
    "KBListTool",
    "KBSearchTool",
    "format_base_line",
    "format_hits",
    "tools",
]
