"""Outils de bases de savoir : ``kb_list``, ``kb_search``, ``kb_add_note``.

Ils s'appuient sur ``ctx.kb`` (``dhaos.kb.manager.KnowledgeManager``) et sont
programmés contre son contrat : ``list_bases()``, ``get_base(name)``,
``create_base(name)``, ``add_text(base, text, title=...)``, ``search(query,
bases=..., top_k=...)``. Les erreurs métier (``KnowledgeError``) sont
renvoyées au modèle comme résultats en erreur, jamais propagées. Sans
gestionnaire (``ctx.kb is None``), les outils répondent « bases de savoir
indisponibles ».
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


def format_hits(hits: list[Any]) -> str:
    if not hits:
        return "aucun résultat"
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        base = getattr(hit, "base", "") or "?"
        source = getattr(hit, "source", "") or getattr(hit, "title", "") or "?"
        score = float(getattr(hit, "score", 0.0) or 0.0)
        text = str(getattr(hit, "text", "") or "").strip()
        blocks.append(f"[{i}] {base}/{source} (score {score:.2f})\n{text}")
    return "\n\n".join(blocks)


class KBListTool(Tool):
    name = "kb_list"
    description = (
        "Liste les bases de savoir locales (catégories de connaissances) avec "
        "leur description et leur taille. Consultez-les avec kb_search."
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
        "Recherche (hybride : sémantique + mots-clés) dans les bases de savoir "
        "locales. Sans `bases`, toutes les bases sont interrogées. Renvoie les "
        "passages les plus pertinents avec leur base, leur source et un score."
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
                "description": "Nombre maximal de passages (défaut : kb.top_k).",
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
        if top_k is not None:
            top_k = max(1, min(int(top_k), MAX_TOP_K))
        try:
            hits = list(kb.search(query, bases=bases, top_k=top_k))
        except KnowledgeError as e:
            raise ToolError(str(e)) from e
        return ToolResult(
            format_hits(hits),
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
        "Ajoute une note textuelle à une base de savoir (créée si elle n'existe "
        "pas) : conclusions, procédures, décisions à retenir pour plus tard."
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
            doc_id = kb.add_text(base, text, title=title or None)
        except KnowledgeError as e:
            raise ToolError(str(e)) from e
        suffix = " (base créée)" if created else ""
        return ToolResult(
            f"note ajoutée à {base} (doc {doc_id}){suffix}",
            data={"base": base, "doc_id": doc_id, "created": created, "title": title},
        )


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
