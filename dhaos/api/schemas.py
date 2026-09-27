"""Modèles pydantic de l'API HTTP : corps de requêtes (validés, bornés) et
réponses.

Les entrées viennent du réseau : chaque champ est typé et borné (longueurs,
intervalles). Les réponses reprennent les dataclasses de ``dhaos.kb.manager``
et de ``dhaos.agent`` sous une forme JSON stable.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Plafonds défensifs (les corps de requêtes sont non fiables).
MAX_BASE_NAME_CHARS = 200
MAX_DESCRIPTION_CHARS = 4_000
MAX_SOURCE_CHARS = 4_096
MAX_SOURCES = 200
MAX_NOTE_CHARS = 1_000_000
MAX_TITLE_CHARS = 500
MAX_QUERY_CHARS = 4_000
MAX_BASES_PER_QUERY = 32
MAX_TOP_K = 50
MAX_MESSAGE_CHARS = 200_000
MAX_SESSION_ID_CHARS = 128
MAX_BACKEND_NAME_CHARS = 32
MAX_MODEL_CHARS = 200
TOOL_RESULT_PREVIEW_CHARS = 300

SearchMode = Literal["hybrid", "vector", "keyword"]


# ============================================================== requêtes
class BaseCreate(BaseModel):
    """Création d'une base de savoir."""

    name: str = Field(min_length=1, max_length=MAX_BASE_NAME_CHARS, description="Nom de la base.")
    description: str = Field("", max_length=MAX_DESCRIPTION_CHARS, description="Description libre.")


class BaseUpdate(BaseModel):
    """Renommage et/ou nouvelle description d'une base."""

    new_name: str | None = Field(None, min_length=1, max_length=MAX_BASE_NAME_CHARS)
    description: str | None = Field(None, max_length=MAX_DESCRIPTION_CHARS)


class DocumentsAdd(BaseModel):
    """Ingestion de fichiers, dossiers ou URLs dans une base."""

    sources: list[str] = Field(
        min_length=1,
        max_length=MAX_SOURCES,
        description="Chemins (fichiers, dossiers) ou URLs http(s).",
    )
    recursive: bool = Field(True, description="Parcourir les dossiers récursivement.")


class NoteAdd(BaseModel):
    """Ajout d'une note textuelle à une base."""

    text: str = Field(min_length=1, max_length=MAX_NOTE_CHARS)
    title: str | None = Field(None, max_length=MAX_TITLE_CHARS)


class DocumentRemove(BaseModel):
    """Suppression d'un document identifié par sa source."""

    source: str = Field(min_length=1, max_length=MAX_SOURCE_CHARS)


class SearchRequest(BaseModel):
    """Recherche dans une, plusieurs ou toutes les bases."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    bases: list[str] | None = Field(None, max_length=MAX_BASES_PER_QUERY, description="Défaut : toutes.")
    top_k: int | None = Field(None, ge=1, le=MAX_TOP_K)
    mode: SearchMode = "hybrid"


class ChatRequest(BaseModel):
    """Un tour de conversation avec l'agent."""

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str | None = Field(None, max_length=MAX_SESSION_ID_CHARS, description="Absent : nouvelle session.")
    backend: str | None = Field(None, max_length=MAX_BACKEND_NAME_CHARS, description="ollama | claude (défaut : config).")
    model: str | None = Field(None, max_length=MAX_MODEL_CHARS)
    stream: bool = Field(True, description="true : flux SSE ; false : réponse JSON complète.")
    no_tools: bool = Field(False, description="true : l'agent répond sans outils.")


class ConfirmAnswer(BaseModel):
    """Réponse de l'utilisateur à une demande de confirmation émise en SSE."""

    id: str = Field(min_length=1, max_length=64)
    answer: bool


# Clés de configuration modifiables depuis l'API (réglages courants, sans secret).
CONFIG_PATCH_KEYS: tuple[str, ...] = (
    "backends.default", "backends.ollama.model", "backends.ollama.host", "backends.ollama.embed_model",
    "backends.ollama.num_ctx", "backends.ollama.keep_alive", "backends.ollama.timeout",
    "backends.claude.model", "backends.claude.effort",
    "tools.write_policy", "tools.shell_policy", "kb.embedder", "kb.tool_top_k", "kb.tool_snippet_chars",
    "web.provider", "web.searxng_url", "agent.language", "agent.auto_kb_search", "agent.max_iterations",
    "agent.collect_traces",
)


class ConfigPatch(BaseModel):
    """Modification d'une clé de configuration (``section.cle``)."""

    key: str = Field(min_length=1, max_length=80)
    value: Any


class ModelsOut(BaseModel):
    backend: str
    default: str = ""
    ok: bool = False
    detail: str = ""
    models: list[str] = Field(default_factory=list)


class ReindexOut(BaseModel):
    chunks: int


# ============================================================== réponses
class _FromAttributes(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class BaseInfoOut(_FromAttributes):
    name: str
    description: str = ""
    n_docs: int = 0
    n_chunks: int = 0
    embedder: str = ""
    created_at: str = ""


class DocInfoOut(_FromAttributes):
    id: int
    base: str
    source: str
    title: str = ""
    n_chunks: int = 0
    size: int = 0
    hash: str = ""
    added_at: str = ""


class BaseDetail(BaseInfoOut):
    documents: list[DocInfoOut] = Field(default_factory=list)


class HitOut(_FromAttributes):
    base: str
    source: str
    title: str = ""
    text: str
    score: float
    chunk_ord: int
    doc_id: int


class IngestReportOut(_FromAttributes):
    base: str
    added: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    errors: list[str] = Field(default_factory=list)
    summary: str = ""


class NoteOut(BaseModel):
    doc_id: int


class RemovedOut(BaseModel):
    removed: bool


class UsageOut(_FromAttributes):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatOut(BaseModel):
    text: str
    session_id: str
    usage: UsageOut
    stop_reason: str
    iterations: int
    tool_calls: int
    error: str | None = None


class SessionInfoOut(BaseModel):
    """``SessionInfo`` sans le chemin du fichier (jamais exposé)."""

    id: str
    title: str = ""
    backend: str = ""
    model: str = ""
    created_at: str = ""
    updated_at: str = ""
    n_messages: int = 0


class SessionDetail(BaseModel):
    id: str
    meta: dict[str, Any]
    messages: list[dict[str, Any]]
    events: list[dict[str, Any]] = Field(default_factory=list)


class BackendHealth(BaseModel):
    name: str
    model: str
    health: dict[str, Any]


class KBHealth(BaseModel):
    bases: int = 0
    documents: int = 0
    chunks: int = 0


class HealthOut(BaseModel):
    status: str = "ok"
    version: str
    backend: BackendHealth
    kb: KBHealth


__all__ = [
    "CONFIG_PATCH_KEYS",
    "ConfigPatch",
    "ConfirmAnswer",
    "ModelsOut",
    "ReindexOut",
    "BackendHealth",
    "BaseCreate",
    "BaseDetail",
    "BaseInfoOut",
    "BaseUpdate",
    "ChatOut",
    "ChatRequest",
    "DocInfoOut",
    "DocumentRemove",
    "DocumentsAdd",
    "HealthOut",
    "HitOut",
    "IngestReportOut",
    "KBHealth",
    "NoteAdd",
    "NoteOut",
    "RemovedOut",
    "SearchMode",
    "SearchRequest",
    "SessionDetail",
    "SessionInfoOut",
    "TOOL_RESULT_PREVIEW_CHARS",
    "UsageOut",
]
