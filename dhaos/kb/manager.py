"""Gestionnaire des bases de savoir — CONTRAT.

Une *base* est une catégorie de savoir nommée (ex. ``developpeur``,
``infra``, ``projet-x``). On y ingère des fichiers, dossiers, URLs ou notes ;
on interroge une base, plusieurs, ou toutes.

Implémentation : SQLite via ``dhaos.kb.store``, découpage via
``dhaos.kb.chunking``, extraction via ``dhaos.kb.ingest``, vecteurs via
``dhaos.kb.embeddings``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from ..config import Settings


@dataclass
class BaseInfo:
    name: str
    description: str = ""
    n_docs: int = 0
    n_chunks: int = 0
    embedder: str = ""  # identifiant "nom:dim" de l'embedder utilisé
    created_at: str = ""


@dataclass
class DocInfo:
    id: int
    base: str
    source: str  # chemin absolu, URL, ou "note:<titre>"
    title: str
    n_chunks: int
    size: int
    hash: str
    added_at: str


@dataclass
class Hit:
    base: str
    source: str
    title: str
    text: str
    score: float
    chunk_ord: int
    doc_id: int


@dataclass
class IngestReport:
    base: str
    added: int = 0
    updated: int = 0
    skipped: int = 0  # inchangés, binaires, trop gros, ignorés
    failed: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"base '{self.base}' : {self.added} ajouté(s), {self.updated} mis à jour, "
            f"{self.skipped} ignoré(s), {self.failed} échec(s), {self.chunks} chunk(s)"
        )


ProgressCallback = Callable[[str], None]


class KnowledgeManager:
    """Façade unique pour la CLI, l'API, l'outil ``kb_*`` et l'entraînement.

    Les noms de base sont normalisés (minuscules, ``[a-z0-9._-]``) ; les
    erreurs métier lèvent ``KnowledgeError``.
    """

    def __init__(self, settings: Settings, *, embedder: Any = None, db_path: Path | None = None):
        raise NotImplementedError

    # ------------------------------------------------------------ bases
    def create_base(self, name: str, description: str = "") -> BaseInfo:
        raise NotImplementedError

    def list_bases(self) -> list[BaseInfo]:
        raise NotImplementedError

    def get_base(self, name: str) -> BaseInfo | None:
        raise NotImplementedError

    def rename_base(self, old: str, new: str) -> BaseInfo:
        raise NotImplementedError

    def set_description(self, name: str, description: str) -> BaseInfo:
        raise NotImplementedError

    def delete_base(self, name: str) -> None:
        raise NotImplementedError

    # -------------------------------------------------------- documents
    def add(
        self,
        base: str,
        sources: list[str | Path],
        *,
        recursive: bool = True,
        on_progress: ProgressCallback | None = None,
    ) -> IngestReport:
        """Ingère fichiers, dossiers (récursif) et URLs (http/https)."""
        raise NotImplementedError

    def add_text(self, base: str, text: str, *, source: str | None = None, title: str | None = None) -> int:
        """Ajoute une note textuelle ; renvoie l'id du document."""
        raise NotImplementedError

    def list_documents(self, base: str) -> list[DocInfo]:
        raise NotImplementedError

    def remove_document(self, base: str, source: str) -> bool:
        raise NotImplementedError

    # ----------------------------------------------------------- recherche
    def search(
        self,
        query: str,
        *,
        bases: list[str] | None = None,
        top_k: int | None = None,
        mode: str = "hybrid",  # "hybrid" | "vector" | "keyword"
    ) -> list[Hit]:
        """``bases=None`` interroge toutes les bases."""
        raise NotImplementedError

    # ------------------------------------------------------------- divers
    def stats(self, base: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def reindex(self, base: str) -> int:
        """Recalcule les embeddings avec l'embedder courant ; renvoie le nb de chunks."""
        raise NotImplementedError

    def export(self, base: str, path: Path) -> int:
        """Exporte les documents d'une base en JSONL ; renvoie le nb de documents."""
        raise NotImplementedError

    def corpus_text(self, bases: list[str] | None = None) -> Iterator[str]:
        """Texte brut des documents (pour l'entraînement)."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class KnowledgeError(Exception):
    """Erreur métier des bases de savoir (base inexistante, nom invalide…)."""
