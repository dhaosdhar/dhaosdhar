"""Bases de savoir locales, organisées en catégories gérables.

Stockage SQLite (``knowledge.db``) : bases → documents → chunks (+ embeddings
+ index FTS5). Recherche hybride (vecteurs + mots-clés, fusion RRF).

API publique : ``KnowledgeManager`` (dhaos.kb.manager) et ``get_embedder``
(dhaos.kb.embeddings).
"""
from .manager import BaseInfo, DocInfo, Hit, IngestReport, KnowledgeManager

__all__ = ["BaseInfo", "DocInfo", "Hit", "IngestReport", "KnowledgeManager"]
