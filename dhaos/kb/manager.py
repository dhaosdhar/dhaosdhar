"""Gestionnaire des bases de savoir.

Une *base* est une catégorie de savoir nommée (ex. ``developpeur``,
``infra``, ``projet-x``). On y ingère des fichiers, dossiers, URLs ou notes ;
on interroge une base, plusieurs, ou toutes.

Implémentation : SQLite via ``dhaos.kb.store``, découpage via
``dhaos.kb.chunking``, extraction via ``dhaos.kb.ingest``, vecteurs via
``dhaos.kb.embeddings``.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from ..config import Settings
from .chunking import chunk_text
from .embeddings import embedder_id, get_embedder
from .ingest import document_kind, extract, extract_url, is_url, iter_files
from .store import Store


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

NAME_RE = re.compile(r"^[a-z0-9._-]{1,64}$")
SEARCH_MODES = ("hybrid", "vector", "keyword")
RRF_K = 60
CANDIDATE_FACTOR = 3


def normalize_name(name: Any) -> str:
    """Nom de base normalisé : strip, minuscules, espaces → ``-`` ;
    ``[a-z0-9._-]{1,64}`` sinon ``KnowledgeError``."""
    raw = str(name if name is not None else "").strip().lower()
    normalized = re.sub(r"\s+", "-", raw)
    if not NAME_RE.match(normalized):
        raise KnowledgeError(
            f"nom de base invalide : {str(name)!r} (autorisé : lettres minuscules, chiffres, '.', '_', '-', 64 caractères max)"
        )
    return normalized


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text_size(text: str) -> int:
    return len(text.encode("utf-8"))


class KnowledgeManager:
    """Façade unique pour la CLI, l'API, l'outil ``kb_*`` et l'entraînement.

    Les noms de base sont normalisés (minuscules, ``[a-z0-9._-]``) ; les
    erreurs métier lèvent ``KnowledgeError``.
    """

    def __init__(self, settings: Settings, *, embedder: Any = None, db_path: Path | None = None):
        self.settings = settings
        self.db_path = Path(db_path) if db_path is not None else settings.kb_db_path
        self.store = Store(self.db_path)
        self._embedder: Any = embedder

    # ---------------------------------------------------------- internes
    @property
    def embedder(self) -> Any:
        """Embedder courant (construit paresseusement selon ``kb.embedder``)."""
        if self._embedder is None:
            self._embedder = get_embedder(self.settings)
        return self._embedder

    def _current_embedder_id(self) -> str:
        try:
            return embedder_id(self.embedder)
        except Exception as e:  # noqa: BLE001 — Ollama injoignable, etc.
            raise KnowledgeError(f"embedder indisponible : {e}") from e

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            matrix = np.asarray(self.embedder.embed(list(texts)), dtype=np.float32)
        except KnowledgeError:
            raise
        except Exception as e:  # noqa: BLE001
            raise KnowledgeError(f"échec du calcul des embeddings : {e}") from e
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise KnowledgeError(f"embeddings de forme inattendue {matrix.shape} pour {len(texts)} texte(s)")
        return matrix

    @staticmethod
    def _to_info(row: sqlite3.Row) -> BaseInfo:
        return BaseInfo(
            name=str(row["name"]),
            description=str(row["description"] or ""),
            n_docs=int(row["n_docs"] or 0),
            n_chunks=int(row["n_chunks"] or 0),
            embedder=str(row["embedder"] or ""),
            created_at=str(row["created_at"] or ""),
        )

    def _require_base(self, name: str) -> sqlite3.Row:
        key = normalize_name(name)
        row = self.store.get_base(key)
        if row is None:
            raise KnowledgeError(f"base inconnue : {key}")
        return row

    def _resolve_bases(self, bases: list[str] | None) -> list[sqlite3.Row]:
        if not bases:
            return self.store.list_bases()
        rows: list[sqlite3.Row] = []
        seen: set[int] = set()
        for name in bases:
            row = self._require_base(name)
            if int(row["id"]) not in seen:
                seen.add(int(row["id"]))
                rows.append(row)
        return rows

    def _ensure_embedder(self, base_row: sqlite3.Row) -> str:
        """Vérifie que l'embedder courant correspond à celui de la base ; une
        base vide est mise à jour, sinon ``KnowledgeError`` (⇒ ``reindex``)."""
        current = self._current_embedder_id()
        recorded = str(base_row["embedder"] or "")
        if recorded == current:
            return current
        if int(base_row["n_docs"] or 0) == 0 and int(base_row["n_chunks"] or 0) == 0:
            self.store.set_embedder(int(base_row["id"]), current)
            return current
        raise KnowledgeError(
            f"la base '{base_row['name']}' a été indexée avec l'embedder {recorded or '(inconnu)'} "
            f"alors que l'embedder courant est {current} : lancez `dhaos kb reindex {base_row['name']}` "
            "(ou changez kb.embedder) avant d'y ajouter des documents"
        )

    def _chunks_for(self, text: str, kind: str) -> list[str]:
        return chunk_text(
            text,
            chunk_chars=self.settings.kb.chunk_chars,
            overlap=self.settings.kb.chunk_overlap,
            kind=kind,
        )

    def _ingest_document(
        self,
        base_row: sqlite3.Row,
        *,
        source: str,
        title: str,
        text: str,
        kind: str,
        report: IngestReport,
        on_progress: ProgressCallback | None,
    ) -> int | None:
        """Ingère un texte extrait ; renvoie l'id du document ou ``None`` si ignoré."""
        base_id = int(base_row["id"])
        if not text.strip():
            report.skipped += 1
            self._progress(on_progress, f"ignoré (vide) : {source}")
            return None
        digest = _text_hash(text)
        existing = self.store.get_document(base_id, source)
        if existing is not None and str(existing["hash"]) == digest:
            report.skipped += 1
            self._progress(on_progress, f"inchangé : {source}")
            return int(existing["id"])
        chunks = self._chunks_for(text, kind)
        embeddings = self._embed(chunks) if chunks else None
        with self.store.transaction():
            doc_id, existed = self.store.upsert_document(
                base_id, source, title=title, hash=digest, size=_text_size(text), text=text, kind=kind
            )
            n = self.store.insert_chunks(doc_id, chunks, embeddings) if chunks else 0
        if existed:
            report.updated += 1
        else:
            report.added += 1
        report.chunks += n
        self._progress(on_progress, f"{'mis à jour' if existed else 'ajouté'} : {source} ({n} chunk(s))")
        return doc_id

    @staticmethod
    def _progress(cb: ProgressCallback | None, message: str) -> None:
        if cb is None:
            return
        try:
            cb(message)
        except Exception:  # noqa: BLE001 — un callback d'affichage ne doit rien casser
            pass

    def _ingest_file(
        self, base_row: sqlite3.Row, path: Path, report: IngestReport, on_progress: ProgressCallback | None
    ) -> None:
        source = str(path)
        try:
            extracted = extract(path, self.settings)
            if extracted is None:
                report.skipped += 1
                self._progress(on_progress, f"ignoré (binaire, trop gros ou vide) : {source}")
                return
            title, text = extracted
            self._ingest_document(
                base_row,
                source=source,
                title=title,
                text=text,
                kind=document_kind(path),
                report=report,
                on_progress=on_progress,
            )
        except Exception as e:  # noqa: BLE001 — collecté, n'interrompt pas l'ingestion
            report.failed += 1
            report.errors.append(f"{source} : {type(e).__name__}: {e}")
            self._progress(on_progress, f"échec : {source} ({e})")

    def _ingest_url(
        self, base_row: sqlite3.Row, url: str, report: IngestReport, on_progress: ProgressCallback | None
    ) -> None:
        try:
            title, text = extract_url(url, self.settings)
            self._ingest_document(
                base_row, source=url, title=title, text=text, kind="text", report=report, on_progress=on_progress
            )
        except Exception as e:  # noqa: BLE001
            report.failed += 1
            report.errors.append(f"{url} : {type(e).__name__}: {e}")
            self._progress(on_progress, f"échec : {url} ({e})")

    # ------------------------------------------------------------ bases
    def create_base(self, name: str, description: str = "") -> BaseInfo:
        key = normalize_name(name)
        if self.store.get_base(key) is not None:
            raise KnowledgeError(f"la base existe déjà : {key}")
        row = self.store.create_base(key, str(description or "").strip(), self._current_embedder_id())
        return self._to_info(row)

    def list_bases(self) -> list[BaseInfo]:
        return [self._to_info(r) for r in self.store.list_bases()]

    def get_base(self, name: str) -> BaseInfo | None:
        try:
            key = normalize_name(name)
        except KnowledgeError:
            return None
        row = self.store.get_base(key)
        return self._to_info(row) if row is not None else None

    def rename_base(self, old: str, new: str) -> BaseInfo:
        row = self._require_base(old)
        new_key = normalize_name(new)
        if new_key == str(row["name"]):
            return self._to_info(row)
        if self.store.get_base(new_key) is not None:
            raise KnowledgeError(f"la base existe déjà : {new_key}")
        self.store.rename_base(int(row["id"]), new_key)
        updated = self.store.get_base(new_key)
        assert updated is not None
        return self._to_info(updated)

    def set_description(self, name: str, description: str) -> BaseInfo:
        row = self._require_base(name)
        self.store.set_description(int(row["id"]), str(description or "").strip())
        updated = self.store.get_base_by_id(int(row["id"]))
        assert updated is not None
        return self._to_info(updated)

    def delete_base(self, name: str) -> None:
        row = self._require_base(name)
        self.store.delete_base(int(row["id"]))

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
        base_row = self._require_base(base)
        report = IngestReport(base=str(base_row["name"]))
        self._ensure_embedder(base_row)
        for raw in sources:
            source = str(raw).strip()
            if not source:
                continue
            if is_url(source):
                self._ingest_url(base_row, source, report, on_progress)
                continue
            path = Path(source).expanduser()
            try:
                path = path.resolve()
            except OSError:
                path = path.absolute()
            if path.is_dir():
                self._progress(on_progress, f"parcours du dossier : {path}")
                for file in iter_files(path, recursive, self.settings):
                    self._ingest_file(base_row, file, report, on_progress)
            elif path.is_file():
                self._ingest_file(base_row, path, report, on_progress)
            else:
                report.failed += 1
                report.errors.append(f"{source} : source introuvable")
                self._progress(on_progress, f"introuvable : {source}")
        return report

    def add_text(self, base: str, text: str, *, source: str | None = None, title: str | None = None) -> int:
        """Ajoute une note textuelle ; renvoie l'id du document."""
        base_row = self._require_base(base)
        text = str(text or "")
        if not text.strip():
            raise KnowledgeError("note vide")
        self._ensure_embedder(base_row)
        title = " ".join(str(title).split()) if title else ""
        if not title:
            first = next((line.strip() for line in text.splitlines() if line.strip()), "")
            title = first[:80]
        src = " ".join(str(source).split()) if source else ""
        if not src:
            src = f"note:{title}" if title and title != text.strip()[:80] else f"note:{_text_hash(text)[:12]}"
        report = IngestReport(base=str(base_row["name"]))
        doc_id = self._ingest_document(
            base_row, source=src, title=title, text=text, kind="text", report=report, on_progress=None
        )
        if doc_id is None:
            raise KnowledgeError("note vide")
        return doc_id

    def list_documents(self, base: str) -> list[DocInfo]:
        row = self._require_base(base)
        name = str(row["name"])
        return [
            DocInfo(
                id=int(d["id"]),
                base=name,
                source=str(d["source"]),
                title=str(d["title"] or ""),
                n_chunks=int(d["n_chunks"] or 0),
                size=int(d["size"] or 0),
                hash=str(d["hash"] or ""),
                added_at=str(d["added_at"] or ""),
            )
            for d in self.store.list_documents(int(row["id"]))
        ]

    def remove_document(self, base: str, source: str) -> bool:
        row = self._require_base(base)
        base_id = int(row["id"])
        candidates = [str(source).strip()]
        if candidates[0] and not is_url(candidates[0]) and not candidates[0].startswith("note:"):
            try:
                candidates.append(str(Path(candidates[0]).expanduser().resolve()))
            except OSError:
                pass
        for candidate in candidates:
            if not candidate:
                continue
            doc = self.store.get_document(base_id, candidate)
            if doc is not None:
                return self.store.delete_document(int(doc["id"]))
        return False

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
        mode = str(mode or "hybrid").strip().lower()
        if mode not in SEARCH_MODES:
            raise KnowledgeError(f"mode de recherche inconnu : {mode!r} (attendu : hybrid, vector, keyword)")
        query = " ".join(str(query or "").split())
        rows = self._resolve_bases(bases)
        if not query or not rows:
            return []
        k = int(top_k) if top_k is not None else int(self.settings.kb.top_k)
        k = max(1, k)
        n_candidates = k * CANDIDATE_FACTOR
        base_ids = [int(r["id"]) for r in rows]

        vector_ranked: list[tuple[int, float]] = []
        if mode in ("hybrid", "vector"):
            vector_ranked = self._vector_candidates(query, rows, n_candidates)
        keyword_ranked: list[tuple[int, float]] = []
        if mode in ("hybrid", "keyword"):
            keyword_ranked = self.store.fts_search(query, base_ids, n_candidates)

        scores: dict[int, float] = {}
        if mode == "vector":
            scores = dict(vector_ranked)
        elif mode == "keyword":
            relevance = [(cid, max(0.0, -s)) for cid, s in keyword_ranked]
            best = max((r for _, r in relevance), default=0.0) or 1.0
            scores = {cid: r / best for cid, r in relevance}
        else:
            for ranked in (vector_ranked, keyword_ranked):
                for rank, (cid, _score) in enumerate(ranked):
                    scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if not scores:
            return []
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        chunk_rows = self.store.get_chunks([cid for cid, _ in ordered])
        hits: list[Hit] = []
        for cid, score in ordered:
            row = chunk_rows.get(cid)
            if row is None:
                continue
            hits.append(
                Hit(
                    base=str(row["base"]),
                    source=str(row["source"]),
                    title=str(row["title"] or ""),
                    text=str(row["text"]),
                    score=float(score),
                    chunk_ord=int(row["ord"]),
                    doc_id=int(row["doc_id"]),
                )
            )
        return hits

    def _vector_candidates(self, query: str, rows: list[sqlite3.Row], n: int) -> list[tuple[int, float]]:
        """Top ``n`` chunks par cosinus, restreint aux bases indexées avec l'embedder courant."""
        current = self._current_embedder_id()
        base_ids = [int(r["id"]) for r in rows if str(r["embedder"] or "") == current]
        if not base_ids:
            return []
        ids, matrix = self.store.vectors(base_ids, int(self.embedder.dim))
        if ids.shape[0] == 0:
            return []
        q = self._embed([query])[0]
        sims = matrix @ q
        n = min(max(1, n), int(sims.shape[0]))
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top], kind="stable")]
        # Cosinus nul ou négatif = aucune caractéristique commune : pas un candidat.
        return [(int(ids[i]), float(sims[i])) for i in top if sims[i] > 0.0]

    # ------------------------------------------------------------- divers
    def stats(self, base: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.store.stats())
        out["embedder"] = self._current_embedder_id()
        if base is not None:
            row = self._require_base(base)
            base_id = int(row["id"])
            out["base"] = {
                "name": str(row["name"]),
                "description": str(row["description"] or ""),
                "documents": int(row["n_docs"] or 0),
                "chunks": int(row["n_chunks"] or 0),
                "text_bytes": self.store.text_bytes(base_id),
                "embedder": str(row["embedder"] or ""),
                "created_at": str(row["created_at"] or ""),
            }
        else:
            out["per_base"] = [
                {
                    "name": str(r["name"]),
                    "documents": int(r["n_docs"] or 0),
                    "chunks": int(r["n_chunks"] or 0),
                    "embedder": str(r["embedder"] or ""),
                }
                for r in self.store.list_bases()
            ]
        return out

    def reindex(self, base: str) -> int:
        """Recalcule les embeddings avec l'embedder courant ; renvoie le nb de chunks."""
        row = self._require_base(base)
        base_id = int(row["id"])
        current = self._current_embedder_id()
        total = 0
        for doc in self.store.list_documents(base_id):
            doc_id = int(doc["id"])
            text = self.store.get_document_text(doc_id)
            kind = str(doc["kind"] or "text")
            chunks = self._chunks_for(text, kind) if text.strip() else []
            if not chunks:
                # Document sans texte conservé (ancien format) : on garde ses chunks.
                existing = self.store.chunks_for_doc(doc_id)
                chunks = [str(c["text"]) for c in existing]
                if not chunks:
                    continue
                embeddings = self._embed(chunks)
                self.store.update_embeddings([int(c["id"]) for c in existing], embeddings)
                total += len(chunks)
                continue
            embeddings = self._embed(chunks)
            with self.store.transaction():
                self.store.delete_chunks(doc_id)
                total += self.store.insert_chunks(doc_id, chunks, embeddings)
        self.store.set_embedder(base_id, current)
        return total

    def export(self, base: str, path: Path) -> int:
        """Exporte les documents d'une base en JSONL ; renvoie le nb de documents."""
        row = self._require_base(base)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for doc in self.store.export_rows(int(row["id"])):
                record = {
                    "source": str(doc["source"]),
                    "title": str(doc["title"] or ""),
                    "text": str(doc["text"] or ""),
                    "added_at": str(doc["added_at"] or ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
        return n

    def corpus_text(self, bases: list[str] | None = None) -> Iterator[str]:
        """Texte brut des documents (pour l'entraînement)."""
        rows = self._resolve_bases(bases)
        base_ids = [int(r["id"]) for r in rows]
        for _doc_id, text in self.store.iter_document_texts(base_ids):
            if text.strip():
                yield text

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "KnowledgeManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class KnowledgeError(Exception):
    """Erreur métier des bases de savoir (base inexistante, nom invalide…)."""


__all__ = [
    "BaseInfo",
    "DocInfo",
    "Hit",
    "IngestReport",
    "KnowledgeError",
    "KnowledgeManager",
    "ProgressCallback",
    "normalize_name",
]
