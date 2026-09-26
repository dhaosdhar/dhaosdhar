"""Stockage SQLite des bases de savoir : bases → documents → chunks (+ FTS5).

Une seule connexion (WAL, clés étrangères activées), protégée par un verrou
réentrant. Les embeddings sont des BLOB float32 (colonne ``dim``) ; l'index
plein texte ``chunks_fts`` est une table FTS5 à contenu externe synchronisée
par triggers avec ``chunks``.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bases (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    embedder    TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id        INTEGER PRIMARY KEY,
    base_id   INTEGER NOT NULL REFERENCES bases(id) ON DELETE CASCADE,
    source    TEXT NOT NULL,
    title     TEXT NOT NULL DEFAULT '',
    hash      TEXT NOT NULL DEFAULT '',
    size      INTEGER NOT NULL DEFAULT 0,
    kind      TEXT NOT NULL DEFAULT 'text',
    text      TEXT NOT NULL DEFAULT '',
    added_at  TEXT NOT NULL,
    UNIQUE (base_id, source)
);
CREATE INDEX IF NOT EXISTS idx_documents_base ON documents(base_id);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    doc_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord       INTEGER NOT NULL,
    text      TEXT NOT NULL,
    embedding BLOB,
    dim       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MAX_FTS_TOKENS = 64


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def build_fts_query(query: str) -> str:
    """Requête FTS5 sûre : tokens ``\\w+`` entre guillemets, joints par OR.
    Chaîne vide si aucun token."""
    tokens = _TOKEN_RE.findall(str(query or ""))[:MAX_FTS_TOKENS]
    return " OR ".join(f'"{t}"' for t in tokens)


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


class Store:
    """Accès bas niveau à ``knowledge.db`` (une connexion, un verrou)."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path) if str(db_path) != ":memory:" else None
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._depth = 0
        self.conn = sqlite3.connect(
            str(self.db_path) if self.db_path is not None else ":memory:",
            check_same_thread=False,
            isolation_level=None,
        )
        self.conn.row_factory = sqlite3.Row
        if self.db_path is not None:
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # executescript valide implicitement : hors de transaction().
        with self._lock:
            self.conn.executescript(_SCHEMA)

    # ---------------------------------------------------------- connexion
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Transaction (réentrante : seule la plus externe engage/valide)."""
        with self._lock:
            outermost = self._depth == 0
            if outermost:
                self.conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self.conn
            except BaseException:
                self._depth -= 1
                if outermost:
                    self.conn.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if outermost:
                    self.conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, tuple(params)).fetchone()

    def _all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, tuple(params)).fetchall()

    # -------------------------------------------------------------- bases
    _BASE_SELECT = """
        SELECT b.id, b.name, b.description, b.embedder, b.created_at,
               (SELECT COUNT(*) FROM documents d WHERE d.base_id = b.id) AS n_docs,
               (SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.doc_id
                 WHERE d.base_id = b.id) AS n_chunks
        FROM bases b
    """

    def create_base(self, name: str, description: str = "", embedder: str = "") -> sqlite3.Row:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO bases(name, description, embedder, created_at) VALUES (?, ?, ?, ?)",
                (name, description, embedder, now_iso()),
            )
        row = self.get_base(name)
        assert row is not None
        return row

    def get_base(self, name: str) -> sqlite3.Row | None:
        return self._one(self._BASE_SELECT + " WHERE b.name = ?", (name,))

    def get_base_by_id(self, base_id: int) -> sqlite3.Row | None:
        return self._one(self._BASE_SELECT + " WHERE b.id = ?", (base_id,))

    def list_bases(self) -> list[sqlite3.Row]:
        return self._all(self._BASE_SELECT + " ORDER BY b.name")

    def rename_base(self, base_id: int, new_name: str) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE bases SET name = ? WHERE id = ?", (new_name, base_id))

    def set_description(self, base_id: int, description: str) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE bases SET description = ? WHERE id = ?", (description, base_id))

    def set_embedder(self, base_id: int, embedder: str) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE bases SET embedder = ? WHERE id = ?", (embedder, base_id))

    def delete_base(self, base_id: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE doc_id IN (SELECT id FROM documents WHERE base_id = ?)", (base_id,)
            )
            conn.execute("DELETE FROM documents WHERE base_id = ?", (base_id,))
            conn.execute("DELETE FROM bases WHERE id = ?", (base_id,))

    # ---------------------------------------------------------- documents
    _DOC_SELECT = """
        SELECT d.id, d.base_id, d.source, d.title, d.hash, d.size, d.kind, d.added_at,
               (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS n_chunks
        FROM documents d
    """

    def get_document(self, base_id: int, source: str) -> sqlite3.Row | None:
        return self._one(self._DOC_SELECT + " WHERE d.base_id = ? AND d.source = ?", (base_id, source))

    def get_document_by_id(self, doc_id: int) -> sqlite3.Row | None:
        return self._one(self._DOC_SELECT + " WHERE d.id = ?", (doc_id,))

    def get_document_text(self, doc_id: int) -> str:
        row = self._one("SELECT text FROM documents WHERE id = ?", (doc_id,))
        return str(row["text"]) if row is not None else ""

    def list_documents(self, base_id: int) -> list[sqlite3.Row]:
        return self._all(self._DOC_SELECT + " WHERE d.base_id = ? ORDER BY d.source", (base_id,))

    def iter_document_texts(self, base_ids: Sequence[int]) -> Iterator[tuple[int, str]]:
        """``(doc_id, texte)`` des documents des bases données (ordre : base, source).

        Seule la liste des identifiants est chargée d'un bloc ; chaque texte est
        lu à la demande, sans tenir le verrou entre deux itérations, pour ne pas
        charger tout le corpus en mémoire."""
        if not base_ids:
            return
        ids = [
            int(r["id"])
            for r in self._all(
                f"SELECT id FROM documents WHERE base_id IN ({_placeholders(len(base_ids))})"
                " ORDER BY base_id, source",
                tuple(base_ids),
            )
        ]
        for doc_id in ids:
            row = self._one("SELECT text FROM documents WHERE id = ?", (doc_id,))
            if row is not None:
                yield doc_id, str(row["text"])

    def export_rows(self, base_id: int) -> list[sqlite3.Row]:
        return self._all(
            "SELECT source, title, text, added_at FROM documents WHERE base_id = ? ORDER BY source", (base_id,)
        )

    def upsert_document(
        self,
        base_id: int,
        source: str,
        *,
        title: str,
        hash: str,
        size: int,
        text: str,
        kind: str = "text",
    ) -> tuple[int, bool]:
        """Crée ou remplace le document ``(base_id, source)`` ; les anciens
        chunks sont supprimés. Renvoie ``(doc_id, existait_déjà)``."""
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT id FROM documents WHERE base_id = ? AND source = ?", (base_id, source)
            ).fetchone()
            if existing is not None:
                doc_id = int(existing["id"])
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                conn.execute(
                    "UPDATE documents SET title = ?, hash = ?, size = ?, kind = ?, text = ?, added_at = ?"
                    " WHERE id = ?",
                    (title, hash, int(size), kind, text, now_iso(), doc_id),
                )
                return doc_id, True
            cur = conn.execute(
                "INSERT INTO documents(base_id, source, title, hash, size, kind, text, added_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (base_id, source, title, hash, int(size), kind, text, now_iso()),
            )
            return int(cur.lastrowid), False

    def delete_document(self, doc_id: int) -> bool:
        with self.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------- chunks
    @staticmethod
    def _blob(vector: np.ndarray | None) -> tuple[bytes | None, int]:
        if vector is None:
            return None, 0
        arr = np.ascontiguousarray(np.asarray(vector, dtype=np.float32).reshape(-1))
        return arr.tobytes(), int(arr.shape[0])

    def insert_chunks(self, doc_id: int, texts: Sequence[str], embeddings: np.ndarray | None = None) -> int:
        """Insère les chunks (ordre = position) avec leurs embeddings (ligne i ↔ texte i)."""
        texts = list(texts)
        if embeddings is not None:
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
                raise ValueError(
                    f"embeddings de forme {embeddings.shape} pour {len(texts)} chunk(s)"
                )
        rows = []
        for i, text in enumerate(texts):
            blob, dim = self._blob(embeddings[i] if embeddings is not None else None)
            rows.append((doc_id, i, text, blob, dim))
        with self.transaction() as conn:
            conn.executemany(
                "INSERT INTO chunks(doc_id, ord, text, embedding, dim) VALUES (?, ?, ?, ?, ?)", rows
            )
        return len(rows)

    def delete_chunks(self, doc_id: int) -> int:
        with self.transaction() as conn:
            return conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,)).rowcount

    def chunks_for_doc(self, doc_id: int) -> list[sqlite3.Row]:
        return self._all("SELECT id, ord, text, dim FROM chunks WHERE doc_id = ? ORDER BY ord", (doc_id,))

    def update_embeddings(self, chunk_ids: Sequence[int], embeddings: np.ndarray) -> int:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunk_ids):
            raise ValueError(f"embeddings de forme {embeddings.shape} pour {len(chunk_ids)} chunk(s)")
        rows = []
        for cid, vec in zip(chunk_ids, embeddings):
            blob, dim = self._blob(vec)
            rows.append((blob, dim, int(cid)))
        with self.transaction() as conn:
            conn.executemany("UPDATE chunks SET embedding = ?, dim = ? WHERE id = ?", rows)
        return len(rows)

    def get_chunks(self, chunk_ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        """Chunks (texte, position, document, source, titre, base) par identifiant."""
        ids = [int(i) for i in chunk_ids]
        if not ids:
            return {}
        out: dict[int, sqlite3.Row] = {}
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            rows = self._all(
                "SELECT c.id, c.ord, c.text, c.doc_id, d.source, d.title, b.name AS base"
                " FROM chunks c JOIN documents d ON d.id = c.doc_id JOIN bases b ON b.id = d.base_id"
                f" WHERE c.id IN ({_placeholders(len(batch))})",
                batch,
            )
            for row in rows:
                out[int(row["id"])] = row
        return out

    def vectors(self, base_ids: Sequence[int], dim: int) -> tuple[np.ndarray, np.ndarray]:
        """``(ids, matrice)`` des embeddings de dimension ``dim`` des chunks des bases
        données : ``ids`` int64 ``(n,)``, matrice float32 ``(n, dim)``."""
        empty = (np.zeros((0,), dtype=np.int64), np.zeros((0, int(dim)), dtype=np.float32))
        if not base_ids or dim <= 0:
            return empty
        rows = self._all(
            "SELECT c.id, c.embedding FROM chunks c JOIN documents d ON d.id = c.doc_id"
            f" WHERE d.base_id IN ({_placeholders(len(base_ids))}) AND c.dim = ? AND c.embedding IS NOT NULL"
            " ORDER BY c.id",
            (*base_ids, int(dim)),
        )
        if not rows:
            return empty
        ids = np.fromiter((int(r["id"]) for r in rows), dtype=np.int64, count=len(rows))
        matrix = np.frombuffer(b"".join(bytes(r["embedding"]) for r in rows), dtype=np.float32)
        return ids, matrix.reshape(len(rows), int(dim))

    # ---------------------------------------------------------------- FTS
    def fts_search(self, query: str, base_ids: Sequence[int], limit: int) -> list[tuple[int, float]]:
        """``[(chunk_id, bm25)]`` triés du plus pertinent au moins pertinent
        (bm25 FTS5 : plus petit = meilleur, en général négatif).
        Requête sans token alphanumérique ou sans base ⇒ ``[]``."""
        match = build_fts_query(query)
        if not match or not base_ids or limit <= 0:
            return []
        rows = self._all(
            "SELECT c.id AS id, bm25(chunks_fts) AS score"
            " FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
            " JOIN documents d ON d.id = c.doc_id"
            f" WHERE chunks_fts MATCH ? AND d.base_id IN ({_placeholders(len(base_ids))})"
            " ORDER BY score LIMIT ?",
            (match, *base_ids, int(limit)),
        )
        return [(int(r["id"]), float(r["score"])) for r in rows]

    # -------------------------------------------------------------- stats
    def count(self, table: str, base_id: int | None = None) -> int:
        if table not in ("bases", "documents", "chunks"):
            raise ValueError(f"table inconnue : {table}")
        if base_id is None:
            row = self._one(f"SELECT COUNT(*) AS n FROM {table}")
        elif table == "documents":
            row = self._one("SELECT COUNT(*) AS n FROM documents WHERE base_id = ?", (base_id,))
        elif table == "chunks":
            row = self._one(
                "SELECT COUNT(*) AS n FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE d.base_id = ?",
                (base_id,),
            )
        else:
            row = self._one("SELECT COUNT(*) AS n FROM bases WHERE id = ?", (base_id,))
        return int(row["n"]) if row is not None else 0

    def text_bytes(self, base_id: int | None = None) -> int:
        if base_id is None:
            row = self._one("SELECT COALESCE(SUM(size), 0) AS n FROM documents")
        else:
            row = self._one("SELECT COALESCE(SUM(size), 0) AS n FROM documents WHERE base_id = ?", (base_id,))
        return int(row["n"]) if row is not None else 0

    def db_size_bytes(self) -> int:
        """Taille sur disque (fichier principal + WAL + SHM)."""
        if self.db_path is None:
            return 0
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self.db_path) + suffix)
            try:
                total += os.path.getsize(p)
            except OSError:
                continue
        return total

    def stats(self) -> dict[str, Any]:
        return {
            "bases": self.count("bases"),
            "documents": self.count("documents"),
            "chunks": self.count("chunks"),
            "text_bytes": self.text_bytes(),
            "db_bytes": self.db_size_bytes(),
            "db_path": str(self.db_path) if self.db_path is not None else ":memory:",
        }


__all__ = ["MAX_FTS_TOKENS", "Store", "build_fts_query", "now_iso"]
