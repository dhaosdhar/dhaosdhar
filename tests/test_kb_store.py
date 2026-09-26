"""Tests du stockage SQLite (schéma, CRUD, FTS5, vecteurs)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dhaos.kb.store import Store, build_fts_query


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "kb" / "knowledge.db")
    yield s
    s.close()


def test_schema_and_pragmas(store: Store) -> None:
    tables = {r["name"] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"bases", "documents", "chunks", "chunks_fts"} <= tables
    assert store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    # réouverture idempotente
    other = Store(store.db_path)
    other.close()


def test_base_crud(store: Store) -> None:
    row = store.create_base("dev", "desc", "hash:512")
    assert row["name"] == "dev" and row["description"] == "desc" and row["embedder"] == "hash:512"
    assert row["n_docs"] == 0 and row["n_chunks"] == 0 and row["created_at"]
    assert store.get_base("dev")["id"] == row["id"]
    assert store.get_base("nope") is None
    store.create_base("infra")
    assert [b["name"] for b in store.list_bases()] == ["dev", "infra"]
    store.rename_base(row["id"], "developpeur")
    store.set_description(row["id"], "nouvelle")
    store.set_embedder(row["id"], "hash:64")
    b = store.get_base_by_id(row["id"])
    assert (b["name"], b["description"], b["embedder"]) == ("developpeur", "nouvelle", "hash:64")
    with pytest.raises(Exception):
        store.create_base("infra")  # UNIQUE
    store.delete_base(row["id"])
    assert store.get_base("developpeur") is None
    assert [b["name"] for b in store.list_bases()] == ["infra"]


def test_documents_and_chunks(store: Store) -> None:
    base = store.create_base("dev")
    doc_id, existed = store.upsert_document(
        base["id"], "/tmp/a.py", title="a.py", hash="h1", size=10, text="def f(): pass", kind="code"
    )
    assert not existed
    emb = np.eye(2, 4, dtype=np.float32)
    assert store.insert_chunks(doc_id, ["chunk zéro alpha", "chunk un beta"], emb) == 2
    doc = store.get_document(base["id"], "/tmp/a.py")
    assert doc["n_chunks"] == 2 and doc["hash"] == "h1" and doc["kind"] == "code"
    assert store.get_document_text(doc_id) == "def f(): pass"
    assert [c["ord"] for c in store.chunks_for_doc(doc_id)] == [0, 1]
    assert store.get_base("dev")["n_docs"] == 1 and store.get_base("dev")["n_chunks"] == 2

    ids, matrix = store.vectors([base["id"]], 4)
    assert ids.shape == (2,) and matrix.shape == (2, 4) and matrix.dtype == np.float32
    assert np.array_equal(matrix, emb)
    assert store.vectors([base["id"]], 3)[0].shape == (0,)  # autre dimension ignorée
    assert store.vectors([], 4)[1].shape == (0, 4)

    chunks = store.get_chunks(list(ids))
    assert chunks[int(ids[0])]["base"] == "dev" and chunks[int(ids[0])]["source"] == "/tmp/a.py"
    assert chunks[int(ids[1])]["text"] == "chunk un beta"

    # upsert : remplace le document et supprime les anciens chunks (aussi dans FTS)
    assert store.fts_search("alpha", [base["id"]], 5)
    doc_id2, existed = store.upsert_document(
        base["id"], "/tmp/a.py", title="a.py", hash="h2", size=12, text="def g(): pass", kind="code"
    )
    assert existed and doc_id2 == doc_id
    assert store.chunks_for_doc(doc_id) == []
    assert store.fts_search("alpha", [base["id"]], 5) == []
    store.insert_chunks(doc_id, ["gamma delta"], np.ones((1, 4), dtype=np.float32))
    assert store.fts_search("gamma", [base["id"]], 5)
    assert store.count("documents", base["id"]) == 1 and store.count("chunks", base["id"]) == 1

    assert store.delete_document(doc_id)
    assert not store.delete_document(doc_id)
    assert store.fts_search("gamma", [base["id"]], 5) == []
    assert store.count("chunks") == 0


def test_insert_chunks_shape_mismatch(store: Store) -> None:
    base = store.create_base("dev")
    doc_id, _ = store.upsert_document(base["id"], "s", title="t", hash="h", size=1, text="x")
    with pytest.raises(ValueError):
        store.insert_chunks(doc_id, ["a", "b"], np.ones((1, 4), dtype=np.float32))
    assert store.insert_chunks(doc_id, ["a", "b"], None) == 2
    assert store.vectors([base["id"]], 4)[0].shape == (0,)


def test_update_embeddings(store: Store) -> None:
    base = store.create_base("dev")
    doc_id, _ = store.upsert_document(base["id"], "s", title="t", hash="h", size=1, text="x")
    store.insert_chunks(doc_id, ["a", "b"], np.zeros((2, 2), dtype=np.float32))
    ids = [c["id"] for c in store.chunks_for_doc(doc_id)]
    store.update_embeddings(ids, np.ones((2, 3), dtype=np.float32))
    got_ids, matrix = store.vectors([base["id"]], 3)
    assert list(got_ids) == ids and matrix.shape == (2, 3)


def test_build_fts_query_is_safe() -> None:
    assert build_fts_query("") == ""
    assert build_fts_query("!!! ---") == ""
    assert build_fts_query('zorglub "(x)" : foo') == '"zorglub" OR "x" OR "foo"'
    assert build_fts_query("élève NOT AND") == '"élève" OR "NOT" OR "AND"'


def test_fts_search_with_special_characters(store: Store) -> None:
    base = store.create_base("dev")
    doc_id, _ = store.upsert_document(base["id"], "s", title="t", hash="h", size=1, text="x")
    store.insert_chunks(doc_id, ["la fonction zorglub(x) renvoie : rien", "un élève studieux", "sans rapport"])
    for query in ['zorglub "(x)" : foo', "fonction(x)", 'renvoie" OR ', "NOT AND OR", "(", '"""', "eleve"]:
        results = store.fts_search(query, [base["id"]], 10)
        assert all(isinstance(cid, int) and isinstance(score, float) for cid, score in results)
    hits = store.fts_search("zorglub", [base["id"]], 10)
    assert len(hits) == 1
    assert store.get_chunks([hits[0][0]])[hits[0][0]]["text"].startswith("la fonction")
    assert store.fts_search("eleve", [base["id"]], 10)  # accents retirés
    assert store.fts_search("", [base["id"]], 10) == []
    assert store.fts_search("zorglub", [], 10) == []
    assert store.fts_search("zorglub", [base["id"]], 0) == []


def test_fts_search_filters_by_base_and_ranks(store: Store) -> None:
    a = store.create_base("a")
    b = store.create_base("b")
    da, _ = store.upsert_document(a["id"], "sa", title="", hash="h", size=1, text="")
    db, _ = store.upsert_document(b["id"], "sb", title="", hash="h", size=1, text="")
    store.insert_chunks(da, ["pomme poire", "pomme pomme pomme banane"])
    store.insert_chunks(db, ["pomme cerise"])
    res = store.fts_search("pomme", [a["id"]], 10)
    assert len(res) == 2
    assert res[0][1] <= res[1][1]  # bm25 : plus petit = meilleur
    assert {cid for cid, _ in store.fts_search("pomme", [a["id"], b["id"]], 10)} == {1, 2, 3}


def test_stats_and_size(store: Store) -> None:
    base = store.create_base("dev")
    doc_id, _ = store.upsert_document(base["id"], "s", title="t", hash="h", size=42, text="texte")
    store.insert_chunks(doc_id, ["a"])
    st = store.stats()
    assert st["bases"] == 1 and st["documents"] == 1 and st["chunks"] == 1
    assert st["text_bytes"] == 42
    assert st["db_bytes"] > 0
    assert st["db_path"].endswith("knowledge.db")
    assert store.text_bytes(base["id"]) == 42
    assert list(store.iter_document_texts([base["id"]])) == [(doc_id, "texte")]
    assert list(store.iter_document_texts([])) == []
    rows = store.export_rows(base["id"])
    assert rows[0]["source"] == "s" and rows[0]["text"] == "texte"


def test_transaction_rollback(store: Store) -> None:
    base = store.create_base("dev")
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.upsert_document(base["id"], "s", title="t", hash="h", size=1, text="x")
            raise RuntimeError("abandon")
    assert store.count("documents") == 0


def test_memory_store() -> None:
    s = Store(":memory:")
    s.create_base("x")
    assert s.stats()["db_bytes"] == 0 and s.stats()["bases"] == 1
    s.close()


# ------------------------------------- régression : iter_document_texts en flux
def test_iter_document_texts_streams_and_keeps_order(store: Store) -> None:
    import threading
    import tracemalloc

    base = store.create_base("b")
    other = store.create_base("a")
    doc_len = 200_000
    for i in range(50):
        store.upsert_document(
            base["id"], f"doc-{i:03d}", title=f"doc {i}", hash=f"h{i}", size=doc_len, text=chr(97 + i % 26) * doc_len
        )
    store.upsert_document(other["id"], "z", title="z", hash="hz", size=1, text="z")

    gen = store.iter_document_texts([base["id"]])
    tracemalloc.start()
    first = next(gen)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(first[1]) == doc_len
    # Un document à la fois, pas tout le corpus (~10 Mo) d'un bloc.
    assert peak < 3 * doc_len

    # Le verrou n'est pas tenu entre deux next() : un autre thread peut lire.
    result: list = []

    def reader() -> None:
        result.append(store.get_base("b") is not None)

    t = threading.Thread(target=reader)
    t.start()
    t.join(timeout=5)
    assert result == [True]
    gen.close()

    # Ordre (base_id, source) conservé, bases multiples.
    ids = [d for d, _ in store.iter_document_texts([other["id"], base["id"]])]
    expected = [
        int(r["id"])
        for r in store.conn.execute(
            "SELECT id FROM documents WHERE base_id IN (?, ?) ORDER BY base_id, source",
            (other["id"], base["id"]),
        )
    ]
    assert ids == expected and len(ids) == 51
    assert list(store.iter_document_texts([])) == []
