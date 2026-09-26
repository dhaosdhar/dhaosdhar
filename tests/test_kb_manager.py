"""Tests du KnowledgeManager (embedder hash, tout sous tmp_path, sans réseau)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from dhaos.config import Settings
from dhaos.kb import ingest
from dhaos.kb.embeddings import HashEmbedder
from dhaos.kb.ingest import document_kind, extract, is_ignored, iter_files
from dhaos.kb.manager import KnowledgeError, KnowledgeManager, normalize_name


@pytest.fixture
def kb(settings: Settings) -> KnowledgeManager:
    manager = KnowledgeManager(settings)
    yield manager
    manager.close()


def _write_pdf(path: Path, text: str) -> None:
    """PDF minimal généré avec pypdf.PdfWriter contenant un flux de texte."""
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, StreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = StreamObject()
    stream._data = f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode("latin-1")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with open(path, "wb") as f:
        writer.write(f)


@dataclass
class FakePage:
    url: str
    title: str
    text: str


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "python.md").write_text(
        "# Guide Python\n\nUtiliser pytest pour les tests unitaires. Les fixtures isolent l'état.\n\n"
        "Le module zorglub gère les kangourous.\n",
        encoding="utf-8",
    )
    (root / "infra.txt").write_text(
        "Déploiement des serveurs : systemd, journald, unités de service. Le tartempion redémarre nginx.\n",
        encoding="utf-8",
    )
    (root / "app.py").write_text(
        "import os\n\ndef compute_quetzalcoatl(x):\n    return x * 2\n\nclass Service:\n    pass\n",
        encoding="utf-8",
    )
    (root / "image.bin").write_bytes(b"\x89PNG\x00\x00\x01\x02" * 50)
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\nsecretgitword = 1\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "lib.js").write_text("var nodemoduleword = 1;\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"\x00\x01")
    (root / "sub").mkdir()
    (root / "sub" / "deep.rst").write_text("Documentation profonde sur les wombats.\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------ noms
def test_normalize_name() -> None:
    assert normalize_name("  Dev Base ") == "dev-base"
    assert normalize_name("Projet.X_1") == "projet.x_1"
    for bad in ("", "   ", "a/b", "é", "x" * 65, "nom!", None):
        with pytest.raises(KnowledgeError):
            normalize_name(bad)


# ---------------------------------------------------------- cycle des bases
def test_base_lifecycle(kb: KnowledgeManager) -> None:
    assert kb.list_bases() == []
    info = kb.create_base("Developpeur", "Bonnes pratiques")
    assert info.name == "developpeur" and info.description == "Bonnes pratiques"
    assert info.embedder == "hash:512" and info.n_docs == 0 and info.created_at
    with pytest.raises(KnowledgeError):
        kb.create_base("developpeur")
    with pytest.raises(KnowledgeError):
        kb.create_base("nom invalide!")
    kb.create_base("infra")
    assert [b.name for b in kb.list_bases()] == ["developpeur", "infra"]
    assert kb.get_base("DEVELOPPEUR").name == "developpeur"
    assert kb.get_base("absent") is None
    assert kb.get_base("nom invalide!") is None

    renamed = kb.rename_base("developpeur", "Dev")
    assert renamed.name == "dev" and renamed.description == "Bonnes pratiques"
    assert kb.get_base("developpeur") is None
    with pytest.raises(KnowledgeError):
        kb.rename_base("dev", "infra")
    with pytest.raises(KnowledgeError):
        kb.rename_base("absent", "x")

    assert kb.set_description("dev", "  Nouvelle description ").description == "Nouvelle description"
    with pytest.raises(KnowledgeError):
        kb.set_description("absent", "x")

    kb.add_text("dev", "une note", title="n")
    kb.delete_base("dev")
    assert [b.name for b in kb.list_bases()] == ["infra"]
    with pytest.raises(KnowledgeError):
        kb.delete_base("dev")
    assert kb.stats()["documents"] == 0 and kb.stats()["chunks"] == 0


def test_db_path_and_reopen(settings: Settings) -> None:
    kb = KnowledgeManager(settings)
    kb.create_base("dev")
    kb.close()
    assert settings.kb_db_path.is_file()
    kb2 = KnowledgeManager(settings)
    assert [b.name for b in kb2.list_bases()] == ["dev"]
    kb2.close()
    custom = settings.data_dir / "sub" / "autre.db"
    kb3 = KnowledgeManager(settings, db_path=custom)
    assert kb3.list_bases() == [] and custom.is_file()
    kb3.close()


# --------------------------------------------------------------- ingestion
def test_ingest_files_and_directory(kb: KnowledgeManager, tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    kb.create_base("dev")
    messages: list[str] = []
    report = kb.add("dev", [root], on_progress=messages.append)
    assert report.base == "dev"
    assert report.added == 4, report.summary()  # python.md, infra.txt, app.py, sub/deep.rst
    assert report.updated == 0 and report.failed == 0 and report.errors == []
    assert report.skipped == 1  # image.bin (binaire)
    assert report.chunks >= 4
    assert any("python.md" in m for m in messages)
    assert "ajouté" in report.summary()

    sources = {Path(d.source).name for d in kb.list_documents("dev")}
    assert sources == {"python.md", "infra.txt", "app.py", "deep.rst"}
    assert not any(d.source.endswith("config") or d.source.endswith("lib.js") for d in kb.list_documents("dev"))
    assert kb.search("secretgitword nodemoduleword", mode="keyword") == []
    docs = {Path(d.source).name: d for d in kb.list_documents("dev")}
    assert docs["app.py"].title == "app.py" and docs["app.py"].n_chunks >= 1
    assert docs["app.py"].size > 0 and len(docs["app.py"].hash) == 64 and docs["app.py"].added_at
    assert docs["app.py"].base == "dev"
    assert kb.get_base("dev").n_docs == 4

    # non récursif : sub/ ignoré
    kb.create_base("flat")
    report = kb.add("flat", [root], recursive=False)
    assert {Path(d.source).name for d in kb.list_documents("flat")} == {"python.md", "infra.txt", "app.py"}


def test_ingest_single_file_relative_and_missing(kb: KnowledgeManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _corpus(tmp_path)
    kb.create_base("dev")
    monkeypatch.chdir(root)
    report = kb.add("dev", ["python.md", "absent.txt", ""])
    assert report.added == 1 and report.failed == 1
    assert "absent.txt" in report.errors[0]
    doc = kb.list_documents("dev")[0]
    assert Path(doc.source).is_absolute()


def test_reingest_unchanged_then_modified(kb: KnowledgeManager, tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    kb.create_base("dev")
    first = kb.add("dev", [root / "python.md"])
    assert (first.added, first.chunks) == (1, 1)
    again = kb.add("dev", [root / "python.md"])
    assert (again.added, again.updated, again.skipped) == (0, 0, 1)
    assert kb.search("kangourous", mode="keyword")

    (root / "python.md").write_text("Contenu totalement nouveau sur les ornithorynques.\n", encoding="utf-8")
    changed = kb.add("dev", [root / "python.md"])
    assert (changed.added, changed.updated, changed.skipped) == (0, 1, 0)
    docs = kb.list_documents("dev")
    assert len(docs) == 1 and docs[0].n_chunks == 1
    assert kb.search("kangourous", mode="keyword") == []
    assert kb.search("kangourous", mode="vector") == []
    hits = kb.search("ornithorynques")
    assert hits and hits[0].doc_id == docs[0].id
    assert kb.stats()["chunks"] == 1


def test_ingest_pdf(kb: KnowledgeManager, tmp_path: Path, settings: Settings) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, "Zorglub pdf tartempion")
    kb.create_base("docs")
    report = kb.add("docs", [pdf])
    assert report.failed == 0 and report.errors == []
    extracted = extract(pdf, settings)
    if extracted is None:
        assert report.skipped == 1
    else:
        assert report.added == 1
        assert "Zorglub" in extracted[1]
        hits = kb.search("tartempion")
        assert hits and hits[0].source == str(pdf)


def test_ingest_html(kb: KnowledgeManager, tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text(
        "<html><head><title>Ma Page</title><script>var x = 'scriptword';</script></head>"
        "<body><h1>Titre</h1><p>Les axolotls respirent sous l'eau.</p></body></html>",
        encoding="utf-8",
    )
    kb.create_base("web")
    assert kb.add("web", [page]).added == 1
    doc = kb.list_documents("web")[0]
    assert doc.title == "Ma Page"
    assert kb.search("axolotls")
    assert kb.search("scriptword", mode="keyword") == []


def test_ingest_too_big_and_binary_are_skipped(kb: KnowledgeManager, tmp_path: Path, settings: Settings) -> None:
    settings.kb.max_file_bytes = 100
    big = tmp_path / "big.txt"
    big.write_text("x" * 500, encoding="utf-8")
    binary = tmp_path / "blob.dat"
    binary.write_bytes(bytes(range(256)) * 4)
    kb.create_base("dev")
    report = kb.add("dev", [big, binary])
    assert report.skipped == 2 and report.added == 0 and report.failed == 0


def test_ingest_url_via_monkeypatch(kb: KnowledgeManager, monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    calls: list[tuple[str, int | None]] = []

    def fake_fetch(s: Settings, url: str, *, max_chars: int | None = None, client: object = None) -> FakePage:
        calls.append((url, max_chars))
        if "boom" in url:
            raise RuntimeError("serveur injoignable")
        return FakePage(url=url, title="Doc distante", text="Les pangolins ont des écailles.\n\nSecond paragraphe.")

    monkeypatch.setattr(ingest, "fetch_page", fake_fetch)
    kb.create_base("web")
    report = kb.add("web", ["https://example.org/doc", "http://example.org/boom"])
    assert report.added == 1 and report.failed == 1
    assert "boom" in report.errors[0] and "serveur injoignable" in report.errors[0]
    assert calls[0][0] == "https://example.org/doc" and calls[0][1] == settings.kb.max_file_bytes
    doc = kb.list_documents("web")[0]
    assert doc.source == "https://example.org/doc" and doc.title == "Doc distante"
    hits = kb.search("pangolins")
    assert hits and hits[0].source == "https://example.org/doc"
    # inchangé au second passage
    assert kb.add("web", ["https://example.org/doc"]).skipped == 1


def test_add_text_notes(kb: KnowledgeManager) -> None:
    kb.create_base("dev")
    doc_id = kb.add_text("dev", "Toujours typer les fonctions.", title="Règle")
    docs = kb.list_documents("dev")
    assert docs[0].id == doc_id and docs[0].source == "note:Règle" and docs[0].title == "Règle"
    # même titre ⇒ remplacement
    kb.add_text("dev", "Toujours typer les fonctions et les documenter.", title="Règle")
    assert len(kb.list_documents("dev")) == 1
    # sans titre : titre = première ligne, source dérivée (première ligne + hash)
    kb.add_text("dev", "Première ligne\nseconde ligne")
    kb.add_text("dev", "Première ligne\nseconde ligne")  # identique ⇒ pas de doublon
    assert len(kb.list_documents("dev")) == 2
    untitled = [d for d in kb.list_documents("dev") if d.title == "Première ligne"]
    assert len(untitled) == 1 and untitled[0].source.startswith("note:Première ligne#")
    # même première ligne, texte différent ⇒ nouvelle note (pas d'écrasement)
    kb.add_text("dev", "Première ligne\ntroisième ligne")
    assert len(kb.list_documents("dev")) == 3
    kb.add_text("dev", "Autre note", source="custom:1")
    assert any(d.source == "custom:1" for d in kb.list_documents("dev"))
    with pytest.raises(KnowledgeError):
        kb.add_text("dev", "   ")
    with pytest.raises(KnowledgeError):
        kb.add_text("absent", "x")
    assert kb.remove_document("dev", "custom:1")
    assert not kb.remove_document("dev", "custom:1")


def test_add_note_reports_replacement(kb: KnowledgeManager) -> None:
    """Régression : ``add_text`` ne disait pas qu'une note de même titre avait
    été remplacée ; ``add_note`` renvoie ``NoteResult.replaced``."""
    kb.create_base("dev")
    first = kb.add_note("dev", "Toujours typer.", title="Règle")
    assert (first.doc_id, first.source, first.title, first.replaced) == (1, "note:Règle", "Règle", False)
    same = kb.add_note("dev", "Toujours typer.", title="Règle")
    assert same.doc_id == 1 and same.replaced is False  # inchangé ⇒ pas un remplacement
    changed = kb.add_note("dev", "Toujours typer et documenter.", title="Règle")
    assert changed.doc_id == 1 and changed.replaced is True
    assert kb.add_text("dev", "Encore autre chose.", title="Règle") == 1
    assert len(kb.list_documents("dev")) == 1
    # sans titre : source dérivée et jamais de remplacement entre textes différents
    a = kb.add_note("dev", "TODO\nfaire A")
    b = kb.add_note("dev", "TODO\nfaire B")
    assert a.doc_id != b.doc_id and a.source != b.source and not a.replaced and not b.replaced
    assert a.title == b.title == "TODO"
    with pytest.raises(KnowledgeError):
        kb.add_note("dev", " ")


def test_remove_document_by_path(kb: KnowledgeManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _corpus(tmp_path)
    kb.create_base("dev")
    kb.add("dev", [root / "infra.txt"])
    monkeypatch.chdir(root)
    assert kb.remove_document("dev", "infra.txt")
    assert kb.list_documents("dev") == []
    assert kb.search("tartempion") == []
    with pytest.raises(KnowledgeError):
        kb.list_documents("absent")


def test_progress_callback_errors_are_ignored(kb: KnowledgeManager, tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    kb.create_base("dev")

    def bad(msg: str) -> None:
        raise RuntimeError("affichage cassé")

    assert kb.add("dev", [root / "infra.txt"], on_progress=bad).added == 1


# --------------------------------------------------------------- recherche
def test_hybrid_search_finds_the_right_document(kb: KnowledgeManager, tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    kb.create_base("dev")
    kb.add("dev", [root])
    hits = kb.search("kangourous zorglub")
    assert hits
    assert hits[0].source.endswith("python.md")
    assert hits[0].base == "dev" and hits[0].title == "python.md"
    assert "zorglub" in hits[0].text and hits[0].chunk_ord == 0 and hits[0].doc_id > 0
    assert 0 < hits[0].score <= 2 / 61 + 1e-9  # score RRF (deux listes, rang 0)
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))

    assert kb.search("tartempion nginx")[0].source.endswith("infra.txt")
    assert kb.search("compute_quetzalcoatl")[0].source.endswith("app.py")
    assert kb.search("wombats")[0].source.endswith("deep.rst")
    assert kb.search("   ") == []
    assert kb.search("xyzzyplughfoobarbaz", mode="keyword") == []


def test_search_filters_by_bases(kb: KnowledgeManager, tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    kb.create_base("py")
    kb.create_base("ops")
    kb.add("py", [root / "python.md"])
    kb.add("ops", [root / "infra.txt"])
    assert {h.base for h in kb.search("kangourous tartempion")} == {"py", "ops"}
    assert {h.base for h in kb.search("kangourous tartempion", bases=["ops"])} == {"ops"}
    assert {h.base for h in kb.search("kangourous tartempion", bases=["PY", "py"])} == {"py"}
    assert {h.base for h in kb.search("kangourous tartempion", bases=[])} == {"py", "ops"}
    with pytest.raises(KnowledgeError):
        kb.search("x", bases=["absente"])
    with pytest.raises(KnowledgeError):
        kb.search("x", bases=["nom invalide!"])


def test_search_modes_and_top_k(kb: KnowledgeManager, tmp_path: Path, settings: Settings) -> None:
    root = _corpus(tmp_path)
    kb.create_base("dev")
    kb.add("dev", [root])
    kw = kb.search("kangourous", mode="keyword")
    assert kw and kw[0].source.endswith("python.md") and kw[0].score == pytest.approx(1.0)
    assert all(0 < h.score <= 1 for h in kw)
    vec = kb.search("kangourous zorglub", mode="vector")
    assert vec and vec[0].source.endswith("python.md") and 0 < vec[0].score <= 1
    assert kb.search('kangourous "(x)" : foo', mode="keyword")  # caractères spéciaux sans erreur SQL
    assert kb.search("(", mode="keyword") == []
    for mode in ("HYBRID", "Vector", "keyword"):
        assert isinstance(kb.search("pytest", mode=mode), list)
    with pytest.raises(KnowledgeError):
        kb.search("x", mode="magie")

    for _ in range(5):
        kb.add_text("dev", "les kangourous kangourous sautent " * 3, title=f"k{_}")
    settings.kb.top_k = 2
    assert len(kb.search("kangourous")) == 2
    assert len(kb.search("kangourous", top_k=3)) == 3
    assert len(kb.search("kangourous", top_k=0)) == 1


def test_search_without_bases_or_documents(kb: KnowledgeManager) -> None:
    assert kb.search("quoi que ce soit") == []
    kb.create_base("vide")
    assert kb.search("quoi que ce soit") == []
    assert kb.search("quoi", mode="vector") == []
    assert kb.search("quoi", mode="keyword") == []


# -------------------------------------------------------- embedder / reindex
def test_embedder_mismatch_requires_reindex(settings: Settings, tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    kb = KnowledgeManager(settings, embedder=HashEmbedder(dim=64))
    kb.create_base("dev")
    assert kb.get_base("dev").embedder == "hash:64"
    kb.add("dev", [root / "python.md"])
    kb.close()

    other = KnowledgeManager(settings, embedder=HashEmbedder(dim=128))
    with pytest.raises(KnowledgeError) as exc:
        other.add("dev", [root / "infra.txt"])
    assert "reindex" in str(exc.value)
    with pytest.raises(KnowledgeError):
        other.add_text("dev", "note")
    # recherche : les vecteurs incompatibles sont ignorés, FTS continue de répondre
    assert other.search("kangourous", mode="vector") == []
    assert other.search("kangourous")
    # base vide : l'embedder est simplement mis à jour
    other.create_base("neuve")
    assert other.get_base("neuve").embedder == "hash:128"
    kb_base = other.store.get_base("neuve")
    other.store.set_embedder(int(kb_base["id"]), "hash:64")
    other.add_text("neuve", "hello")
    assert other.get_base("neuve").embedder == "hash:128"

    n = other.reindex("dev")
    assert n == other.get_base("dev").n_chunks >= 1
    assert other.get_base("dev").embedder == "hash:128"
    assert other.search("kangourous", mode="vector")
    assert other.add("dev", [root / "infra.txt"]).added == 1
    with pytest.raises(KnowledgeError):
        other.reindex("absente")
    other.close()


def test_reindex_applies_new_chunking(kb: KnowledgeManager, settings: Settings) -> None:
    kb.create_base("dev")
    kb.add_text("dev", "\n\n".join(f"Paragraphe {i} " + "mot " * 30 for i in range(10)), title="long")
    before = kb.get_base("dev").n_chunks
    settings.kb.chunk_chars = 200
    settings.kb.chunk_overlap = 20
    n = kb.reindex("dev")
    assert n == kb.get_base("dev").n_chunks > before
    assert kb.search("Paragraphe 7", mode="keyword")


# ------------------------------------------------------------------ divers
def test_export_corpus_and_stats(kb: KnowledgeManager, tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    kb.create_base("dev")
    kb.create_base("ops")
    kb.add("dev", [root / "python.md", root / "app.py"])
    kb.add("ops", [root / "infra.txt"])

    out = tmp_path / "exports" / "dev.jsonl"
    assert kb.export("dev", out) == 2
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert {l["source"] for l in lines} == {str(root / "python.md"), str(root / "app.py")}
    assert all(set(l) == {"source", "title", "text", "added_at"} for l in lines)
    assert any("kangourous" in l["text"] for l in lines)
    with pytest.raises(KnowledgeError):
        kb.export("absente", out)

    corpus = list(kb.corpus_text())
    assert len(corpus) == 3 and any("tartempion" in t for t in corpus)
    assert len(list(kb.corpus_text(["ops"]))) == 1
    with pytest.raises(KnowledgeError):
        list(kb.corpus_text(["absente"]))

    st = kb.stats()
    assert st["bases"] == 2 and st["documents"] == 3 and st["chunks"] >= 3
    assert st["db_bytes"] > 0 and st["embedder"] == "hash:512"
    assert {b["name"] for b in st["per_base"]} == {"dev", "ops"}
    st_dev = kb.stats("dev")
    assert st_dev["base"]["name"] == "dev" and st_dev["base"]["documents"] == 2
    assert st_dev["base"]["chunks"] >= 2 and st_dev["base"]["embedder"] == "hash:512"
    with pytest.raises(KnowledgeError):
        kb.stats("absente")


# ------------------------------------------------------------- ingest utils
def test_is_ignored_and_iter_files(tmp_path: Path, settings: Settings) -> None:
    root = _corpus(tmp_path)
    patterns = settings.kb.ignore_patterns
    assert is_ignored(".git/config", patterns)
    assert is_ignored("a/node_modules/b.js", patterns)
    assert is_ignored("x.pyc", patterns)
    assert is_ignored("lib.min.js", patterns)
    assert not is_ignored("src/main.py", patterns)
    assert not is_ignored("anything", [])

    files = {p.relative_to(root).as_posix() for p in iter_files(root, True, settings)}
    assert files == {"python.md", "infra.txt", "app.py", "image.bin", "sub/deep.rst"}
    flat = {p.name for p in iter_files(root, False, settings)}
    assert flat == {"python.md", "infra.txt", "app.py", "image.bin"}
    assert list(iter_files(root / "absent", True, settings)) == []

    # liens symboliques non suivis
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    os.symlink(outside, root / "link_dir")
    os.symlink(outside / "secret.txt", root / "link_file.txt")
    files = {p.relative_to(root).as_posix() for p in iter_files(root, True, settings)}
    assert not any(p.startswith("link") for p in files)


def test_document_kind_and_extract(tmp_path: Path, settings: Settings) -> None:
    assert document_kind("a.py") == "code" and document_kind("Makefile") == "code"
    assert document_kind("x.json") == "code" and document_kind("x.yml") == "code"
    assert document_kind("x.md") == "text" and document_kind("x.html") == "text"
    assert document_kind("sans_extension") == "text"

    latin = tmp_path / "latin.txt"
    latin.write_bytes("caf\xe9 cr\xe8me".encode("latin-1"))
    title, text = extract(latin, settings)
    assert title == "latin.txt" and "caf" in text  # décodage tolérant
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    assert extract(empty, settings) is None
    with pytest.raises(OSError):
        extract(tmp_path / "absent.txt", settings)


# ------------------------------------------------- régressions : reindex atomique
class _FlakyEmbedder:
    """Embedder ``hash:<dim>`` qui échoue au ``fail_at``-ième appel de ``embed``."""

    name = "hash"

    def __init__(self, dim: int, fail_at: int = 2) -> None:
        self.dim = dim
        self.calls = 0
        self.fail_at = fail_at
        self._inner = HashEmbedder(dim=dim)

    def embed(self, texts):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls >= self.fail_at:
            raise RuntimeError("Ollama a coupé")
        return self._inner.embed(texts)


def _chunk_dims(kb: KnowledgeManager, base: str) -> set[int]:
    base_id = int(kb.store.get_base(base)["id"])
    rows = kb.store.conn.execute(
        "SELECT c.dim FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE d.base_id = ?", (base_id,)
    ).fetchall()
    return {int(r["dim"]) for r in rows}


def test_reindex_failure_leaves_base_untouched(settings: Settings) -> None:
    kb = KnowledgeManager(settings, embedder=HashEmbedder(dim=64))
    kb.create_base("notes")
    kb.add_text("notes", "alpha bravo charlie delta", title="n1")
    kb.add_text("notes", "echo foxtrot golf hotel", title="n2")
    assert _chunk_dims(kb, "notes") == {64}
    kb.close()

    flaky = _FlakyEmbedder(dim=128, fail_at=2)
    other = KnowledgeManager(settings, embedder=flaky)
    with pytest.raises(KnowledgeError):
        other.reindex("notes")
    assert flaky.calls == 2
    # Rien n'a été écrit : dimensions homogènes, étiquette inchangée.
    assert _chunk_dims(other, "notes") == {64}
    assert other.get_base("notes").embedder == "hash:64"
    other.close()

    back = KnowledgeManager(settings, embedder=HashEmbedder(dim=64))
    assert [h.title for h in back.search("alpha bravo", mode="vector")][:1] == ["n1"]
    assert [h.title for h in back.search("echo foxtrot", mode="vector")][:1] == ["n2"]
    back.close()


def test_reindex_success_rewrites_all_chunks(settings: Settings) -> None:
    kb = KnowledgeManager(settings, embedder=HashEmbedder(dim=64))
    kb.create_base("notes")
    kb.add_text("notes", "alpha bravo charlie delta", title="n1")
    kb.add_text("notes", "echo foxtrot golf hotel", title="n2")
    kb.close()

    other = KnowledgeManager(settings, embedder=HashEmbedder(dim=128))
    n = other.reindex("notes")
    assert n == other.get_base("notes").n_chunks == 2
    assert _chunk_dims(other, "notes") == {128}
    assert other.get_base("notes").embedder == "hash:128"
    assert [h.title for h in other.search("alpha bravo", mode="vector")][:1] == ["n1"]
    assert [h.title for h in other.search("echo foxtrot", mode="vector")][:1] == ["n2"]
    other.close()
