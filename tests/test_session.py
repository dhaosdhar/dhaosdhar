"""Tests des sessions persistées (JSONL) et du magasin de sessions."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dhaos.agent.session import Session, SessionStore
from dhaos.config import Settings
from dhaos.types import Message, ToolCall

ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def _sample_messages() -> list[Message]:
    return [
        Message(role="user", content="Bonjour — accents : éàü"),
        Message(
            role="assistant",
            content="Je lis le fichier.",
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
            raw=[{"type": "text", "text": "Je lis le fichier."}],
        ),
        Message(role="tool", content="print('x')", tool_call_id="c1", name="read_file"),
        Message(role="tool", content="Outil inconnu : nope", tool_call_id="c2", name="nope", is_error=True),
        Message(role="assistant", content="Voilà."),
    ]


# ---------------------------------------------------------------- Session


def test_session_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "s1.jsonl"
    session = Session(id="s1", path=path, meta={"title": "Test", "backend": "fake"})
    for m in _sample_messages():
        session.append(m)
    session.log_event("tool", name="read_file", is_error=False, chars=9)
    session.log_event("usage", input_tokens=10, output_tokens=5)
    session.save()

    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))

    lines = path.read_text(encoding="utf-8").splitlines()
    kinds = [json.loads(ln)["type"] for ln in lines]
    assert kinds == ["meta"] + ["message"] * 5 + ["event"] * 2
    first = json.loads(lines[0])
    assert first["id"] == "s1" and first["title"] == "Test"
    assert "éàü" in lines[1]  # pas d'échappement ASCII forcé

    loaded = Session.load(path)
    assert loaded.id == "s1"
    assert loaded.meta["title"] == "Test" and loaded.meta["backend"] == "fake"
    assert [m.to_dict() for m in loaded.messages] == [m.to_dict() for m in session.messages]
    assert loaded.messages[1].tool_calls[0].arguments == {"path": "a.py"}
    assert loaded.messages[1].raw == [{"type": "text", "text": "Je lis le fichier."}]
    assert loaded.messages[3].is_error is True and loaded.messages[3].tool_call_id == "c2"
    assert [e["kind"] for e in loaded.events] == ["tool", "usage"]
    assert loaded.events[0]["name"] == "read_file" and "at" in loaded.events[0]
    assert loaded.events[1]["input_tokens"] == 10


def test_session_save_overwrites_atomically(tmp_path: Path) -> None:
    path = tmp_path / "s2.jsonl"
    session = Session(id="s2", path=path)
    session.append(Message(role="user", content="un"))
    session.save()
    session.append(Message(role="assistant", content="deux"))
    session.save()
    loaded = Session.load(path)
    assert [m.content for m in loaded.messages] == ["un", "deux"]
    assert sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if '"meta"' in ln) == 1
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]


def test_session_load_ignores_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "s3.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "meta", "id": "s3", "title": "T"}),
                "ceci n'est pas du JSON",
                json.dumps(["une", "liste"]),
                json.dumps({"type": "message", "role": "user", "content": "ok"}),
                json.dumps({"type": "message", "role": "martien", "content": "?"}),
                json.dumps({"type": "message", "content": "sans rôle"}),
                json.dumps({"type": "inconnu", "x": 1}),
                "",
                json.dumps({"type": "event", "kind": "usage", "input_tokens": 1}),
                json.dumps({"type": "message", "role": "assistant", "content": "fin"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = Session.load(path)
    assert loaded.id == "s3" and loaded.meta["title"] == "T"
    assert [m.content for m in loaded.messages] == ["ok", "fin"]
    assert loaded.events == [{"kind": "usage", "input_tokens": 1}]


def test_session_load_without_meta_uses_filename(tmp_path: Path) -> None:
    path = tmp_path / "abcd.jsonl"
    path.write_text(json.dumps({"type": "message", "role": "user", "content": "x"}) + "\n", encoding="utf-8")
    loaded = Session.load(path)
    assert loaded.id == "abcd" and loaded.meta["id"] == "abcd"
    assert len(loaded.messages) == 1


def test_session_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Session.load(tmp_path / "absent.jsonl")


# ------------------------------------------------------------ SessionStore


def test_store_create_writes_file_and_meta(settings: Settings) -> None:
    store = SessionStore(settings)
    session = store.create(title="Titre", backend="fake", model="fake-model")
    assert ID_RE.match(session.id), session.id
    assert session.path == settings.sessions_dir / f"{session.id}.jsonl"
    assert session.path.is_file()
    meta = session.meta
    assert meta["id"] == session.id
    assert meta["title"] == "Titre" and meta["backend"] == "fake" and meta["model"] == "fake-model"
    assert meta["created_at"] and meta["updated_at"] == meta["created_at"]
    assert meta["traces"] is settings.agent.collect_traces
    on_disk = json.loads(session.path.read_text(encoding="utf-8").splitlines()[0])
    assert on_disk["type"] == "meta" and on_disk["id"] == session.id and on_disk["traces"] is True


def test_store_create_honours_collect_traces(settings: Settings) -> None:
    settings.agent.collect_traces = False
    session = SessionStore(settings).create()
    assert session.meta["traces"] is False


def test_store_ids_are_unique(settings: Settings) -> None:
    store = SessionStore(settings)
    ids = {store.create().id for _ in range(5)}
    assert len(ids) == 5


def test_store_get_roundtrip_and_missing(settings: Settings) -> None:
    store = SessionStore(settings)
    session = store.create(title="A")
    session.append(Message(role="user", content="salut"))
    session.save()
    again = store.get(session.id)
    assert again.id == session.id and again.meta["title"] == "A"
    assert [m.to_dict() for m in again.messages] == [m.to_dict() for m in session.messages]
    with pytest.raises(KeyError):
        store.get("20260101-000000-ffff")


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "..", "a b", "é", "x.jsonl", "\x00"])
def test_store_rejects_unsafe_ids(settings: Settings, bad: str) -> None:
    store = SessionStore(settings)
    with pytest.raises(KeyError):
        store.get(bad)
    with pytest.raises(KeyError):  # ne doit jamais toucher hors du dossier
        store.delete(bad)
    assert not (settings.sessions_dir.parent / "x.jsonl").exists()


def test_store_delete_never_escapes_directory(settings: Settings) -> None:
    store = SessionStore(settings)
    outside = settings.sessions_dir.parent / "victime.jsonl"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(KeyError):
        store.delete("../victime")
    assert outside.is_file()


def test_store_list_sorted_most_recent_first(settings: Settings) -> None:
    store = SessionStore(settings)
    stamps = ["2026-01-01T10:00:00+00:00", "2026-03-01T10:00:00+00:00", "2026-02-01T10:00:00+00:00"]
    created = []
    for i, stamp in enumerate(stamps):
        s = store.create(title=f"s{i}", backend="fake", model="m")
        s.meta["updated_at"] = stamp
        for _ in range(i):
            s.append(Message(role="user", content="x"))
        s.save()
        created.append(s)

    infos = store.list()
    assert [i.title for i in infos] == ["s1", "s2", "s0"]
    assert [i.n_messages for i in infos] == [1, 2, 0]
    by_id = {i.id: i for i in infos}
    for s in created:
        info = by_id[s.id]
        assert info.backend == "fake" and info.model == "m"
        assert info.created_at == s.meta["created_at"]
        assert info.updated_at == s.meta["updated_at"]
        assert info.path == s.path


def test_store_list_ignores_unreadable_files(settings: Settings) -> None:
    store = SessionStore(settings)
    good = store.create(title="ok")
    (settings.sessions_dir / "20260101-000000-dead.jsonl").write_text("pas du json\n", encoding="utf-8")
    (settings.sessions_dir / "vide.jsonl").write_text("", encoding="utf-8")
    (settings.sessions_dir / "notes.txt").write_text("{}", encoding="utf-8")
    infos = store.list()
    ids = [i.id for i in infos]
    assert good.id in ids and "notes" not in ids
    assert all(isinstance(i.n_messages, int) for i in infos)


def test_store_delete(settings: Settings) -> None:
    store = SessionStore(settings)
    session = store.create()
    assert store.delete(session.id) is True
    assert not session.path.exists()
    assert store.delete(session.id) is False
    with pytest.raises(KeyError):
        store.get(session.id)
    assert session.id not in [i.id for i in store.list()]


def test_store_list_empty(settings: Settings) -> None:
    assert SessionStore(settings).list() == []
