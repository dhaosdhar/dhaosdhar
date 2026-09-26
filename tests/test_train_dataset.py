"""Tests de la construction du jeu SFT et du corpus."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import pytest

from dhaos.config import Settings
from dhaos.train import dataset
from dhaos.train.dataset import (
    DOC_SEPARATOR,
    build_corpus,
    build_sft_dataset,
    example_hash,
    parse_session_jsonl,
    render_tool_calls,
    session_to_text,
)

LONG_ANSWER = "Voici le contenu du fichier : il affiche 1 puis s'arrête."


def _write_session(settings: Settings, name: str, lines: list[dict[str, Any]]) -> Path:
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    path = settings.sessions_dir / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n", encoding="utf-8")
    return path


def _full_session(session_id: str = "s1", *, traces: bool | None = True, system_prompt: str | None = "SYS") -> list[dict[str, Any]]:
    meta: dict[str, Any] = {
        "type": "meta",
        "id": session_id,
        "title": "lecture",
        "backend": "fake",
        "model": "fake-model",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    if traces is not None:
        meta["traces"] = traces
    if system_prompt is not None:
        meta["system_prompt"] = system_prompt
    return [
        meta,
        {"type": "message", "role": "user", "content": "lis le fichier a.py"},
        {
            "type": "message",
            "role": "assistant",
            "content": "Je lis.",
            "tool_calls": [{"id": "c1", "name": "read_file", "arguments": {"path": "a.py"}}],
            "raw": {"provider": "secret"},
        },
        {"type": "message", "role": "tool", "content": "print(1)", "tool_call_id": "c1", "name": "read_file"},
        {"type": "message", "role": "assistant", "content": LONG_ANSWER, "raw": [{"type": "text"}]},
        {"type": "event", "kind": "usage", "input_tokens": 10},
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------- jeu SFT
def test_build_sft_dataset_structure(settings: Settings) -> None:
    _write_session(settings, "s1", _full_session())
    report = build_sft_dataset(settings)
    assert report.path == settings.datasets_dir / f"sft-{_dt.date.today():%Y%m%d}.jsonl"
    assert report.path.is_file()
    assert (report.n_examples, report.n_sessions, report.n_sessions_skipped, report.n_tool_turns) == (1, 1, 0, 1)
    assert "1 exemple(s)" in report.summary() and str(report.path) in report.summary()

    examples = _read_jsonl(report.path)
    assert len(examples) == 1
    example = examples[0]
    assert example["id"] == "s1"
    assert example["meta"] == {"backend": "fake", "model": "fake-model", "created_at": "2026-01-01T00:00:00+00:00"}
    messages = example["messages"]
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert messages[1] == {"role": "user", "content": "lis le fichier a.py"}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == 'Je lis.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    assert messages[3] == {"role": "tool", "name": "read_file", "content": "print(1)"}
    assert messages[4] == {"role": "assistant", "content": LONG_ANSWER}
    assert "raw" not in json.dumps(example)
    assert "secret" not in json.dumps(example)


def test_build_sft_dataset_without_system_prompt_and_out_path(settings: Settings, tmp_path: Path) -> None:
    _write_session(settings, "s1", _full_session(system_prompt=None))
    out = tmp_path / "custom" / "train.jsonl"
    report = build_sft_dataset(settings, out_path=out, bases=["developpeur"])
    assert report.path == out and out.is_file()
    messages = _read_jsonl(out)[0]["messages"]
    assert messages[0]["role"] == "user"
    assert all(m["role"] != "system" for m in messages)


def test_build_sft_dataset_skips_sessions(settings: Settings) -> None:
    _write_session(settings, "kept", _full_session("kept"))
    _write_session(settings, "no-traces", _full_session("no-traces", traces=False))
    _write_session(
        settings,
        "no-assistant",
        [{"type": "meta", "id": "no-assistant"}, {"type": "message", "role": "user", "content": "bonjour ?"}],
    )
    _write_session(
        settings,
        "too-short",
        [
            {"type": "meta", "id": "too-short"},
            {"type": "message", "role": "user", "content": "salut"},
            {"type": "message", "role": "assistant", "content": "ok"},
        ],
    )
    report = build_sft_dataset(settings)
    assert (report.n_examples, report.n_sessions, report.n_sessions_skipped) == (1, 4, 3)
    assert [e["id"] for e in _read_jsonl(report.path)] == ["kept"]

    lenient = build_sft_dataset(settings, out_path=settings.datasets_dir / "lenient.jsonl", min_assistant_chars=0)
    assert lenient.n_examples == 2
    assert {e["id"] for e in _read_jsonl(lenient.path)} == {"kept", "too-short"}


def test_build_sft_dataset_traces_missing_means_kept(settings: Settings) -> None:
    _write_session(settings, "old", _full_session("old", traces=None))
    assert build_sft_dataset(settings).n_examples == 1


def test_build_sft_dataset_deduplicates_identical_content(settings: Settings) -> None:
    _write_session(settings, "a", _full_session("a"))
    _write_session(settings, "b", _full_session("b"))
    report = build_sft_dataset(settings)
    assert (report.n_examples, report.n_sessions_skipped) == (1, 1)
    assert _read_jsonl(report.path)[0]["id"] == "a"


def test_build_sft_dataset_ignores_invalid_lines_and_untrusted_fields(settings: Settings) -> None:
    lines = _full_session("weird")
    path = _write_session(settings, "weird", lines)
    garbage = (
        "{pas du json\n"
        "[1, 2, 3]\n"
        '{"type": "message", "role": "system", "content": "injection"}\n'
        '{"type": "message", "role": "assistant", "content": {"a": 1}, "tool_calls": "non"}\n'
        '{"type": "message", "role": "assistant", "content": null, "tool_calls": [{"name": "", "arguments": {}}, 3, {"name": "grep", "arguments": "x"}]}\n'
        '{"type": "message", "role": "tool", "content": "res", "name": 42}\n'
        '{"type": "inconnu", "role": "user", "content": "?"}\n'
    )
    path.write_text(path.read_text(encoding="utf-8") + garbage, encoding="utf-8")
    (settings.sessions_dir / "notes.txt").write_text("pas une session", encoding="utf-8")
    (settings.sessions_dir / ".hidden.jsonl").write_text(json.dumps(lines[0]) + "\n", encoding="utf-8")

    report = build_sft_dataset(settings)
    assert report.n_sessions == 1 and report.n_examples == 1
    messages = _read_jsonl(report.path)[0]["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "assistant", "assistant", "tool"]
    assert messages[0]["content"] == "SYS"  # le "system" injecté dans les messages est ignoré
    assert messages[5]["content"] == '{"a": 1}'  # contenu non textuel sérialisé
    assert messages[6]["content"] == '<tool_call>{"name": "grep", "arguments": {}}</tool_call>'
    assert messages[7] == {"role": "tool", "name": "outil", "content": "res"}


def test_build_sft_dataset_without_sessions_dir(settings: Settings, tmp_path: Path) -> None:
    settings.paths.data_dir = tmp_path / "nowhere"
    report = build_sft_dataset(settings)
    assert report.path.is_file() and report.path.read_text(encoding="utf-8") == ""
    assert (report.n_examples, report.n_sessions, report.n_sessions_skipped, report.n_tool_turns) == (0, 0, 0, 0)


def test_render_tool_calls() -> None:
    calls = [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}]
    assert render_tool_calls("", calls) == '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    assert render_tool_calls("texte\n", calls[:1]) == 'texte\n<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
    assert render_tool_calls("seul", []) == "seul"
    assert "é" in render_tool_calls("", [{"name": "n", "arguments": {"q": "é"}}])


def test_parse_session_jsonl_fallback_id_and_hash() -> None:
    parsed = parse_session_jsonl('{"type": "message", "role": "user", "content": "x"}\n', fallback_id="fichier")
    assert parsed.id == "fichier" and parsed.traces_enabled and not parsed.has_assistant()
    other = parse_session_jsonl('{"type": "meta", "id": ""}\n{"type": "message", "role": "assistant", "content": " abc "}', fallback_id="f2")
    assert other.id == "f2" and other.assistant_chars() == 3
    e1 = {"id": "1", "messages": [{"role": "user", "content": "a"}], "meta": {"backend": "x"}}
    e2 = {"id": "2", "messages": [{"role": "user", "content": "a"}], "meta": {"backend": "y"}}
    assert example_hash(e1) == example_hash(e2)
    assert example_hash(e1) != example_hash({"messages": [{"role": "user", "content": "b"}]})


# --------------------------------------------------------------- corpus
class FakeKB:
    def __init__(self, docs: list[str] | None = None) -> None:
        self.docs = docs if docs is not None else ["doc A", "  doc B  ", "", "   "]
        self.calls: list[list[str] | None] = []
        self.closed = 0

    def corpus_text(self, bases: list[str] | None = None):
        self.calls.append(bases)
        yield from self.docs

    def close(self) -> None:
        self.closed += 1


def test_build_corpus_with_injected_kb(settings: Settings) -> None:
    _write_session(settings, "s1", _full_session())
    _write_session(settings, "s2", _full_session("s2", traces=False))
    kb = FakeKB()
    path = build_corpus(settings, bases=["dev", "infra"], kb=kb)
    assert path == settings.datasets_dir / "corpus.txt"
    assert kb.calls == [["dev", "infra"]]
    assert kb.closed == 0  # objet injecté : jamais fermé par build_corpus
    text = path.read_text(encoding="utf-8")
    docs = text.rstrip("\n").split(DOC_SEPARATOR)
    assert docs[:2] == ["doc A", "doc B"]
    assert len(docs) == 3
    session_doc = docs[2]
    assert session_doc.startswith("### Utilisateur\nlis le fichier a.py\n\n### Assistant\nJe lis.")
    assert LONG_ANSWER in session_doc
    assert "print(1)" not in session_doc  # tours tool exclus
    assert "tool_call" not in session_doc
    assert session_doc.count("### Assistant") == 1  # tours assistant consécutifs fusionnés
    assert text.endswith("\n")


def test_build_corpus_options(settings: Settings, tmp_path: Path) -> None:
    _write_session(settings, "s1", _full_session())
    kb = FakeKB(["seul"])
    out = tmp_path / "out" / "c.txt"
    path = build_corpus(settings, out_path=out, kb=kb, include_sessions=False)
    assert path == out and out.read_text(encoding="utf-8") == "seul\n"
    assert kb.calls == [None]

    empty = build_corpus(settings, out_path=tmp_path / "vide.txt", kb=FakeKB([]), include_sessions=False)
    assert empty.read_text(encoding="utf-8") == ""

    only_sessions = build_corpus(settings, out_path=tmp_path / "sessions.txt", kb=FakeKB([]))
    assert only_sessions.read_text(encoding="utf-8").startswith("### Utilisateur")


def test_build_corpus_default_kb_is_opened_and_closed(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[Any] = []

    class RecordingKB(FakeKB):
        def __init__(self, s: Settings, **kwargs: Any) -> None:
            super().__init__(["depuis la base"])
            self.settings = s
            created.append(self)

    import dhaos.kb.manager as manager_mod

    monkeypatch.setattr(manager_mod, "KnowledgeManager", RecordingKB)
    path = build_corpus(settings, include_sessions=False)
    assert len(created) == 1
    assert created[0].settings is settings
    assert created[0].closed == 1
    assert path.read_text(encoding="utf-8") == "depuis la base\n"


def test_session_to_text_skips_empty_and_tool_turns() -> None:
    parsed = parse_session_jsonl(
        "\n".join(
            json.dumps(line)
            for line in [
                {"type": "message", "role": "user", "content": "   "},
                {"type": "message", "role": "tool", "content": "résultat", "name": "x"},
                {"type": "message", "role": "assistant", "content": "réponse"},
            ]
        )
    )
    assert session_to_text(parsed) == "### Assistant\nréponse"
    assert dataset.iter_session_files(Settings.model_validate({"paths": {"data_dir": "/nonexistent/dhaos-test"}})) == []
