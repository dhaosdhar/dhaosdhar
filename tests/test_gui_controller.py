"""Contrôleur de l'interface de bureau, sans Tk : conversations simultanées,
événements, confirmations, reprise de session."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from dhaos.config import Settings
from dhaos.gui.controller import Event, GuiController
from dhaos.kb.manager import KnowledgeManager
from dhaos.types import ChatResponse

from .fakes import FakeBackend, tool_call, tool_response


class Factory:
    """Une liste de réponses par conversation (ordre de création des backends)."""

    def __init__(self, scripts: list[list[str | ChatResponse]]) -> None:
        self.scripts = list(scripts)
        self.backends: list[FakeBackend] = []

    def __call__(self, settings: Settings, name: str | None, model: str | None) -> FakeBackend:
        fb = FakeBackend(self.scripts.pop(0) if self.scripts else ["ok"])
        fb.name = name or "ollama"
        fb.model = model or "dhaos"
        self.backends.append(fb)
        return fb


def collect(ctrl: GuiController, conv_ids: set[str], *, until: str = "done", timeout: float = 20.0, on_event=None) -> dict[str, list[Event]]:
    """Consomme les événements jusqu'à ``until`` pour chaque conversation."""
    seen: dict[str, list[Event]] = {c: [] for c in conv_ids}
    finished: set[str] = set()
    deadline = time.monotonic() + timeout
    while finished != conv_ids and time.monotonic() < deadline:
        for ev in ctrl.drain(timeout=0.2):
            if ev.conv_id in seen:
                seen[ev.conv_id].append(ev)
                if on_event:
                    on_event(ev)
                if ev.kind in (until, "error"):
                    finished.add(ev.conv_id)
    assert finished == conv_ids, f"conversations sans fin de tour : {conv_ids - finished}"
    return seen


@pytest.fixture
def ctrl(settings: Settings):
    controller = GuiController(settings, backend_factory=Factory([]), kb_factory=KnowledgeManager)
    yield controller
    controller.shutdown()


def test_simple_turn_streams_text_and_done(settings: Settings) -> None:
    ctrl = GuiController(settings, backend_factory=Factory([["Bonjour !"]]), kb_factory=KnowledgeManager)
    conv = ctrl.new_conversation()
    assert conv.title == "Nouvelle conversation" and ctrl.send(conv.id, "salut")
    assert ctrl.send(conv.id, "encore") is False  # occupée
    events = collect(ctrl, {conv.id})[conv.id]
    kinds = [e.kind for e in events]
    assert kinds[0] == "status" and "text" in kinds and kinds[-1] == "done"
    done = events[-1].data
    assert done["text"] == "Bonjour !" and done["session_id"] == conv.session.id and done["title"] == "salut"
    assert conv.busy is False and conv.title == "salut"
    assert [m.role for m in conv.session.messages] == ["user", "assistant"]
    assert ctrl.list_sessions()[0].id == conv.session.id
    ctrl.shutdown()


def test_two_conversations_run_concurrently(settings: Settings) -> None:
    class SlowBackend(FakeBackend):
        def chat(self, *a: Any, **kw: Any) -> ChatResponse:
            time.sleep(0.4)
            return super().chat(*a, **kw)

    class SlowFactory(Factory):
        def __call__(self, settings: Settings, name: str | None, model: str | None) -> FakeBackend:
            fb = SlowBackend(self.scripts.pop(0)); fb.name = name or "ollama"; self.backends.append(fb); return fb

    ctrl = GuiController(settings, backend_factory=SlowFactory([["réponse A"], ["réponse B"]]), kb_factory=KnowledgeManager)
    a = ctrl.new_conversation(); b = ctrl.new_conversation(backend="claude", model="claude-sonnet-5")
    t0 = time.monotonic()
    assert ctrl.send(a.id, "question A") and ctrl.send(b.id, "question B")
    seen = collect(ctrl, {a.id, b.id})
    elapsed = time.monotonic() - t0
    assert seen[a.id][-1].data["text"] == "réponse A" and seen[b.id][-1].data["text"] == "réponse B"
    assert elapsed < 0.75, f"les deux tours auraient dû tourner en parallèle ({elapsed:.2f}s)"
    assert b.backend_name == "claude" and b.session.meta["backend"] == "claude"
    ctrl.shutdown()


def test_tool_round_with_inline_confirmation(settings: Settings, project_root: Path) -> None:
    settings.tools.shell_policy = "ask"
    settings.api.confirm_timeout = 10
    script: list[str | ChatResponse] = [tool_response(tool_call("run_command", "c1", command="touch gui-ok.txt")), "Fait."]
    ctrl = GuiController(settings, backend_factory=Factory([script]), kb_factory=KnowledgeManager)
    conv = ctrl.new_conversation()
    ctrl.send(conv.id, "crée le fichier")

    def on_event(ev: Event) -> None:
        if ev.kind == "confirm":
            assert ev.data["prompt"].startswith("Exécuter : touch gui-ok.txt") and ctrl.pending_confirmations() == 1
            assert ctrl.answer_confirm(ev.data["id"], True) is True

    events = collect(ctrl, {conv.id}, on_event=on_event)[conv.id]
    kinds = [e.kind for e in events if e.kind != "status"]
    assert kinds == ["tool_call", "confirm", "tool_result", "text", "done"], kinds
    assert events[[e.kind for e in events].index("tool_result")].data["is_error"] is False
    assert (project_root / "gui-ok.txt").exists() and ctrl.pending_confirmations() == 0
    assert ctrl.answer_confirm("inconnu", True) is False
    ctrl.shutdown()


def test_confirmation_refused(settings: Settings, project_root: Path) -> None:
    settings.tools.shell_policy = "ask"
    script: list[str | ChatResponse] = [tool_response(tool_call("run_command", "c1", command="touch non.txt")), "ok"]
    ctrl = GuiController(settings, backend_factory=Factory([script]), kb_factory=KnowledgeManager)
    conv = ctrl.new_conversation()
    ctrl.send(conv.id, "x")
    events = collect(ctrl, {conv.id}, on_event=lambda ev: ev.kind == "confirm" and ctrl.answer_confirm(ev.data["id"], False))[conv.id]
    result = next(e for e in events if e.kind == "tool_result").data
    assert result["is_error"] is True and "refus" in result["preview"] and not (project_root / "non.txt").exists()
    ctrl.shutdown()


def test_open_session_keeps_thread_and_continues(settings: Settings) -> None:
    ctrl = GuiController(settings, backend_factory=Factory([["première"], ["seconde"]]), kb_factory=KnowledgeManager)
    conv = ctrl.new_conversation()
    ctrl.send(conv.id, "un"); collect(ctrl, {conv.id})
    sid = conv.session.id
    ctrl.close_conversation(conv.id)
    assert ctrl.get(conv.id) is None and ctrl.list_sessions()[0].id == sid
    again = ctrl.open_session(sid)
    assert [m.content for m in again.session.messages] == ["un", "première"] and again.title == "un"
    ctrl.send(again.id, "deux"); collect(ctrl, {again.id})
    assert [m.content for m in ctrl.store.get(sid).messages] == ["un", "première", "deux", "seconde"]
    ctrl.shutdown()


def test_unused_conversation_leaves_no_session(ctrl: GuiController) -> None:
    conv = ctrl.new_conversation()
    assert ctrl.list_sessions() and ctrl.list_sessions()[0].id == conv.session.id
    ctrl.close_conversation(conv.id)
    assert ctrl.list_sessions() == []
    assert ctrl.delete_session("inexistant") is False


def test_backend_error_becomes_error_event(settings: Settings) -> None:
    class Broken(Factory):
        def __call__(self, settings: Settings, name: str | None, model: str | None) -> FakeBackend:
            raise RuntimeError("pas de backend")

    ctrl = GuiController(settings, backend_factory=Broken([]), kb_factory=KnowledgeManager)
    conv = ctrl.new_conversation()
    ctrl.send(conv.id, "x")
    events = collect(ctrl, {conv.id})[conv.id]
    assert events[-1].kind == "error" and "pas de backend" in events[-1].data["detail"] and conv.busy is False
    assert ctrl.new_conversation(backend="ollama").backend_name == "ollama"
    with pytest.raises(ValueError):
        ctrl.new_conversation(backend="inconnu")
    ctrl.shutdown()


def test_models_and_health(ctrl: GuiController) -> None:
    assert "claude-opus-5" in ctrl.models("claude")
    assert ctrl.health("ollama")["ok"] is True  # FakeBackend
    assert ctrl.default_model("ollama") == "dhaos" and ctrl.default_model("claude") == "claude-opus-5"
