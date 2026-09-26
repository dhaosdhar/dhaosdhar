"""Tests de la boucle agentique (backend scripté, outil factice)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dhaos.agent.loop import ITERATION_LIMIT_NOTICE, Agent, AgentResult
from dhaos.agent.session import Session, SessionStore
from dhaos.backends.base import Backend, BackendError
from dhaos.config import Settings
from dhaos.tools.base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult
from dhaos.types import ChatResponse, Message, ToolCall, Usage

from .fakes import FakeBackend, tool_call, tool_response


class EchoTool(Tool):
    name = "echo"
    description = "Renvoie ses arguments."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(json.dumps(args, ensure_ascii=False))


class FailingTool(Tool):
    name = "boom"
    description = "Échoue toujours."
    parameters = {"type": "object", "properties": {}}

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise ToolError("panne volontaire")


class ErrorBackend(Backend):
    name = "broken"
    model = "none"

    def chat(self, messages, *, system="", tools=None, on_text=None, on_thinking=None) -> ChatResponse:
        raise BackendError("connexion refusée")


@pytest.fixture
def echo() -> EchoTool:
    return EchoTool()


@pytest.fixture
def registry(echo: EchoTool) -> ToolRegistry:
    return ToolRegistry([echo, FailingTool()])


def make_agent(settings: Settings, ctx: ToolContext, registry: ToolRegistry, responses: list, **kw) -> tuple[Agent, FakeBackend]:
    backend = FakeBackend(responses)
    return Agent(settings, backend, registry, ctx, **kw), backend


# ------------------------------------------------------------- réponses simples


def test_simple_reply(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    agent, backend = make_agent(settings, ctx, registry, ["Bonjour !"])
    result = agent.run("salut")
    assert isinstance(result, AgentResult)
    assert result.text == "Bonjour !"
    assert result.stop_reason == "end_turn"
    assert result.iterations == 0 and result.tool_calls == 0 and result.error is None
    assert [m.role for m in agent.messages] == ["user", "assistant"]
    assert agent.messages[0].content == "salut"

    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["system"] == agent.system_prompt and "# Rôle" in call["system"]
    assert sorted(t.name for t in call["tools"]) == ["boom", "echo"]
    assert call["messages"][0].role == "user"


def test_on_text_called(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    agent, _ = make_agent(settings, ctx, registry, ["morceau"])
    chunks: list[str] = []
    agent.run("x", on_text=chunks.append)
    assert chunks == ["morceau"]


def test_empty_registry_sends_no_tools(settings: Settings, ctx: ToolContext) -> None:
    agent, backend = make_agent(settings, ctx, ToolRegistry(), ["ok"])
    agent.run("x")
    assert backend.calls[0]["tools"] == []
    assert "Aucun outil" in agent.system_prompt


# ------------------------------------------------------------- outils


def test_tool_chain(settings: Settings, ctx: ToolContext, registry: ToolRegistry, echo: EchoTool) -> None:
    agent, backend = make_agent(
        settings, ctx, registry, [tool_response(tool_call("echo", "call_42", text="hi"), text="Je regarde."), "Fini"]
    )
    seen_calls: list[ToolCall] = []
    seen_results: list[tuple[ToolCall, ToolResult]] = []
    result = agent.run("fais écho", on_tool_call=seen_calls.append, on_tool_result=lambda c, r: seen_results.append((c, r)))

    assert result.text == "Fini"
    assert result.stop_reason == "end_turn"
    assert result.iterations == 1 and result.tool_calls == 1
    assert result.usage.input_tokens == 10 and result.usage.output_tokens == 5
    assert echo.calls == [{"text": "hi"}]

    roles = [m.role for m in agent.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assistant = agent.messages[1]
    assert assistant.content == "Je regarde." and assistant.tool_calls[0].id == "call_42"
    tool_msg = agent.messages[2]
    assert tool_msg.tool_call_id == "call_42"
    assert tool_msg.name == "echo"
    assert tool_msg.is_error is False
    assert json.loads(tool_msg.content) == {"text": "hi"}

    assert len(backend.calls) == 2
    second = backend.calls[1]
    assert second["system"] == agent.system_prompt
    assert [t.name for t in second["tools"]] == ["echo", "boom"]
    assert [m.role for m in second["messages"]] == ["user", "assistant", "tool"]

    assert [c.id for c in seen_calls] == ["call_42"]
    assert seen_results[0][0].id == "call_42" and seen_results[0][1].content == tool_msg.content


def test_multiple_calls_in_one_turn(settings: Settings, ctx: ToolContext, registry: ToolRegistry, echo: EchoTool) -> None:
    agent, backend = make_agent(
        settings,
        ctx,
        registry,
        [tool_response(tool_call("echo", "c1", text="a"), tool_call("echo", "c2", text="b")), "ok"],
    )
    result = agent.run("deux")
    assert result.iterations == 1 and result.tool_calls == 2
    assert echo.calls == [{"text": "a"}, {"text": "b"}]
    tools = [m for m in agent.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tools] == ["c1", "c2"]
    assert [m.name for m in tools] == ["echo", "echo"]
    assert [m.role for m in backend.calls[1]["messages"]] == ["user", "assistant", "tool", "tool"]


def test_unknown_tool_is_error_and_loop_continues(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    agent, backend = make_agent(settings, ctx, registry, [tool_response(tool_call("nope", "c9", x=1)), "après"])
    result = agent.run("x")
    assert result.text == "après" and result.stop_reason == "end_turn"
    tool_msg = agent.messages[2]
    assert tool_msg.role == "tool" and tool_msg.is_error is True
    assert tool_msg.tool_call_id == "c9" and tool_msg.name == "nope"
    assert "Outil inconnu" in tool_msg.content
    assert len(backend.calls) == 2


def test_tool_failure_and_invalid_args_are_errors(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    agent, _ = make_agent(
        settings,
        ctx,
        registry,
        [tool_response(tool_call("boom", "c1"), tool_call("echo", "c2", text=123)), "fin"],
    )
    result = agent.run("x")
    assert result.text == "fin"
    tools = [m for m in agent.messages if m.role == "tool"]
    assert [m.is_error for m in tools] == [True, True]
    assert "panne volontaire" in tools[0].content
    assert "INVALID_JSON" in tools[1].content


# ------------------------------------------------------------- arrêts


def test_refusal_stops_cleanly(settings: Settings, ctx: ToolContext, registry: ToolRegistry, echo: EchoTool) -> None:
    resp = ChatResponse(text="Je refuse.", tool_calls=[tool_call("echo", text="x")], stop_reason="refusal")
    agent, backend = make_agent(settings, ctx, registry, [resp, "jamais"])
    result = agent.run("x")
    assert result.stop_reason == "refusal"
    assert result.text == "Je refuse."
    assert result.tool_calls == 0 and echo.calls == []
    assert len(backend.calls) == 1
    assert [m.role for m in agent.messages] == ["user", "assistant"]


def test_max_tokens_does_not_execute_tools(settings: Settings, ctx: ToolContext, registry: ToolRegistry, echo: EchoTool) -> None:
    resp = ChatResponse(text="trop long", tool_calls=[tool_call("echo", text="x")], stop_reason="max_tokens")
    agent, backend = make_agent(settings, ctx, registry, [resp, "jamais"])
    result = agent.run("x")
    assert result.stop_reason == "max_tokens"
    assert result.text == "trop long"
    assert echo.calls == [] and len(backend.calls) == 1


def test_max_iterations(settings: Settings, ctx: ToolContext, registry: ToolRegistry, echo: EchoTool) -> None:
    settings.agent.max_iterations = 2
    responses = [tool_response(tool_call("echo", f"c{i}", text=str(i))) for i in range(6)]
    agent, backend = make_agent(settings, ctx, registry, responses)
    result = agent.run("boucle")

    assert result.stop_reason == "max_iterations"
    assert result.iterations == 2 and result.tool_calls == 2
    assert echo.calls == [{"text": "0"}, {"text": "1"}]
    assert len(backend.calls) == 3
    assert backend.calls[0]["tools"] and backend.calls[1]["tools"]
    assert backend.calls[2]["tools"] == []  # dernier appel sans outils
    last_user = [m for m in backend.calls[2]["messages"] if m.role == "user"][-1]
    assert last_user.content == ITERATION_LIMIT_NOTICE
    assert last_user.content.startswith("[système]")
    assert agent.messages[-1].role == "assistant"
    assert result.text == agent.messages[-1].content


def test_backend_error(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    store = SessionStore(settings)
    session = store.create(backend="broken")
    agent = Agent(settings, ErrorBackend(), registry, ctx, session=session)
    result = agent.run("x")
    assert result.stop_reason == "error"
    assert result.text == ""
    assert result.error == "connexion refusée"
    assert [m.role for m in agent.messages] == ["user"]
    reloaded = Session.load(session.path)
    assert [m.content for m in reloaded.messages] == ["x"]
    usage_events = [e for e in reloaded.events if e["kind"] == "usage"]
    assert usage_events and usage_events[-1]["stop_reason"] == "error"
    assert usage_events[-1]["error"] == "connexion refusée"


# ------------------------------------------------------------- session


def test_session_persisted_and_reloadable(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    store = SessionStore(settings)
    session = store.create(backend="fake", model="fake-model")
    created_at = session.meta["created_at"]
    agent, backend = make_agent(
        settings, ctx, registry, [tool_response(tool_call("echo", "c1", text="a")), "réponse finale"], session=session
    )
    result = agent.run("première question\nsuite")
    assert result.text == "réponse finale"

    assert session.meta["title"] == "première question"
    assert session.meta["updated_at"] >= created_at
    assert [m.role for m in session.messages] == ["user", "assistant", "tool", "assistant"]

    reloaded = Session.load(session.path)
    assert [m.to_dict() for m in reloaded.messages] == [m.to_dict() for m in agent.messages]
    kinds = [e["kind"] for e in reloaded.events]
    assert kinds == ["tool", "usage"]
    assert reloaded.events[0]["name"] == "echo" and reloaded.events[0]["is_error"] is False
    assert reloaded.events[1]["tool_calls"] == 1 and reloaded.events[1]["stop_reason"] == "end_turn"
    assert reloaded.meta["backend"] == "fake" and reloaded.meta["model"] == "fake-model"
    assert store.get(session.id).meta["title"] == "première question"

    # Reprise : un nouvel agent sur la session rechargée conserve l'historique.
    agent2, backend2 = make_agent(settings, ctx, registry, ["encore"], session=reloaded)
    assert [m.to_dict() for m in agent2.messages] == [m.to_dict() for m in agent.messages]
    agent2.run("deuxième")
    assert [m.role for m in backend2.calls[0]["messages"]] == ["user", "assistant", "tool", "assistant", "user"]
    again = Session.load(session.path)
    assert len(again.messages) == 6
    assert [e["kind"] for e in again.events] == ["tool", "usage", "usage"]


def test_session_title_not_overwritten(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    session = SessionStore(settings).create(title="Mon titre")
    agent, _ = make_agent(settings, ctx, registry, ["ok"], session=session)
    agent.run("question")
    assert session.meta["title"] == "Mon titre"


def test_reset(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    session = SessionStore(settings).create()
    agent, backend = make_agent(settings, ctx, registry, ["un", "deux"], session=session)
    agent.run("a")
    assert len(agent.messages) == 2
    agent.reset()
    assert agent.messages == []
    assert session.messages == [] and session.events == []
    reloaded = Session.load(session.path)
    assert reloaded.messages == [] and reloaded.events == []
    assert reloaded.meta["id"] == session.id
    agent.run("b")
    assert [m.role for m in backend.calls[1]["messages"]] == ["user"]


def test_without_session_nothing_written(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    agent, _ = make_agent(settings, ctx, registry, ["ok"])
    agent.run("x")
    assert list(settings.sessions_dir.glob("*.jsonl")) == []


# ------------------------------------------------------------- prompt système


def test_system_prompt_includes_bases_and_language(settings: Settings, policy, registry: ToolRegistry) -> None:
    from dhaos.policy import Journal, auto_confirm

    kb = SimpleNamespace(
        list_bases=lambda: [SimpleNamespace(name="dev", description="Base développeur", n_docs=3)]
    )
    ctx = ToolContext(settings=settings, policy=policy, journal=Journal(settings.journal_path), confirm=auto_confirm, kb=kb)
    agent, backend = make_agent(settings, ctx, registry, ["ok"])
    prompt = agent.system_prompt
    assert "- dev : Base développeur (3 docs)" in prompt
    assert "kb_search" in prompt
    assert "Réponds en français" in prompt
    assert str(ctx.project_root) in prompt
    assert "fake" in prompt
    assert "echo" in prompt and "boom" in prompt
    agent.run("x")
    assert backend.calls[0]["system"] == prompt


def test_system_prompt_survives_broken_kb(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    def broken() -> list:
        raise RuntimeError("base corrompue")

    ctx.kb = SimpleNamespace(list_bases=broken)
    agent, _ = make_agent(settings, ctx, registry, ["ok"])
    assert "Aucune base de savoir" in agent.system_prompt


def test_custom_system_prompt(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    agent, backend = make_agent(settings, ctx, registry, ["ok"], system_prompt="PROMPT PERSO")
    assert agent.system_prompt == "PROMPT PERSO"
    agent.run("x")
    assert backend.calls[0]["system"] == "PROMPT PERSO"


def test_messages_property_is_a_copy(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    agent, _ = make_agent(settings, ctx, registry, ["ok"])
    agent.run("x")
    snapshot = agent.messages
    snapshot.append(Message(role="user", content="intrus"))
    assert len(agent.messages) == 2


def test_usage_accumulates(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    responses = [
        tool_response(tool_call("echo", "c1", text="a")),
        ChatResponse(text="fin", usage=Usage(input_tokens=100, output_tokens=7)),
    ]
    agent, _ = make_agent(settings, ctx, registry, responses)
    result = agent.run("x")
    assert (result.usage.input_tokens, result.usage.output_tokens) == (110, 12)
