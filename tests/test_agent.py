"""Tests de la boucle agentique (backend scripté, outil factice)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dhaos.agent.loop import (
    INTERRUPTED_TOOL_NOTICE,
    ITERATION_LIMIT_NOTICE,
    LIMIT_TOOL_NOTICE,
    ORPHAN_TOOL_NOTICE,
    REFUSED_TOOL_NOTICE,
    TRUNCATED_TOOL_NOTICE,
    Agent,
    AgentResult,
    repair_orphan_tool_calls,
)
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


def orphan_tool_calls(messages: list[Message]) -> list[str]:
    """Ids d'appels d'outils (``tool_calls`` ou blocs ``tool_use`` de ``raw``)
    d'un message assistant sans message ``tool`` correspondant juste après."""
    orphans: list[str] = []
    for i, m in enumerate(messages):
        if m.role != "assistant":
            continue
        ids = [c.id for c in m.tool_calls]
        if isinstance(m.raw, list):
            ids += [b["id"] for b in m.raw if isinstance(b, dict) and b.get("type") == "tool_use" and b["id"] not in ids]
        answered = set()
        for nxt in messages[i + 1 :]:
            if nxt.role != "tool":
                break
            answered.add(nxt.tool_call_id)
        orphans.extend(t for t in ids if t not in answered)
    return orphans


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
    # L'appel refusé n'est pas exécuté mais reçoit un résultat en erreur.
    assert [m.role for m in agent.messages] == ["user", "assistant", "tool"]
    assert agent.messages[2].is_error and agent.messages[2].content == REFUSED_TOOL_NOTICE


def test_max_tokens_does_not_execute_tools(settings: Settings, ctx: ToolContext, registry: ToolRegistry, echo: EchoTool) -> None:
    resp = ChatResponse(text="trop long", tool_calls=[tool_call("echo", text="x")], stop_reason="max_tokens")
    agent, backend = make_agent(settings, ctx, registry, [resp, "suite"])
    result = agent.run("x")
    assert result.stop_reason == "max_tokens"
    assert result.text == "trop long"
    assert echo.calls == [] and len(backend.calls) == 1
    # Régression : au tour suivant, aucun appel d'outil ne reste sans résultat.
    agent.run("continue")
    assert orphan_tool_calls(agent.messages) == []
    assert [m.role for m in backend.calls[1]["messages"]] == ["user", "assistant", "tool", "user"]


def test_max_iterations(settings: Settings, ctx: ToolContext, registry: ToolRegistry, echo: EchoTool) -> None:
    settings.agent.max_iterations = 2
    responses = [tool_response(tool_call("echo", f"c{i}", text=str(i))) for i in range(2)] + ["conclusion"]
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
    assert result.text == agent.messages[-1].content == "conclusion"


def test_max_iterations_closes_ignored_tool_calls(settings: Settings, ctx: ToolContext, registry: ToolRegistry, echo: EchoTool) -> None:
    """Le modèle ignore la consigne de clôture et redemande un outil : l'appel
    n'est pas exécuté mais reçoit un résultat en erreur (pas d'orphelin)."""
    settings.agent.max_iterations = 1
    responses = [tool_response(tool_call("echo", "c0", text="0")), tool_response(tool_call("echo", "c1", text="1"), text="encore")]
    agent, backend = make_agent(settings, ctx, registry, responses)
    result = agent.run("boucle")
    assert result.stop_reason == "max_iterations" and result.tool_calls == 1
    assert echo.calls == [{"text": "0"}]
    assert len(backend.calls) == 2
    last = agent.messages[-1]
    assert last.role == "tool" and last.tool_call_id == "c1" and last.is_error and last.content == LIMIT_TOOL_NOTICE
    assert orphan_tool_calls(agent.messages) == []


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


# ------------------------------------------------------------- régressions : appels d'outils sans résultat


class InterruptingTool(Tool):
    """Simule un Ctrl-C pendant l'exécution (``KeyboardInterrupt`` n'est pas
    une ``Exception`` : le registre ne l'attrape pas)."""

    name = "interrupt"
    description = "Lève KeyboardInterrupt."
    parameters = {"type": "object", "properties": {}}

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise KeyboardInterrupt


def claude_raw(*calls: ToolCall, text: str = "je lance") -> list[dict[str, Any]]:
    """Blocs de contenu tels que le backend Claude les conserve dans ``raw``."""
    return [{"type": "text", "text": text}] + [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in calls
    ]


def two_calls_response() -> ChatResponse:
    calls = [tool_call("echo", "toolu_1", text="a"), tool_call("interrupt", "toolu_2")]
    return ChatResponse(text="je lance", tool_calls=calls, stop_reason="tool_use", raw=claude_raw(*calls))


def test_interrupted_tool_round_closes_pending_calls(settings: Settings, ctx: ToolContext, echo: EchoTool) -> None:
    """Ctrl-C pendant le 2e outil du tour : la session persistée contient un
    résultat pour chaque appel (le 2e en erreur) et reste exploitable."""
    from dhaos.backends.claude import ClaudeBackend

    registry = ToolRegistry([echo, InterruptingTool()])
    store = SessionStore(settings)
    session = store.create(backend="fake")
    agent, backend = make_agent(settings, ctx, registry, [two_calls_response(), "suite ok"], session=session)

    with pytest.raises(KeyboardInterrupt):
        agent.run("fais")
    assert echo.calls == [{"text": "a"}]

    reloaded = store.get(session.id)
    assert [(m.role, m.tool_call_id) for m in reloaded.messages] == [
        ("user", None), ("assistant", None), ("tool", "toolu_1"), ("tool", "toolu_2"),
    ]
    closing = reloaded.messages[3]
    assert closing.is_error and closing.name == "interrupt" and closing.content == INTERRUPTED_TOOL_NOTICE
    assert reloaded.messages[2].is_error is False
    usage = [e for e in reloaded.events if e["kind"] == "usage"][-1]
    assert usage["stop_reason"] == "interrupted" and "KeyboardInterrupt" in usage["error"]

    # Même agent (ChatLoop.turn continue après « tour interrompu ») : historique valide.
    result = agent.run("suite")
    assert result.text == "suite ok" and result.stop_reason == "end_turn"
    assert orphan_tool_calls(agent.messages) == []
    sent = backend.calls[1]["messages"]
    assert [m.role for m in sent] == ["user", "assistant", "tool", "tool", "user"]
    claude_messages = ClaudeBackend.to_claude_messages(sent)
    assistant_blocks = claude_messages[1]["content"]
    results = {b["tool_use_id"] for b in claude_messages[2]["content"] if b["type"] == "tool_result"}
    assert {b["id"] for b in assistant_blocks if b["type"] == "tool_use"} == results == {"toolu_1", "toolu_2"}

    # Reprise depuis le disque (`dhaos chat --session`) : idem.
    agent2, backend2 = make_agent(settings, ctx, registry, ["reprise"], session=store.get(session.id))
    assert orphan_tool_calls(agent2.messages) == []
    assert agent2.run("encore").text == "reprise"


def test_failing_callback_closes_pending_calls(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    """Un callback ``on_tool_result`` qui lève interrompt le tour sans laisser
    d'appel sans résultat."""
    calls = [tool_call("echo", "A", text="a"), tool_call("echo", "B", text="b")]
    resp = ChatResponse(text="", tool_calls=calls, stop_reason="tool_use", raw=claude_raw(*calls, text=""))
    agent, _ = make_agent(settings, ctx, registry, [resp, "après"])

    def boom(call: ToolCall, result: ToolResult) -> None:
        raise RuntimeError("affichage cassé")

    with pytest.raises(RuntimeError, match="affichage cassé"):
        agent.run("x", on_tool_result=boom)
    tools = [m for m in agent.messages if m.role == "tool"]
    assert [(m.tool_call_id, m.is_error) for m in tools] == [("A", False), ("B", True)]
    assert orphan_tool_calls(agent.messages) == []
    assert agent.run("suite").text == "après"


@pytest.mark.parametrize("stop", ["max_tokens", "refusal"])
def test_dropped_tool_calls_are_closed_even_when_only_in_raw(
    settings: Settings, ctx: ToolContext, registry: ToolRegistry, echo: EchoTool, stop: str
) -> None:
    """Réponse tronquée/refusée : le backend a vidé ``tool_calls`` mais ``raw``
    contient encore le bloc ``tool_use`` ; les tours suivants (même agent ou
    session rechargée) ne rejouent aucun appel sans résultat."""
    call = tool_call("echo", "toolu_9", text="x")
    resp = ChatResponse(text="coupé", tool_calls=[], stop_reason=stop, raw=claude_raw(call))  # type: ignore[arg-type]
    store = SessionStore(settings)
    session = store.create()
    agent, backend = make_agent(settings, ctx, registry, [resp, "q2", "q3"], session=session)

    assert agent.run("q1").stop_reason == stop
    assert echo.calls == []
    closing = agent.messages[-1]
    assert closing.role == "tool" and closing.tool_call_id == "toolu_9" and closing.name == "echo" and closing.is_error
    assert closing.content == (TRUNCATED_TOOL_NOTICE if stop == "max_tokens" else REFUSED_TOOL_NOTICE)

    assert agent.run("q2").text == "q2"
    assert orphan_tool_calls(backend.calls[1]["messages"]) == []

    agent2, backend2 = make_agent(settings, ctx, registry, ["q3"], session=Session.load(store.path_for(session.id)))
    agent2.run("q3")
    assert orphan_tool_calls(backend2.calls[0]["messages"]) == []


def test_repair_orphan_tool_calls_inserts_after_existing_results() -> None:
    calls = [tool_call("echo", "A", text="a"), tool_call("echo", "B", text="b")]
    messages = [
        Message("user", "q"),
        Message("assistant", "", tool_calls=calls, raw=claude_raw(*calls)),
        Message("tool", "ok", tool_call_id="A", name="echo"),
        Message("user", "continue"),
        Message("assistant", "", raw=claude_raw(tool_call("echo", "C", text="c"))),  # tool_calls vidés, raw seul
        Message("assistant", "fin"),
    ]
    assert repair_orphan_tool_calls(messages) == 2
    assert [(m.role, m.tool_call_id) for m in messages] == [
        ("user", None), ("assistant", None), ("tool", "A"), ("tool", "B"),
        ("user", None), ("assistant", None), ("tool", "C"), ("assistant", None),
    ]
    assert messages[3].is_error and messages[3].content == ORPHAN_TOOL_NOTICE and messages[3].name == "echo"
    assert orphan_tool_calls(messages) == []
    assert repair_orphan_tool_calls(messages) == 0  # idempotent


def test_loaded_session_with_orphan_tool_use_is_repaired(settings: Settings, ctx: ToolContext, registry: ToolRegistry) -> None:
    """Session écrite par une version antérieure (tool_use sans tool_result) :
    l'agent la complète au chargement et la persiste réparée."""
    store = SessionStore(settings)
    session = store.create()
    call = tool_call("echo", "toolu_a", text="a")
    session.append(Message("user", "fais"))
    session.append(Message("assistant", "je lance", tool_calls=[call], raw=claude_raw(call)))
    session.append(Message("user", "continue"))
    session.save()

    agent, backend = make_agent(settings, ctx, registry, ["ok"], session=store.get(session.id))
    assert [(m.role, m.tool_call_id) for m in agent.messages] == [
        ("user", None), ("assistant", None), ("tool", "toolu_a"), ("user", None),
    ]
    assert agent.messages[2].is_error and agent.messages[2].content == ORPHAN_TOOL_NOTICE
    agent.run("suite")
    assert orphan_tool_calls(backend.calls[0]["messages"]) == []
    assert orphan_tool_calls(store.get(session.id).messages) == []


# ------------------------------------------------------------- régressions : clôture max_iterations avec le backend Claude


def _claude_block(**fields: Any) -> SimpleNamespace:
    ns = SimpleNamespace(**fields)
    ns.model_dump = lambda mode="python", exclude_none=False: dict(fields)  # noqa: ARG005
    return ns


def _claude_final(*blocks: SimpleNamespace, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
        model="claude-opus-5",
    )


class _ClaudeStream:
    def __init__(self, final: Any):
        self.final = final

    def __enter__(self) -> "_ClaudeStream":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self) -> Any:
        return self.final


class _ClaudeMessages:
    def __init__(self, finals: list[Any]):
        self.finals = list(finals)
        self.calls: list[dict[str, Any]] = []

    def stream(self, **params: Any) -> _ClaudeStream:
        self.calls.append(params)
        return _ClaudeStream(self.finals.pop(0))


def test_max_iterations_closing_call_is_valid_for_claude(settings: Settings, ctx: ToolContext, registry: ToolRegistry, echo: EchoTool) -> None:
    """Avec le vrai ClaudeBackend (client simulé) : le dernier appel, historique
    avec tool_use/tool_result, définit ``tools`` (exigence de l'API) et interdit
    tout nouvel appel via ``tool_choice = none`` ; l'agent obtient sa conclusion."""
    from dhaos.backends.claude import ClaudeBackend

    settings.agent.max_iterations = 1
    messages = _ClaudeMessages([
        _claude_final(
            _claude_block(type="text", text="je lis"),
            _claude_block(type="tool_use", id="toolu_01", name="echo", input={"text": "a"}),
            stop_reason="tool_use",
        ),
        _claude_final(_claude_block(type="text", text="conclusion")),
    ])
    backend = ClaudeBackend(settings, client=SimpleNamespace(beta=SimpleNamespace(messages=messages)))
    agent = Agent(settings, backend, registry, ctx)
    result = agent.run("boucle")

    assert result.stop_reason == "max_iterations" and result.error is None
    assert result.text == "conclusion" and echo.calls == [{"text": "a"}]
    assert len(messages.calls) == 2
    last = messages.calls[1]
    kinds = [b["type"] for m in last["messages"] if isinstance(m["content"], list) for b in m["content"]]
    assert "tool_use" in kinds and "tool_result" in kinds
    assert [t["name"] for t in last["tools"]] == ["echo", "boom"]
    assert last["tool_choice"] == {"type": "none"}
    assert last["messages"][-1] == {"role": "user", "content": ITERATION_LIMIT_NOTICE}


# ------------------------------------------------------------- régressions : max_iterations ≥ 1


@pytest.mark.parametrize("value", ["0", "-1", 0])
def test_max_iterations_below_one_is_rejected(settings: Settings, value: Any) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        settings.with_override("agent.max_iterations", value)
    assert settings.with_override("agent.max_iterations", "1").agent.max_iterations == 1


def test_max_iterations_zero_in_toml_is_rejected(tmp_path: Path) -> None:
    from pydantic import ValidationError

    path = tmp_path / "config.toml"
    path.write_text("[agent]\nmax_iterations = 0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Settings.load(path=path, use_env=False)
