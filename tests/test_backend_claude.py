"""Tests du backend Claude : conversion, paramètres, mapping de la réponse,
reprise sur JSON d'outil illisible, erreurs du SDK, santé.

Aucun réseau ni clé : le client Anthropic est remplacé par un faux objet
(``SimpleNamespace``) dont ``beta.messages.stream`` renvoie un gestionnaire
de contexte itérable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from dhaos.agent.loop import ITERATION_LIMIT_NOTICE, Agent
from dhaos.backends import get_backend
from dhaos.backends.base import BackendError
from dhaos.backends.claude import FALLBACK_BETA, ClaudeBackend
from dhaos.config import Settings
from dhaos.tools.base import ToolContext, ToolRegistry
from dhaos.types import Message, ToolCall, ToolSpec

# ---------------------------------------------------------------- doubles


def block(**fields: Any) -> SimpleNamespace:
    """Bloc de contenu factice avec ``model_dump`` (comme un modèle pydantic)."""
    ns = SimpleNamespace(**fields)
    ns.model_dump = lambda mode="python", exclude_none=False: dict(fields)  # noqa: ARG005
    return ns


def final_message(
    *blocks: SimpleNamespace,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5",
    stop_details: Any = None,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        ),
        model=model,
    )


class FakeStream:
    """Gestionnaire de contexte itérable : une exception dans ``events`` est levée
    pendant l'itération (comme le ``ValueError`` du parseur JSON du SDK)."""

    def __init__(self, events: list[Any], final: Any = None):
        self.events = events
        self.final = final
        self.entered = False
        self.exited = False

    def __enter__(self) -> "FakeStream":
        self.entered = True
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.exited = True
        return False

    def __iter__(self):
        for ev in self.events:
            if isinstance(ev, BaseException):
                raise ev
            yield ev

    def get_final_message(self) -> Any:
        return self.final


class FakeMessages:
    def __init__(self, streams: list[Any]):
        self.streams = list(streams)
        self.calls: list[dict[str, Any]] = []

    def stream(self, **params: Any) -> Any:
        self.calls.append(params)
        item = self.streams.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def fake_client(*streams: Any) -> SimpleNamespace:
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(list(streams))))


def make_backend(settings: Settings, *streams: Any, model: str | None = None) -> tuple[ClaudeBackend, FakeMessages]:
    client = fake_client(*streams)
    return ClaudeBackend(settings, model=model, client=client), client.beta.messages


def status_error(cls: type[anthropic.APIStatusError], status: int, message: str, headers: dict[str, str] | None = None):
    # Le SDK lit response.status_code, response.headers et response.request ;
    # aucun objet httpx n'est nécessaire (ni souhaitable : SDK sur httpx2).
    response = SimpleNamespace(status_code=status, headers=headers or {}, request=SimpleNamespace())
    return cls(message, response=response, body={"error": {"message": message}})  # type: ignore[arg-type]


TOOL = ToolSpec(
    name="read_file",
    description="Lit un fichier",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


# ---------------------------------------------------------------- construction


def test_defaults_and_lazy_client(settings: Settings) -> None:
    backend = ClaudeBackend(settings)
    assert backend.name == "claude"
    assert backend.model == settings.backends.claude.model
    assert backend._client is None  # construction paresseuse, sans réseau
    via_registry = get_backend(settings, "claude", model="claude-sonnet-5")
    assert isinstance(via_registry, ClaudeBackend) and via_registry.model == "claude-sonnet-5"


def test_lazy_client_builds_anthropic_client(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    settings.backends.claude.base_url = "http://127.0.0.1:9/"
    backend = ClaudeBackend(settings)
    client = backend.client
    assert isinstance(client, anthropic.Anthropic)
    assert str(client.base_url).startswith("http://127.0.0.1:9")
    assert backend.client is client


def test_injected_client_is_not_replaced(settings: Settings) -> None:
    sentinel = fake_client()
    backend = ClaudeBackend(settings, client=sentinel)
    assert backend.client is sentinel


# ---------------------------------------------------------------- conversion des messages


def test_message_conversion_groups_tool_results_and_replays_raw() -> None:
    raw = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "je lis"},
        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a"}},
        {"type": "tool_use", "id": "toolu_2", "name": "grep", "input": {"pattern": "x"}},
    ]
    messages = [
        Message("user", "bonjour"),
        Message(
            "assistant",
            "je lis",
            tool_calls=[ToolCall("toolu_1", "read_file", {"path": "a"}), ToolCall("toolu_2", "grep", {"pattern": "x"})],
            raw=raw,
        ),
        Message("tool", "contenu a", tool_call_id="toolu_1", name="read_file"),
        Message("tool", "rien", tool_call_id="toolu_2", name="grep", is_error=True),
        Message("assistant", "fini"),
    ]
    out = ClaudeBackend.to_claude_messages(messages)
    assert out == [
        {"role": "user", "content": "bonjour"},
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "contenu a"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "rien", "is_error": True},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "fini"}]},
    ]


def test_message_conversion_rebuilds_blocks_without_raw_and_skips_empty() -> None:
    messages = [
        Message("user", ""),  # vide : ignoré
        Message("user", "q"),
        Message("assistant", ""),  # vide sans outils ni raw : ignoré
        Message("assistant", "", tool_calls=[ToolCall("toolu_9", "grep", {"pattern": "a"})]),
        Message("tool", "", tool_call_id="toolu_9", name="grep"),
        Message("assistant", "texte", raw={"role": "assistant", "content": "texte"}),  # raw Ollama : reconstruit
    ]
    out = ClaudeBackend.to_claude_messages(messages)
    assert out == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_9", "name": "grep", "input": {"pattern": "a"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_9", "content": "(aucune sortie)"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "texte"}]},
    ]


def test_trailing_tool_results_are_flushed() -> None:
    out = ClaudeBackend.to_claude_messages(
        [Message("user", "q"), Message("tool", "r1", tool_call_id="a"), Message("tool", "r2", tool_call_id="b")]
    )
    assert out[-1]["role"] == "user"
    assert [b["tool_use_id"] for b in out[-1]["content"]] == ["a", "b"]


# ---------------------------------------------------------------- paramètres envoyés


def test_request_params_with_fallbacks(settings: Settings) -> None:
    settings.backends.claude.fallbacks = True
    settings.backends.claude.effort = "high"
    settings.backends.claude.thinking_display = "summarized"
    settings.backends.claude.max_tokens = 4096
    backend, messages = make_backend(settings, FakeStream([], final_message(block(type="text", text="ok"))))
    backend.chat([Message("user", "q")], system="SYS", tools=[TOOL])

    params = messages.calls[0]
    assert params["model"] == settings.backends.claude.model
    assert params["max_tokens"] == 4096
    assert params["system"] == [{"type": "text", "text": "SYS"}]
    assert params["messages"] == [{"role": "user", "content": "q"}]
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"] == {"effort": "high"}
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert params["fallbacks"] == "default"
    assert params["tools"] == [
        {
            "name": "read_file",
            "description": "Lit un fichier",
            "input_schema": TOOL.parameters,
            "eager_input_streaming": True,
        }
    ]
    assert "stream" not in params


def test_request_params_without_fallbacks_system_or_tools(settings: Settings) -> None:
    settings.backends.claude.fallbacks = False
    backend, messages = make_backend(settings, FakeStream([], final_message(block(type="text", text="ok"))), model="claude-sonnet-5")
    backend.chat([Message("user", "q")])
    params = messages.calls[0]
    assert params["model"] == "claude-sonnet-5"
    assert "betas" not in params and "fallbacks" not in params
    assert "system" not in params and "tools" not in params
    assert params["thinking"] == {"type": "adaptive", "display": settings.backends.claude.thinking_display}
    assert params["output_config"] == {"effort": settings.backends.claude.effort}


# ---------------------------------------------------------------- mapping de la réponse


def test_text_and_tool_use_mapping(settings: Settings) -> None:
    events = [
        SimpleNamespace(type="thinking", thinking="hmm"),
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="je "),
        SimpleNamespace(type="text", text="lis"),
        SimpleNamespace(type="input_json", partial_json='{"path"'),
        SimpleNamespace(type="text", text=""),
    ]
    final = final_message(
        block(type="thinking", thinking="hmm", signature="abc"),
        block(type="text", text="je lis"),
        block(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"}),
        block(type="tool_use", id="toolu_2", name="grep", input="pas un dict"),
        stop_reason="tool_use",
        model="claude-opus-5-served",
        input_tokens=11,
        output_tokens=3,
    )
    stream = FakeStream(events, final)
    backend, _ = make_backend(settings, stream)
    texts: list[str] = []
    thoughts: list[str] = []
    resp = backend.chat([Message("user", "q")], on_text=texts.append, on_thinking=thoughts.append)

    assert stream.entered and stream.exited
    assert texts == ["je ", "lis"]
    assert thoughts == ["hmm"]
    assert resp.text == "je lis"
    assert resp.thinking == "hmm"
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls == [
        ToolCall("toolu_1", "read_file", {"path": "a.py"}),
        ToolCall("toolu_2", "grep", {"_raw": "pas un dict"}),
    ]
    assert resp.usage.input_tokens == 11 and resp.usage.output_tokens == 3
    assert resp.model == "claude-opus-5-served"
    assert resp.raw == [
        {"type": "thinking", "thinking": "hmm", "signature": "abc"},
        {"type": "text", "text": "je lis"},
        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}},
        {"type": "tool_use", "id": "toolu_2", "name": "grep", "input": "pas un dict"},
    ]
    # Le message assistant produit rejoue exactement les blocs.
    replay = ClaudeBackend.to_claude_messages([resp.to_assistant_message()])
    assert replay == [{"role": "assistant", "content": resp.raw}]


def test_end_turn_with_tool_use_blocks_is_tool_use(settings: Settings) -> None:
    final = final_message(block(type="tool_use", id="t", name="grep", input={}), stop_reason="end_turn")
    backend, _ = make_backend(settings, FakeStream([], final))
    assert backend.chat([Message("user", "q")]).stop_reason == "tool_use"


@pytest.mark.parametrize("api_stop", ["end_turn", "stop_sequence", "pause_turn", None, "inconnu"])
def test_plain_stop_reasons_map_to_end_turn(settings: Settings, api_stop: str | None) -> None:
    backend, _ = make_backend(settings, FakeStream([], final_message(block(type="text", text="x"), stop_reason=api_stop)))
    resp = backend.chat([Message("user", "q")])
    assert resp.stop_reason == "end_turn" and resp.text == "x"


def test_max_tokens_with_tool_use_drops_tool_calls(settings: Settings) -> None:
    final = final_message(
        block(type="text", text="partiel"),
        block(type="tool_use", id="toolu_1", name="write_file", input={"path": "a", "content": "tronq"}),
        stop_reason="max_tokens",
    )
    backend, _ = make_backend(settings, FakeStream([], final))
    resp = backend.chat([Message("user", "q")])
    assert resp.stop_reason == "max_tokens"
    assert resp.tool_calls == []
    assert resp.text == "partiel"
    # Le texte reste rejouable ; le tool_use abandonné (jamais suivi d'un
    # tool_result) est retiré de la charge brute.
    assert resp.raw == [{"type": "text", "text": "partiel"}]


def test_refusal_drops_tool_calls_and_annotates_text(settings: Settings) -> None:
    final = final_message(
        block(type="text", text="Je ne peux"),
        block(type="tool_use", id="toolu_1", name="run_command", input={"command": "x"}),
        stop_reason="refusal",
        stop_details=SimpleNamespace(type="refusal", category="cyber", explanation="contenu sensible"),
    )
    backend, _ = make_backend(settings, FakeStream([], final))
    resp = backend.chat([Message("user", "q")])
    assert resp.stop_reason == "refusal"
    assert resp.tool_calls == []
    assert resp.text == "Je ne peux\n[refus : cyber — contenu sensible]"


def test_refusal_without_details_leaves_text_untouched(settings: Settings) -> None:
    final = final_message(block(type="text", text="non"), stop_reason="refusal")  # stop_details=None
    backend, _ = make_backend(settings, FakeStream([], final))
    resp = backend.chat([Message("user", "q")])
    assert resp.stop_reason == "refusal"
    assert resp.text == "non"
    assert resp.tool_calls == []


def test_usage_tolerates_missing_fields(settings: Settings) -> None:
    final = SimpleNamespace(content=[block(type="text", text="x")], stop_reason="end_turn", usage=None, model=None)
    backend, _ = make_backend(settings, FakeStream([], final))
    resp = backend.chat([Message("user", "q")])
    assert resp.usage.input_tokens == 0 and resp.usage.output_tokens == 0
    assert resp.model == settings.backends.claude.model


# ---------------------------------------------------------------- reprise sur ValueError


def test_value_error_retries_then_succeeds(settings: Settings) -> None:
    ok = FakeStream([SimpleNamespace(type="text", text="ok")], final_message(block(type="text", text="ok")))
    backend, messages = make_backend(
        settings,
        FakeStream([SimpleNamespace(type="text", text="a"), ValueError("bad json")]),
        FakeStream([ValueError("bad json")]),
        ok,
    )
    texts: list[str] = []
    resp = backend.chat([Message("user", "q")], on_text=texts.append)
    assert resp.text == "ok"
    assert len(messages.calls) == 3
    # Le texte du tour avorté ("a") diverge du tour réussi : il n'est pas
    # concaténé silencieusement, un marqueur de réémission le sépare.
    assert "".join(texts) == "a\n[réémission]\nok"
    assert "".join(texts).endswith(resp.text)


def test_value_error_gives_up_after_two_retries(settings: Settings) -> None:
    backend, messages = make_backend(
        settings,
        FakeStream([ValueError("1")]),
        FakeStream([ValueError("2")]),
        FakeStream([ValueError("3")]),
        FakeStream([], final_message(block(type="text", text="jamais"))),
    )
    with pytest.raises(BackendError, match="JSON illisible"):
        backend.chat([Message("user", "q")])
    assert len(messages.calls) == 3


# ---------------------------------------------------------------- erreurs du SDK


def test_authentication_error(settings: Settings) -> None:
    backend, _ = make_backend(settings, status_error(anthropic.AuthenticationError, 401, "invalid x-api-key"))
    with pytest.raises(BackendError, match="ANTHROPIC_API_KEY"):
        backend.chat([Message("user", "q")])


def test_credentials_error(settings: Settings) -> None:
    backend, _ = make_backend(settings, anthropic.CredentialsError("no credentials"))
    with pytest.raises(BackendError, match="ant auth login"):
        backend.chat([Message("user", "q")])


def test_not_found_error(settings: Settings) -> None:
    backend, _ = make_backend(settings, status_error(anthropic.NotFoundError, 404, "model: nope"), model="nope")
    with pytest.raises(BackendError, match="modèle inconnu : nope"):
        backend.chat([Message("user", "q")])


def test_rate_limit_error(settings: Settings) -> None:
    backend, _ = make_backend(
        settings, status_error(anthropic.RateLimitError, 429, "rate limited", headers={"retry-after": "7"})
    )
    with pytest.raises(BackendError, match=r"limite de débit.*7 s"):
        backend.chat([Message("user", "q")])


def test_status_errors(settings: Settings) -> None:
    backend, _ = make_backend(settings, status_error(anthropic.InternalServerError, 500, "boom"))
    with pytest.raises(BackendError, match="HTTP 500"):
        backend.chat([Message("user", "q")])
    backend, _ = make_backend(settings, status_error(anthropic.BadRequestError, 400, "bad"))
    with pytest.raises(BackendError, match="HTTP 400"):
        backend.chat([Message("user", "q")])


def test_connection_error(settings: Settings) -> None:
    backend, _ = make_backend(settings, anthropic.APIConnectionError(message="down", request=SimpleNamespace()))
    with pytest.raises(BackendError, match="impossible de joindre"):
        backend.chat([Message("user", "q")])


def test_unrelated_exceptions_propagate(settings: Settings) -> None:
    backend, _ = make_backend(settings, RuntimeError("bug interne"))
    with pytest.raises(RuntimeError, match="bug interne"):
        backend.chat([Message("user", "q")])


# ---------------------------------------------------------------- healthcheck (sans réseau)


@pytest.fixture
def no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(key, raising=False)


def write_profile(config_dir: Path, profile: str = "default") -> None:
    """Profil `ant auth login` minimal (configs/<p>.json + credentials/<p>.json)."""
    (config_dir / "configs").mkdir(parents=True, exist_ok=True)
    (config_dir / "credentials").mkdir(parents=True, exist_ok=True)
    (config_dir / "configs" / f"{profile}.json").write_text(
        json.dumps({"authentication": {"type": "user_oauth"}}), encoding="utf-8"
    )
    creds = config_dir / "credentials" / f"{profile}.json"
    creds.write_text(
        json.dumps({"access_token": "sk-ant-oat01-fake", "refresh_token": "r", "expires_at": 4102444800}),
        encoding="utf-8",
    )
    creds.chmod(0o600)


def test_healthcheck_without_credentials(settings: Settings, no_credentials: None) -> None:
    hc = ClaudeBackend(settings).healthcheck()
    assert hc["ok"] is False
    assert hc["backend"] == "claude" and hc["model"] == settings.backends.claude.model
    assert "ANTHROPIC_API_KEY" in hc["detail"] and "ant auth login" in hc["detail"]


def test_healthcheck_with_api_key(settings: Settings, no_credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    hc = ClaudeBackend(settings).healthcheck()
    assert hc["ok"] is True and "ANTHROPIC_API_KEY" in hc["detail"]


def test_healthcheck_with_auth_token(settings: Settings, no_credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    assert ClaudeBackend(settings).healthcheck()["ok"] is True


def test_healthcheck_with_profile_dir(settings: Settings, no_credentials: None) -> None:
    profile = Path(os.environ["HOME"]) / ".config" / "anthropic"
    write_profile(profile)
    hc = ClaudeBackend(settings).healthcheck()
    assert hc["ok"] is True and "ant auth login" in hc["detail"]
    assert str(profile) in hc["detail"] and "`default`" in hc["detail"]


# ---------------------------------------------------------------- régressions : rejeu après arrêt max_tokens / refusal


def orphan_tool_uses(claude_messages: list[dict[str, Any]]) -> list[str]:
    """Ids ``tool_use`` d'un message assistant non suivis d'un ``tool_result``
    dans le message user immédiatement suivant (règle de l'API Messages)."""
    orphans: list[str] = []
    for i, msg in enumerate(claude_messages):
        if msg["role"] != "assistant" or not isinstance(msg["content"], list):
            continue
        ids = [b["id"] for b in msg["content"] if b.get("type") == "tool_use"]
        if not ids:
            continue
        nxt = claude_messages[i + 1] if i + 1 < len(claude_messages) else None
        answered = set()
        if nxt and nxt["role"] == "user" and isinstance(nxt["content"], list):
            answered = {b.get("tool_use_id") for b in nxt["content"] if b.get("type") == "tool_result"}
        orphans.extend(t for t in ids if t not in answered)
    return orphans


@pytest.mark.parametrize("api_stop", ["max_tokens", "model_context_window_exceeded", "refusal"])
def test_dropped_tool_use_is_not_replayed(settings: Settings, api_stop: str) -> None:
    final = final_message(
        block(type="thinking", thinking="", signature="sig"),
        block(type="text", text="Je lis"),
        block(type="tool_use", id="toolu_1", name="read_file", input={"path": "/etc/pas"}),
        stop_reason=api_stop,
    )
    backend, _ = make_backend(settings, FakeStream([], final))
    resp = backend.chat([Message("user", "lis le fichier")])
    assert resp.tool_calls == []
    assert all(b["type"] != "tool_use" for b in resp.raw or [])
    assert any(b["type"] == "text" for b in resp.raw)  # le texte reste rejoué tel quel

    params = backend.build_params([Message("user", "q"), resp.to_assistant_message(), Message("user", "continue")])
    assert orphan_tool_uses(params["messages"]) == []
    assert params["messages"][-1] == {"role": "user", "content": "continue"}


def test_dropped_tool_use_without_text_yields_no_raw(settings: Settings) -> None:
    final = final_message(
        block(type="thinking", thinking="", signature="sig"),
        block(type="tool_use", id="toolu_1", name="write_file", input={"path": "a"}),
        stop_reason="max_tokens",
    )
    backend, _ = make_backend(settings, FakeStream([], final))
    resp = backend.chat([Message("user", "q")])
    assert resp.raw is None and resp.text == ""
    # Le message assistant vide est ignoré au rejeu : pas de tool_use orphelin.
    out = ClaudeBackend.to_claude_messages([Message("user", "q"), resp.to_assistant_message(), Message("user", "suite")])
    assert out == [{"role": "user", "content": "q"}, {"role": "user", "content": "suite"}]


def test_agent_continues_after_max_tokens_with_tool_use(settings: Settings, ctx: ToolContext) -> None:
    """Deux ``agent.run`` successifs (REPL chat) après un max_tokens avec tool_use :
    le second appel capturé ne contient aucun tool_use orphelin."""
    truncated = final_message(
        block(type="text", text="Je lis"),
        block(type="tool_use", id="toolu_1", name="read_file", input={"path": "/etc/pas"}),
        stop_reason="max_tokens",
    )
    backend, messages = make_backend(
        settings, FakeStream([], truncated), FakeStream([], final_message(block(type="text", text="ok")))
    )
    agent = Agent(settings, backend, ToolRegistry([]), ctx)
    first = agent.run("lis le fichier")
    assert first.stop_reason == "max_tokens" and first.tool_calls == 0
    second = agent.run("continue")
    assert second.stop_reason == "end_turn" and second.text == "ok"
    assert len(messages.calls) == 2
    assert orphan_tool_uses(messages.calls[1]["messages"]) == []


# ---------------------------------------------------------------- régressions : clôture max_iterations avec historique d'outils


def tool_history() -> list[Message]:
    return [
        Message("user", "fais"),
        Message("assistant", "", tool_calls=[ToolCall("toolu_1", "read_file", {"path": "a"})],
                raw=[{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a"}}]),
        Message("tool", "contenu", tool_call_id="toolu_1", name="read_file"),
        Message("user", ITERATION_LIMIT_NOTICE),
    ]


def test_closing_turn_keeps_tools_with_tool_choice_none(settings: Settings) -> None:
    backend, messages = make_backend(
        settings,
        FakeStream([], final_message(block(type="tool_use", id="t", name="read_file", input={}), stop_reason="tool_use")),
        FakeStream([], final_message(block(type="text", text="résumé"))),
    )
    backend.chat([Message("user", "fais")], tools=[TOOL])  # tour normal : outils transmis
    resp = backend.chat(tool_history(), system="sys", tools=None)  # clôture max_iterations
    assert resp.text == "résumé"
    params = messages.calls[1]
    assert params["tools"] == [ClaudeBackend._tool_to_claude(TOOL)]  # définitions rejouées
    assert params["tool_choice"] == {"type": "none"}
    assert "tool_choice" not in messages.calls[0]


def test_build_params_defines_tools_whenever_history_has_tool_blocks(settings: Settings) -> None:
    backend = ClaudeBackend(settings, client=fake_client())
    params = backend.build_params(tool_history(), system="sys", tools=None)
    assert "tools" in params and params["tool_choice"] == {"type": "none"}
    assert [t["name"] for t in params["tools"]] == ["read_file"]  # déduit de l'historique
    # Sans bloc d'outil, rien ne change.
    plain = backend.build_params([Message("user", "q")], tools=None)
    assert "tools" not in plain and "tool_choice" not in plain
    # Avec des outils, pas de tool_choice.
    with_tools = backend.build_params(tool_history(), tools=[TOOL])
    assert with_tools["tools"] == [ClaudeBackend._tool_to_claude(TOOL)] and "tool_choice" not in with_tools


# ---------------------------------------------------------------- régressions : erreur SSE mi-flux


@pytest.mark.parametrize("err_type", ["overloaded_error", "api_error"])
def test_mid_stream_error_event_is_server_error(settings: Settings, err_type: str) -> None:
    body = {"type": "error", "error": {"type": err_type, "message": "Overloaded"}}
    response = SimpleNamespace(status_code=200, headers={}, request=SimpleNamespace())
    exc = anthropic.APIStatusError(str(body), response=response, body=body)  # type: ignore[arg-type]
    backend, _ = make_backend(settings, FakeStream([SimpleNamespace(type="text", text="Bonjour"), exc]))
    with pytest.raises(BackendError, match="erreur côté serveur") as info:
        backend.chat([Message("user", "salut")])
    text = str(info.value)
    assert "refusée" not in text and "HTTP 200" not in text
    assert err_type in text and "Overloaded" in text


def test_mid_stream_rate_limit_event(settings: Settings) -> None:
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    response = SimpleNamespace(status_code=200, headers={}, request=SimpleNamespace())
    exc = anthropic.APIStatusError(str(body), response=response, body=body)  # type: ignore[arg-type]
    backend, _ = make_backend(settings, FakeStream([exc]))
    with pytest.raises(BackendError, match="limite de débit"):
        backend.chat([Message("user", "salut")])


def test_mid_stream_error_through_real_sdk(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chemin réel du SDK (httpx2.MockTransport → _streaming → _make_status_error)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sse = (
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message",'
        '"role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,'
        '"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Bonjour"}}\n\n'
        'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())

    client = anthropic.Anthropic(
        api_key="sk-ant-test", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), max_retries=0
    )
    backend = ClaudeBackend(settings, client=client)
    texts: list[str] = []
    with pytest.raises(BackendError, match="erreur côté serveur Anthropic \\(overloaded_error\\)") as info:
        backend.chat([Message("user", "salut")], on_text=texts.append)
    assert texts == ["Bonjour"]
    assert "HTTP 200" not in str(info.value)


# ---------------------------------------------------------------- régressions : réémission sans doublon


def test_retry_does_not_replay_already_shown_prefix(settings: Settings) -> None:
    backend, _ = make_backend(
        settings,
        FakeStream([SimpleNamespace(type="text", text="Voici "), ValueError("bad json")]),
        FakeStream(
            [SimpleNamespace(type="text", text="Voici "), SimpleNamespace(type="text", text="la suite")],
            final_message(block(type="text", text="Voici la suite")),
        ),
    )
    texts: list[str] = []
    resp = backend.chat([Message("user", "q")], on_text=texts.append)
    assert "".join(texts) == "Voici la suite"
    assert resp.text == "Voici la suite"


def test_retry_with_divergent_text_is_marked(settings: Settings) -> None:
    backend, _ = make_backend(
        settings,
        FakeStream([SimpleNamespace(type="text", text="a"), ValueError("bad json")]),
        FakeStream([SimpleNamespace(type="text", text="b")], final_message(block(type="text", text="b"))),
    )
    texts: list[str] = []
    resp = backend.chat([Message("user", "q")], on_text=texts.append)
    shown = "".join(texts)
    assert shown != "ab" and "[réémission]" in shown
    assert shown.endswith(resp.text)


def test_retry_with_chunk_boundaries_shifted(settings: Settings) -> None:
    """Le prolongement est calculé sur le texte cumulé, pas sur les morceaux."""
    backend, _ = make_backend(
        settings,
        FakeStream([SimpleNamespace(type="text", text="ab"), SimpleNamespace(type="text", text="cd"), ValueError("x")]),
        FakeStream(
            [SimpleNamespace(type="text", text="a"), SimpleNamespace(type="text", text="bcde"), SimpleNamespace(type="text", text="f")],
            final_message(block(type="text", text="abcdef")),
        ),
    )
    texts: list[str] = []
    resp = backend.chat([Message("user", "q")], on_text=texts.append)
    assert "".join(texts) == "abcdef" == resp.text


def test_retry_does_not_replay_thinking(settings: Settings) -> None:
    backend, _ = make_backend(
        settings,
        FakeStream([SimpleNamespace(type="thinking", thinking="hmm "), ValueError("bad json")]),
        FakeStream(
            [SimpleNamespace(type="thinking", thinking="hmm "), SimpleNamespace(type="thinking", thinking="ok")],
            final_message(block(type="thinking", thinking="hmm ok", signature="s"), block(type="text", text="r")),
        ),
    )
    thoughts: list[str] = []
    resp = backend.chat([Message("user", "q")], on_thinking=thoughts.append)
    assert "".join(thoughts) == "hmm ok" == resp.thinking


# ---------------------------------------------------------------- régressions : usage avec cache


def test_usage_includes_cached_input_tokens(settings: Settings) -> None:
    final = final_message(
        block(type="text", text="x"),
        input_tokens=12,
        output_tokens=5,
        cache_read_input_tokens=18000,
        cache_creation_input_tokens=2000,
    )
    backend, messages = make_backend(settings, FakeStream([], final))
    resp = backend.chat([Message("user", "q")])
    assert messages.calls[0]["cache_control"] == {"type": "ephemeral"}
    assert resp.usage.input_tokens == 20012 and resp.usage.output_tokens == 5


def test_usage_cache_fields_none(settings: Settings) -> None:
    final = final_message(block(type="text", text="x"), input_tokens=7, cache_read_input_tokens=None)
    backend, _ = make_backend(settings, FakeStream([], final))
    assert backend.chat([Message("user", "q")]).usage.input_tokens == 7


# ---------------------------------------------------------------- régressions : healthcheck et profil du SDK


def test_healthcheck_empty_profile_dir_is_not_ok(settings: Settings, no_credentials: None) -> None:
    (Path(os.environ["HOME"]) / ".config" / "anthropic").mkdir(parents=True)
    hc = ClaudeBackend(settings).healthcheck()
    assert hc["ok"] is False and "ant auth login" in hc["detail"]


def test_healthcheck_profile_via_config_dir(
    settings: Settings, no_credentials: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = tmp_path / "anthropic-cfg"
    write_profile(config_dir)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(config_dir))
    hc = ClaudeBackend(settings).healthcheck()
    assert hc["ok"] is True and str(config_dir) in hc["detail"]


def test_healthcheck_named_profile(
    settings: Settings, no_credentials: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = Path(os.environ["HOME"]) / ".config" / "anthropic"
    write_profile(config_dir, profile="dev")
    monkeypatch.setenv("ANTHROPIC_PROFILE", "dev")
    hc = ClaudeBackend(settings).healthcheck()
    assert hc["ok"] is True and "`dev`" in hc["detail"]
    # Un profil nommé absent est un diagnostic négatif, pas une exception.
    monkeypatch.setenv("ANTHROPIC_PROFILE", "absent")
    assert ClaudeBackend(settings).healthcheck()["ok"] is False


# ---------------------------------------------------------------- régressions : absence d'identifiants


def test_chat_without_credentials_raises_backend_error(settings: Settings, no_credentials: None) -> None:
    backend = ClaudeBackend(settings)  # vrai client SDK, aucune méthode d'authentification
    with pytest.raises(BackendError, match="clé API absente"):
        backend.chat([Message("user", "bonjour")])


def test_unrelated_type_error_propagates(settings: Settings) -> None:
    backend, _ = make_backend(settings, TypeError("bug interne"))
    with pytest.raises(TypeError, match="bug interne"):
        backend.chat([Message("user", "q")])
