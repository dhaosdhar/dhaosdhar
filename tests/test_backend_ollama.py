"""Tests du backend Ollama : flux NDJSON, conversion, embeddings, santé, erreurs.

Aucun réseau : le client ``httpx`` est construit sur ``httpx.MockTransport``.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from dhaos.backends import get_backend
from dhaos.backends.base import BackendError
from dhaos.backends.ollama import OllamaBackend
from dhaos.config import Settings
from dhaos.types import Message, ToolCall, ToolSpec

Handler = Callable[[httpx.Request], httpx.Response]


def ndjson(*objs: dict[str, Any]) -> bytes:
    return b"".join(json.dumps(o).encode("utf-8") + b"\n" for o in objs)


def make_backend(settings: Settings, handler: Handler, **kwargs: Any) -> tuple[OllamaBackend, list[httpx.Request]]:
    """Backend branché sur un transport factice ; renvoie aussi les requêtes vues."""
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(wrapped))
    return OllamaBackend(settings, client=client, **kwargs), seen


def chat_stream(*objs: dict[str, Any]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, content=ndjson(*objs))

    return handler


def done_chunk(reason: str = "stop", prompt: int = 12, evals: int = 7) -> dict[str, Any]:
    return {
        "model": "qwen2.5-coder:7b",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": reason,
        "prompt_eval_count": prompt,
        "eval_count": evals,
    }


TOOL = ToolSpec(
    name="read_file",
    description="Lit un fichier",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


# ---------------------------------------------------------------- construction


def test_defaults_from_settings(settings: Settings) -> None:
    backend, _ = make_backend(settings, chat_stream())
    assert backend.name == "ollama"
    assert backend.model == settings.backends.ollama.model
    assert backend.host == "http://127.0.0.1:11434"
    assert backend.supports_embeddings() is True


def test_model_override_and_registry(settings: Settings) -> None:
    backend, _ = make_backend(settings, chat_stream(), model="llama3.1:8b")
    assert backend.model == "llama3.1:8b"
    via_registry = get_backend(settings, "ollama", model="codellama")
    assert isinstance(via_registry, OllamaBackend)
    assert via_registry.model == "codellama"


# ---------------------------------------------------------------- chat : texte


def test_chat_streams_text_and_collects_usage(settings: Settings) -> None:
    handler = chat_stream(
        {"model": "qwen2.5-coder:7b", "message": {"role": "assistant", "content": "Bon"}, "done": False},
        {"model": "qwen2.5-coder:7b", "message": {"role": "assistant", "content": "jour"}, "done": False},
        done_chunk("stop", prompt=30, evals=4),
    )
    backend, seen = make_backend(settings, handler)
    chunks: list[str] = []
    resp = backend.chat([Message("user", "salut")], system="Tu es utile.", on_text=chunks.append)

    assert resp.text == "Bonjour"
    assert chunks == ["Bon", "jour"]
    assert resp.stop_reason == "end_turn"
    assert resp.tool_calls == []
    assert resp.usage.input_tokens == 30 and resp.usage.output_tokens == 4
    assert resp.model == "qwen2.5-coder:7b"
    assert resp.raw == {"role": "assistant", "content": "Bonjour"}

    payload = json.loads(seen[0].content)
    assert payload["model"] == settings.backends.ollama.model
    assert payload["stream"] is True
    assert payload["messages"] == [
        {"role": "system", "content": "Tu es utile."},
        {"role": "user", "content": "salut"},
    ]
    assert payload["options"] == {"num_ctx": settings.backends.ollama.num_ctx}
    assert "tools" not in payload
    assert "keep_alive" not in payload


def test_chat_thinking_is_forwarded(settings: Settings) -> None:
    handler = chat_stream(
        {"message": {"role": "assistant", "content": "", "thinking": "je réfléchis"}, "done": False},
        {"message": {"role": "assistant", "content": "ok"}, "done": False},
        done_chunk(),
    )
    backend, _ = make_backend(settings, handler)
    thoughts: list[str] = []
    resp = backend.chat([Message("user", "?")], on_thinking=thoughts.append)
    assert thoughts == ["je réfléchis"]
    assert resp.thinking == "je réfléchis"
    assert resp.text == "ok"
    assert resp.raw["thinking"] == "je réfléchis"


def test_chat_length_maps_to_max_tokens(settings: Settings) -> None:
    handler = chat_stream({"message": {"role": "assistant", "content": "tronq"}, "done": False}, done_chunk("length"))
    backend, _ = make_backend(settings, handler)
    resp = backend.chat([Message("user", "?")])
    assert resp.stop_reason == "max_tokens"
    assert resp.text == "tronq"


def test_chat_num_ctx_none_and_keep_alive(settings: Settings) -> None:
    settings.backends.ollama.num_ctx = None
    settings.backends.ollama.keep_alive = "30m"
    backend, seen = make_backend(settings, chat_stream(done_chunk()))
    backend.chat([Message("user", "?")])
    payload = json.loads(seen[0].content)
    assert "options" not in payload
    assert payload["keep_alive"] == "30m"


def test_chat_skips_blank_lines_and_ignores_garbage_fields(settings: Settings) -> None:
    body = b"\n" + ndjson({"message": {"content": "a", "tool_calls": "pas une liste"}, "done": False}) + b"\n\n" + ndjson(done_chunk())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend, _ = make_backend(settings, handler)
    resp = backend.chat([Message("user", "?")])
    assert resp.text == "a"
    assert resp.tool_calls == []


def test_chat_invalid_ndjson_line_raises(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"message": {"content": "a"}}\nceci n est pas du json\n')

    backend, _ = make_backend(settings, handler)
    with pytest.raises(BackendError, match="non JSON"):
        backend.chat([Message("user", "?")])


def test_chat_inline_error_object_raises(settings: Settings) -> None:
    backend, _ = make_backend(settings, chat_stream({"error": "model requires more memory"}))
    with pytest.raises(BackendError, match="model requires more memory"):
        backend.chat([Message("user", "?")])


# ---------------------------------------------------------------- chat : outils


def test_chat_tool_calls_parsing_and_ids(settings: Settings) -> None:
    handler = chat_stream(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "a.py"}}},
                    {"function": {"name": "grep", "arguments": '{"pattern": "TODO"}'}},
                    {"function": {"name": "run_command", "arguments": "{pas du json"}},
                    {"function": {"name": "", "arguments": {}}},
                    {"pas": "une fonction"},
                    "n'importe quoi",
                ],
            },
            "done": False,
        },
        done_chunk("stop"),
    )
    backend, seen = make_backend(settings, handler)
    resp = backend.chat([Message("user", "lis a.py")], tools=[TOOL])

    assert resp.stop_reason == "tool_use"
    ids = [c.id for c in resp.tool_calls]
    assert len(ids) == 3 and len(set(ids)) == 3 and all(i.startswith("call_") for i in ids)
    assert resp.tool_calls[0] == ToolCall(ids[0], "read_file", {"path": "a.py"})
    assert resp.tool_calls[1].arguments == {"pattern": "TODO"}
    assert resp.tool_calls[2].arguments == {"_raw": "{pas du json"}
    assert resp.raw["tool_calls"] == [
        {"function": {"name": "read_file", "arguments": {"path": "a.py"}}},
        {"function": {"name": "grep", "arguments": {"pattern": "TODO"}}},
        {"function": {"name": "run_command", "arguments": {"_raw": "{pas du json"}}},
    ]

    payload = json.loads(seen[0].content)
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {"name": "read_file", "description": "Lit un fichier", "parameters": TOOL.parameters},
        }
    ]


def test_chat_json_array_arguments_become_raw(settings: Settings) -> None:
    handler = chat_stream(
        {"message": {"content": "", "tool_calls": [{"function": {"name": "f", "arguments": "[1, 2]"}}]}, "done": False},
        done_chunk(),
    )
    backend, _ = make_backend(settings, handler)
    resp = backend.chat([Message("user", "?")])
    assert resp.tool_calls[0].arguments == {"_raw": "[1, 2]"}


# ---------------------------------------------------------------- conversion


def test_message_conversion() -> None:
    raw_assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "x"}}}],
    }
    messages = [
        Message("user", "bonjour"),
        Message("assistant", "", tool_calls=[ToolCall("call_1", "read_file", {"path": "x"})], raw=raw_assistant),
        Message("tool", "contenu", tool_call_id="call_1", name="read_file"),
        Message("tool", "introuvable", tool_call_id="call_2", name="grep", is_error=True),
        Message("assistant", "voici", tool_calls=[ToolCall("call_3", "grep", {"pattern": "a"})], raw=None),
        Message("assistant", "fin", raw=[{"type": "text", "text": "fin"}]),  # raw Claude : ignoré
    ]
    out = OllamaBackend.to_ollama_messages(messages, system="SYS")
    assert out == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "bonjour"},
        raw_assistant,
        {"role": "tool", "content": "contenu", "tool_name": "read_file"},
        {"role": "tool", "content": "[erreur] introuvable", "tool_name": "grep"},
        {
            "role": "assistant",
            "content": "voici",
            "tool_calls": [{"function": {"name": "grep", "arguments": {"pattern": "a"}}}],
        },
        {"role": "assistant", "content": "fin"},
    ]


def test_message_conversion_without_system() -> None:
    out = OllamaBackend.to_ollama_messages([Message("user", "x")], system="")
    assert out == [{"role": "user", "content": "x"}]


# ---------------------------------------------------------------- erreurs réseau / HTTP


def test_connect_error_is_actionable(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend, _ = make_backend(settings, handler)
    with pytest.raises(BackendError, match=r"injoignable sur http://127.0.0.1:11434 — lancez `ollama serve`"):
        backend.chat([Message("user", "?")])


def test_timeout_error_is_actionable(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("trop lent", request=request)

    backend, _ = make_backend(settings, handler)
    with pytest.raises(BackendError, match="timeout"):
        backend.chat([Message("user", "?")])


def test_404_means_missing_model(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'x' not found"})

    backend, _ = make_backend(settings, handler, model="absent:7b")
    with pytest.raises(BackendError, match="modèle absent : ollama pull absent:7b"):
        backend.chat([Message("user", "?")])


def test_other_status_includes_body(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "out of memory"})

    backend, _ = make_backend(settings, handler)
    with pytest.raises(BackendError, match="HTTP 500 : out of memory"):
        backend.chat([Message("user", "?")])


# ---------------------------------------------------------------- embeddings


def test_embed(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        assert payload == {"model": settings.backends.ollama.embed_model, "input": ["a", "b"]}
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    backend, _ = make_backend(settings, handler)
    assert backend.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert backend.embed([]) == []


def test_embed_invalid_response(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1]]})

    backend, _ = make_backend(settings, handler)
    with pytest.raises(BackendError, match="embedding invalide"):
        backend.embed(["a", "b"])


def test_embed_404(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    backend, _ = make_backend(settings, handler)
    with pytest.raises(BackendError, match="ollama pull"):
        backend.embed(["a"])


# ---------------------------------------------------------------- healthcheck


def tags(*names: str) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": n, "size": 1} for n in names]})

    return handler


def test_healthcheck_model_present_with_latest_tag(settings: Settings) -> None:
    backend, _ = make_backend(settings, tags("qwen2.5-coder:7b", "nomic-embed-text:latest"), model="nomic-embed-text")
    hc = backend.healthcheck()
    assert hc["ok"] is True
    assert hc["backend"] == "ollama" and hc["model"] == "nomic-embed-text"
    assert hc["models"] == ["qwen2.5-coder:7b", "nomic-embed-text:latest"]
    assert "présent" in hc["detail"]


def test_healthcheck_model_absent(settings: Settings) -> None:
    backend, _ = make_backend(settings, tags("llama3:latest"))
    hc = backend.healthcheck()
    assert hc["ok"] is False
    assert f"ollama pull {settings.backends.ollama.model}" in hc["detail"]
    assert hc["models"] == ["llama3:latest"]


def test_healthcheck_unreachable(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    backend, _ = make_backend(settings, handler)
    hc = backend.healthcheck()
    assert hc == {
        "ok": False,
        "backend": "ollama",
        "model": settings.backends.ollama.model,
        "detail": "Ollama injoignable sur http://127.0.0.1:11434 — lancez `ollama serve`",
        "models": [],
    }


# ---------------------------------------------------------------- régressions : identifiants d'appel uniques


def tool_turn(*names: str) -> Handler:
    return chat_stream(
        {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": n, "arguments": {}}} for n in names]}, "done": False},
        done_chunk(),
    )


def test_tool_call_ids_unique_across_turns_and_backends(settings: Settings) -> None:
    backend, _ = make_backend(settings, tool_turn("read_file", "grep"))
    first = {c.id for c in backend.chat([Message("user", "a")]).tool_calls}
    second = {c.id for c in backend.chat([Message("user", "b")]).tool_calls}
    other, _ = make_backend(settings, tool_turn("read_file", "grep"))
    third = {c.id for c in other.chat([Message("user", "c")]).tool_calls}
    assert len(first) == len(second) == len(third) == 2
    assert first.isdisjoint(second) and first.isdisjoint(third) and second.isdisjoint(third)


def test_ollama_history_replays_on_claude_without_duplicate_ids(settings: Settings) -> None:
    """Session Ollama à deux tours d'outils poursuivie sur Claude (`/backend claude`)."""
    from dhaos.backends.claude import ClaudeBackend

    backend, _ = make_backend(settings, tool_turn("read_file", "grep"))
    history: list[Message] = [Message("user", "q")]
    for _ in range(2):
        resp = backend.chat(history)
        history.append(resp.to_assistant_message())
        for call in resp.tool_calls:
            history.append(Message("tool", "ok", tool_call_id=call.id, name=call.name))
    history.append(Message("user", "fin"))

    out = ClaudeBackend.to_claude_messages(history)
    tool_use_ids = [b["id"] for m in out if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    result_ids = [
        b["tool_use_id"] for m in out if m["role"] == "user" and isinstance(m["content"], list) for b in m["content"]
    ]
    assert len(tool_use_ids) == 4 and len(set(tool_use_ids)) == 4
    assert set(result_ids) == set(tool_use_ids)


# ---------------------------------------------------------------- régressions : flux interrompu / erreurs non textuelles


def test_chat_stream_without_done_raises(settings: Settings) -> None:
    backend, _ = make_backend(settings, chat_stream({"message": {"role": "assistant", "content": "début de rép"}, "done": False}))
    with pytest.raises(BackendError, match="interrompu"):
        backend.chat([Message("user", "salut")])


def test_chat_empty_body_raises(settings: Settings) -> None:
    backend, _ = make_backend(settings, chat_stream())
    with pytest.raises(BackendError, match="interrompu"):
        backend.chat([Message("user", "salut")])


def test_chat_lines_after_done_are_ignored(settings: Settings) -> None:
    handler = chat_stream(
        {"message": {"role": "assistant", "content": "ok"}, "done": False},
        done_chunk("stop", prompt=3, evals=1),
        {"message": {"role": "assistant", "content": "parasite"}, "done": False},
    )
    backend, _ = make_backend(settings, handler)
    resp = backend.chat([Message("user", "?")])
    assert resp.text == "ok" and resp.usage.input_tokens == 3


def test_truncated_stream_gives_agent_error_and_no_partial_message(settings: Settings, ctx) -> None:
    from dhaos.agent.loop import Agent
    from dhaos.agent.session import Session, SessionStore
    from dhaos.tools.base import ToolRegistry

    backend, _ = make_backend(settings, chat_stream({"message": {"content": "début"}, "done": False}))
    session = SessionStore(settings).create(backend="ollama")
    result = Agent(settings, backend, ToolRegistry([]), ctx, session=session).run("question")
    assert result.stop_reason == "error" and "interrompu" in (result.error or "")
    assert [m.role for m in Session.load(session.path).messages] == ["user"]


@pytest.mark.parametrize("err", [{"message": "model requires more system memory"}, 42, ["boom"], True])
def test_chat_inline_non_string_error_raises(settings: Settings, err: Any) -> None:
    backend, _ = make_backend(settings, chat_stream({"error": err}))
    with pytest.raises(BackendError, match="erreur Ollama") as info:
        backend.chat([Message("user", "?")])
    if isinstance(err, dict):
        assert "model requires more system memory" in str(info.value)


def test_status_error_with_non_string_error_body(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "out of memory"}})

    backend, _ = make_backend(settings, handler)
    with pytest.raises(BackendError, match="HTTP 500 : .*out of memory"):
        backend.chat([Message("user", "?")])


# --------------------------------------------------------------------------
# Appels d'outils écrits en texte par le modèle (petits modèles locaux)
# --------------------------------------------------------------------------
from dhaos.backends.ollama import rescue_text_tool_calls  # noqa: E402

LIST_DIR = ToolSpec(
    name="list_dir",
    description="Liste un dossier",
    parameters={"type": "object", "properties": {"path": {"type": "string"}, "depth": {"type": "integer"}}},
)
NAMES = {"list_dir", "read_file"}


@pytest.mark.parametrize(
    "text, expected_names, remainder",
    [
        ('{"name": "list_dir", "arguments": {"path": "/x", "depth": 2}}', ["list_dir"], ""),
        ('Je regarde.\n<tool_call>\n{"name":"read_file","arguments":{"path":"a.py"}}\n</tool_call>', ["read_file"], "Je regarde."),
        ('<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}', ["read_file"], ""),  # balise non fermée
        ('```json\n{"name":"read_file","parameters":"{\\"path\\": \\"b.py\\"}"}\n```', ["read_file"], ""),
        ('{"function": {"name": "list_dir", "arguments": {"path": "."}}}', ["list_dir"], ""),
        ('{"tool_calls": [{"name": "list_dir", "arguments": {}}, {"name": "read_file", "arguments": {"path": "x"}}]}', ["list_dir", "read_file"], ""),
        ('{"name":"list_dir","arguments":{"path":"."}}\n{"name":"read_file","arguments":{"path":"x"}}', ["list_dir", "read_file"], ""),
    ],
)
def test_rescue_text_tool_calls_formats(text: str, expected_names: list[str], remainder: str) -> None:
    rest, calls = rescue_text_tool_calls(text, NAMES)
    assert [c["name"] for c in calls] == expected_names
    assert rest == remainder
    for c in calls:
        assert isinstance(c["arguments"], dict)


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "inconnu", "arguments": {}}',  # outil non déclaré
        'Voici {"a": 1} un objet quelconque.',
        '{"name": "list_dir", "arguments": "pas du json"}',
        '{"name": "list_dir", "arguments": [1, 2]}',
        "Réponse normale sans accolade.",
        "",
    ],
)
def test_rescue_text_tool_calls_leaves_non_calls_alone(text: str) -> None:
    rest, calls = rescue_text_tool_calls(text, NAMES)
    assert calls == [] and rest == text


def test_rescue_ignores_without_tool_names() -> None:
    assert rescue_text_tool_calls('{"name": "list_dir", "arguments": {}}', set()) == ('{"name": "list_dir", "arguments": {}}', [])


def test_chat_rescues_text_tool_call_and_holds_output(settings: Settings) -> None:
    """Le modèle écrit l'appel en JSON dans content : dhaos le convertit en appel
    structuré, ne diffuse pas le JSON et rejoue l'appel comme tool_calls."""
    chunks = ['{"name": "list', '_dir", "arguments": ', '{"path": "/home/js", "depth": 2}}']
    stream = [{"message": {"role": "assistant", "content": c}, "done": False} for c in chunks]
    backend, seen = make_backend(settings, chat_stream(*stream, done_chunk()))
    shown: list[str] = []
    resp = backend.chat([Message(role="user", content="liste")], tools=[LIST_DIR], on_text=shown.append)
    assert resp.stop_reason == "tool_use"
    assert [c.name for c in resp.tool_calls] == ["list_dir"]
    assert resp.tool_calls[0].arguments == {"path": "/home/js", "depth": 2}
    assert resp.tool_calls[0].id.startswith("call_")
    assert resp.text == "" and shown == []
    assert resp.raw["tool_calls"] == [{"function": {"name": "list_dir", "arguments": {"path": "/home/js", "depth": 2}}}]
    assert resp.raw["content"] == ""
    body = json.loads(seen[0].content)
    assert body["tools"][0]["function"]["name"] == "list_dir"


def test_chat_rescue_keeps_prose_around_call(settings: Settings) -> None:
    chunks = ["Je vais lister. ", '<tool_call>{"name":"list_dir","arguments":{"path":"."}}</tool_call>']
    stream = [{"message": {"role": "assistant", "content": c}, "done": False} for c in chunks]
    backend, _ = make_backend(settings, chat_stream(*stream, done_chunk()))
    shown: list[str] = []
    resp = backend.chat([Message(role="user", content="liste")], tools=[LIST_DIR], on_text=shown.append)
    # La prose ne commence pas comme un appel : elle est diffusée immédiatement, le reste aussi
    assert shown[0] == "Je vais lister. "
    assert resp.text == "Je vais lister." and [c.name for c in resp.tool_calls] == ["list_dir"]


def test_chat_normal_text_is_streamed_immediately(settings: Settings) -> None:
    stream = [{"message": {"role": "assistant", "content": c}, "done": False} for c in ["Bon", "jour !"]]
    backend, _ = make_backend(settings, chat_stream(*stream, done_chunk()))
    shown: list[str] = []
    resp = backend.chat([Message(role="user", content="salut")], tools=[LIST_DIR], on_text=shown.append)
    assert shown == ["Bon", "jour !"] and resp.text == "Bonjour !" and resp.tool_calls == []
    assert resp.stop_reason == "end_turn"


def test_chat_json_that_is_not_a_call_is_flushed_at_end(settings: Settings) -> None:
    stream = [{"message": {"role": "assistant", "content": c}, "done": False} for c in ['{"a": ', "1}"]]
    backend, _ = make_backend(settings, chat_stream(*stream, done_chunk()))
    shown: list[str] = []
    resp = backend.chat([Message(role="user", content="json"), ], tools=[LIST_DIR], on_text=shown.append)
    assert shown == ['{"a": 1}'] and resp.text == '{"a": 1}' and resp.tool_calls == []


def test_chat_without_tools_does_not_hold_or_rescue(settings: Settings) -> None:
    text = '{"name": "list_dir", "arguments": {}}'
    stream = [{"message": {"role": "assistant", "content": text}, "done": False}]
    backend, _ = make_backend(settings, chat_stream(*stream, done_chunk()))
    shown: list[str] = []
    resp = backend.chat([Message(role="user", content="x")], tools=None, on_text=shown.append)
    assert shown == [text] and resp.text == text and resp.tool_calls == []


def test_chat_structured_tool_call_wins_over_text(settings: Settings) -> None:
    """Quand Ollama fournit message.tool_calls, le texte n'est pas réinterprété."""
    stream = [
        {"message": {"role": "assistant", "content": "ok", "tool_calls": [{"function": {"name": "list_dir", "arguments": {"path": "."}}}]}, "done": False},
    ]
    backend, _ = make_backend(settings, chat_stream(*stream, done_chunk()))
    resp = backend.chat([Message(role="user", content="x")], tools=[LIST_DIR])
    assert [c.name for c in resp.tool_calls] == ["list_dir"] and resp.text == "ok"


def _stream_chunks(*chunks: str) -> list[dict[str, Any]]:
    return [{"message": {"role": "assistant", "content": c}, "done": False} for c in chunks]


def test_gate_holds_json_after_prose_and_drops_it_when_rescued(settings: Settings) -> None:
    """Prose puis appel JSON sur une nouvelle ligne : la prose est diffusée, le JSON jamais."""
    stream = _stream_chunks("Je regarde.\n", '{"name": "list_dir", ', '"arguments": {"path": "."}}')
    backend, _ = make_backend(settings, chat_stream(*stream, done_chunk()))
    shown: list[str] = []
    resp = backend.chat([Message(role="user", content="x")], tools=[LIST_DIR], on_text=shown.append)
    assert shown == ["Je regarde.\n"]
    assert [c.name for c in resp.tool_calls] == ["list_dir"] and resp.text == "Je regarde."


def test_gate_inline_brace_in_prose_streams_normally(settings: Settings) -> None:
    stream = _stream_chunks("Le dict ", '{"a": 1} est ', "valide.")
    backend, _ = make_backend(settings, chat_stream(*stream, done_chunk()))
    shown: list[str] = []
    resp = backend.chat([Message(role="user", content="x")], tools=[LIST_DIR], on_text=shown.append)
    assert "".join(shown) == resp.text == 'Le dict {"a": 1} est valide.' and len(shown) == 3


def test_gate_partial_tool_call_tag_across_chunks(settings: Settings) -> None:
    stream = _stream_chunks("Voici <tool_", 'call>{"name":"list_dir","arguments":{"path":"."}}</tool_call>')
    backend, _ = make_backend(settings, chat_stream(*stream, done_chunk()))
    shown: list[str] = []
    resp = backend.chat([Message(role="user", content="x")], tools=[LIST_DIR], on_text=shown.append)
    assert "".join(shown) == "Voici " and [c.name for c in resp.tool_calls] == ["list_dir"]


def test_gate_python_code_block_streams(settings: Settings) -> None:
    text = "Exemple :\n```python\nd = {\"a\": 1}\nprint(d)\n```\nVoilà."
    stream = _stream_chunks(*[text[i : i + 7] for i in range(0, len(text), 7)])
    backend, _ = make_backend(settings, chat_stream(*stream, done_chunk()))
    shown: list[str] = []
    resp = backend.chat([Message(role="user", content="x")], tools=[LIST_DIR], on_text=shown.append)
    assert "".join(shown) == text and resp.tool_calls == []


def test_gate_json_block_not_a_call_is_flushed(settings: Settings) -> None:
    text = 'Config :\n```json\n{"debug": true}\n```'
    stream = _stream_chunks(text[:12], text[12:])
    backend, _ = make_backend(settings, chat_stream(*stream, done_chunk()))
    shown: list[str] = []
    resp = backend.chat([Message(role="user", content="x")], tools=[LIST_DIR], on_text=shown.append)
    assert "".join(shown) == text and resp.text == text and resp.tool_calls == []


def test_rescue_flattened_arguments_next_to_name() -> None:
    """Arguments écrits à plat à côté du nom : récupérés si ce sont des paramètres connus."""
    known = {"kb_search": {"query", "bases", "top_k"}, "list_dir": {"path", "depth"}}
    rest, calls = rescue_text_tool_calls('{"name": "kb_search", "query": "politique d\'accès"}', known)
    assert rest == "" and calls == [{"name": "kb_search", "arguments": {"query": "politique d'accès"}}]
    # clé inconnue à plat : pas d'arguments devinés
    rest, calls = rescue_text_tool_calls('{"name": "kb_search", "foo": 1}', known)
    assert calls == [{"name": "kb_search", "arguments": {}}]
    # avec un simple ensemble de noms, rien n'est deviné
    rest, calls = rescue_text_tool_calls('{"name": "kb_search", "query": "x"}', {"kb_search"})
    assert calls == [{"name": "kb_search", "arguments": {}}]
