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
    assert [c.id for c in resp.tool_calls] == ["call_1", "call_2", "call_3"]
    assert resp.tool_calls[0] == ToolCall("call_1", "read_file", {"path": "a.py"})
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
