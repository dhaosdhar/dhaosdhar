"""Doubles de test partagés."""
from __future__ import annotations

from typing import Any

from dhaos.backends.base import Backend
from dhaos.types import ChatResponse, Message, TextCallback, ToolCall, ToolSpec, Usage


class FakeBackend(Backend):
    """Backend scripté : renvoie les réponses dans l'ordre, enregistre les appels.

    Une réponse peut être une chaîne (texte final) ou un ``ChatResponse``.
    """

    name = "fake"
    model = "fake-model"

    def __init__(self, responses: list[str | ChatResponse] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        on_text: TextCallback | None = None,
        on_thinking: TextCallback | None = None,
    ) -> ChatResponse:
        self.calls.append({"messages": list(messages), "system": system, "tools": list(tools or [])})
        if not self.responses:
            resp: ChatResponse = ChatResponse(text="(fin)", stop_reason="end_turn")
        else:
            item = self.responses.pop(0)
            resp = ChatResponse(text=item, stop_reason="end_turn") if isinstance(item, str) else item
        if resp.tool_calls and resp.stop_reason == "end_turn":
            resp.stop_reason = "tool_use"
        resp.model = resp.model or self.model
        if on_text and resp.text:
            on_text(resp.text)
        return resp


def tool_call(name: str, _id: str = "call_1", **arguments: Any) -> ToolCall:
    return ToolCall(id=_id, name=name, arguments=arguments)


def tool_response(*calls: ToolCall, text: str = "") -> ChatResponse:
    return ChatResponse(text=text, tool_calls=list(calls), stop_reason="tool_use", usage=Usage(10, 5))
