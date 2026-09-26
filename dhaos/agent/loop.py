"""Boucle agentique — CONTRAT.

``Agent.run(user_input)`` : ajoute le message utilisateur, appelle le backend
avec le prompt système et les outils, exécute les appels d'outils demandés
(tous ceux d'un même tour, résultats renvoyés ensemble), boucle jusqu'à une
réponse sans outil ou ``agent.max_iterations``, persiste la session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..backends.base import Backend
from ..config import Settings
from ..tools.base import ToolContext, ToolRegistry, ToolResult
from ..types import Message, TextCallback, ToolCall, Usage
from .session import Session

ToolCallCallback = Callable[[ToolCall], None]
ToolResultCallback = Callable[[ToolCall, ToolResult], None]


@dataclass
class AgentResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    iterations: int = 0
    tool_calls: int = 0
    stop_reason: str = "end_turn"  # end_turn | max_iterations | max_tokens | refusal | error
    error: str | None = None


class Agent:
    def __init__(
        self,
        settings: Settings,
        backend: Backend,
        registry: ToolRegistry,
        ctx: ToolContext,
        *,
        session: Session | None = None,
        system_prompt: str | None = None,
    ):
        raise NotImplementedError

    @property
    def messages(self) -> list[Message]:
        raise NotImplementedError

    def run(
        self,
        user_input: str,
        *,
        on_text: TextCallback | None = None,
        on_thinking: TextCallback | None = None,
        on_tool_call: ToolCallCallback | None = None,
        on_tool_result: ToolResultCallback | None = None,
    ) -> AgentResult:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
