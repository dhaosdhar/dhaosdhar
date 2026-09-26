"""Boucle agentique.

``Agent.run(user_input)`` : ajoute le message utilisateur, appelle le backend
avec le prompt système et les outils, exécute les appels d'outils demandés
(tous ceux d'un même tour, résultats renvoyés ensemble), boucle jusqu'à une
réponse sans outil ou ``agent.max_iterations``, persiste la session.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ..backends.base import Backend, BackendError
from ..config import Settings
from ..tools.base import ToolContext, ToolRegistry, ToolResult
from ..types import Message, TextCallback, ToolCall, Usage
from .prompts import build_system_prompt
from .session import Session, now_iso

log = logging.getLogger("dhaos.agent")

ToolCallCallback = Callable[[ToolCall], None]
ToolResultCallback = Callable[[ToolCall, ToolResult], None]

ITERATION_LIMIT_NOTICE = "[système] limite d'itérations atteinte : conclus maintenant sans nouvel outil"
_TITLE_MAX_CHARS = 80


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
        self.settings = settings
        self.backend = backend
        self.registry = registry
        self.ctx = ctx
        self.session = session
        self._messages: list[Message] = list(session.messages) if session is not None else []
        self.system_prompt = system_prompt or build_system_prompt(
            settings,
            project_root=ctx.project_root,
            bases=self._list_bases(ctx),
            backend_name=backend.name,
            tool_names=registry.names,
        )

    @staticmethod
    def _list_bases(ctx: ToolContext) -> list[Any]:
        if ctx.kb is None:
            return []
        try:
            return list(ctx.kb.list_bases())
        except Exception as e:  # noqa: BLE001 — une base illisible ne doit pas empêcher de démarrer
            log.warning("bases de savoir indisponibles pour le prompt système : %s", e)
            return []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    # ------------------------------------------------------------ interne
    def _add(self, message: Message) -> None:
        """Ajoute un message à l'historique et à la session (au fil de l'eau)."""
        self._messages.append(message)
        if self.session is not None:
            self.session.append(message)

    def _tools(self) -> list | None:
        specs = self.registry.specs()
        return specs or None

    def _persist(self, usage: Usage, iterations: int, tool_calls: int, stop_reason: str, error: str | None) -> None:
        if self.session is None:
            return
        self.session.meta["updated_at"] = now_iso()
        event: dict[str, Any] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "iterations": iterations,
            "tool_calls": tool_calls,
            "stop_reason": stop_reason,
        }
        if error:
            event["error"] = error
        self.session.log_event("usage", **event)
        try:
            self.session.save()
        except OSError as e:
            log.warning("échec de la sauvegarde de la session %s : %s", self.session.id, e)

    def _set_title_if_missing(self, user_input: str) -> None:
        if self.session is None or self.session.meta.get("title"):
            return
        first_line = next((ln.strip() for ln in user_input.splitlines() if ln.strip()), "")
        if first_line:
            self.session.meta["title"] = first_line[:_TITLE_MAX_CHARS]

    # ------------------------------------------------------------- public
    def run(
        self,
        user_input: str,
        *,
        on_text: TextCallback | None = None,
        on_thinking: TextCallback | None = None,
        on_tool_call: ToolCallCallback | None = None,
        on_tool_result: ToolResultCallback | None = None,
    ) -> AgentResult:
        usage = Usage()
        iterations = 0
        tool_calls_count = 0
        last_text = ""
        stop_reason = "end_turn"
        error: str | None = None
        max_iterations = self.settings.agent.max_iterations
        tools = self._tools()

        self._add(Message(role="user", content=user_input))
        self._set_title_if_missing(user_input)

        try:
            while True:
                try:
                    response = self.backend.chat(
                        self._messages,
                        system=self.system_prompt,
                        tools=tools,
                        on_text=on_text,
                        on_thinking=on_thinking,
                    )
                except BackendError as e:
                    stop_reason, error = "error", str(e)
                    break
                self._add(response.to_assistant_message())
                usage.add(response.usage)
                last_text = response.text

                if response.stop_reason in ("refusal", "max_tokens"):
                    # Ne jamais exécuter des appels d'outils tronqués ou refusés.
                    stop_reason = response.stop_reason
                    break
                if response.stop_reason == "error":
                    stop_reason = "error"
                    error = response.text or "le backend a signalé une erreur"
                    break
                if not response.tool_calls:
                    stop_reason = "end_turn"
                    break

                for call in response.tool_calls:
                    if on_tool_call is not None:
                        on_tool_call(call)
                    result = self.registry.execute(call, self.ctx)
                    tool_calls_count += 1
                    if on_tool_result is not None:
                        on_tool_result(call, result)
                    self._add(
                        Message(
                            role="tool",
                            content=result.content,
                            tool_call_id=call.id,
                            name=call.name,
                            is_error=result.is_error,
                        )
                    )
                    if self.session is not None:
                        self.session.log_event(
                            "tool", name=call.name, is_error=result.is_error, chars=len(result.content)
                        )
                iterations += 1

                if iterations >= max_iterations:
                    self._add(Message(role="user", content=ITERATION_LIMIT_NOTICE))
                    try:
                        final = self.backend.chat(
                            self._messages,
                            system=self.system_prompt,
                            tools=None,
                            on_text=on_text,
                            on_thinking=on_thinking,
                        )
                    except BackendError as e:
                        stop_reason, error = "error", str(e)
                        break
                    self._add(final.to_assistant_message())
                    usage.add(final.usage)
                    last_text = final.text
                    stop_reason = "max_iterations"
                    break
        finally:
            self._persist(usage, iterations, tool_calls_count, stop_reason, error)

        return AgentResult(
            text="" if stop_reason == "error" else last_text,
            usage=usage,
            iterations=iterations,
            tool_calls=tool_calls_count,
            stop_reason=stop_reason,
            error=error,
        )

    def reset(self) -> None:
        """Vide l'historique (et la session persistée, le cas échéant)."""
        self._messages = []
        if self.session is not None:
            self.session.messages.clear()
            self.session.events.clear()
            self.session.meta["updated_at"] = now_iso()
            try:
                self.session.save()
            except OSError as e:
                log.warning("échec de la sauvegarde de la session %s : %s", self.session.id, e)
