"""Boucle agentique.

``Agent.run(user_input)`` : ajoute le message utilisateur, appelle le backend
avec le prompt système et les outils, exécute les appels d'outils demandés
(tous ceux d'un même tour, résultats renvoyés ensemble), boucle jusqu'à une
réponse sans outil ou ``agent.max_iterations``, persiste la session.

Invariant : tout appel d'outil présent dans l'historique (``tool_calls`` ou
bloc ``tool_use`` de ``raw``) est suivi d'un message ``tool`` — au besoin
synthétique et ``is_error`` (interruption, réponse tronquée ou refusée,
limite d'itérations, session rechargée incomplète).
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
# Résultats d'outil synthétiques (``is_error``) qui ferment un appel jamais
# exécuté : l'API Claude refuse tout historique où un ``tool_use`` n'est pas
# immédiatement suivi de son ``tool_result``, et une session ainsi persistée
# serait inutilisable à chaque reprise.
INTERRUPTED_TOOL_NOTICE = "[système] exécution interrompue : appel non exécuté"
TRUNCATED_TOOL_NOTICE = "[système] réponse tronquée : appel non exécuté"
REFUSED_TOOL_NOTICE = "[système] réponse refusée : appel non exécuté"
LIMIT_TOOL_NOTICE = "[système] limite d'itérations atteinte : appel non exécuté"
ORPHAN_TOOL_NOTICE = "[système] appel resté sans résultat (session interrompue) : appel non exécuté"
_TITLE_MAX_CHARS = 80


def _tool_use_ids(message: Message) -> list[tuple[str, str]]:
    """``(id, nom)`` des appels d'outils portés par un message assistant.

    Lit ``tool_calls`` et, à défaut, les blocs ``tool_use`` de ``raw`` (un
    backend peut vider ``tool_calls`` — réponse tronquée — tout en rejouant la
    charge brute, qui contient encore les blocs).
    """
    seen: dict[str, str] = {}
    for call in message.tool_calls:
        seen.setdefault(str(call.id), str(call.name))
    if isinstance(message.raw, list):
        for block in message.raw:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                seen.setdefault(str(block["id"]), str(block.get("name") or ""))
    return list(seen.items())


def _closing_message(call_id: str, name: str, reason: str) -> Message:
    return Message(role="tool", content=reason, tool_call_id=call_id, name=name or None, is_error=True)


def repair_orphan_tool_calls(messages: list[Message], reason: str = ORPHAN_TOOL_NOTICE) -> int:
    """Complète *en place* chaque appel d'outil sans résultat par un message
    ``tool`` en erreur, inséré juste après les résultats existants du tour.

    Renvoie le nombre de messages insérés. Sert à réparer une session
    persistée après une interruption (Ctrl-C pendant un tour d'outils) ou
    écrite par une version antérieure.
    """
    inserted = 0
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.role != "assistant":
            i += 1
            continue
        j = i + 1
        answered: set[str | None] = set()
        while j < len(messages) and messages[j].role == "tool":
            answered.add(messages[j].tool_call_id)
            j += 1
        missing = [(cid, name) for cid, name in _tool_use_ids(message) if cid not in answered]
        for offset, (cid, name) in enumerate(missing):
            messages.insert(j + offset, _closing_message(cid, name, reason))
        inserted += len(missing)
        i = j + len(missing)
    return inserted


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
        if session is not None:
            repaired = repair_orphan_tool_calls(session.messages)
            if repaired:
                log.warning("session %s : %d appel(s) d'outil sans résultat complété(s)", session.id, repaired)
        self._messages: list[Message] = list(session.messages) if session is not None else []
        self.system_prompt = system_prompt or build_system_prompt(
            settings,
            project_root=ctx.project_root,
            bases=self._list_bases(ctx),
            backend_name=backend.name,
            tool_names=registry.names,
        )
        # Prompt système réellement envoyé au backend, persisté avec la session
        # (matière première du jeu SFT : ``train/dataset.session_to_example``).
        if session is not None and not session.meta.get("system_prompt"):
            session.meta["system_prompt"] = self.system_prompt

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

    def _close_unanswered(self, assistant: Message, reason: str) -> int:
        """Ajoute un résultat en erreur pour chaque appel de ``assistant``
        (dernier message assistant de l'historique) resté sans message ``tool``."""
        answered = {m.tool_call_id for m in self._messages if m.role == "tool"}
        missing = [(cid, name) for cid, name in _tool_use_ids(assistant) if cid not in answered]
        for cid, name in missing:
            self._add(_closing_message(cid, name, reason))
        return len(missing)

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
                assistant = response.to_assistant_message()
                self._add(assistant)
                usage.add(response.usage)
                last_text = response.text

                if response.stop_reason in ("refusal", "max_tokens"):
                    # Ne jamais exécuter des appels d'outils tronqués ou refusés,
                    # mais les clore : un tool_use sans tool_result rendrait
                    # l'historique irrecevable à chaque tour suivant.
                    stop_reason = response.stop_reason
                    self._close_unanswered(
                        assistant, REFUSED_TOOL_NOTICE if stop_reason == "refusal" else TRUNCATED_TOOL_NOTICE
                    )
                    break
                if response.stop_reason == "error":
                    stop_reason = "error"
                    error = response.text or "le backend a signalé une erreur"
                    break
                if not response.tool_calls:
                    stop_reason = "end_turn"
                    break

                try:
                    for call in response.tool_calls:
                        if on_tool_call is not None:
                            on_tool_call(call)
                        result = self.registry.execute(call, self.ctx)
                        tool_calls_count += 1
                        # Le résultat réel est enregistré avant d'être affiché :
                        # un callback défaillant ne doit pas le faire perdre.
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
                        if on_tool_result is not None:
                            on_tool_result(call, result)
                except BaseException as exc:
                    # KeyboardInterrupt (Ctrl-C pendant un outil), callback
                    # défaillant… : clore les appels du tour restés sans
                    # résultat avant que ``finally`` ne persiste la session.
                    stop_reason, error = "interrupted", f"tour interrompu : {type(exc).__name__}"
                    self._close_unanswered(assistant, INTERRUPTED_TOOL_NOTICE)
                    raise
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
                    closing = final.to_assistant_message()
                    self._add(closing)
                    usage.add(final.usage)
                    last_text = final.text
                    stop_reason = "max_iterations"
                    # Le modèle a pu ignorer la consigne : ses appels ne sont
                    # pas exécutés, mais reçoivent un résultat en erreur.
                    self._close_unanswered(closing, LIMIT_TOOL_NOTICE)
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
