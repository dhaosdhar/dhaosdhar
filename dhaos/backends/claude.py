"""Backend Claude (API Anthropic, SDK ``anthropic`` ≥ 1.8).

Le SDK 1.x repose sur ``httpx2`` : on ne lui passe jamais d'objets du paquet
``httpx``. Le client est construit paresseusement ; la clé est résolue par le
SDK lui-même (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` ou profil
``ant auth login``).

Chaque tour passe par ``client.beta.messages.stream`` avec réflexion adaptative,
niveau d'effort, mise en cache automatique et, si configuré, bascule serveur
(``fallbacks="default"``). Les blocs de contenu renvoyés sont conservés tels
quels dans ``ChatResponse.raw`` pour être rejoués à l'identique lors d'un
enchaînement d'outils (blocs de réflexion compris).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config import Settings
from ..types import ChatResponse, Message, StopReason, TextCallback, ToolCall, ToolSpec, Usage
from .base import Backend, BackendError

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_JSON_RETRIES = 2  # réémissions d'un tour dont le JSON d'outil est imparsable

_MAX_TOKENS_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})
_EMPTY_RESULT = "(aucune sortie)"


def _jsonable(value: Any) -> Any:
    """Réduit une valeur inattendue à quelque chose de sérialisable."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _block_to_dict(block: Any) -> dict[str, Any]:
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        data = dump(mode="json", exclude_none=True)
        if isinstance(data, dict):
            return data
    if isinstance(block, dict):
        return dict(block)
    return {"type": str(getattr(block, "type", "unknown"))}


def _looks_like_key_material(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


class ClaudeBackend(Backend):
    name = "claude"

    def __init__(self, settings: Settings, model: str | None = None, *, client: Any = None):
        self.settings = settings
        self.cfg = settings.backends.claude
        self.model = model or self.cfg.model
        self._client = client

    # ------------------------------------------------------------ client
    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(base_url=self.cfg.base_url or None, timeout=self.cfg.timeout)
        return self._client

    # ------------------------------------------------------------ conversion
    @staticmethod
    def _tool_to_claude(tool: ToolSpec) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
            "eager_input_streaming": True,
        }

    @staticmethod
    def to_claude_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Historique neutre → messages Claude.

        Les messages ``tool`` consécutifs sont regroupés en un seul message
        ``user`` de blocs ``tool_result`` ; un message assistant est rejoué
        depuis ``raw`` (liste de blocs) quand il est disponible.
        """
        out: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []

        def flush() -> None:
            if pending:
                out.append({"role": "user", "content": list(pending)})
                pending.clear()

        for m in messages:
            if m.role == "tool":
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content or _EMPTY_RESULT,
                }
                if m.is_error:
                    block["is_error"] = True
                pending.append(block)
                continue
            flush()
            if m.role == "user":
                if not m.content:
                    continue
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                raw = m.raw
                if isinstance(raw, list) and raw and all(isinstance(b, dict) for b in raw):
                    out.append({"role": "assistant", "content": [dict(b) for b in raw]})
                    continue
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for call in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
                if not blocks:
                    continue
                out.append({"role": "assistant", "content": blocks})
        flush()
        return out

    def build_params(
        self, messages: list[Message], *, system: str = "", tools: list[ToolSpec] | None = None
    ) -> dict[str, Any]:
        """Paramètres de ``client.beta.messages.stream`` pour un tour."""
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.cfg.max_tokens,
            "messages": self.to_claude_messages(messages),
            "thinking": {"type": "adaptive", "display": self.cfg.thinking_display},
            "output_config": {"effort": self.cfg.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            params["system"] = [{"type": "text", "text": system}]
        if tools:
            params["tools"] = [self._tool_to_claude(t) for t in tools]
        if self.cfg.fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    # ------------------------------------------------------------ chat
    def chat(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        on_text: TextCallback | None = None,
        on_thinking: TextCallback | None = None,
    ) -> ChatResponse:
        import anthropic

        params = self.build_params(messages, system=system, tools=tools)
        final: Any = None
        attempts = 0
        while final is None:
            try:
                with self.client.beta.messages.stream(**params) as stream:
                    for event in stream:
                        etype = getattr(event, "type", None)
                        if etype == "text":
                            chunk = getattr(event, "text", "")
                            if on_text and chunk:
                                on_text(chunk)
                        elif etype == "thinking":
                            chunk = getattr(event, "thinking", "")
                            if on_thinking and chunk:
                                on_thinking(chunk)
                    final = stream.get_final_message()
            except ValueError as exc:
                # JSON d'appel d'outil que le SDK n'a pas pu parser : le bloc
                # n'a pas d'identifiant à répondre, on réémet le tour (borné).
                attempts += 1
                if attempts > MAX_JSON_RETRIES:
                    raise BackendError(
                        f"Claude a produit un appel d'outil au JSON illisible {attempts} fois de suite : {exc}"
                    ) from exc
                continue
            except (anthropic.AuthenticationError, anthropic.CredentialsError) as exc:
                raise BackendError(
                    "clé API absente ou invalide : définissez ANTHROPIC_API_KEY (ou `ant auth login`)"
                ) from exc
            except anthropic.NotFoundError as exc:
                raise BackendError(
                    f"modèle inconnu : {self.model} (vérifiez backends.claude.model ; {exc.message})"
                ) from exc
            except anthropic.RateLimitError as exc:
                retry_after = ""
                headers = getattr(getattr(exc, "response", None), "headers", None)
                if headers is not None:
                    value = headers.get("retry-after")
                    if value:
                        retry_after = f" (réessayez dans {value} s)"
                raise BackendError(f"limite de débit Anthropic atteinte{retry_after} : {exc.message}") from exc
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    raise BackendError(
                        f"erreur côté serveur Anthropic (HTTP {exc.status_code}) : réessayez plus tard — {exc.message}"
                    ) from exc
                raise BackendError(f"requête refusée par l'API Anthropic (HTTP {exc.status_code}) : {exc.message}") from exc
            except anthropic.APIConnectionError as exc:
                raise BackendError(
                    "impossible de joindre l'API Anthropic : vérifiez la connexion réseau, le proxy "
                    f"ou backends.claude.base_url ({exc})"
                ) from exc
        return self._to_response(final)

    # ------------------------------------------------------------ mapping
    def _to_response(self, final: Any) -> ChatResponse:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content = getattr(final, "content", None) or []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(str(getattr(block, "text", "") or ""))
            elif btype == "tool_use":
                inp = getattr(block, "input", None)
                arguments = dict(inp) if isinstance(inp, dict) else {"_raw": _jsonable(inp)}
                tool_calls.append(
                    ToolCall(id=str(getattr(block, "id", "") or ""), name=str(getattr(block, "name", "") or ""), arguments=arguments)
                )
            elif btype == "thinking":
                thinking_parts.append(str(getattr(block, "thinking", "") or ""))
        raw = [_block_to_dict(b) for b in content]

        text = "".join(text_parts)
        api_stop = getattr(final, "stop_reason", None)
        stop_reason: StopReason
        if api_stop in _MAX_TOKENS_REASONS:
            stop_reason = "max_tokens"
            tool_calls = []  # entrée d'outil potentiellement tronquée : on n'exécute rien
        elif api_stop == "refusal":
            stop_reason = "refusal"
            tool_calls = []
            details = getattr(final, "stop_details", None)
            if details is not None:
                category = getattr(details, "category", None)
                explanation = getattr(details, "explanation", None)
                parts = [str(p) for p in (category, explanation) if p]
                note = f"[refus{' : ' + ' — '.join(parts) if parts else ''}]"
                text = f"{text}\n{note}" if text else note
        elif tool_calls:
            # "tool_use", ou fin de tour avec des appels : on les exécute.
            stop_reason = "tool_use"
        else:
            # end_turn, stop_sequence, pause_turn ou valeur inconnue.
            stop_reason = "end_turn"

        usage_obj = getattr(final, "usage", None)
        usage = Usage(
            input_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
        )
        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=str(getattr(final, "model", "") or self.model),
            raw=raw,
            thinking="".join(thinking_parts),
        )

    # ------------------------------------------------------------ santé
    def healthcheck(self) -> dict[str, Any]:
        """Sans appel réseau : vérifie qu'une clé ou un profil est détectable."""
        result: dict[str, Any] = {"ok": False, "backend": self.name, "model": self.model, "detail": "", "models": []}
        if _looks_like_key_material(os.environ.get("ANTHROPIC_API_KEY")):
            result["ok"] = True
            result["detail"] = f"clé détectée via ANTHROPIC_API_KEY ; modèle {self.model}"
            return result
        if _looks_like_key_material(os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            result["ok"] = True
            result["detail"] = f"jeton détecté via ANTHROPIC_AUTH_TOKEN ; modèle {self.model}"
            return result
        profile_dir = Path(os.path.expanduser("~/.config/anthropic"))
        if profile_dir.is_dir():
            result["ok"] = True
            result["detail"] = f"profil `ant auth login` détecté ({profile_dir}) ; modèle {self.model}"
            return result
        result["detail"] = (
            "aucune clé API détectée : définissez ANTHROPIC_API_KEY (ou ANTHROPIC_AUTH_TOKEN) "
            "ou lancez `ant auth login`"
        )
        return result


__all__ = ["ClaudeBackend", "FALLBACK_BETA", "MAX_JSON_RETRIES"]
