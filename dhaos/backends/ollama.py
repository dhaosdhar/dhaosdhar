"""Backend Ollama (modèles locaux).

- ``POST /api/chat`` en flux NDJSON (texte, réflexion, appels d'outils),
- ``POST /api/embed`` pour les vecteurs,
- ``GET /api/tags`` pour l'état de santé.

Le client ``httpx`` est injectable (les tests utilisent ``httpx.MockTransport``).
Toutes les données reçues sont considérées comme non fiables et validées.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import Settings
from ..types import ChatResponse, Message, TextCallback, ToolCall, ToolSpec, Usage
from .base import Backend, BackendError

_ERR_PREFIX = "[erreur] "


def _strip_latest(name: str) -> str:
    return name[: -len(":latest")] if name.endswith(":latest") else name


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Arguments d'un appel d'outil : dict, chaîne JSON, ou n'importe quoi."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {"_raw": raw}
        return dict(parsed) if isinstance(parsed, dict) else {"_raw": raw}
    try:
        return {"_raw": json.loads(json.dumps(raw))}
    except (TypeError, ValueError):
        return {"_raw": str(raw)}


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, settings: Settings, model: str | None = None, *, client: httpx.Client | None = None):
        self.settings = settings
        self.cfg = settings.backends.ollama
        self.host = self.cfg.host.rstrip("/")
        self.model = model or self.cfg.model
        self.timeout = self.cfg.timeout
        self._client = client or httpx.Client(timeout=self.timeout)

    # ------------------------------------------------------------ conversion
    @staticmethod
    def _tool_to_ollama(tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def to_ollama_messages(messages: list[Message], system: str = "") -> list[dict[str, Any]]:
        """Historique neutre → liste de messages Ollama (système en tête si non vide)."""
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                if isinstance(m.raw, dict) and m.raw.get("role") == "assistant":
                    out.append(dict(m.raw))
                    continue
                msg: dict[str, Any] = {"role": "assistant", "content": m.content}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
                    ]
                out.append(msg)
            elif m.role == "tool":
                content = (_ERR_PREFIX + m.content) if m.is_error else m.content
                msg = {"role": "tool", "content": content}
                if m.name:
                    msg["tool_name"] = m.name
                out.append(msg)
        return out

    # ------------------------------------------------------------ réseau
    def _network_error(self, exc: httpx.HTTPError) -> BackendError:
        if isinstance(exc, httpx.ConnectError):
            return BackendError(f"Ollama injoignable sur {self.host} — lancez `ollama serve`")
        if isinstance(exc, httpx.TimeoutException):
            return BackendError(
                f"Ollama n'a pas répondu dans le délai ({self.timeout:g} s) — "
                "augmentez backends.ollama.timeout ou choisissez un modèle plus léger"
            )
        return BackendError(f"erreur réseau avec Ollama ({self.host}) : {type(exc).__name__}: {exc}")

    def _status_error(self, status: int, body: str) -> BackendError:
        if status == 404:
            return BackendError(f"modèle absent : ollama pull {self.model}")
        detail = body.strip()
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
                detail = parsed["error"]
        except (json.JSONDecodeError, ValueError):
            pass
        return BackendError(f"Ollama a renvoyé HTTP {status} : {detail[:2000]}")

    def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            resp = self._client.post(self.host + path, json=payload, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        if resp.status_code >= 400:
            raise self._status_error(resp.status_code, resp.text)
        try:
            return resp.json()
        except ValueError as exc:
            raise BackendError(f"réponse Ollama invalide sur {path} : JSON illisible") from exc

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_ollama_messages(messages, system),
            "stream": True,
        }
        if tools:
            payload["tools"] = [self._tool_to_ollama(t) for t in tools]
        if self.cfg.num_ctx:
            payload["options"] = {"num_ctx": self.cfg.num_ctx}
        if self.cfg.keep_alive is not None:
            payload["keep_alive"] = self.cfg.keep_alive

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        raw_tool_calls: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        done_reason: str | None = None
        usage = Usage()
        model_name = self.model

        try:
            with self._client.stream("POST", self.host + "/api/chat", json=payload, timeout=self.timeout) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise self._status_error(resp.status_code, resp.text)
                for line in resp.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise BackendError(f"flux Ollama invalide : ligne non JSON ({line[:120]!r})") from exc
                    if not isinstance(obj, dict):
                        raise BackendError("flux Ollama invalide : objet attendu")
                    if isinstance(obj.get("error"), str):
                        raise BackendError(f"erreur Ollama : {obj['error']}")

                    if isinstance(obj.get("model"), str):
                        model_name = obj["model"]
                    msg = obj.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content:
                            text_parts.append(content)
                            if on_text:
                                on_text(content)
                        thinking = msg.get("thinking")
                        if isinstance(thinking, str) and thinking:
                            thinking_parts.append(thinking)
                            if on_thinking:
                                on_thinking(thinking)
                        calls = msg.get("tool_calls")
                        if isinstance(calls, list):
                            for item in calls:
                                if not isinstance(item, dict):
                                    continue
                                fn = item.get("function")
                                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
                                    continue
                                arguments = _parse_arguments(fn.get("arguments"))
                                raw_tool_calls.append({"function": {"name": fn["name"], "arguments": arguments}})
                                tool_calls.append(
                                    ToolCall(id=f"call_{len(tool_calls) + 1}", name=fn["name"], arguments=arguments)
                                )
                    if obj.get("done"):
                        dr = obj.get("done_reason")
                        done_reason = dr if isinstance(dr, str) else None
                        usage = Usage(
                            input_tokens=_as_int(obj.get("prompt_eval_count")),
                            output_tokens=_as_int(obj.get("eval_count")),
                        )
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc

        text = "".join(text_parts)
        raw: dict[str, Any] = {"role": "assistant", "content": text}
        if thinking_parts:
            raw["thinking"] = "".join(thinking_parts)
        if raw_tool_calls:
            raw["tool_calls"] = raw_tool_calls

        if tool_calls:
            stop_reason = "tool_use"
        elif done_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=model_name,
            raw=raw,
            thinking="".join(thinking_parts),
        )

    # ------------------------------------------------------------ embeddings
    def supports_embeddings(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._post_json("/api/embed", {"model": self.cfg.embed_model, "input": list(texts)})
        embeddings = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise BackendError(
                f"réponse d'embedding invalide d'Ollama (modèle {self.cfg.embed_model}) : "
                f"{len(embeddings) if isinstance(embeddings, list) else 'aucun'} vecteur(s) pour {len(texts)} texte(s)"
            )
        out: list[list[float]] = []
        for vec in embeddings:
            if not isinstance(vec, list):
                raise BackendError("réponse d'embedding invalide d'Ollama : vecteur attendu")
            try:
                out.append([float(x) for x in vec])
            except (TypeError, ValueError) as exc:
                raise BackendError("réponse d'embedding invalide d'Ollama : valeur non numérique") from exc
        return out

    # ------------------------------------------------------------ santé
    def healthcheck(self) -> dict[str, Any]:
        result: dict[str, Any] = {"ok": False, "backend": self.name, "model": self.model, "detail": "", "models": []}
        try:
            resp = self._client.get(self.host + "/api/tags", timeout=min(self.timeout, 10.0))
        except httpx.HTTPError:
            result["detail"] = f"Ollama injoignable sur {self.host} — lancez `ollama serve`"
            return result
        if resp.status_code >= 400:
            result["detail"] = f"Ollama a renvoyé HTTP {resp.status_code} sur /api/tags"
            return result
        try:
            data = resp.json()
        except ValueError:
            result["detail"] = "réponse invalide d'Ollama sur /api/tags"
            return result
        models: list[str] = []
        entries = data.get("models") if isinstance(data, dict) else None
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                models.append(entry["name"])
        result["models"] = models
        wanted = _strip_latest(self.model)
        present = any(_strip_latest(m) == wanted for m in models)
        result["ok"] = present
        if present:
            result["detail"] = f"Ollama joignable sur {self.host} ; modèle {self.model} présent"
        else:
            result["detail"] = (
                f"Ollama joignable sur {self.host} mais le modèle {self.model} est absent : "
                f"ollama pull {self.model}"
            )
        return result


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


__all__ = ["OllamaBackend"]
