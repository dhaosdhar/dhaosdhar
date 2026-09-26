"""Backend Ollama (modèles locaux).

- ``POST /api/chat`` en flux NDJSON (texte, réflexion, appels d'outils),
- ``POST /api/embed`` pour les vecteurs,
- ``GET /api/tags`` pour l'état de santé.

Le client ``httpx`` est injectable (les tests utilisent ``httpx.MockTransport``).
Toutes les données reçues sont considérées comme non fiables et validées.
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from typing import Any

import httpx

from ..config import Settings
from ..types import ChatResponse, Message, TextCallback, ToolCall, ToolSpec, Usage
from .base import Backend, BackendError

_ERR_PREFIX = "[erreur] "


def _new_call_id() -> str:
    """Identifiant d'appel d'outil unique (Ollama n'en fournit pas).

    Unique par processus et entre sessions rechargées : un même historique
    peut être poursuivi sur Claude, dont l'API exige des ids ``tool_use``
    distincts.
    """
    return f"call_{uuid.uuid4().hex[:12]}"


def _error_detail(err: Any) -> str:
    """Texte d'un champ ``error`` renvoyé par Ollama, quelle que soit sa forme."""
    if isinstance(err, str):
        return err
    try:
        return json.dumps(err, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(err)


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



_DEBUG = bool(os.environ.get("DHAOS_DEBUG"))
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|tool_call|tool)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_TEXT_CALL_PREFIXES = ("{", "[", "<tool_call>", "```")


def _debug(msg: str) -> None:
    if _DEBUG:
        sys.stderr.write(f"[dhaos/ollama] {msg}\n")
        sys.stderr.flush()


def _iter_json_values(text: str):
    """Itère ``(début, fin, valeur)`` pour chaque valeur JSON de premier niveau
    (objet ou tableau) décodable dans ``text``, dans l'ordre d'apparition."""
    decoder = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        starts = [k for k in (text.find("{", i), text.find("[", i)) if k != -1]
        if not starts:
            return
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        yield start, end, value
        i = end


def _call_from_object(obj: Any, tool_names: set[str]) -> dict[str, Any] | None:
    """``{"name": outil, "arguments": {...}}`` (ou variantes) → appel normalisé."""
    if not isinstance(obj, dict):
        return None
    fn = obj["function"] if isinstance(obj.get("function"), dict) else obj
    name = fn.get("name")
    if not isinstance(name, str) or name not in tool_names:
        return None
    args: Any = {}
    for key in ("arguments", "parameters", "input", "args"):
        if key in fn:
            args = fn[key]
            break
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def _calls_from_value(value: Any, tool_names: set[str]) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    items = value if isinstance(value, list) else [value]
    calls = []
    for item in items:
        call = _call_from_object(item, tool_names)
        if call:
            calls.append(call)
    return calls


def rescue_text_tool_calls(text: str, tool_names: set[str]) -> tuple[str, list[dict[str, Any]]]:
    """Récupère les appels d'outils qu'un modèle a écrits *en texte* au lieu
    d'utiliser le mécanisme structuré : JSON nu ``{"name": …, "arguments": …}``,
    balises ``<tool_call>…</tool_call>`` ou bloc ```json. Seuls les noms
    d'outils connus sont acceptés. Renvoie le texte débarrassé des appels et la
    liste des appels ``{"name", "arguments"}`` dans l'ordre d'apparition."""
    if not text or not tool_names or "{" not in text:
        return text, []
    found: list[tuple[int, int, list[dict[str, Any]]]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(s < end and start < e for s, e, _ in found)

    for regex in (_TOOL_CALL_TAG_RE, _FENCE_RE):
        for m in regex.finditer(text):
            if overlaps(m.start(), m.end()):
                continue
            calls = [c for _, _, v in _iter_json_values(m.group(1)) for c in _calls_from_value(v, tool_names)]
            if calls:
                found.append((m.start(), m.end(), calls))
    for start, end, value in _iter_json_values(text):
        if overlaps(start, end):
            continue
        calls = _calls_from_value(value, tool_names)
        if calls:
            found.append((start, end, calls))
    if not found:
        return text, []
    found.sort(key=lambda t: t[0])
    pieces, pos, calls = [], 0, []
    for start, end, group in found:
        pieces.append(text[pos:start])
        calls.extend(group)
        pos = end
    pieces.append(text[pos:])
    return "".join(pieces).strip(), calls


class _TextGate:
    """Retient la diffusion du texte tant qu'il peut s'agir d'un appel d'outil
    écrit en clair (début ``{``, ``[``, ``<tool_call>`` ou ```) ; tout autre
    début est diffusé immédiatement, puis le reste au fil de l'eau."""

    def __init__(self, on_text: TextCallback | None, enabled: bool):
        self.on_text = on_text
        self.buffer = ""
        self.passthrough = on_text is None or not enabled

    def feed(self, chunk: str) -> None:
        if self.on_text is None:
            return
        if self.passthrough:
            self.on_text(chunk)
            return
        self.buffer += chunk
        head = self.buffer.lstrip()
        if not head:
            return
        if any(head.startswith(p) or p.startswith(head) for p in _TEXT_CALL_PREFIXES):
            return  # candidat : on retient
        self.passthrough = True
        self.on_text(self.buffer)
        self.buffer = ""

    def finish(self, replacement: str | None = None) -> None:
        if self.on_text is None or self.passthrough:
            return
        text = self.buffer if replacement is None else replacement
        self.buffer = ""
        if text:
            self.on_text(text)


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
            if isinstance(parsed, dict) and parsed.get("error") is not None:
                detail = _error_detail(parsed["error"])
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
        done_seen = False
        usage = Usage()
        model_name = self.model
        gate = _TextGate(on_text, enabled=bool(tools))
        _debug(
            f"POST /api/chat model={self.model} messages={len(payload['messages'])} "
            f"tools={len(payload.get('tools', []))} options={payload.get('options')}"
        )

        try:
            with self._client.stream("POST", self.host + "/api/chat", json=payload, timeout=self.timeout) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise self._status_error(resp.status_code, resp.text)
                for line in resp.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    _debug(f"<< {line[:400]}")
                    try:
                        obj = json.loads(line)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise BackendError(f"flux Ollama invalide : ligne non JSON ({line[:120]!r})") from exc
                    if not isinstance(obj, dict):
                        raise BackendError("flux Ollama invalide : objet attendu")
                    if obj.get("error") is not None:
                        raise BackendError(f"erreur Ollama : {_error_detail(obj['error'])[:2000]}")

                    if isinstance(obj.get("model"), str):
                        model_name = obj["model"]
                    msg = obj.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content:
                            text_parts.append(content)
                            gate.feed(content)
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
                                tool_calls.append(ToolCall(id=_new_call_id(), name=fn["name"], arguments=arguments))
                    if obj.get("done"):
                        done_seen = True
                        dr = obj.get("done_reason")
                        done_reason = dr if isinstance(dr, str) else None
                        usage = Usage(
                            input_tokens=_as_int(obj.get("prompt_eval_count")),
                            output_tokens=_as_int(obj.get("eval_count")),
                        )
                        break  # objet final : ignorer d'éventuelles lignes ultérieures
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        if not done_seen:
            # Fin de flux propre sans objet final : la réponse est tronquée
            # (runner arrêté, proxy qui coupe…) et ne doit pas passer pour complète.
            raise BackendError(
                "flux Ollama interrompu avant la fin de la réponse (objet final `done` absent) — relancez la requête"
            )

        text = "".join(text_parts)
        if tools and not tool_calls:
            # Petits modèles locaux : l'appel d'outil arrive parfois en texte
            # (JSON nu, <tool_call>, bloc ```json) au lieu de message.tool_calls.
            remainder, rescued = rescue_text_tool_calls(text, {t.name for t in tools})
            if rescued:
                _debug(f"appel(s) d'outil récupéré(s) depuis le texte : {[c['name'] for c in rescued]}")
                for call in rescued:
                    raw_tool_calls.append({"function": {"name": call["name"], "arguments": call["arguments"]}})
                    tool_calls.append(ToolCall(id=_new_call_id(), name=call["name"], arguments=call["arguments"]))
                text = remainder
                gate.finish(replacement=remainder)
        gate.finish()
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
