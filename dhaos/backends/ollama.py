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
from pathlib import Path
from typing import Any, Callable

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


def parse_modelfile(text: str) -> dict[str, Any]:
    """Lit un Modelfile Ollama : FROM, TEMPLATE, SYSTEM (blocs \"\"\"…\"\"\") et
    PARAMETER (``stop`` répété → liste)."""
    spec: dict[str, Any] = {"from": "", "template": "", "system": "", "parameters": {}}
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith("#"):
            continue
        keyword, _, rest = raw.partition(" ")
        key = keyword.upper()
        rest = rest.strip()
        if key in ("TEMPLATE", "SYSTEM"):
            if rest.startswith('"""'):
                body = rest[3:]
                if body.endswith('"""') and len(body) >= 3:
                    value = body[:-3]
                else:
                    chunk = [body]
                    while i < len(lines):
                        line = lines[i]
                        i += 1
                        if line.rstrip().endswith('"""'):
                            chunk.append(line.rstrip()[:-3])
                            break
                        chunk.append(line)
                    value = "\n".join(chunk)
            else:
                value = rest.strip('"')
            spec[key.lower()] = value
        elif key == "FROM":
            spec["from"] = rest
        elif key == "PARAMETER":
            name, _, value = rest.partition(" ")
            value = value.strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                parsed: Any = value[1:-1]
            else:
                try:
                    parsed = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed = value
            params = spec["parameters"]
            if name == "stop":
                params.setdefault("stop", []).append(parsed)
            else:
                params[name] = parsed
    return spec


def missing_model_hint(model: str, base: str = "") -> str:
    """Conseil quand le modèle configuré est absent d'Ollama."""
    if model.split(":")[0] == "dhaos":
        return f"modèle {model} absent : lancez `dhaos model create` (base : {base or 'backends.ollama.base_model'})"
    return f"modèle absent : ollama pull {model}"


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


_ARG_KEYS = ("arguments", "parameters", "input", "args")
_META_KEYS = {"name", "function", "type", "id", "index"}


def _call_from_object(obj: Any, tool_names: "set[str] | dict[str, set[str]]") -> dict[str, Any] | None:
    """``{"name": outil, "arguments": {...}}`` (ou variantes) → appel normalisé.

    ``tool_names`` peut associer à chaque outil ses noms de paramètres : un
    petit modèle écrit parfois les arguments *à plat* à côté du nom
    (``{"name": "kb_search", "query": "…"}``) ; ils sont alors récupérés."""
    if not isinstance(obj, dict):
        return None
    fn = obj["function"] if isinstance(obj.get("function"), dict) else obj
    name = fn.get("name")
    if not isinstance(name, str) or name not in tool_names:
        return None
    args: Any = {}
    for key in _ARG_KEYS:
        if key in fn:
            args = fn[key]
            break
    else:
        params = tool_names.get(name) if isinstance(tool_names, dict) else None
        flat = {k: v for k, v in fn.items() if k not in _META_KEYS}
        if params and flat and set(flat) <= set(params):
            args = flat
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


def rescue_text_tool_calls(
    text: str, tool_names: "set[str] | dict[str, set[str]]"
) -> tuple[str, list[dict[str, Any]]]:
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


_SCHEMA_KEEP_KEYS = ("type", "properties", "required", "items", "enum", "default")


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Schéma allégé pour les modèles locaux : noms, types, requis, énumérations ;
    ni descriptions de propriétés ni bornes (minLength, maximum…). Les schémas
    complets pèsent ~2 000 tokens relus à chaque premier tour sur CPU ; les
    entrées restent validées côté registre avec le schéma complet."""
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key in _SCHEMA_KEEP_KEYS:
        if key not in schema:
            continue
        value = schema[key]
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: compact_schema(spec) if isinstance(spec, dict) else spec for name, spec in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = compact_schema(value)
        else:
            out[key] = value
    return out


class _TextGate:
    """Diffuse le texte au fil de l'eau, mais retient la fin de la réponse dès
    qu'elle ressemble à un appel d'outil écrit en clair : ``{`` ou ``[`` en
    début de ligne, ``<tool_call>`` n'importe où, ou un bloc ```json /
    ```tool_call. À la fin du tour, le texte retenu est abandonné si des
    appels ont été récupérés dedans, sinon diffusé tel quel."""

    LINE_MARKERS = ("{", "[", "```json", "```tool_call", "```tool")
    ANY_MARKERS = ("<tool_call>",)

    def __init__(self, on_text: TextCallback | None, enabled: bool):
        self.on_text = on_text
        self.enabled = bool(enabled) and on_text is not None
        self.buffer = ""  # texte retenu (à partir du marqueur)
        self.pending = ""  # fin de chunk qui pourrait être l'amorce d'un marqueur
        self.holding = False
        self.at_line_start = True

    # ----------------------------------------------------------- détection
    def _marker_index(self, text: str) -> int | None:
        best: int | None = None
        for marker in self.ANY_MARKERS:
            i = text.find(marker)
            if i != -1 and (best is None or i < best):
                best = i
        pos = 0
        while True:
            if pos > 0 or self.at_line_start:
                stripped = text[pos:].lstrip(" \t")
                offset = pos + (len(text) - pos - len(stripped))
                if any(stripped.startswith(m) for m in self.LINE_MARKERS) and (best is None or offset < best):
                    best = offset
            nl = text.find("\n", pos)
            if nl == -1:
                return best
            pos = nl + 1

    @classmethod
    def _partial_marker_len(cls, text: str) -> int:
        """Longueur du suffixe de ``text`` qui est un préfixe strict d'un marqueur
        multi-caractères (« <tool_ », « `` »…) : on l'attend avant de diffuser."""
        best = 0
        for marker in cls.ANY_MARKERS + tuple(m for m in cls.LINE_MARKERS if len(m) > 1):
            for k in range(1, len(marker)):
                if text.endswith(marker[:k]):
                    best = max(best, k)
        return best

    # -------------------------------------------------------------- flux
    def feed(self, chunk: str) -> None:
        if self.on_text is None:
            return
        if not self.enabled:
            self.on_text(chunk)
            return
        if self.holding:
            self.buffer += chunk
            return
        text = self.pending + chunk
        self.pending = ""
        idx = self._marker_index(text)
        if idx is not None:
            head, tail = text[:idx], text[idx:]
            if head:
                self.on_text(head)
            self.holding = True
            self.buffer = tail
            return
        keep = self._partial_marker_len(text)
        if keep:
            self.pending = text[-keep:]
            text = text[:-keep]
        if text:
            self.on_text(text)
            self.at_line_start = text.rstrip(" \t").endswith("\n") or (text.strip() == "" and self.at_line_start)

    def finish(self, rescued: bool = False) -> None:
        if self.on_text is None or not self.enabled:
            return
        leftover = self.buffer if self.holding else self.pending
        self.buffer = self.pending = ""
        self.holding = False
        if leftover and not rescued:
            self.on_text(leftover)


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
                "parameters": compact_schema(tool.parameters),
            },
        }

    # ------------------------------------------------------- gestion des modèles
    def list_models(self) -> list[str]:
        data = self._get_json("/api/tags")
        models = data.get("models") if isinstance(data, dict) else None
        return [str(m.get("name")) for m in (models or []) if isinstance(m, dict) and m.get("name")]

    def show_model(self, name: str) -> dict[str, Any]:
        data = self._post_json("/api/show", {"model": name})
        return data if isinstance(data, dict) else {}

    def create_model(
        self,
        name: str,
        base: str,
        *,
        system: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Crée un modèle Ollama dérivé de ``base`` (``POST /api/create``)."""
        payload: dict[str, Any] = {"model": name, "from": base, "stream": False}
        if system:
            payload["system"] = system
        if parameters:
            payload["parameters"] = parameters
        data = self._post_json("/api/create", payload)
        return data if isinstance(data, dict) else {"status": str(data)}

    def blob_exists(self, digest: str) -> bool:
        try:
            resp = self._client.head(self.host + f"/api/blobs/sha256:{digest}", timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        return resp.status_code == 200

    def upload_blob(self, digest: str, path: Path) -> None:
        """Téléverse un fichier de poids dans le dépôt d'Ollama (``POST /api/blobs/sha256:…``)."""
        try:
            with open(path, "rb") as handle:
                resp = self._client.post(
                    self.host + f"/api/blobs/sha256:{digest}", content=handle, timeout=max(self.timeout, 3600.0)
                )
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        if resp.status_code >= 400:
            raise self._status_error(resp.status_code, resp.text)

    def import_modelfile(
        self, modelfile: Path, name: str = "dhaos", *, on_log: Callable[[str], None] | None = None
    ) -> dict[str, Any]:
        """Crée ``name`` à partir d'un Modelfile dont FROM désigne un fichier
        GGUF : le blob est téléversé s'il manque, puis ``/api/create`` reçoit
        gabarit, identité et paramètres (aucune dépendance à la CLI ollama)."""
        from ..utils import sha256_file

        log = on_log or (lambda _msg: None)
        spec = parse_modelfile(Path(modelfile).read_text(encoding="utf-8"))
        source = spec.get("from") or ""
        if not source:
            raise BackendError("Modelfile sans FROM")
        weights = Path(source)
        if not weights.is_absolute():
            weights = Path(modelfile).parent / weights
        if not weights.is_file():
            raise BackendError(f"fichier de poids introuvable : {weights}")
        size_gb = weights.stat().st_size / 1e9
        log(f"empreinte des poids ({size_gb:.2f} Go)…")
        digest = sha256_file(weights)
        if self.blob_exists(digest):
            log("poids déjà présents dans Ollama")
        else:
            log(f"téléversement des poids vers Ollama ({size_gb:.2f} Go)…")
            self.upload_blob(digest, weights)
        payload: dict[str, Any] = {"model": name, "files": {weights.name: f"sha256:{digest}"}, "stream": False}
        if spec.get("template"):
            payload["template"] = spec["template"]
        if spec.get("system"):
            payload["system"] = spec["system"]
        if spec.get("parameters"):
            payload["parameters"] = spec["parameters"]
        log(f"création du modèle {name}…")
        data = self._post_json("/api/create", payload)
        return data if isinstance(data, dict) else {"status": str(data)}

    def delete_model(self, name: str) -> None:
        try:
            resp = self._client.request("DELETE", self.host + "/api/delete", json={"model": name}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        if resp.status_code >= 400:
            raise self._status_error(resp.status_code, resp.text)

    def _get_json(self, path: str) -> Any:
        try:
            resp = self._client.get(self.host + path, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        if resp.status_code >= 400:
            raise self._status_error(resp.status_code, resp.text)
        try:
            return resp.json()
        except ValueError as exc:
            raise BackendError(f"réponse invalide d'Ollama sur {path}") from exc

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
            return BackendError(missing_model_hint(self.model, self.cfg.base_model))
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
        rescued_any = False
        if tools and not tool_calls:
            # Petits modèles locaux : l'appel d'outil arrive parfois en texte
            # (JSON nu, <tool_call>, bloc ```json) au lieu de message.tool_calls.
            known = {t.name: set((t.parameters or {}).get("properties", {}) or {}) for t in tools}
            remainder, rescued = rescue_text_tool_calls(text, known)
            if rescued:
                _debug(f"appel(s) d'outil récupéré(s) depuis le texte : {[c['name'] for c in rescued]}")
                for call in rescued:
                    raw_tool_calls.append({"function": {"name": call["name"], "arguments": call["arguments"]}})
                    tool_calls.append(ToolCall(id=_new_call_id(), name=call["name"], arguments=call["arguments"]))
                text = remainder
                rescued_any = True
        gate.finish(rescued=rescued_any)
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
                f"Ollama joignable sur {self.host} mais le {missing_model_hint(self.model, self.cfg.base_model)}"
            )
        return result


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


__all__ = ["OllamaBackend"]
