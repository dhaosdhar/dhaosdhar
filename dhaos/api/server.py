"""API HTTP de dhaos (FastAPI).

``create_app(settings, *, backend_factory=None, kb_factory=None)`` construit
l'application :

- les bases de savoir sont ouvertes une fois (``app.state.kb``) et fermées à
  l'arrêt ;
- chaque ``POST /chat`` assemble une exécution complète via
  ``dhaos.runtime.build_runtime`` (backend, outils, session persistée) ;
- la confirmation des actions sensibles suit ``api.auto_confirm`` (sinon
  elles sont refusées : l'API n'a personne à qui demander) ;
- toutes les routes exigent ``Authorization: Bearer <token>`` : ``api.token``
  ou, à défaut, un jeton aléatoire généré à la construction et exposé dans
  ``app.state.token`` (affiché par ``dhaos serve``). L'API donne le même accès
  disque que la CLI ; elle n'est donc jamais servie sans jeton. ``/health``
  reste joignable sans jeton mais ne renvoie alors qu'un état minimal
  (ni détail du backend, ni appel sortant) ;
- un garde Host/Origin (``HostOriginGuard``) refuse les requêtes dont
  l'en-tête ``Host`` n'est pas attendu (400 ; ``api.allowed_hosts``, défaut :
  localhost, 127.0.0.1, ::1 et ``api.host``) et celles portant un ``Origin``
  non listé dans ``api.allowed_origins`` (403 ; vide par défaut), ce qui
  neutralise le DNS rebinding depuis un navigateur ;
- l'ingestion (``POST /kb/{name}/documents``) soumet chaque fichier, y
  compris ceux d'un dossier parcouru, à ``AccessPolicy.check_read``.

Le serveur n'est pas lancé ici : la CLI (``dhaos serve``) s'en charge ;
``main()`` est un point d'entrée autonome facultatif.
"""
from __future__ import annotations

import contextlib
import hmac
import json
import logging
import queue
import secrets
import sys
import threading
import time
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, Iterator
from urllib.parse import urlsplit

import tomli_w
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from .. import __version__
from ..agent.loop import Agent, AgentResult
from ..agent.session import Session, SessionStore
from ..backends import BACKEND_NAMES, get_backend
from ..backends.base import Backend, BackendError
from ..config import Settings, default_config_path
from ..kb.ingest import iter_files
from ..kb.manager import KnowledgeError, normalize_name
from ..policy import AccessPolicy, Confirmer, Journal, auto_confirm, never_confirm
from ..runtime import build_runtime
from ..tools.base import ToolResult
from ..types import ToolCall
from . import schemas as S

log = logging.getLogger("dhaos.api")

BackendFactory = Callable[[Settings, str | None, str | None], Backend]
KBFactory = Callable[[Settings], Any]

SECRET_KEYS: tuple[str, ...] = ("web.brave_api_key", "api.token")
MASK = "***"
KEEPALIVE_SECONDS = 15.0
MAX_JOURNAL_ENTRIES = 1000
LOCAL_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")

_END = object()  # sentinelle de fin de flux


# ================================================================ utilitaires
def default_backend_factory(settings: Settings, name: str | None, model: str | None) -> Backend:
    """Fabrique par défaut : ``dhaos.backends.get_backend``."""
    return get_backend(settings, name, model=model)


def default_kb_factory(settings: Settings) -> Any:
    """Fabrique par défaut : ``dhaos.kb.manager.KnowledgeManager``."""
    from ..kb.manager import KnowledgeManager

    return KnowledgeManager(settings)


def masked_config(settings: Settings) -> dict[str, Any]:
    """``settings.model_dump(mode="json")`` avec les secrets remplacés par ``***``."""
    data = settings.model_dump(mode="json", exclude={"source_path"})
    for dotted in SECRET_KEYS:
        section, _, key = dotted.partition(".")
        node = data.get(section)
        if isinstance(node, dict) and node.get(key):
            node[key] = MASK
    data["source_path"] = str(settings.source_path or default_config_path())
    data["project_root"] = str(settings.resolve_project_root())
    return data


def sse(event: str, data: Any) -> bytes:
    """Encode un événement Server-Sent Events (``event:`` + ``data:`` JSON)."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _kb_status(exc: KnowledgeError) -> int:
    message = str(exc)
    if "inconnue" in message or "introuvable" in message:
        return status.HTTP_404_NOT_FOUND
    return status.HTTP_400_BAD_REQUEST


@contextlib.contextmanager
def _kb_errors() -> Iterator[None]:
    """Traduit ``KnowledgeError`` en 400/404 JSON."""
    try:
        yield
    except KnowledgeError as e:
        raise HTTPException(_kb_status(e), str(e)) from e


def _key_error_message(exc: KeyError, fallback: str) -> str:
    return str(exc.args[0]) if exc.args else fallback


def _usage_dict(result: AgentResult) -> dict[str, int]:
    return {"input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens}


def _done_payload(result: AgentResult, session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "usage": _usage_dict(result),
        "stop_reason": result.stop_reason,
        "iterations": result.iterations,
        "tool_calls": result.tool_calls,
        "error": result.error,
    }


UI_DIR = Path(__file__).resolve().parent.parent / "ui"
# Modèles Claude proposés dans l'interface (l'API Anthropic n'est pas interrogée).
CLAUDE_MODELS: tuple[str, ...] = (
    "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-fable-5-1", "claude-opus-4-8",
)


def host_name(header: str) -> str:
    """Nom d'hôte d'un en-tête ``Host`` (port et crochets IPv6 retirés, minuscules)."""
    value = str(header or "").strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end > 0 else value
    if value.count(":") == 1:
        value = value.rsplit(":", 1)[0]
    return value


def allowed_hosts_for(settings: Settings) -> set[str]:
    """Hôtes acceptés : ``api.allowed_hosts`` ou, à défaut, les adresses locales
    et ``api.host`` (``*`` désactive le contrôle)."""
    configured = [host_name(h) for h in settings.api.allowed_hosts if str(h).strip()]
    if configured:
        return set(configured)
    return {*LOCAL_HOSTS, host_name(settings.api.host)} - {""}


def normalize_origin(origin: str) -> str:
    """Origine navigateur comparable : minuscules, sans barre oblique finale."""
    return str(origin or "").strip().rstrip("/").lower()


class HostOriginGuard:
    """Middleware ASGI : 400 si l'en-tête ``Host`` n'est pas attendu (DNS
    rebinding, hôte de proxy non prévu), 403 si un en-tête ``Origin`` est
    présent sans figurer dans les origines autorisées (requête navigateur
    inter-sites ou après rebinding). Les autres requêtes passent telles quelles."""

    def __init__(self, app: ASGIApp, *, allowed_hosts: Iterable[str], allowed_origins: Iterable[str]) -> None:
        self.app = app
        self.hosts = {host_name(h) for h in allowed_hosts} - {""}
        self.origins = {normalize_origin(o) for o in allowed_origins} - {""}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = host_name(headers.get("host", ""))
        if "*" not in self.hosts and host not in self.hosts:
            response = JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"en-tête Host non autorisé : {host or '(absent)'} (voir api.allowed_hosts)"},
            )
            await response(scope, receive, send)
            return
        origin = headers.get("origin")
        same_origin = origin is not None and urlsplit(origin).netloc.lower() == headers.get("host", "").strip().lower()
        if (
            origin is not None
            and not same_origin  # l'interface servie par cette API (Host déjà validé)
            and "*" not in self.origins
            and normalize_origin(origin) not in self.origins
        ):
            response = JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "origine navigateur non autorisée (voir api.allowed_origins)"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class ConfirmBroker:
    """Demandes de confirmation en attente d'une réponse de l'interface web.

    ``ask`` (appelé depuis le fil de l'agent) pousse un événement SSE
    ``confirm`` {id, prompt} puis attend ``answer`` (``POST /chat/confirm``)
    jusqu'à ``timeout`` secondes ; sans réponse, l'action est refusée."""

    def __init__(self, timeout: float) -> None:
        self.timeout = max(1.0, float(timeout))
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}

    def ask(self, prompt: str, session_id: str, events: "queue.Queue[Any]") -> bool:
        cid = secrets.token_urlsafe(12)
        entry: dict[str, Any] = {"event": threading.Event(), "answer": None, "session_id": session_id, "created": time.time()}
        with self._lock:
            self._pending[cid] = entry
        events.put(("confirm", {"id": cid, "prompt": str(prompt), "session_id": session_id, "timeout": self.timeout}))
        answered = entry["event"].wait(self.timeout)
        with self._lock:
            self._pending.pop(cid, None)
        return bool(entry["answer"]) if answered else False

    def answer(self, cid: str, value: bool) -> bool:
        with self._lock:
            entry = self._pending.get(cid)
            if entry is None:
                return False
            entry["answer"] = bool(value)
            entry["event"].set()
            return True

    def confirmer(self, session_id: str, events: "queue.Queue[Any]") -> Confirmer:
        return lambda prompt: self.ask(prompt, session_id, events)

    def pending(self) -> int:
        with self._lock:
            return len(self._pending)


class SessionLocks:
    """Une session ne sert qu'un tour à la fois (le fichier JSONL est réécrit
    intégralement par l'agent) ; un second appel concurrent reçoit 409."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._busy: set[str] = set()

    def acquire(self, session_id: str) -> bool:
        with self._guard:
            if session_id in self._busy:
                return False
            self._busy.add(session_id)
            return True

    def release(self, session_id: str) -> None:
        with self._guard:
            self._busy.discard(session_id)

    def is_busy(self, session_id: str) -> bool:
        with self._guard:
            return session_id in self._busy


# ============================================================ flux de chat
def run_agent_to_queue(agent: Agent, session_id: str, message: str, events: "queue.Queue[Any]") -> bool:
    """Exécute ``Agent.run`` (synchrone) en poussant les événements SSE dans
    ``events`` ; termine toujours par la sentinelle de fin. Renvoie ``False``
    si l'agent a levé une exception (événement ``error`` émis)."""

    def put(event: str, data: dict[str, Any]) -> None:
        events.put((event, data))

    def on_tool_result(call: ToolCall, result: ToolResult) -> None:
        put(
            "tool_result",
            {
                "id": call.id,
                "name": call.name,
                "is_error": bool(result.is_error),
                "preview": str(result.content)[: S.TOOL_RESULT_PREVIEW_CHARS],
            },
        )

    try:
        result = agent.run(
            message,
            on_text=lambda text: put("text", {"text": text}),
            on_thinking=lambda text: put("thinking", {"text": text}),
            on_tool_call=lambda call: put(
                "tool_call", {"id": call.id, "name": call.name, "arguments": call.arguments}
            ),
            on_tool_result=on_tool_result,
        )
        put("done", _done_payload(result, session_id))
        return True
    except Exception as e:  # noqa: BLE001 — le flux doit toujours se terminer proprement
        log.exception("échec de l'agent (session %s)", session_id)
        put("error", {"detail": f"{type(e).__name__}: {e}"})
        return False
    finally:
        events.put(_END)


def iter_sse(events: "queue.Queue[Any]") -> Iterator[bytes]:
    """Consomme la file jusqu'à la sentinelle ; commentaire de maintien de
    connexion toutes les ``KEEPALIVE_SECONDS`` sans événement."""
    while True:
        try:
            item = events.get(timeout=KEEPALIVE_SECONDS)
        except queue.Empty:
            yield b": keep-alive\n\n"
            continue
        if item is _END:
            return
        event, data = item
        yield sse(event, data)


# ============================================================== application
def create_app(
    settings: Settings,
    *,
    backend_factory: BackendFactory | None = None,
    kb_factory: KBFactory | None = None,
) -> FastAPI:
    """Construit l'application FastAPI (voir la docstring du module)."""
    make_backend: BackendFactory = backend_factory or default_backend_factory
    make_kb: KBFactory = kb_factory or default_kb_factory
    confirm = auto_confirm if settings.api.auto_confirm else never_confirm
    token_generated = not settings.api.token
    token = settings.api.token or secrets.token_urlsafe(32)
    token_bytes = token.encode("utf-8")

    settings.ensure_dirs()
    kb = make_kb(settings)
    policy = AccessPolicy(settings)
    kb_lock = threading.RLock()
    session_locks = SessionLocks()
    confirm_broker = ConfirmBroker(settings.api.confirm_timeout)

    def close_kb() -> None:
        manager = getattr(app.state, "kb", None)
        if manager is None:
            return
        try:
            manager.close()
        except Exception as e:  # noqa: BLE001
            log.warning("fermeture des bases de savoir : %s", e)
        app.state.kb = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            close_kb()

    app = FastAPI(
        title="dhaos",
        version=__version__,
        description="API HTTP de l'assistant de codage dhaos (bases de savoir locales).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.token = token
    app.state.token_generated = token_generated
    app.state.kb = kb
    app.state.kb_lock = kb_lock
    app.state.session_locks = session_locks
    app.state.close_kb = close_kb
    app.state.confirm_broker = confirm_broker
    app.add_middleware(
        HostOriginGuard,
        allowed_hosts=allowed_hosts_for(settings),
        allowed_origins=list(settings.api.allowed_origins),
    )

    # ----------------------------------------------------------- erreurs
    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("erreur interne de l'API")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"erreur interne : {type(exc).__name__}: {exc}"},
        )

    # ------------------------------------------------------ authentification
    def authenticated(request: Request) -> bool:
        """``True`` si la requête porte le jeton (``Authorization: Bearer``).
        Comparaison en octets : ``compare_digest`` refuse les ``str`` non ASCII
        (un en-tête latin-1 ne doit pas provoquer de 500)."""
        header = request.headers.get("authorization", "")
        scheme, _, value = header.partition(" ")
        presented = value.strip().encode("utf-8", "surrogateescape")
        return scheme.lower() == "bearer" and hmac.compare_digest(presented, token_bytes)

    async def require_token(request: Request) -> None:
        if not authenticated(request):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "jeton d'accès requis ou invalide (en-tête Authorization: Bearer <token>)",
                headers={"WWW-Authenticate": "Bearer"},
            )

    router = APIRouter(dependencies=[Depends(require_token)])

    # ------------------------------------------------------------ helpers
    def current_kb() -> Any:
        manager = app.state.kb
        if manager is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "bases de savoir fermées")
        return manager

    def require_base(name: str) -> Any:
        """``BaseInfo`` de la base ; 400 si le nom est invalide, 404 si absente."""
        with _kb_errors():
            key = normalize_name(name)
        info = current_kb().get_base(key)
        if info is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"base inconnue : {key}")
        return info

    def load_session(store: SessionStore, session_id: str) -> Session:
        try:
            store.validate_id(session_id)
        except KeyError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, _key_error_message(e, "identifiant invalide")) from e
        try:
            return store.get(session_id)
        except KeyError as e:
            raise HTTPException(status.HTTP_404_NOT_FOUND, _key_error_message(e, "session introuvable")) from e

    def build_backend(name: str | None, model: str | None) -> Backend:
        try:
            return make_backend(settings, name, model)
        except (ValueError, BackendError) as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"backend indisponible : {e}") from e

    # ------------------------------------------------------ interface web
    if UI_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

        @app.get("/", include_in_schema=False)
        def ui_index() -> FileResponse:
            return FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------- health
    @app.get("/health", response_model=S.HealthOut)
    def health(request: Request) -> S.HealthOut:
        """État du service, du backend par défaut et des bases de savoir.
        Sans jeton : état minimal (ni détail du backend, ni appel sortant)."""
        backend_name = settings.backends.default
        if not authenticated(request):
            return S.HealthOut(
                status="ok",
                version=__version__,
                backend=S.BackendHealth(name=str(backend_name), model="", health={}),
                kb=S.KBHealth(),
            )
        model = getattr(getattr(settings.backends, backend_name, None), "model", "")
        backend_health: dict[str, Any]
        try:
            backend = make_backend(settings, None, None)
            backend_name, model = backend.name, backend.model
            try:
                backend_health = dict(backend.healthcheck())
            finally:
                closer = getattr(backend, "close", None)
                if callable(closer):
                    with contextlib.suppress(Exception):
                        closer()
        except Exception as e:  # noqa: BLE001 — le healthcheck ne fait jamais échouer la route
            backend_health = {"ok": False, "backend": backend_name, "model": model, "detail": str(e), "models": []}
        kb_health = S.KBHealth()
        manager = app.state.kb
        if manager is not None:
            try:
                stats = manager.stats()
                kb_health = S.KBHealth(
                    bases=int(stats.get("bases", 0)),
                    documents=int(stats.get("documents", 0)),
                    chunks=int(stats.get("chunks", 0)),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("statistiques des bases indisponibles : %s", e)
        return S.HealthOut(
            status="ok",
            version=__version__,
            backend=S.BackendHealth(name=str(backend_name), model=str(model or ""), health=backend_health),
            kb=kb_health,
        )

    # ------------------------------------------------------------- config
    @router.get("/models", response_model=S.ModelsOut)
    def models(backend: str | None = Query(None, max_length=S.MAX_BACKEND_NAME_CHARS)) -> S.ModelsOut:
        """Modèles proposés pour un backend : liste Ollama (``/api/tags``) ou
        modèles Claude connus ; ``default`` = modèle configuré."""
        name = (backend or settings.backends.default).lower()
        if name not in BACKEND_NAMES:
            return S.ModelsOut(backend=name, ok=False, detail=f"backend inconnu : {name} (attendu : {', '.join(BACKEND_NAMES)})")
        if name == "claude":
            cfg = settings.backends.claude
            return S.ModelsOut(backend="claude", default=cfg.model, ok=True, models=list(CLAUDE_MODELS))
        try:
            instance = build_backend(name, None)
            health = instance.healthcheck()
        except HTTPException as e:
            return S.ModelsOut(backend=name, ok=False, detail=str(e.detail))
        except Exception as e:  # noqa: BLE001
            return S.ModelsOut(backend=name, ok=False, detail=f"{type(e).__name__}: {e}")
        return S.ModelsOut(
            backend=name,
            default=str(getattr(instance, "model", "") or ""),
            ok=bool(health.get("ok")),
            detail=str(health.get("detail") or ""),
            models=[str(m) for m in (health.get("models") or [])],
        )

    @router.patch("/config")
    def config_patch(body: S.ConfigPatch) -> dict[str, Any]:
        """Modifie une clé courante (liste blanche), l'enregistre dans le
        fichier TOML et l'applique à chaud."""
        if body.key not in S.CONFIG_PATCH_KEYS:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"clé non modifiable par l'API : {body.key}")
        try:
            updated = settings.with_override(body.key, body.value)
        except ValidationError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"valeur invalide pour {body.key} : {e.errors()[0].get('msg', e)}") from e
        parts = body.key.split(".")
        new_value = updated
        for part in parts:
            new_value = getattr(new_value, part)
        # Fichier : on repart du TOML brut (sans les surcharges d'environnement).
        path = settings.source_path or default_config_path()
        data: dict[str, Any] = {}
        if path.is_file():
            with contextlib.suppress(tomllib.TOMLDecodeError, OSError):
                data = tomllib.loads(path.read_text(encoding="utf-8"))
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        serialized = new_value.model_dump(mode="json") if hasattr(new_value, "model_dump") else new_value
        if isinstance(serialized, Path):
            serialized = str(serialized)
        if serialized is None:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = serialized
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(tomli_w.dumps(data), encoding="utf-8")
        except OSError as e:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"écriture de la configuration impossible : {e}") from e
        # Application à chaud : on modifie les objets existants (partagés par la politique, les backends…).
        target: Any = settings
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], new_value)
        return {"key": body.key, "value": serialized, "path": str(path)}

    @router.get("/config")
    def config() -> dict[str, Any]:
        """Configuration effective, secrets masqués."""
        return masked_config(settings)

    # --------------------------------------------------------------- kb
    @router.get("/kb", response_model=list[S.BaseInfoOut])
    def kb_list() -> list[S.BaseInfoOut]:
        with kb_lock, _kb_errors():
            return [S.BaseInfoOut.model_validate(b) for b in current_kb().list_bases()]

    @router.post("/kb", response_model=S.BaseInfoOut, status_code=status.HTTP_201_CREATED)
    def kb_create(body: S.BaseCreate) -> S.BaseInfoOut:
        with kb_lock, _kb_errors():
            return S.BaseInfoOut.model_validate(current_kb().create_base(body.name, body.description))

    @router.post("/kb/search", response_model=list[S.HitOut])
    def kb_search(body: S.SearchRequest) -> list[S.HitOut]:
        bases = [b for b in (" ".join(str(x).split()) for x in body.bases or []) if b] or None
        with kb_lock, _kb_errors():
            hits = current_kb().search(body.query, bases=bases, top_k=body.top_k, mode=body.mode)
        return [S.HitOut.model_validate(h) for h in hits]

    @router.get("/kb/stats")
    def kb_stats() -> dict[str, Any]:
        with kb_lock, _kb_errors():
            return dict(current_kb().stats())

    @router.get("/kb/{name}", response_model=S.BaseDetail)
    def kb_get(name: str) -> S.BaseDetail:
        with kb_lock, _kb_errors():
            info = require_base(name)
            docs = current_kb().list_documents(info.name)
        return S.BaseDetail(
            **S.BaseInfoOut.model_validate(info).model_dump(),
            documents=[S.DocInfoOut.model_validate(d) for d in docs],
        )

    @router.patch("/kb/{name}", response_model=S.BaseInfoOut)
    def kb_update(name: str, body: S.BaseUpdate) -> S.BaseInfoOut:
        with kb_lock, _kb_errors():
            info = require_base(name)
            manager = current_kb()
            if body.description is not None:
                info = manager.set_description(info.name, body.description)
            if body.new_name is not None:
                info = manager.rename_base(info.name, body.new_name)
        return S.BaseInfoOut.model_validate(info)

    @router.delete("/kb/{name}", status_code=status.HTTP_204_NO_CONTENT)
    def kb_delete(name: str) -> Response:
        with kb_lock, _kb_errors():
            info = require_base(name)
            current_kb().delete_base(info.name)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.get("/kb/{name}/stats")
    def kb_base_stats(name: str) -> dict[str, Any]:
        with kb_lock, _kb_errors():
            info = require_base(name)
            return dict(current_kb().stats(info.name))

    @router.post("/kb/{name}/documents", response_model=S.IngestReportOut)
    def kb_add_documents(name: str, body: S.DocumentsAdd) -> S.IngestReportOut:
        """Ingère fichiers, dossiers et URLs ; les chemins refusés par la
        politique d'accès sont comptés en échec (jamais lus). Un dossier est
        parcouru ici (``iter_files``) pour soumettre chacun de ses fichiers à
        ``check_read`` : les motifs ``tools.deny_patterns`` s'appliquent aussi
        au contenu des dossiers, pas seulement à la source de premier niveau."""
        allowed: list[str] = []
        denied: list[str] = []

        def screen(candidate: str | Path) -> bool:
            decision = policy.check_read(candidate)
            if decision.allowed:
                return True
            denied.append(f"{candidate} : {decision.reason or 'lecture refusée'}")
            return False

        for raw in body.sources:
            source = str(raw).strip()
            if not source:
                continue
            if source.startswith(("http://", "https://")):
                allowed.append(source)
                continue
            if not screen(source):
                continue
            path = Path(source).expanduser()
            try:
                path = path.resolve()
            except OSError:
                path = path.absolute()
            if path.is_dir():
                allowed.extend(str(f) for f in iter_files(path, body.recursive, settings) if screen(f))
            else:
                allowed.append(source)
        with kb_lock, _kb_errors():
            info = require_base(name)
            manager = current_kb()
            if allowed:
                report = manager.add(info.name, allowed, recursive=body.recursive)
            else:
                from ..kb.manager import IngestReport

                report = IngestReport(base=info.name)
        report.failed += len(denied)
        report.errors.extend(denied)
        return S.IngestReportOut(
            base=report.base,
            added=report.added,
            updated=report.updated,
            skipped=report.skipped,
            failed=report.failed,
            chunks=report.chunks,
            errors=list(report.errors),
            summary=report.summary(),
        )

    @router.post("/kb/{name}/reindex", response_model=S.ReindexOut)
    def kb_reindex(name: str) -> S.ReindexOut:
        info = require_base(name)
        with kb_lock, _kb_errors():
            return S.ReindexOut(chunks=int(current_kb().reindex(info.name)))

    @router.post("/kb/{name}/notes", response_model=S.NoteOut)
    def kb_add_note(name: str, body: S.NoteAdd) -> S.NoteOut:
        title = " ".join(body.title.split()) if body.title else None
        with kb_lock, _kb_errors():
            info = require_base(name)
            doc_id = current_kb().add_text(info.name, body.text, title=title or None)
        return S.NoteOut(doc_id=int(doc_id))

    @router.delete("/kb/{name}/documents", response_model=S.RemovedOut)
    def kb_remove_document(
        name: str,
        body: S.DocumentRemove | None = None,
        source: str | None = Query(None, min_length=1, max_length=S.MAX_SOURCE_CHARS),
    ) -> S.RemovedOut:
        """Supprime un document par sa source (corps JSON ``{source}`` ou
        paramètre ``?source=``)."""
        target = body.source if body is not None else source
        if not target or not target.strip():
            raise HTTPException(422, "source manquante")
        with kb_lock, _kb_errors():
            info = require_base(name)
            removed = current_kb().remove_document(info.name, target.strip())
        return S.RemovedOut(removed=bool(removed))

    # --------------------------------------------------------------- chat
    @router.post("/chat")
    def chat(body: S.ChatRequest) -> Any:
        """Un tour d'agent : JSON complet (``stream=false``) ou flux SSE."""
        backend = build_backend(body.backend, body.model)
        store = SessionStore(settings)
        created = body.session_id is None
        if created:
            session = store.create(backend=backend.name, model=backend.model)
        else:
            session = load_session(store, body.session_id)

        def release(*, failed: bool = False) -> None:
            """Libère la session ; après un échec, une session créée par cet
            appel et restée vide est supprimée (aucun fichier orphelin)."""
            session_locks.release(session.id)
            if failed and created and not session.messages:
                with contextlib.suppress(Exception):
                    store.delete(session.id)

        if not session_locks.acquire(session.id):
            raise HTTPException(status.HTTP_409_CONFLICT, f"session occupée : {session.id}")
        events: "queue.Queue[Any]" = queue.Queue()
        # En flux, l'interface peut répondre aux confirmations ; sinon la
        # politique configurée s'applique (auto_confirm ou refus).
        turn_confirm = confirm if (settings.api.auto_confirm or not body.stream) else confirm_broker.confirmer(session.id, events)
        try:
            runtime = build_runtime(
                settings,
                backend=backend,
                confirm=turn_confirm,
                session=session,
                kb=app.state.kb,
                tools=not body.no_tools,
            )
        except Exception:
            release(failed=True)
            raise

        if not body.stream:
            failed = True
            try:
                result = runtime.agent.run(body.message)
                failed = False
            except Exception as e:  # noqa: BLE001
                log.exception("échec de l'agent (session %s)", session.id)
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"échec de l'agent : {e}") from e
            finally:
                release(failed=failed)
            return S.ChatOut(
                text=result.text,
                session_id=session.id,
                usage=S.UsageOut(**_usage_dict(result)),
                stop_reason=result.stop_reason,
                iterations=result.iterations,
                tool_calls=result.tool_calls,
                error=result.error,
            )

        def worker() -> None:
            failed = True
            try:
                failed = not run_agent_to_queue(runtime.agent, session.id, body.message, events)
            finally:
                release(failed=failed)

        threading.Thread(target=worker, name=f"dhaos-chat-{session.id}", daemon=True).start()
        return StreamingResponse(
            iter_sse(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Session-Id": session.id},
        )

    @router.post("/chat/confirm")
    def chat_confirm(body: S.ConfirmAnswer) -> dict[str, Any]:
        """Réponse à une demande de confirmation reçue en SSE (``event: confirm``)."""
        if not confirm_broker.answer(body.id, body.answer):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "demande de confirmation inconnue ou expirée")
        return {"accepted": True, "id": body.id, "answer": body.answer}

    # ------------------------------------------------------------ sessions
    @router.get("/sessions", response_model=list[S.SessionInfoOut])
    def sessions_list() -> list[S.SessionInfoOut]:
        store = SessionStore(settings)
        return [
            S.SessionInfoOut(
                id=i.id,
                title=i.title,
                backend=i.backend,
                model=i.model,
                created_at=i.created_at,
                updated_at=i.updated_at,
                n_messages=i.n_messages,
            )
            for i in store.list()
        ]

    @router.get("/sessions/{session_id}", response_model=S.SessionDetail)
    def sessions_get(session_id: str) -> S.SessionDetail:
        session = load_session(SessionStore(settings), session_id)
        return S.SessionDetail(
            id=session.id,
            meta=dict(session.meta),
            messages=[m.to_dict() for m in session.messages],
            events=list(session.events),
        )

    @router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def sessions_delete(session_id: str) -> Response:
        store = SessionStore(settings)
        try:
            removed = store.delete(session_id)
        except KeyError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, _key_error_message(e, "identifiant invalide")) from e
        if not removed:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"session introuvable : {session_id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # ------------------------------------------------------------- journal
    @router.get("/journal")
    def journal(n: int = Query(50, ge=1, le=MAX_JOURNAL_ENTRIES)) -> list[dict[str, Any]]:
        """Dernières entrées du journal des actions (écritures, commandes)."""
        return Journal(settings.journal_path).tail(n)

    app.include_router(router)
    return app


def main() -> None:
    """Point d'entrée autonome (``python -m dhaos.api.server``) : uvicorn sur
    ``api.host:api.port``. La CLI ``dhaos serve`` reste la voie normale."""
    import uvicorn

    settings = Settings.load()
    application = create_app(settings)
    if application.state.token_generated:
        print(f"jeton d'accès (api.token absent) : {application.state.token}", file=sys.stderr)
    uvicorn.run(application, host=settings.api.host, port=settings.api.port)


__all__ = [
    "BackendFactory",
    "ConfirmBroker",
    "HostOriginGuard",
    "KBFactory",
    "LOCAL_HOSTS",
    "MASK",
    "SECRET_KEYS",
    "SessionLocks",
    "allowed_hosts_for",
    "create_app",
    "host_name",
    "normalize_origin",
    "default_backend_factory",
    "default_kb_factory",
    "iter_sse",
    "main",
    "masked_config",
    "run_agent_to_queue",
    "sse",
]


if __name__ == "__main__":
    main()
