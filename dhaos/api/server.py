"""API HTTP de dhaos (FastAPI).

``create_app(settings, *, backend_factory=None, kb_factory=None)`` construit
l'application :

- les bases de savoir sont ouvertes une fois (``app.state.kb``) et fermées à
  l'arrêt ;
- chaque ``POST /chat`` assemble une exécution complète via
  ``dhaos.runtime.build_runtime`` (backend, outils, session persistée) ;
- la confirmation des actions sensibles suit ``api.auto_confirm`` (sinon
  elles sont refusées : l'API n'a personne à qui demander) ;
- si ``api.token`` est défini, toutes les routes sauf ``/health`` exigent
  ``Authorization: Bearer <token>``.

Le serveur n'est pas lancé ici : la CLI (``dhaos serve``) s'en charge ;
``main()`` est un point d'entrée autonome facultatif.
"""
from __future__ import annotations

import contextlib
import hmac
import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Iterator

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from .. import __version__
from ..agent.loop import Agent, AgentResult
from ..agent.session import Session, SessionStore
from ..backends import get_backend
from ..backends.base import Backend, BackendError
from ..config import Settings
from ..kb.manager import KnowledgeError, normalize_name
from ..policy import AccessPolicy, Journal, auto_confirm, never_confirm
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
def run_agent_to_queue(agent: Agent, session_id: str, message: str, events: "queue.Queue[Any]") -> None:
    """Exécute ``Agent.run`` (synchrone) en poussant les événements SSE dans
    ``events`` ; termine toujours par la sentinelle de fin."""

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
    except Exception as e:  # noqa: BLE001 — le flux doit toujours se terminer proprement
        log.exception("échec de l'agent (session %s)", session_id)
        put("error", {"detail": f"{type(e).__name__}: {e}"})
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
    token = settings.api.token or None

    settings.ensure_dirs()
    kb = make_kb(settings)
    policy = AccessPolicy(settings)
    kb_lock = threading.RLock()
    session_locks = SessionLocks()

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
    app.state.kb = kb
    app.state.kb_lock = kb_lock
    app.state.session_locks = session_locks
    app.state.close_kb = close_kb

    # ----------------------------------------------------------- erreurs
    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("erreur interne de l'API")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"erreur interne : {type(exc).__name__}: {exc}"},
        )

    # ------------------------------------------------------ authentification
    async def require_token(request: Request) -> None:
        if token is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, value = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(value.strip(), token):
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

    # ------------------------------------------------------------- health
    @app.get("/health", response_model=S.HealthOut)
    def health() -> S.HealthOut:
        """État du service, du backend par défaut et des bases de savoir."""
        backend_name = settings.backends.default
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
        politique d'accès sont comptés en échec (jamais lus)."""
        allowed: list[str] = []
        denied: list[str] = []
        for raw in body.sources:
            source = str(raw).strip()
            if not source:
                continue
            if source.startswith(("http://", "https://")):
                allowed.append(source)
                continue
            decision = policy.check_read(source)
            if decision.allowed:
                allowed.append(source)
            else:
                denied.append(f"{source} : {decision.reason or 'lecture refusée'}")
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
        if body.session_id is None:
            session = store.create(backend=backend.name, model=backend.model)
        else:
            session = load_session(store, body.session_id)

        if not session_locks.acquire(session.id):
            raise HTTPException(status.HTTP_409_CONFLICT, f"session occupée : {session.id}")
        try:
            runtime = build_runtime(
                settings,
                backend=backend,
                confirm=confirm,
                session=session,
                kb=app.state.kb,
                tools=not body.no_tools,
            )
        except Exception:
            session_locks.release(session.id)
            raise

        if not body.stream:
            try:
                result = runtime.agent.run(body.message)
            except Exception as e:  # noqa: BLE001
                log.exception("échec de l'agent (session %s)", session.id)
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"échec de l'agent : {e}") from e
            finally:
                session_locks.release(session.id)
            return S.ChatOut(
                text=result.text,
                session_id=session.id,
                usage=S.UsageOut(**_usage_dict(result)),
                stop_reason=result.stop_reason,
                iterations=result.iterations,
                tool_calls=result.tool_calls,
                error=result.error,
            )

        events: "queue.Queue[Any]" = queue.Queue()

        def worker() -> None:
            try:
                run_agent_to_queue(runtime.agent, session.id, body.message, events)
            finally:
                session_locks.release(session.id)

        threading.Thread(target=worker, name=f"dhaos-chat-{session.id}", daemon=True).start()
        return StreamingResponse(
            iter_sse(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Session-Id": session.id},
        )

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
    uvicorn.run(create_app(settings), host=settings.api.host, port=settings.api.port)


__all__ = [
    "BackendFactory",
    "KBFactory",
    "MASK",
    "SECRET_KEYS",
    "SessionLocks",
    "create_app",
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
