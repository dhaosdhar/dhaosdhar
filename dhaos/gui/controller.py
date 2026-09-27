"""Contrôleur de l'interface de bureau — aucune dépendance à Tk.

Chaque conversation possède sa session, son agent et tourne dans son propre
fil ; la vue consomme une file d'événements (texte, appels d'outils,
confirmations, fin de tour) depuis le fil principal. Les demandes de
confirmation bloquent le fil de la conversation jusqu'à la réponse de
l'utilisateur (ou l'expiration).
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..agent.session import Session, SessionInfo, SessionStore
from ..backends import BACKEND_NAMES, get_backend
from ..backends.base import Backend
from ..config import Settings
from ..runtime import Runtime, build_runtime
from ..tools.base import ToolResult
from ..types import ToolCall

BackendFactory = Callable[[Settings, str | None, str | None], Backend]
TOOL_PREVIEW_CHARS = 800


@dataclass
class Event:
    """Un événement pour la vue : ``kind`` ∈ text, thinking, tool_call,
    tool_result, confirm, done, error, status."""

    conv_id: str
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfirmRequest:
    id: str
    conv_id: str
    prompt: str
    event: threading.Event = field(default_factory=threading.Event)
    answer: bool | None = None


class Conversation:
    """Une conversation ouverte dans un onglet."""

    def __init__(
        self,
        conv_id: str,
        session: Session,
        *,
        backend_name: str | None,
        model: str | None,
        use_tools: bool,
    ) -> None:
        self.id = conv_id
        self.session = session
        self.backend_name = backend_name
        self.model = model
        self.use_tools = use_tools
        self.runtime: Runtime | None = None
        self.busy = False
        self.thread: threading.Thread | None = None
        self.created = time.time()

    @property
    def title(self) -> str:
        meta_title = str(self.session.meta.get("title") or "").strip()
        if meta_title:
            return meta_title
        for m in self.session.messages:
            if m.role == "user" and m.content.strip():
                return m.content.strip().splitlines()[0][:60]
        return "Nouvelle conversation"


class GuiController:
    def __init__(
        self,
        settings: Settings,
        *,
        backend_factory: BackendFactory | None = None,
        kb_factory: Callable[[Settings], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.settings.ensure_dirs()
        self.events: "queue.Queue[Event]" = queue.Queue()
        self.conversations: dict[str, Conversation] = {}
        self.store = SessionStore(settings)
        self._backend_factory = backend_factory
        self._kb_factory = kb_factory
        self._kb: Any = None
        self._kb_lock = threading.Lock()
        self._pending: dict[str, ConfirmRequest] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ ressources
    @property
    def kb(self) -> Any:
        """Gestionnaire de bases de savoir partagé (SQLite thread-safe), créé à la demande."""
        with self._kb_lock:
            if self._kb is None:
                if self._kb_factory is not None:
                    self._kb = self._kb_factory(self.settings)
                else:
                    from ..kb.manager import KnowledgeManager

                    self._kb = KnowledgeManager(self.settings)
            return self._kb

    def _make_backend(self, name: str | None, model: str | None) -> Backend:
        if self._backend_factory is not None:
            return self._backend_factory(self.settings, name, model)
        return get_backend(self.settings, name, model=model)

    def health(self, backend_name: str | None = None) -> dict[str, Any]:
        try:
            return self._make_backend(backend_name, None).healthcheck()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "backend": backend_name or self.settings.backends.default, "model": "", "detail": str(e), "models": []}

    def models(self, backend_name: str | None = None) -> list[str]:
        name = (backend_name or self.settings.backends.default).lower()
        if name == "claude":
            return ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-fable-5-1", "claude-opus-4-8"]
        health = self.health(name)
        return [str(m) for m in (health.get("models") or [])]

    def default_model(self, backend_name: str | None = None) -> str:
        name = (backend_name or self.settings.backends.default).lower()
        return self.settings.backends.claude.model if name == "claude" else self.settings.backends.ollama.model

    # ---------------------------------------------------------- conversations
    def new_conversation(
        self, *, backend: str | None = None, model: str | None = None, tools: bool = True
    ) -> Conversation:
        backend_name = (backend or self.settings.backends.default).lower()
        if backend_name not in BACKEND_NAMES:
            raise ValueError(f"backend inconnu : {backend_name}")
        session = self.store.create(backend=backend_name, model=model or self.default_model(backend_name))
        conv = Conversation(uuid.uuid4().hex[:8], session, backend_name=backend_name, model=model, use_tools=tools)
        with self._lock:
            self.conversations[conv.id] = conv
        return conv

    def open_session(self, session_id: str, *, tools: bool = True) -> Conversation:
        """Reprend une session existante dans une nouvelle conversation (le fil
        complet est disponible dans ``conversation.session.messages``)."""
        session = self.store.get(session_id)
        backend_name = str(session.meta.get("backend") or self.settings.backends.default).lower()
        if backend_name not in BACKEND_NAMES:
            backend_name = self.settings.backends.default
        model = str(session.meta.get("model") or "") or None
        conv = Conversation(uuid.uuid4().hex[:8], session, backend_name=backend_name, model=model, use_tools=tools)
        with self._lock:
            self.conversations[conv.id] = conv
        return conv

    def get(self, conv_id: str) -> Conversation | None:
        return self.conversations.get(conv_id)

    def close_conversation(self, conv_id: str) -> None:
        with self._lock:
            conv = self.conversations.pop(conv_id, None)
        if conv is None:
            return
        if conv.runtime is not None:
            try:
                conv.runtime.registry  # noqa: B018 — le runtime partage la base : ne pas la fermer ici
            except Exception:  # noqa: BLE001
                pass
        # Une conversation jamais utilisée ne laisse pas de fichier vide.
        if not conv.session.messages and not conv.busy:
            try:
                self.store.delete(conv.session.id)
            except Exception:  # noqa: BLE001
                pass

    def list_sessions(self) -> list[SessionInfo]:
        return self.store.list()

    def delete_session(self, session_id: str) -> bool:
        return self.store.delete(session_id)

    # ---------------------------------------------------------------- envoi
    def send(self, conv_id: str, text: str) -> bool:
        """Lance un tour d'agent dans le fil de la conversation ; ``False`` si
        elle est occupée ou si le texte est vide."""
        conv = self.conversations.get(conv_id)
        if conv is None or conv.busy or not text.strip():
            return False
        conv.busy = True
        thread = threading.Thread(target=self._run_turn, args=(conv, text), name=f"dhaos-gui-{conv.id}", daemon=True)
        conv.thread = thread
        thread.start()
        return True

    def _emit(self, conv: Conversation, kind: str, **data: Any) -> None:
        self.events.put(Event(conv.id, kind, data))

    def _confirmer(self, conv: Conversation) -> Callable[[str], bool]:
        def confirm(prompt: str) -> bool:
            request = ConfirmRequest(uuid.uuid4().hex[:10], conv.id, str(prompt))
            with self._lock:
                self._pending[request.id] = request
            self._emit(conv, "confirm", id=request.id, prompt=request.prompt)
            answered = request.event.wait(self.settings.api.confirm_timeout)
            with self._lock:
                self._pending.pop(request.id, None)
            return bool(request.answer) if answered else False

        return confirm

    def answer_confirm(self, request_id: str, answer: bool) -> bool:
        with self._lock:
            request = self._pending.get(request_id)
            if request is None:
                return False
            request.answer = bool(answer)
            request.event.set()
            return True

    def pending_confirmations(self) -> int:
        with self._lock:
            return len(self._pending)

    def _ensure_runtime(self, conv: Conversation) -> Runtime:
        if conv.runtime is None:
            backend = self._make_backend(conv.backend_name, conv.model)
            conv.runtime = build_runtime(
                self.settings,
                backend=backend,
                confirm=self._confirmer(conv),
                session=conv.session,
                kb=self.kb,
                tools=conv.use_tools,
            )
        return conv.runtime

    def _run_turn(self, conv: Conversation, text: str) -> None:
        self._emit(conv, "status", label="le modèle lit le contexte et génère…")
        try:
            runtime = self._ensure_runtime(conv)

            def on_tool_result(call: ToolCall, result: ToolResult) -> None:
                self._emit(
                    conv,
                    "tool_result",
                    id=call.id,
                    name=call.name,
                    is_error=bool(result.is_error),
                    preview=str(result.content)[:TOOL_PREVIEW_CHARS],
                )
                self._emit(conv, "status", label="le modèle poursuit…")

            result = runtime.agent.run(
                text,
                on_text=lambda chunk: self._emit(conv, "text", text=chunk),
                on_thinking=lambda chunk: self._emit(conv, "thinking", text=chunk),
                on_tool_call=lambda call: (
                    self._emit(conv, "tool_call", id=call.id, name=call.name, arguments=call.arguments),
                    self._emit(conv, "status", label=f"exécution de {call.name}…"),
                ),
                on_tool_result=on_tool_result,
            )
            self._emit(
                conv,
                "done",
                text=result.text,
                usage={"input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens},
                stop_reason=result.stop_reason,
                iterations=result.iterations,
                tool_calls=result.tool_calls,
                error=result.error,
                session_id=conv.session.id,
                title=conv.title,
            )
        except Exception as e:  # noqa: BLE001 — la vue affiche l'erreur, le fil se termine proprement
            self._emit(conv, "error", detail=f"{type(e).__name__}: {e}")
        finally:
            conv.busy = False

    # -------------------------------------------------------------- divers
    def drain(self, *, timeout: float = 0.0) -> list[Event]:
        """Événements disponibles (la vue l'appelle depuis son minuteur)."""
        out: list[Event] = []
        deadline = time.monotonic() + timeout
        while True:
            try:
                remaining = max(0.0, deadline - time.monotonic())
                out.append(self.events.get(timeout=remaining) if timeout else self.events.get_nowait())
            except queue.Empty:
                return out
            if not timeout:
                continue
            if time.monotonic() >= deadline:
                return out

    def wait_idle(self, conv_id: str, timeout: float = 30.0) -> bool:
        conv = self.conversations.get(conv_id)
        if conv is None or conv.thread is None:
            return True
        conv.thread.join(timeout)
        return not conv.thread.is_alive()

    def shutdown(self) -> None:
        with self._kb_lock:
            if self._kb is not None:
                try:
                    self._kb.close()
                except Exception:  # noqa: BLE001
                    pass
                self._kb = None
