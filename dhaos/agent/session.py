"""Sessions de conversation persistées.

Une session = un fichier JSONL dans ``settings.sessions_dir`` :
première ligne ``{"type": "meta", ...}`` (id, titre, backend, modèle, dates),
puis une ligne ``{"type": "message", ...}`` par ``Message.to_dict()``, et des
lignes ``{"type": "event", ...}`` optionnelles (usage, outils) pour les traces
d'entraînement.

Le fichier est réécrit intégralement et de façon atomique à chaque
``save()`` (fichier temporaire puis ``os.replace``). Les lignes invalides sont
ignorées à la relecture.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..types import Message

# Identifiants acceptés par le magasin : pas de séparateur de chemin, pas de
# ``..`` — évite toute traversée de répertoire depuis une entrée non fiable.
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SESSION_SUFFIX = ".jsonl"
_VALID_ROLES = ("user", "assistant", "tool")


def now_iso() -> str:
    """Horodatage ISO 8601 local, à la seconde (même format que le journal)."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _json_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


@dataclass
class SessionInfo:
    id: str
    title: str
    backend: str
    model: str
    created_at: str
    updated_at: str
    n_messages: int
    path: Path


@dataclass
class Session:
    id: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def append(self, message: Message) -> None:
        """Ajoute un message à l'historique (en mémoire ; ``save()`` persiste)."""
        self.messages.append(message)

    def log_event(self, kind: str, **fields: Any) -> None:
        """Enregistre un événement (usage, outil…) en mémoire, horodaté."""
        event: dict[str, Any] = {"kind": str(kind), "at": now_iso()}
        for key, value in fields.items():
            if key not in ("type", "kind", "at"):
                event[key] = value
        self.events.append(event)

    # ------------------------------------------------------------ persistance
    def _lines(self) -> list[str]:
        meta = {"id": self.id, **{k: v for k, v in self.meta.items() if k != "type"}}
        lines = [_json_line({"type": "meta", **meta})]
        lines.extend(_json_line({"type": "message", **m.to_dict()}) for m in self.messages)
        lines.extend(_json_line({"type": "event", **ev}) for ev in self.events)
        return lines

    def save(self) -> None:
        """Écrit tout le fichier JSONL de façon atomique (meta, messages, événements)."""
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self._lines()) + "\n"
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.stem}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: Path) -> "Session":
        """Relit un fichier JSONL ; les lignes invalides ou inconnues sont ignorées.

        Lève ``FileNotFoundError`` si le fichier n'existe pas.
        """
        path = Path(path)
        meta: dict[str, Any] = {}
        messages: list[Message] = []
        events: list[dict[str, Any]] = []
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            kind = obj.get("type")
            body = {k: v for k, v in obj.items() if k != "type"}
            if kind == "meta":
                meta.update(body)
            elif kind == "message":
                if body.get("role") not in _VALID_ROLES:
                    continue
                try:
                    messages.append(Message.from_dict(body))
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
            elif kind == "event":
                events.append(body)
        session_id = str(meta.get("id") or path.stem)
        meta["id"] = session_id
        return cls(id=session_id, path=path, meta=meta, messages=messages, events=events)


def _scan(path: Path) -> tuple[dict[str, Any], int]:
    """Lecture légère : meta + nombre de messages (sans construire les Message)."""
    meta: dict[str, Any] = {}
    n_messages = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            kind = obj.get("type")
            if kind == "meta":
                meta.update({k: v for k, v in obj.items() if k != "type"})
            elif kind == "message":
                n_messages += 1
    return meta, n_messages


class SessionStore:
    """Magasin de sessions sur disque (``settings.sessions_dir``)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.directory = Path(settings.sessions_dir)
        self.directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ identifiants
    @staticmethod
    def validate_id(session_id: str) -> str:
        """Renvoie l'identifiant s'il est sûr, sinon lève ``KeyError``."""
        sid = str(session_id or "").strip()
        if not sid or len(sid) > 128 or not SESSION_ID_RE.match(sid):
            raise KeyError(f"identifiant de session invalide : {session_id!r}")
        return sid

    def path_for(self, session_id: str) -> Path:
        return self.directory / f"{self.validate_id(session_id)}{SESSION_SUFFIX}"

    def new_id(self) -> str:
        """Horodatage compact + 4 hex aléatoires, ex. ``20260926-153012-a1b2``."""
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        for _ in range(64):
            sid = f"{stamp}-{secrets.token_hex(2)}"
            if not (self.directory / f"{sid}{SESSION_SUFFIX}").exists():
                return sid
        raise RuntimeError("impossible de générer un identifiant de session unique")

    # ------------------------------------------------------------------- CRUD
    def create(self, *, title: str = "", backend: str = "", model: str = "") -> Session:
        """Crée une session et écrit son fichier immédiatement."""
        sid = self.new_id()
        now = now_iso()
        meta: dict[str, Any] = {
            "id": sid,
            "title": str(title or ""),
            "backend": str(backend or ""),
            "model": str(model or ""),
            "created_at": now,
            "updated_at": now,
            "traces": bool(self.settings.agent.collect_traces),
        }
        session = Session(id=sid, path=self.path_for(sid), meta=meta)
        session.save()
        return session

    def get(self, session_id: str) -> Session:
        """Charge une session ; ``KeyError`` si l'identifiant est invalide ou absent."""
        path = self.path_for(session_id)
        if not path.is_file():
            raise KeyError(f"session introuvable : {session_id}")
        try:
            return Session.load(path)
        except OSError as e:
            raise KeyError(f"session illisible : {session_id} ({e})") from e

    def list(self) -> list[SessionInfo]:
        """Sessions du plus récent au plus ancien (fichiers illisibles ignorés)."""
        infos: list[SessionInfo] = []
        for path in self.directory.glob(f"*{SESSION_SUFFIX}"):
            if not path.is_file() or not SESSION_ID_RE.match(path.stem):
                continue
            try:
                meta, n_messages = _scan(path)
            except OSError:
                continue
            fallback = _dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
                timespec="seconds"
            )
            created = str(meta.get("created_at") or fallback)
            infos.append(
                SessionInfo(
                    id=str(meta.get("id") or path.stem),
                    title=str(meta.get("title") or ""),
                    backend=str(meta.get("backend") or ""),
                    model=str(meta.get("model") or ""),
                    created_at=created,
                    updated_at=str(meta.get("updated_at") or created),
                    n_messages=n_messages,
                    path=path,
                )
            )
        infos.sort(key=lambda i: (i.updated_at, i.created_at, i.id), reverse=True)
        return infos

    def delete(self, session_id: str) -> bool:
        """Supprime le fichier de la session ; ``True`` si elle existait."""
        path = self.path_for(session_id)
        if not path.is_file():
            return False
        path.unlink()
        return True
