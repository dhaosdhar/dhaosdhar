"""Sessions de conversation persistées — CONTRAT.

Une session = un fichier JSONL dans ``settings.sessions_dir`` :
première ligne ``{"type": "meta", ...}`` (id, titre, backend, modèle, dates),
puis une ligne ``{"type": "message", ...}`` par ``Message.to_dict()``, et des
lignes ``{"type": "event", ...}`` optionnelles (usage, outils) pour les traces
d'entraînement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..types import Message


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

    def append(self, message: Message) -> None:
        raise NotImplementedError

    def log_event(self, kind: str, **fields: Any) -> None:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "Session":
        raise NotImplementedError


class SessionStore:
    def __init__(self, settings: Settings):
        raise NotImplementedError

    def create(self, *, title: str = "", backend: str = "", model: str = "") -> Session:
        raise NotImplementedError

    def get(self, session_id: str) -> Session:
        raise NotImplementedError

    def list(self) -> list[SessionInfo]:
        raise NotImplementedError

    def delete(self, session_id: str) -> bool:
        raise NotImplementedError
