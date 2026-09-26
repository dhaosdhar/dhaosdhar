"""Registre des backends : ``get_backend(settings, "ollama" | "claude")``."""
from __future__ import annotations

from ..config import Settings
from .base import Backend, BackendError

BACKEND_NAMES: tuple[str, ...] = ("ollama", "claude")


def get_backend(settings: Settings, name: str | None = None, *, model: str | None = None) -> Backend:
    name = (name or settings.backends.default).lower()
    if name == "ollama":
        from .ollama import OllamaBackend

        return OllamaBackend(settings, model=model)
    if name == "claude":
        from .claude import ClaudeBackend

        return ClaudeBackend(settings, model=model)
    raise ValueError(f"backend inconnu : {name!r} (attendu : {', '.join(BACKEND_NAMES)})")


__all__ = ["Backend", "BackendError", "BACKEND_NAMES", "get_backend"]
