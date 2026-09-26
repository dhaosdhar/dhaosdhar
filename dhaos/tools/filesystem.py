"""Outils filesystem — à implémenter (voir docs/ARCHITECTURE.md)."""
from __future__ import annotations

from ..config import Settings
from .base import Tool


def tools(settings: Settings) -> list[Tool]:
    raise NotImplementedError
