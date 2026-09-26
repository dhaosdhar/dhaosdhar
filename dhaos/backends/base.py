"""Contrat commun des backends de modèle (Ollama, Claude).

Un backend reçoit l'historique neutre (``dhaos.types.Message``), le prompt
système et les outils, diffuse le texte au fil de l'eau via ``on_text`` et
renvoie un ``ChatResponse`` complet (texte, appels d'outils, raison d'arrêt,
usage, charge brute à rejouer).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import ChatResponse, Message, TextCallback, ToolSpec


class BackendError(RuntimeError):
    """Erreur d'un backend (réseau, modèle absent, réponse invalide)."""


class Backend(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        on_text: TextCallback | None = None,
        on_thinking: TextCallback | None = None,
    ) -> ChatResponse:
        """Un tour de génération. Doit lever ``BackendError`` en cas d'échec
        irrécupérable et renvoyer ``stop_reason="tool_use"`` quand
        ``tool_calls`` n'est pas vide."""

    def supports_embeddings(self) -> bool:
        return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(f"le backend {self.name} ne fournit pas d'embeddings")

    def healthcheck(self) -> dict[str, Any]:
        """``{"ok": bool, "backend": str, "model": str, "detail": str, "models": [...]}``."""
        return {"ok": True, "backend": self.name, "model": self.model, "detail": "", "models": []}
