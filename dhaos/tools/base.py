"""Contrat des outils exposés au modèle et registre d'exécution.

Un outil déclare un nom, une description et un schéma JSON de paramètres,
puis implémente ``run(args, ctx)``. Le registre valide les arguments contre
le schéma (les entrées du modèle sont *non fiables*), exécute, capture les
exceptions et tronque les sorties trop longues.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from ..config import Settings
from ..policy import AccessPolicy, Confirmer, Journal, never_confirm
from ..types import ToolCall, ToolSpec
from ..utils import truncate


class ToolError(Exception):
    """Erreur attendue d'un outil, renvoyée au modèle comme résultat en erreur."""


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    data: Any = None  # charge structurée optionnelle (API, tests)


@dataclass
class ToolContext:
    """Tout ce dont un outil a besoin pour s'exécuter."""

    settings: Settings
    policy: AccessPolicy
    journal: Journal
    confirm: Confirmer = never_confirm
    project_root: Path | None = None
    kb: Any = None  # dhaos.kb.manager.KnowledgeManager (optionnel)
    backend: Any = None  # dhaos.backends.base.Backend (optionnel)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.project_root is None:
            self.project_root = self.policy.project_root


class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    # Indicatif : l'outil peut déclencher une confirmation utilisateur.
    may_require_confirmation: bool = False

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


def describe_parameters(schema: dict[str, Any]) -> str:
    """« query (string, obligatoire), bases (array), top_k (integer) » depuis un schéma JSON."""
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    if not props:
        return "aucun"
    parts = []
    for name, spec in props.items():
        kind = spec.get("type", "any") if isinstance(spec, dict) else "any"
        if isinstance(kind, list):
            kind = "/".join(str(k) for k in kind)
        parts.append(f"{name} ({kind}, obligatoire)" if name in required else f"{name} ({kind})")
    return ", ".join(parts)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("un outil doit avoir un nom")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(f"Outil inconnu : {call.name}", is_error=True)
        args = call.arguments if isinstance(call.arguments, dict) else {}
        try:
            jsonschema.validate(args, tool.parameters)
        except jsonschema.ValidationError as e:
            # Message lisible par un petit modèle local : ce qui manque, ce qui
            # est attendu, et l'invitation explicite à rappeler l'outil.
            received = json.dumps(call.arguments, ensure_ascii=False)
            return ToolResult(
                f"INVALID_JSON — appel de {call.name} invalide : {e.message}. "
                f"Arguments reçus : {received}. Paramètres attendus : {describe_parameters(tool.parameters)}. "
                f"Rappelle {call.name} avec des arguments corrects.",
                is_error=True,
            )
        try:
            result = tool.run(args, ctx)
        except ToolError as e:
            result = ToolResult(str(e), is_error=True)
        except Exception as e:  # noqa: BLE001 — jamais laisser un outil casser la boucle
            result = ToolResult(f"{type(e).__name__}: {e}", is_error=True)
        limit = ctx.settings.tools.max_output_chars
        if len(result.content) > limit:
            result.content = truncate(result.content, limit)
        return result
