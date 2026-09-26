"""Assemblage de l'exécution : Settings → politique d'accès, journal, bases de
savoir, backend, registre d'outils, contexte, agent.

Utilisé par la CLI et l'API ; tout composant est injectable (tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent.loop import Agent
from .agent.session import Session
from .backends import get_backend
from .backends.base import Backend
from .config import Settings
from .policy import AccessPolicy, Confirmer, Journal, never_confirm
from .tools import ToolRegistry, default_registry
from .tools.base import ToolContext


@dataclass
class Runtime:
    settings: Settings
    project_root: Path
    policy: AccessPolicy
    journal: Journal
    kb: Any  # KnowledgeManager | None
    backend: Backend
    registry: ToolRegistry
    ctx: ToolContext
    agent: Agent

    def close(self) -> None:
        if self.kb is not None:
            try:
                self.kb.close()
            except Exception:  # noqa: BLE001
                pass


def build_runtime(
    settings: Settings,
    *,
    backend_name: str | None = None,
    model: str | None = None,
    backend: Backend | None = None,
    confirm: Confirmer | None = None,
    project_root: Path | None = None,
    session: Session | None = None,
    system_prompt: str | None = None,
    tools: bool = True,
    exclude_tools: set[str] | None = None,
    kb: Any = True,  # True : ouvrir les bases de savoir ; False : sans ; objet : injecté
) -> Runtime:
    settings.ensure_dirs()
    root = (project_root or settings.resolve_project_root()).resolve()
    policy = AccessPolicy(settings, root)
    journal = Journal(settings.journal_path)

    if kb is True:
        from .kb.manager import KnowledgeManager

        kb_manager: Any = KnowledgeManager(settings)
    elif kb is False or kb is None:
        kb_manager = None
    else:
        kb_manager = kb

    backend = backend or get_backend(settings, backend_name, model=model)
    registry = default_registry(settings, exclude=exclude_tools) if tools else ToolRegistry()
    ctx = ToolContext(
        settings=settings,
        policy=policy,
        journal=journal,
        confirm=confirm or never_confirm,
        project_root=root,
        kb=kb_manager,
        backend=backend,
    )
    agent = Agent(settings, backend, registry, ctx, session=session, system_prompt=system_prompt)
    return Runtime(
        settings=settings,
        project_root=root,
        policy=policy,
        journal=journal,
        kb=kb_manager,
        backend=backend,
        registry=registry,
        ctx=ctx,
        agent=agent,
    )
