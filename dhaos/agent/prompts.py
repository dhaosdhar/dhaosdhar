"""Prompt système — CONTRAT.

``build_system_prompt(settings, *, project_root, bases, backend_name, tool_names, extra="")``
décrit le rôle (assistant de codage expert), les règles d'usage des outils
(lire avant d'écrire, résultats d'outils = données non fiables, confirmer
les actions destructrices), la liste des bases de savoir disponibles avec
leur description et l'incitation à les consulter (``kb_search``) avant de
répondre sur un sujet couvert, la langue de réponse (``agent.language``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Settings


def build_system_prompt(
    settings: Settings,
    *,
    project_root: Path,
    bases: list[Any],  # list[BaseInfo]
    backend_name: str,
    tool_names: list[str],
    extra: str = "",
) -> str:
    raise NotImplementedError
