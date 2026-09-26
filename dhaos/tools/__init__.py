"""Outils de l'agent : disque, shell, web, bases de savoir.

Chaque module expose ``tools(settings) -> list[Tool]`` ; ``default_registry``
les assemble. Noms d'outils (stables, référencés par le prompt système) :

- filesystem : ``read_file``, ``list_dir``, ``find_files``, ``grep``,
  ``write_file``, ``edit_file``
- shell      : ``run_command``
- web        : ``web_search``, ``fetch_url``
- knowledge  : ``kb_list``, ``kb_search``, ``kb_add_note``
"""
from __future__ import annotations

from ..config import Settings
from .base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult


def default_registry(settings: Settings, *, exclude: set[str] | None = None) -> ToolRegistry:
    from . import filesystem, knowledge, shell, web

    exclude = exclude or set()
    registry = ToolRegistry()
    for module in (filesystem, shell, web, knowledge):
        for tool in module.tools(settings):
            if tool.name not in exclude:
                registry.register(tool)
    return registry


__all__ = ["Tool", "ToolContext", "ToolError", "ToolRegistry", "ToolResult", "default_registry"]
