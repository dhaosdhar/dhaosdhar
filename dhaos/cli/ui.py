"""Aides d'affichage (rich) partagées par les commandes de la CLI.

Tout texte d'origine externe (noms de bases, sources, extraits, arguments
d'outils…) est *non fiable* : il passe systématiquement par
``rich.markup.escape`` avant d'être affiché, pour ne jamais être interprété
comme balisage rich.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, Iterable

import typer
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

# Les consoles résolvent ``sys.stdout`` / ``sys.stderr`` à chaque écriture,
# ce qui les rend compatibles avec ``typer.testing.CliRunner``.
console = Console()
err_console = Console(stderr=True)

_ELLIPSIS = "…"


# ------------------------------------------------------------------ messages
def fail(message: str, code: int = 1) -> None:
    """Affiche une erreur en rouge (stderr) et termine avec le code donné."""
    err_console.print(f"[bold red]Erreur :[/] {escape(str(message))}")
    raise typer.Exit(code)


def warn(message: str) -> None:
    err_console.print(f"[yellow]{escape(str(message))}[/]")


def success(message: str) -> None:
    console.print(f"[green]{escape(str(message))}[/]")


def note(message: str) -> None:
    console.print(f"[dim]{escape(str(message))}[/]")


# ------------------------------------------------------------ confirmations
def stdin_is_interactive() -> bool:
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def ask_confirm(question: str, *, default: bool = False) -> bool:
    """Pose une question oui/non ; ``False`` si l'entrée est fermée ou interrompue."""
    try:
        return bool(Confirm.ask(escape(str(question)), default=default, console=console))
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False


def interactive_confirm(question: str) -> bool:
    """Confirmation destinée aux outils de l'agent : refuse d'office quand
    l'entrée standard n'est pas interactive (script, tube, service)."""
    if not stdin_is_interactive():
        return False
    return ask_confirm(question)


# ------------------------------------------------------------------- textes
def shorten(value: Any, limit: int = 120) -> str:
    """Aplatit une valeur sur une ligne et la tronque à ``limit`` caractères."""
    text = " ".join(str(value if value is not None else "").split())
    if limit > 0 and len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + _ELLIPSIS
    return text


def format_args(args: Any, limit: int = 90) -> str:
    """Représentation abrégée des arguments d'un appel d'outil."""
    if not isinstance(args, dict) or not args:
        return ""
    parts: list[str] = []
    for key, value in args.items():
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        parts.append(f"{key}={shorten(rendered, 40)!r}" if isinstance(value, str) else f"{key}={shorten(rendered, 40)}")
    return shorten(", ".join(parts), limit)


def render_scalar(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if value is None:
        return ""
    return str(value)


# ------------------------------------------------------------- flux de texte
class StreamPrinter:
    """Imprime le texte diffusé par le modèle au fil de l'eau (stdout, flush)
    et sait si le curseur est en début de ligne."""

    def __init__(self) -> None:
        self.at_line_start = True
        self.chars = 0

    def __call__(self, chunk: str) -> None:
        if not chunk:
            return
        sys.stdout.write(chunk)
        sys.stdout.flush()
        self.chars += len(chunk)
        self.at_line_start = chunk.endswith("\n")

    def ensure_newline(self) -> None:
        if not self.at_line_start:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.at_line_start = True



class Activity:
    """Indicateur d'activité sur stderr pendant que rien ne s'affiche : spinner,
    phase en cours et temps écoulé (« ⠋ le modèle lit le contexte… 37 s »).
    Inactif quand stderr n'est pas un terminal (tube, service, tests)."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, stream: Any = None, enabled: bool | None = None) -> None:
        self.stream = stream or sys.stderr
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)()) if enabled is None else enabled
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._label = ""
        self._t0 = 0.0

    def start(self, label: str) -> None:
        if not self.enabled:
            return
        self.stop()
        self._label = label
        self._t0 = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, name="dhaos-activity", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        i = 0
        while not self._stop.wait(0.2):
            elapsed = int(time.monotonic() - self._t0)
            frame = self.FRAMES[i % len(self.FRAMES)]
            self.stream.write(f"\r\x1b[2K{frame} {self._label} {elapsed} s")
            self.stream.flush()
            i += 1

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1)
        self._thread = None
        self.stream.write("\r\x1b[2K")
        self.stream.flush()

    def __enter__(self) -> "Activity":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


# ----------------------------------------------------------------- tableaux
def make_table(*columns: str, title: str | None = None) -> Table:
    table = Table(title=title, show_lines=False, header_style="bold", pad_edge=False)
    for name in columns:
        justify = "right" if name.lower() in ("docs", "chunks", "messages", "taille", "score", "n") else "left"
        table.add_column(name, justify=justify, overflow="fold")
    return table


def print_bases_table(bases: Iterable[Any]) -> None:
    rows = list(bases)
    if not rows:
        note("Aucune base de savoir. Créez-en une avec : dhaos kb create NOM")
        return
    table = make_table("Nom", "Description", "Docs", "Chunks", "Embedder")
    for b in rows:
        table.add_row(
            escape(str(getattr(b, "name", ""))),
            escape(shorten(getattr(b, "description", ""), 80)),
            str(getattr(b, "n_docs", 0)),
            str(getattr(b, "n_chunks", 0)),
            escape(str(getattr(b, "embedder", ""))),
        )
    console.print(table)


def print_mapping(data: Any, *, title: str | None = None) -> None:
    """Affiche un dictionnaire clé/valeur (valeurs imbriquées en JSON)."""
    if not isinstance(data, dict):
        console.print(escape(render_scalar(data)))
        return
    if not data:
        note("(vide)")
        return
    table = make_table("Clé", "Valeur", title=title)
    for key, value in data.items():
        table.add_row(escape(str(key)), escape(shorten(render_scalar(value), 200)))
    console.print(table)


def print_json(data: Any) -> None:
    """JSON brut sur stdout (sans coloration, pour être analysable)."""
    sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")
    sys.stdout.flush()


__all__ = [
    "StreamPrinter",
    "ask_confirm",
    "console",
    "err_console",
    "fail",
    "format_args",
    "interactive_confirm",
    "make_table",
    "note",
    "print_bases_table",
    "print_json",
    "print_mapping",
    "render_scalar",
    "shorten",
    "stdin_is_interactive",
    "success",
    "warn",
]
