"""Outil shell : exécution d'une commande bash sous politique d'accès.

``run_command`` exécute ``/bin/bash -c <commande>`` dans un répertoire de
travail lisible (``check_read``), après ``check_command`` (liste blanche ou
confirmation via ``ctx.confirm`` ; une commande en liste blanche dont un
argument désigne un chemin protégé — secrets, clés — ou une option d'écriture
repasse en confirmation), avec un délai maximal (``tools.command_timeout``),
un environnement expurgé des variables sensibles (clés, jetons, secrets, mots
de passe, sockets d'agent SSH/GPG, URL avec identifiants) et une sortie
tronquée. Chaque exécution est journalisée (``ctx.journal``).
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from ..config import Settings
from ..utils import truncate
from .base import Tool, ToolContext, ToolError, ToolResult

# Fragments de noms de variables d'environnement jamais transmis aux commandes.
SENSITIVE_ENV_FRAGMENTS: tuple[str, ...] = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
# Variables jamais transmises quel que soit leur nom : accès aux agents SSH / GPG.
SENSITIVE_ENV_NAMES: tuple[str, ...] = ("SSH_AUTH_SOCK", "SSH_AGENT_PID", "GPG_AGENT_INFO")
# Valeur ressemblant à une URL porteuse d'identifiants (``schéma://user:mdp@hôte``),
# forme usuelle de DATABASE_URL, REDIS_URL, *_DSN…
CREDENTIAL_URL_RE = re.compile(r"^\s*[a-z][a-z0-9+.-]*://[^/@\s]*:[^/@\s]+@", re.IGNORECASE)


def is_sensitive_env(name: str, value: str) -> bool:
    """Vrai si la variable ne doit pas être transmise aux commandes.

    Le nom évoque un secret (``SENSITIVE_ENV_FRAGMENTS``), désigne un agent
    (``SENSITIVE_ENV_NAMES``), ou la valeur est une URL avec identifiants.
    """
    upper = name.upper()
    if upper in SENSITIVE_ENV_NAMES or any(frag in upper for frag in SENSITIVE_ENV_FRAGMENTS):
        return True
    return bool(CREDENTIAL_URL_RE.match(value))


def scrub_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copie de l'environnement sans les variables sensibles (voir ``is_sensitive_env``)."""
    source = os.environ if environ is None else environ
    return {k: v for k, v in source.items() if not is_sensitive_env(k, v)}


def _bash_path() -> str:
    if os.path.exists("/bin/bash"):
        return "/bin/bash"
    return shutil.which("bash") or "/bin/sh"


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _ask_confirmation(ctx: ToolContext, prompt: str) -> bool:
    try:
        return bool(ctx.confirm(prompt))
    except Exception:  # noqa: BLE001 — un callback défaillant vaut refus
        return False


def _section(text: str, limit: int) -> str:
    return truncate(text, limit) if text else "(vide)"


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Exécute une commande shell (bash -c) dans le projet ou dans cwd, et "
        "renvoie le code de sortie, stdout et stderr (tronqués). Non interactive : "
        "pas d'entrée standard. Délai maximal configurable (timeout en secondes, "
        "plafonné par la configuration). Les commandes hors liste blanche "
        "(ls, cat, git status, pytest…) demandent une confirmation à "
        "l'utilisateur selon la politique ; pour lire ou modifier des fichiers, "
        "préférez read_file / edit_file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1, "description": "Commande bash à exécuter."},
            "cwd": {"type": "string", "minLength": 1, "description": "Répertoire de travail (défaut : racine du projet)."},
            "timeout": {"type": "number", "exclusiveMinimum": 0, "description": "Délai maximal en secondes."},
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    may_require_confirmation = True

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            return self._run(args, ctx)
        except ToolError as e:
            return ToolResult(str(e), is_error=True)

    def _run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(args["command"])
        if not command.strip():
            raise ToolError("commande refusée : commande vide")

        # Répertoire de travail : lisible et existant.
        cwd_arg = args.get("cwd")
        cwd = ctx.policy.resolve(cwd_arg) if cwd_arg else Path(ctx.project_root or ctx.policy.project_root)
        decision = ctx.policy.check_read(cwd)
        if not decision.allowed:
            raise ToolError(f"commande refusée : répertoire de travail interdit ({decision.reason}) : {cwd}")
        if not cwd.is_dir():
            raise ToolError(f"commande refusée : répertoire de travail introuvable : {cwd}")

        # Politique de commande et confirmation.
        decision = ctx.policy.check_command(command, cwd=cwd)
        if not decision.allowed:
            raise ToolError(f"commande refusée : {decision.reason}")
        confirmed = False
        if decision.needs_confirmation:
            if not _ask_confirmation(ctx, f"Exécuter : {command} ? "):
                raise ToolError(f"commande refusée : confirmation refusée par l'utilisateur ({decision.reason})")
            confirmed = True

        # Délai : celui demandé, plafonné par la configuration.
        limit = float(ctx.settings.tools.command_timeout)
        requested = args.get("timeout")
        timeout: float | None
        if limit <= 0:
            timeout = float(requested) if requested else None
        else:
            timeout = min(float(requested), limit) if requested else limit

        env = scrub_env()
        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                [_bash_path(), "-c", command],
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
        except OSError as e:
            raise ToolError(f"impossible de lancer la commande : {e.strerror or e}") from None

        half = max(1000, ctx.settings.tools.max_output_chars // 2)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            duration = round(time.monotonic() - started, 3)
            ctx.journal.record(
                "run_command", command=command, cwd=str(cwd), exit=None, duration=duration,
                confirmed=confirmed, timeout=timeout,
            )
            shown = f"{timeout:g}" if timeout is not None else "?"
            msg = (
                f"délai dépassé ({shown}s) : commande interrompue\n"
                f"--- stdout ---\n{_section(stdout or '', half)}\n--- stderr ---\n{_section(stderr or '', half)}"
            )
            return ToolResult(msg, is_error=True, data={"exit": None, "timeout": timeout, "duration": duration})

        duration = round(time.monotonic() - started, 3)
        rc = proc.returncode
        ctx.journal.record(
            "run_command", command=command, cwd=str(cwd), exit=rc, duration=duration, confirmed=confirmed
        )
        text = f"exit code {rc}\n--- stdout ---\n{_section(stdout, half)}\n--- stderr ---\n{_section(stderr, half)}"
        return ToolResult(
            text,
            data={"exit": rc, "duration": duration, "cwd": str(cwd), "confirmed": confirmed,
                  "stdout_chars": len(stdout), "stderr_chars": len(stderr)},
        )


def tools(settings: Settings) -> list[Tool]:
    """Outil shell : run_command."""
    return [RunCommandTool()]


__all__ = ["CREDENTIAL_URL_RE", "RunCommandTool", "SENSITIVE_ENV_FRAGMENTS", "SENSITIVE_ENV_NAMES",
           "is_sensitive_env", "scrub_env", "tools"]
