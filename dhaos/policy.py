"""Politique d'accès de l'agent : lecture du disque, écritures, commandes shell,
et journal des actions.

Principe : **lecture libre** sur les racines configurées (tout le disque par
défaut) sauf motifs protégés ; **écriture** libre dans le projet courant et
confirmée ailleurs ; **commandes** confirmées sauf liste blanche. Chaque
écriture et chaque commande est journalisée (``journal.jsonl``).
"""
from __future__ import annotations

import datetime as _dt
import fnmatch
import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Settings

# Callback de confirmation : reçoit une question, renvoie True pour autoriser.
Confirmer = Callable[[str], bool]


def auto_confirm(_prompt: str) -> bool:
    return True


def never_confirm(_prompt: str) -> bool:
    return False


@dataclass
class Decision:
    allowed: bool
    needs_confirmation: bool = False
    reason: str = ""


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Traduit un glob (avec ``**``) en expression régulière ancrée."""
    i, out = 0, []
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if pattern.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern.startswith("**", i):
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out) + "$")


class AccessPolicy:
    """Décide ce que l'agent peut lire, écrire et exécuter."""

    def __init__(self, settings: Settings, project_root: Path | None = None):
        self.settings = settings
        self.tools = settings.tools
        self.project_root = (project_root or settings.resolve_project_root()).resolve()
        self.read_roots = [Path(os.path.expanduser(str(r))).resolve() for r in self.tools.read_roots]
        self._deny: list[tuple[str, str, re.Pattern[str] | None]] = []
        for pat in self.tools.deny_patterns:
            expanded = os.path.expanduser(pat) if pat.startswith("~") else pat
            if "/" in expanded:
                self._deny.append(("path", pat, glob_to_regex(expanded)))
            else:
                self._deny.append(("name", pat, None))

    # ------------------------------------------------------------- chemins
    def resolve(self, path: str | Path) -> Path:
        p = Path(os.path.expanduser(str(path)))
        if not p.is_absolute():
            p = self.project_root / p
        return p.resolve()

    def is_inside_project(self, path: str | Path) -> bool:
        p = self.resolve(path)
        return p == self.project_root or p.is_relative_to(self.project_root)

    def denied_reason(self, path: str | Path) -> str | None:
        p = self.resolve(path)
        s = str(p)
        for kind, original, regex in self._deny:
            if kind == "path":
                if regex is not None and regex.match(s):
                    return f"chemin protégé par la politique ({original})"
            else:
                for part in p.parts:
                    if fnmatch.fnmatchcase(part, original):
                        return f"nom protégé par la politique ({original})"
        return None

    def check_read(self, path: str | Path) -> Decision:
        p = self.resolve(path)
        reason = self.denied_reason(p)
        if reason:
            return Decision(False, False, reason)
        if not any(p == root or p.is_relative_to(root) for root in self.read_roots):
            return Decision(False, False, "hors des racines de lecture autorisées")
        return Decision(True)

    def check_write(self, path: str | Path) -> Decision:
        p = self.resolve(path)
        reason = self.denied_reason(p)
        if reason:
            return Decision(False, False, reason)
        policy = self.tools.write_policy
        if policy == "deny":
            return Decision(False, False, "écriture désactivée (write_policy = deny)")
        if policy == "all":
            return Decision(True)
        if policy == "ask":
            return Decision(True, True, "confirmation requise (write_policy = ask)")
        if self.is_inside_project(p):
            return Decision(True)
        return Decision(True, True, f"hors du projet {self.project_root} : confirmation requise")

    # ----------------------------------------------------------- commandes
    def is_auto_allowed(self, command: str) -> bool:
        cmd = command.strip()
        if not cmd or re.search(r"[;&|<>`$\n\\]", cmd):
            return False
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return False
        if not tokens:
            return False
        for entry in self.tools.shell_auto_allow:
            words = entry.split()
            if words and tokens[: len(words)] == words:
                return True
        return False

    def check_command(self, command: str) -> Decision:
        policy = self.tools.shell_policy
        if policy == "deny":
            return Decision(False, False, "shell désactivé (shell_policy = deny)")
        if policy == "auto" or self.is_auto_allowed(command):
            return Decision(True)
        return Decision(True, True, "commande hors liste blanche : confirmation requise")


class Journal:
    """Journal JSONL des actions sensibles (écritures, commandes, confirmations)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def record(self, kind: str, **fields: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ts": time.time(),
            "iso": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "kind": kind,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
