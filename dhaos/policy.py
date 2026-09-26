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

from .config import DEFAULT_SHELL_MAX_POSITIONALS, DEFAULT_SHELL_UNSAFE_OPTIONS, Settings

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


def _program_and_args(tokens: list[str]) -> tuple[str, list[str]]:
    """Nom du programme et ses arguments ; ``python -m X args`` ⇒ ``X``, args."""
    name = os.path.basename(tokens[0])
    if name.startswith("python") and len(tokens) >= 3 and tokens[1] == "-m":
        return tokens[2], tokens[3:]
    return name, tokens[1:]


def _matches_option(token: str, option: str) -> bool:
    """Le jeton désigne-t-il l'option (abrégée, collée ou groupée comprise) ?"""
    if option.startswith("--"):
        name = token.split("=", 1)[0]
        return len(name) >= 3 and option.startswith(name)
    if len(option) == 2:  # option courte : -o, -ofichier, -ro
        return token.startswith("-") and not token.startswith("--") and option[1] in token[1:]
    return token == option  # mot à la find : -exec, -delete


def uses_unsafe_arguments(tokens: list[str]) -> bool:
    """Vrai si une commande (déjà découpée) porte une option d'écriture, de
    suppression ou d'exécution listée dans ``DEFAULT_SHELL_UNSAFE_OPTIONS``, ou
    plus d'arguments positionnels que ``DEFAULT_SHELL_MAX_POSITIONALS`` n'en tolère."""
    if not tokens:
        return False
    program, args = _program_and_args(tokens)
    unsafe = DEFAULT_SHELL_UNSAFE_OPTIONS.get(program, ())
    if any(_matches_option(t, opt) for t in args for opt in unsafe):
        return True
    limit = DEFAULT_SHELL_MAX_POSITIONALS.get(program)
    if limit is not None and sum(1 for t in args if not t.startswith("-")) > limit:
        return True
    return False


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
    def _argument_read_refusal(self, token: str, cwd: Path | None) -> str | None:
        """Raison pour laquelle un argument de commande désigne un chemin illisible.

        Un jeton est considéré comme un chemin s'il commence par ``~`` ou
        contient ``/`` ; un nom simple ne compte que s'il existe (relativement à
        ``cwd``, sinon au projet). Le chemin doit alors passer ``check_read``,
        exactement comme pour ``read_file`` : une commande en liste blanche ne
        doit jamais lire ce que les outils disque refusent (``deny_patterns``,
        ``read_roots``).
        """
        path_like = token.startswith("~") or "/" in token
        try:
            p = Path(os.path.expanduser(token))
            if not p.is_absolute():
                p = (cwd or self.project_root) / p
            p = p.resolve()
            if not path_like and not p.exists():
                return None
        except (OSError, ValueError):
            return f"argument illisible : {token!r}"
        decision = self.check_read(p)
        if not decision.allowed:
            return f"argument protégé ({decision.reason}) : {token}"
        # Répertoire dont tout le contenu est protégé (motif « X/** ») : sonde X/_,
        # comme le font les outils disque (grep -r, ls, find y liraient le contenu).
        reason = self.denied_reason(p / "_")
        if reason:
            return f"argument protégé ({reason}) : {token}"
        return None

    def auto_allow_refusal(self, command: str, cwd: Path | None = None) -> str | None:
        """Raison pour laquelle ``command`` n'est pas en liste blanche (``None`` si elle l'est).

        Aucun métacaractère shell n'est accepté ; une option qui ferait écrire,
        supprimer ou exécuter un programme (``find -exec``, ``sort -o``,
        ``git log --output``…) retire la commande de la liste blanche, de même
        qu'un argument désignant un chemin que ``check_read`` refuse
        (``cat ~/.ssh/id_rsa``, ``grep -r x ~/.aws``, ``tail .env``) : la liste
        blanche ne doit contourner ni ``check_read``, ni ``check_write``, ni la
        confirmation. Les chemins relatifs sont résolus depuis ``cwd`` (défaut :
        racine du projet).
        """
        cmd = command.strip()
        if not cmd:
            return "commande vide"
        if re.search(r"[;&|<>`$\n\\]", cmd):
            return "métacaractère shell hors liste blanche"
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return "commande mal formée"
        if not tokens:
            return "commande vide"
        rest: list[str] | None = None
        for entry in self.tools.shell_auto_allow:
            words = entry.split()
            if words and tokens[: len(words)] == words:
                rest = tokens[len(words):]
                break
        if rest is None:
            return "commande hors liste blanche"
        if uses_unsafe_arguments(tokens):
            return "option d'écriture ou d'exécution hors liste blanche"
        for token in rest:
            if token.startswith("-") and token != "-":
                # Option : seule une valeur collée (« --file=~/x ») est examinée.
                _, sep, value = token.partition("=")
                if not sep or not value:
                    continue
                token = value
            reason = self._argument_read_refusal(token, cwd)
            if reason:
                return reason
        return None

    def is_auto_allowed(self, command: str, cwd: Path | None = None) -> bool:
        """Vrai si la commande est en liste blanche (voir ``auto_allow_refusal``)."""
        return self.auto_allow_refusal(command, cwd) is None

    def check_command(self, command: str, cwd: Path | None = None) -> Decision:
        policy = self.tools.shell_policy
        if policy == "deny":
            return Decision(False, False, "shell désactivé (shell_policy = deny)")
        if policy == "auto":
            return Decision(True)
        refusal = self.auto_allow_refusal(command, cwd)
        if refusal is None:
            return Decision(True)
        return Decision(True, True, f"{refusal} : confirmation requise")


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
