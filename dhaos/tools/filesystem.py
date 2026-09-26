"""Outils disque : lecture, listage, recherche, écriture et édition de fichiers.

Tous les chemins passent par ``ctx.policy.resolve`` (chemin relatif = relatif
au projet) puis par ``check_read`` / ``check_write``. Les écritures sont
confirmées quand la politique l'exige (``ctx.confirm``), sauvegardées dans
``settings.backups_dir`` (si ``tools.backup_before_write``), effectuées de
façon atomique et journalisées (``ctx.journal``).

Les arguments viennent du modèle : ils sont validés par le schéma JSON du
registre puis revérifiés ici (existence, type de fichier, bornes).
"""
from __future__ import annotations

import datetime as _dt
import difflib
import fnmatch
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

from ..config import Settings
from ..utils import human_size, is_probably_binary, truncate
from .base import Tool, ToolContext, ToolError, ToolResult

_HEAD_BYTES = 8192  # échantillon pour la détection binaire
_MAX_READ_BYTES = 50_000_000  # garde-fou : jamais charger plus en mémoire
_LIST_LIMIT = 500  # entrées max renvoyées par list_dir
_SCAN_LIMIT = 200_000  # entrées max parcourues par find_files / grep
_GREP_LINE_CHARS = 300
_DIFF_LINES = 60
_BACKUP_NAME_CHARS = 150


# ----------------------------------------------------------------- helpers


def _display(ctx: ToolContext, p: Path) -> str:
    """Chemin relatif au projet quand le fichier y est, absolu sinon."""
    root = ctx.policy.project_root
    try:
        rel = p.relative_to(root)
    except ValueError:
        return str(p)
    s = str(rel)
    return "." if s == "." else s


def _is_ignored(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pat) for pat in patterns)


def _glob_match(rel: str, name: str, pattern: str) -> bool:
    """Glob (fnmatch) sur le chemin relatif OU le nom ; ``**/`` en tête toléré."""
    if fnmatch.fnmatchcase(rel, pattern) or fnmatch.fnmatchcase(name, pattern):
        return True
    if pattern.startswith("**/"):
        return fnmatch.fnmatchcase(rel, pattern[3:])
    return False


def _resolve_readable(ctx: ToolContext, path_arg: str) -> Path:
    """Résout un chemin et applique ``check_read`` ; lève ``ToolError`` si refusé."""
    p = ctx.policy.resolve(path_arg)
    decision = ctx.policy.check_read(p)
    if not decision.allowed:
        raise ToolError(f"lecture refusée : {decision.reason} ({p})")
    return p


class _DenyFilter:
    """Filtre des chemins protégés lors d'un parcours, exact et peu coûteux.

    ``AccessPolicy.denied_reason`` résout chaque chemin (coûteux sur des
    milliers de fichiers). Ici :

    - un dossier est élagué s'il est protégé ou si une *sonde* ``dossier/_``
      l'est (motifs ``X/**`` : ``X`` lui-même n'est pas protégé mais tout son
      contenu l'est) ;
    - un fichier n'est vérifié par la politique que si son nom correspond au
      dernier segment d'un motif de refus (les composants de dossier ont déjà
      été contrôlés à l'élagage).
    """

    def __init__(self, ctx: ToolContext):
        self.policy = ctx.policy
        globs: list[str] = []
        for pat in ctx.settings.tools.deny_patterns:
            segment = pat.rsplit("/", 1)[-1] if "/" in pat else pat
            if not segment or set(segment) <= {"*"}:
                continue  # motif « tout le contenu » : couvert par la sonde
            globs.append(fnmatch.translate(segment))
        self.name_re: re.Pattern[str] | None = re.compile("|".join(globs)) if globs else None

    def dir_reason(self, d: Path) -> str | None:
        return self.policy.denied_reason(d) or self.policy.denied_reason(d / "_")

    def file_reason(self, p: Path, name: str) -> str | None:
        if self.name_re is None or not self.name_re.match(name):
            return None
        return self.policy.denied_reason(p)


def _resolve_readable_dir(ctx: ToolContext, path_arg: str, deny: _DenyFilter) -> Path:
    """Comme ``_resolve_readable`` pour un répertoire dont le contenu doit être lisible."""
    p = _resolve_readable(ctx, path_arg)
    if p.is_dir():
        reason = deny.dir_reason(p)
        if reason:
            raise ToolError(f"lecture refusée : contenu du répertoire {p} : {reason}")
    return p


def _read_bytes_checked(p: Path) -> bytes:
    """Lit un fichier régulier texte ; erreurs claires sinon."""
    if not p.exists():
        raise ToolError(f"fichier introuvable : {p}")
    if p.is_dir():
        raise ToolError(f"{p} est un répertoire (utilisez list_dir)")
    if not p.is_file():
        raise ToolError(f"{p} n'est pas un fichier régulier")
    try:
        size = p.stat().st_size
        if size > _MAX_READ_BYTES:
            raise ToolError(
                f"fichier trop volumineux : {p} ({human_size(size)}, maximum {human_size(_MAX_READ_BYTES)})"
            )
        with open(p, "rb") as f:
            head = f.read(_HEAD_BYTES)
            if is_probably_binary(head):
                raise ToolError(f"fichier binaire : {p} (lecture refusée)")
            rest = f.read()
    except PermissionError:
        raise ToolError(f"permission refusée par le système : {p}") from None
    except OSError as e:
        raise ToolError(f"erreur de lecture de {p} : {e.strerror or e}") from None
    return head + rest


def _load_text(ctx: ToolContext, path_arg: str, *, strict: bool = False) -> tuple[Path, str]:
    p = _resolve_readable(ctx, path_arg)
    data = _read_bytes_checked(p)
    if strict:
        try:
            return p, data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"fichier non décodable en UTF-8 : {p} (édition refusée, utilisez write_file)") from None
    return p, data.decode("utf-8", errors="replace")


def _split_lines(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line.rstrip("\r") for line in lines]


def _ask_confirmation(ctx: ToolContext, prompt: str) -> bool:
    try:
        return bool(ctx.confirm(prompt))
    except Exception:  # noqa: BLE001 — un callback défaillant vaut refus
        return False


def _backup(ctx: ToolContext, target: Path) -> Path | None:
    """Copie l'ancien contenu dans ``backups_dir/<horodatage>__<nom-aplati>``."""
    if not (target.is_file() and ctx.settings.tools.backup_before_write):
        return None
    backups = ctx.settings.backups_dir
    backups.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    flat = str(target).strip("/").replace("/", "_") or "racine"
    if len(flat) > _BACKUP_NAME_CHARS:
        flat = flat[-_BACKUP_NAME_CHARS:]
    dest = backups / f"{stamp}__{flat}"
    n = 1
    while dest.exists():
        dest = backups / f"{stamp}-{n}__{flat}"
        n += 1
    shutil.copy2(target, dest)
    return dest


def _atomic_write(target: Path, data: bytes) -> None:
    """Écrit via un fichier temporaire du même dossier puis ``os.replace``."""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if target.exists():
            shutil.copymode(target, tmp)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _write_with_policy(
    ctx: ToolContext,
    target: Path,
    content: str,
    *,
    kind: str,
    create_dirs: bool,
    **journal_extra: Any,
) -> tuple[int, Path | None, bool]:
    """Chemin d'écriture commun à write_file / edit_file.

    Politique (``check_write``), confirmation, sauvegarde, écriture atomique,
    journal. Renvoie ``(octets écrits, sauvegarde, confirmé)`` ; lève
    ``ToolError`` en cas de refus.
    """
    decision = ctx.policy.check_write(target)
    if not decision.allowed:
        raise ToolError(f"écriture refusée : {decision.reason} ({target})")
    confirmed = False
    if decision.needs_confirmation:
        prompt = f"Écrire dans {target} ({len(content)} caractères) ? "
        if not _ask_confirmation(ctx, prompt):
            raise ToolError(f"écriture refusée : confirmation refusée par l'utilisateur ({decision.reason})")
        confirmed = True
    if target.is_dir():
        raise ToolError(f"écriture refusée : {target} est un répertoire")
    if target.exists() and not target.is_file():
        raise ToolError(f"écriture refusée : {target} n'est pas un fichier régulier")
    parent = target.parent
    if not parent.is_dir():
        if not create_dirs:
            raise ToolError(f"répertoire parent inexistant : {parent} (create_dirs=false)")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ToolError(f"impossible de créer {parent} : {e.strerror or e}") from None
    data = content.encode("utf-8")
    try:
        backup = _backup(ctx, target)
        _atomic_write(target, data)
    except PermissionError:
        raise ToolError(f"permission refusée par le système : {target}") from None
    except OSError as e:
        raise ToolError(f"erreur d'écriture de {target} : {e.strerror or e}") from None
    ctx.journal.record(
        kind,
        path=str(target),
        bytes=len(data),
        backup=str(backup) if backup else None,
        confirmed=confirmed,
        **journal_extra,
    )
    return len(data), backup, confirmed


def _walk_files(
    ctx: ToolContext, root: Path, *, state: dict[str, Any]
) -> Iterator[tuple[Path, str, str]]:
    """Parcourt ``root`` (sans suivre les liens) en ignorant ``kb.ignore_patterns``.

    Produit ``(chemin, chemin relatif à root, nom)`` pour chaque fichier, dans
    l'ordre trié ; ``state["scanned"]`` compte les entrées vues et
    ``state["interrupted"]`` passe à True si ``_SCAN_LIMIT`` est atteint.
    """
    patterns = ctx.settings.kb.ignore_patterns
    deny = _DenyFilter(ctx)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=None):
        base = Path(dirpath)
        kept: list[str] = []
        for d in sorted(dirnames):
            if _is_ignored(d, patterns) or deny.dir_reason(base / d):
                continue
            kept.append(d)
        dirnames[:] = kept
        for name in sorted(filenames):
            state["scanned"] = state.get("scanned", 0) + 1
            if state["scanned"] > _SCAN_LIMIT:
                state["interrupted"] = True
                return
            if _is_ignored(name, patterns):
                continue
            p = base / name
            if deny.file_reason(p, name):
                continue
            yield p, str(p.relative_to(root)), name
        state["scanned"] = state.get("scanned", 0) + len(dirnames)


# ------------------------------------------------------------------- base


class _FSTool(Tool):
    """Convertit les ``ToolError`` en résultats d'erreur (même hors registre)."""

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            return self._run(args, ctx)
        except ToolError as e:
            return ToolResult(str(e), is_error=True)

    def _run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:  # pragma: no cover
        raise NotImplementedError


# -------------------------------------------------------------- read_file


class ReadFileTool(_FSTool):
    name = "read_file"
    description = (
        "Lit un fichier texte et renvoie son contenu avec les numéros de ligne "
        "(format « N| texte »). Utilisez start_line / end_line (1-based, inclus) "
        "pour ne lire qu'une partie d'un gros fichier. Chemin relatif = relatif au "
        "projet. Les fichiers binaires et les fichiers protégés (secrets, clés) "
        "sont refusés ; la sortie est tronquée au-delà de tools.max_file_chars."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "description": "Chemin du fichier (absolu ou relatif au projet)."},
            "start_line": {"type": "integer", "minimum": 1, "description": "Première ligne à renvoyer (1-based)."},
            "end_line": {"type": "integer", "minimum": 1, "description": "Dernière ligne à renvoyer (incluse)."},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def _run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        p, text = _load_text(ctx, args["path"])
        lines = _split_lines(text)
        total = len(lines)
        if total == 0:
            return ToolResult(f"{p} (fichier vide)", data={"path": str(p), "total_lines": 0})
        start = int(args.get("start_line") or 1)
        end = int(args.get("end_line") or total)
        if start > total:
            raise ToolError(f"start_line={start} dépasse le nombre de lignes du fichier ({total})")
        if end < start:
            raise ToolError(f"end_line ({end}) doit être supérieur ou égal à start_line ({start})")
        end = min(end, total)
        body = "\n".join(f"{i}| {lines[i - 1]}" for i in range(start, end + 1))
        out = f"{p} (lignes {start}–{end} sur {total})\n{body}"
        out = truncate(out, ctx.settings.tools.max_file_chars)
        return ToolResult(out, data={"path": str(p), "total_lines": total, "start": start, "end": end})


# --------------------------------------------------------------- list_dir


class ListDirTool(_FSTool):
    name = "list_dir"
    description = (
        "Liste le contenu d'un répertoire sous forme d'arborescence indentée : "
        "type (d = dossier, f = fichier, l = lien symbolique), nom et taille. "
        "depth (1 à 3) contrôle la profondeur. Par défaut les entrées cachées "
        "(nom commençant par un point) et les dossiers techniques (.git, "
        "node_modules, __pycache__, .venv…) sont masqués : show_hidden=true pour "
        "tout afficher. Limité à 500 entrées."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "description": "Répertoire à lister (absolu ou relatif au projet)."},
            "depth": {"type": "integer", "minimum": 1, "maximum": 3, "description": "Profondeur (1 = enfants directs)."},
            "show_hidden": {"type": "boolean", "description": "Afficher aussi les entrées cachées et ignorées."},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def _run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        deny = _DenyFilter(ctx)
        p = _resolve_readable_dir(ctx, args["path"], deny)
        if not p.exists():
            raise ToolError(f"répertoire introuvable : {p}")
        if not p.is_dir():
            raise ToolError(f"{p} n'est pas un répertoire (utilisez read_file)")
        depth = int(args.get("depth") or 1)
        show_hidden = bool(args.get("show_hidden", False))
        patterns = ctx.settings.kb.ignore_patterns
        state = {"count": 0, "hidden": 0, "truncated": False}
        lines: list[str] = [f"{p} (profondeur {depth})"]

        def walk(directory: Path, level: int) -> None:
            indent = "  " * level
            try:
                entries = list(os.scandir(directory))
            except PermissionError:
                lines.append(f"{indent}(accès refusé par le système)")
                return
            except OSError as e:
                lines.append(f"{indent}(erreur : {e.strerror or e})")
                return
            entries.sort(key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
            for entry in entries:
                if not show_hidden and (entry.name.startswith(".") or _is_ignored(entry.name, patterns)):
                    state["hidden"] += 1
                    continue
                if state["count"] >= _LIST_LIMIT:
                    state["truncated"] = True
                    return
                state["count"] += 1
                if entry.is_symlink():
                    try:
                        target = os.readlink(entry.path)
                    except OSError:
                        target = "?"
                    lines.append(f"{indent}l {entry.name} -> {target}")
                elif entry.is_dir(follow_symlinks=False):
                    denied = deny.dir_reason(Path(entry.path))
                    suffix = f"  ({denied})" if denied else ""
                    lines.append(f"{indent}d {entry.name}/{suffix}")
                    if level + 1 < depth and not denied:
                        walk(Path(entry.path), level + 1)
                        if state["truncated"]:
                            return
                else:
                    try:
                        size = entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        size = 0
                    denied = deny.file_reason(Path(entry.path), entry.name)
                    suffix = f"  ({denied})" if denied else ""
                    lines.append(f"{indent}f {entry.name}  {human_size(size)}{suffix}")

        walk(p, 0)
        if state["count"] == 0:
            lines.append("(répertoire vide)" if state["hidden"] == 0 else "(aucune entrée visible)")
        if state["truncated"]:
            lines.append(f"… [liste tronquée à {_LIST_LIMIT} entrées]")
        if state["hidden"]:
            lines.append(f"({state['hidden']} entrée(s) masquée(s) ; show_hidden=true pour tout voir)")
        return ToolResult(
            "\n".join(lines),
            data={"path": str(p), "entries": state["count"], "hidden": state["hidden"], "truncated": state["truncated"]},
        )


# ------------------------------------------------------------- find_files


class FindFilesTool(_FSTool):
    name = "find_files"
    description = (
        "Recherche des fichiers par motif glob (ex. « *.py », « src/**/*.ts », "
        "« test_*.py ») sous un répertoire (root, par défaut le projet). Le motif "
        "est comparé au chemin relatif et au nom du fichier. Les dossiers "
        "techniques (.git, node_modules, .venv…) sont ignorés, les liens "
        "symboliques ne sont pas suivis. Résultats triés, limités à max_results."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "minLength": 1, "description": "Motif glob (fnmatch)."},
            "root": {"type": "string", "minLength": 1, "description": "Répertoire de départ (défaut : projet)."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 5000, "description": "Nombre max de résultats (défaut 200)."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def _run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = _resolve_readable_dir(ctx, args.get("root") or str(ctx.project_root), _DenyFilter(ctx))
        if not root.is_dir():
            raise ToolError(f"répertoire introuvable : {root}")
        pattern = str(args["pattern"])
        max_results = int(args.get("max_results") or 200)
        state: dict[str, Any] = {}
        matches: list[str] = []
        for p, rel, name in _walk_files(ctx, root, state=state):
            if _glob_match(rel, name, pattern):
                matches.append(_display(ctx, p))
        matches.sort()
        truncated = len(matches) > max_results
        shown = matches[:max_results]
        lines = [f"{len(matches)} fichier(s) pour « {pattern} » sous {root}"]
        lines.extend(shown)
        if truncated:
            lines.append(f"… [tronqué : {max_results} résultats affichés sur {len(matches)}]")
        if state.get("interrupted"):
            lines.append(f"(exploration interrompue après {_SCAN_LIMIT} entrées ; précisez root)")
        return ToolResult(
            "\n".join(lines),
            data={"root": str(root), "count": len(matches), "shown": len(shown), "truncated": truncated},
        )


# ------------------------------------------------------------------- grep


class GrepTool(_FSTool):
    name = "grep"
    description = (
        "Recherche une expression régulière (syntaxe Python) dans les fichiers "
        "texte d'un répertoire (root, par défaut le projet ; peut aussi être un "
        "fichier). glob restreint les fichiers examinés (ex. « *.py »). Renvoie "
        "des lignes « chemin:numéro: texte » (texte tronqué à 300 caractères). "
        "Binaires, fichiers trop gros et fichiers protégés sont ignorés."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "minLength": 1, "description": "Expression régulière Python."},
            "root": {"type": "string", "minLength": 1, "description": "Répertoire ou fichier de départ (défaut : projet)."},
            "glob": {"type": ["string", "null"], "description": "Motif de nom de fichier (fnmatch), ex. « *.py »."},
            "case_insensitive": {"type": "boolean", "description": "Ignorer la casse."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 5000, "description": "Nombre max de lignes (défaut 200)."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def _run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = str(args["pattern"])
        flags = re.IGNORECASE if args.get("case_insensitive") else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise ToolError(f"expression régulière invalide « {pattern} » : {e}") from None
        root = _resolve_readable_dir(ctx, args.get("root") or str(ctx.project_root), _DenyFilter(ctx))
        if not root.exists():
            raise ToolError(f"chemin introuvable : {root}")
        glob = args.get("glob") or None
        max_results = int(args.get("max_results") or 200)
        max_bytes = ctx.settings.kb.max_file_bytes
        state: dict[str, Any] = {}

        if root.is_file():
            candidates: Iterator[tuple[Path, str, str]] = iter([(root, root.name, root.name)])
        else:
            candidates = _walk_files(ctx, root, state=state)

        out: list[str] = []
        files_scanned = 0
        truncated = False
        for p, rel, name in candidates:
            if glob and not _glob_match(rel, name, glob):
                continue
            try:
                if not p.is_file() or p.stat().st_size > max_bytes:
                    continue
                with open(p, "rb") as f:
                    head = f.read(_HEAD_BYTES)
                    if is_probably_binary(head):
                        continue
                    data = head + f.read()
            except OSError:
                continue
            files_scanned += 1
            disp = _display(ctx, p)
            for i, line in enumerate(data.decode("utf-8", errors="replace").split("\n"), start=1):
                if regex.search(line):
                    text = line.rstrip("\r")
                    if len(text) > _GREP_LINE_CHARS:
                        text = text[:_GREP_LINE_CHARS] + "…"
                    out.append(f"{disp}:{i}: {text}")
                    if len(out) >= max_results:
                        truncated = True
                        break
            if truncated:
                break

        header = f"{len(out)} correspondance(s) pour /{pattern}/ dans {files_scanned} fichier(s) sous {root}"
        if not out:
            header = f"aucune correspondance pour /{pattern}/ ({files_scanned} fichier(s) examiné(s) sous {root})"
        lines = [header, *out]
        if truncated:
            lines.append(f"… [tronqué : {max_results} correspondances affichées ; affinez le motif ou root]")
        if state.get("interrupted"):
            lines.append(f"(exploration interrompue après {_SCAN_LIMIT} entrées ; précisez root)")
        return ToolResult(
            "\n".join(lines),
            data={"root": str(root), "count": len(out), "files": files_scanned, "truncated": truncated},
        )


# ------------------------------------------------------------- write_file


class WriteFileTool(_FSTool):
    name = "write_file"
    description = (
        "Écrit un fichier texte (UTF-8) en remplaçant intégralement son contenu, "
        "ou le crée (create_dirs crée les dossiers parents). Pour modifier une "
        "partie d'un fichier existant, préférez edit_file. Écriture libre dans le "
        "projet, confirmée par l'utilisateur ailleurs (selon la politique) ; "
        "l'ancien contenu est sauvegardé et l'action journalisée. Fichiers "
        "protégés (secrets, clés) refusés."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "description": "Chemin du fichier (absolu ou relatif au projet)."},
            "content": {"type": "string", "description": "Contenu complet à écrire."},
            "create_dirs": {"type": "boolean", "description": "Créer les dossiers parents manquants (défaut true)."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }
    may_require_confirmation = True

    def _run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        p = ctx.policy.resolve(args["path"])
        content = str(args["content"])
        create_dirs = bool(args.get("create_dirs", True))
        existed = p.is_file()
        n, backup, confirmed = _write_with_policy(ctx, p, content, kind="write_file", create_dirs=create_dirs)
        msg = f"écrit {p} ({n} octets)"
        if backup:
            msg += f"\nsauvegarde de l'ancien contenu : {backup}"
        return ToolResult(
            msg,
            data={"path": str(p), "bytes": n, "backup": str(backup) if backup else None,
                  "confirmed": confirmed, "created": not existed},
        )


# -------------------------------------------------------------- edit_file


class EditFileTool(_FSTool):
    name = "edit_file"
    description = (
        "Remplace un passage exact d'un fichier texte : old_string doit apparaître "
        "tel quel (indentation comprise) et une seule fois, sauf replace_all=true "
        "qui remplace toutes les occurrences. Lisez le fichier avant pour copier "
        "le passage exact. Même politique que write_file (confirmation hors "
        "projet, sauvegarde, journal). Renvoie un extrait de diff."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1, "description": "Chemin du fichier (absolu ou relatif au projet)."},
            "old_string": {"type": "string", "minLength": 1, "description": "Texte exact à remplacer."},
            "new_string": {"type": "string", "description": "Texte de remplacement (peut être vide)."},
            "replace_all": {"type": "boolean", "description": "Remplacer toutes les occurrences (défaut false)."},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }
    may_require_confirmation = True

    def _run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        p, text = _load_text(ctx, args["path"], strict=True)
        old = str(args["old_string"])
        new = str(args["new_string"])
        replace_all = bool(args.get("replace_all", False))
        if old == new:
            raise ToolError("old_string et new_string sont identiques : aucun changement")
        count = text.count(old)
        if count == 0:
            raise ToolError(f"old_string introuvable dans {p} (le texte doit correspondre exactement)")
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_string apparaît {count} fois dans {p} : ajoutez du contexte pour le rendre unique "
                "ou utilisez replace_all=true"
            )
        replaced = count if replace_all else 1
        new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        n, backup, confirmed = _write_with_policy(
            ctx, p, new_text, kind="edit_file", create_dirs=False, replacements=replaced
        )
        disp = _display(ctx, p)
        diff_lines = list(
            difflib.unified_diff(
                text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{disp}",
                tofile=f"b/{disp}",
                n=2,
            )
        )
        omitted = max(0, len(diff_lines) - _DIFF_LINES)
        diff = "".join(line if line.endswith("\n") else line + "\n" for line in diff_lines[:_DIFF_LINES])
        if omitted:
            diff += f"… [diff tronqué : {omitted} ligne(s) omise(s)]\n"
        msg = f"modifié {p} ({replaced} remplacement(s), {n} octets)\n{diff}".rstrip("\n")
        return ToolResult(
            msg,
            data={"path": str(p), "bytes": n, "replacements": replaced,
                  "backup": str(backup) if backup else None, "confirmed": confirmed},
        )


def tools(settings: Settings) -> list[Tool]:
    """Outils disque : read_file, list_dir, find_files, grep, write_file, edit_file."""
    return [ReadFileTool(), ListDirTool(), FindFilesTool(), GrepTool(), WriteFileTool(), EditFileTool()]


__all__ = [
    "EditFileTool",
    "FindFilesTool",
    "GrepTool",
    "ListDirTool",
    "ReadFileTool",
    "WriteFileTool",
    "tools",
]
