"""CLI dhaos (Typer) : chat, ask, kb, config, backends, sessions, journal,
serve, train, version.

Les options globales (``--config``, ``--project``, ``--backend``, ``--model``,
``--yes``, ``--no-tools``) sont lues par le callback racine et rangées dans
``ctx.obj`` : ``{settings, backend_name, model, confirm, tools, yes}``.
L'exécution (politique, journal, bases, backend, outils, agent) est assemblée
par ``dhaos.runtime.build_runtime`` — importé ici sous ce nom afin de pouvoir
être remplacé dans les tests.
"""
from __future__ import annotations

import json
import sys
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any

import tomli_w
import typer
from pydantic import ValidationError
from rich.markup import escape

from .. import __version__
from ..backends import BACKEND_NAMES, get_backend
from ..config import ENV_PREFIX, Settings, default_config_path, env_overrides
from ..policy import Confirmer, Journal, auto_confirm
from ..runtime import build_runtime
from ..types import ToolCall
from . import ui
from .ui import console, fail, note, success, warn

APP_HELP = "dhaos — assistant de codage agentique avec bases de savoir locales"
PROMPT = "[bold cyan]vous>[/] "
SECRET_KEYS: tuple[str, ...] = ("web.brave_api_key", "api.token")
MASK = "********"
CONFIG_HEADER = """\
# Configuration de dhaos — assistant de codage agentique.
# Fichier généré par `dhaos config init`.
# Priorité : valeurs par défaut < ce fichier < variables d'environnement
# DHAOS__SECTION__CLE (ex. DHAOS__BACKENDS__DEFAULT=claude).
# Sections : paths (données, projet), backends (ollama / claude), tools
# (politique de lecture, d'écriture et de shell), web (recherche), kb (bases
# de savoir), agent, api (serveur HTTP), train (entraînement).
# Modifier une clé : `dhaos config set backends.default claude`.
"""

app = typer.Typer(no_args_is_help=True, help=APP_HELP)
kb_app = typer.Typer(no_args_is_help=True, help="Bases de savoir locales (catégories gérables).")
config_app = typer.Typer(no_args_is_help=True, help="Configuration (fichier TOML).")
sessions_app = typer.Typer(no_args_is_help=True, help="Sessions de conversation persistées.")
train_app = typer.Typer(no_args_is_help=True, help="Entraînement : jeux de données, modèle nano, LoRA.")
app.add_typer(kb_app, name="kb")
app.add_typer(config_app, name="config")
app.add_typer(sessions_app, name="sessions")
app.add_typer(train_app, name="train")


class BackendChoice(str, Enum):
    ollama = "ollama"
    claude = "claude"


class SearchMode(str, Enum):
    hybrid = "hybrid"
    vector = "vector"
    keyword = "keyword"


# =============================================================== options globales
@app.callback()
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", help="Fichier de configuration TOML."),
    project: Path | None = typer.Option(
        None, "--project", help="Racine du projet (défaut : répertoire courant).",
        exists=True, file_okay=False, resolve_path=True,
    ),
    backend: BackendChoice | None = typer.Option(None, "--backend", help="Backend : ollama ou claude."),
    model: str | None = typer.Option(None, "--model", help="Nom du modèle à utiliser."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirmer automatiquement les actions sensibles."),
    no_tools: bool = typer.Option(False, "--no-tools", help="Désactiver tous les outils de l'agent."),
) -> None:
    """dhaos — assistant de codage agentique avec bases de savoir locales."""
    try:
        settings = Settings.load(config)
    except (OSError, ValueError, ValidationError) as e:
        fail(f"configuration illisible : {e}")
        return
    if project is not None:
        settings.paths.project_root = project
    confirm: Confirmer = auto_confirm if yes else ui.interactive_confirm
    ctx.obj = {
        "settings": settings,
        "backend_name": backend.value if backend else None,
        "model": model,
        "confirm": confirm,
        "tools": not no_tools,
        "yes": yes,
    }


def _opts(ctx: typer.Context) -> dict[str, Any]:
    obj = ctx.obj if isinstance(ctx.obj, dict) else None
    if obj is None or "settings" not in obj:
        # Commande invoquée sans le callback racine (ne devrait pas arriver).
        settings = Settings.load()
        obj = {
            "settings": settings, "backend_name": None, "model": None,
            "confirm": ui.interactive_confirm, "tools": True, "yes": False,
        }
        ctx.obj = obj
    return obj


def _settings(ctx: typer.Context) -> Settings:
    return _opts(ctx)["settings"]


def _make_runtime(opts: dict[str, Any], *, backend_name: str | None, model: str | None, session: Any = None) -> Any:
    """Assemble l'exécution ; toute erreur devient un message rouge + exit 1."""
    try:
        return build_runtime(
            opts["settings"],
            backend_name=backend_name,
            model=model,
            confirm=opts["confirm"],
            tools=bool(opts["tools"]),
            session=session,
        )
    except Exception as e:  # noqa: BLE001 — backend absent, base illisible, clé manquante…
        fail(f"impossible de démarrer l'agent : {e}")
        return None


def _session_store(settings: Settings) -> Any:
    from ..agent.session import SessionStore

    return SessionStore(settings)


def _drop_if_empty(store: Any, session: Any) -> None:
    """Supprime une session créée pour ce lancement qui n'a reçu aucune
    réponse de l'assistant (sortie immédiate, backend injoignable…)."""
    if session is None:
        return
    messages = getattr(session, "messages", None) or []
    if any(getattr(m, "role", "") == "assistant" for m in messages):
        return
    try:
        store.delete(session.id)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------ rappels d'outils
def _on_tool_call(call: ToolCall) -> None:
    console.print(f"[dim]⚙ {escape(str(call.name))}({escape(ui.format_args(call.arguments))})[/]", highlight=False)


def _on_tool_result(_call: ToolCall, result: Any) -> None:
    status = "erreur" if getattr(result, "is_error", False) else "ok"
    n = len(getattr(result, "content", "") or "")
    console.print(f"[dim]  ↳ {status} ({n} car.)[/]", highlight=False)


def _print_usage(result: Any) -> None:
    usage = getattr(result, "usage", None)
    tokens_in = getattr(usage, "input_tokens", 0)
    tokens_out = getattr(usage, "output_tokens", 0)
    note(
        f"usage : {tokens_in} jetons en entrée, {tokens_out} en sortie · "
        f"{getattr(result, 'iterations', 0)} itération(s) · {getattr(result, 'tool_calls', 0)} appel(s) d'outil · "
        f"arrêt : {getattr(result, 'stop_reason', '')}"
    )


def _run_turn(runtime: Any, text: str, *, stream: bool = True) -> Any:
    """Un tour d'agent avec affichage en flux ; renvoie l'AgentResult.

    Tant que rien ne s'affiche (lecture du contexte, génération d'un appel
    d'outil, exécution), un indicateur d'activité tourne sur stderr avec la
    phase et le temps écoulé : l'utilisateur sait que ça travaille."""
    printer = ui.StreamPrinter()
    activity = ui.Activity(enabled=False if not stream else None)

    def on_text(chunk: str) -> None:
        activity.stop()
        printer(chunk)

    def tool_call(call: ToolCall) -> None:
        activity.stop()
        printer.ensure_newline()
        _on_tool_call(call)
        activity.start(f"exécution de {call.name}…")

    def tool_result(call: ToolCall, result: Any) -> None:
        activity.stop()
        _on_tool_result(call, result)
        activity.start("le modèle poursuit…")

    activity.start("le modèle lit le contexte et génère…")
    try:
        result = runtime.agent.run(
            text,
            on_text=on_text if stream else None,
            on_tool_call=tool_call if stream else None,
            on_tool_result=tool_result if stream else None,
        )
    finally:
        activity.stop()
    if stream:
        printer.ensure_newline()
    return result


# ======================================================================== chat
class ChatLoop:
    """Boucle interactive : commandes ``/…`` et tours d'agent."""

    COMMANDS = (
        ("/help", "cette aide"),
        ("/quit, /exit", "quitter"),
        ("/reset", "vider l'historique de la conversation"),
        ("/backend NOM", "changer de backend (ollama, claude)"),
        ("/model NOM", "changer de modèle"),
        ("/kb", "lister les bases de savoir"),
        ("/tools", "lister les outils disponibles"),
        ("/session", "afficher l'identifiant de la session"),
    )

    def __init__(self, opts: dict[str, Any], runtime: Any, session: Any, store: Any) -> None:
        self.opts = opts
        self.settings: Settings = opts["settings"]
        self.runtime = runtime
        self.session = session
        self.store = store
        self.backend_name: str | None = getattr(runtime.backend, "name", None) or opts.get("backend_name")
        self.model: str | None = opts.get("model")

    # ------------------------------------------------------------ affichage
    def header(self) -> None:
        backend = getattr(self.runtime.backend, "name", "?")
        model = getattr(self.runtime.backend, "model", "") or "(défaut)"
        project = getattr(self.runtime, "project_root", self.settings.resolve_project_root())
        n_bases = self._count_bases()
        console.print(
            f"[bold]dhaos[/] · backend [cyan]{escape(str(backend))}[/] · modèle [cyan]{escape(str(model))}[/] · "
            f"projet [cyan]{escape(str(project))}[/] · {n_bases} base(s) de savoir"
        )
        note("/help pour les commandes, /quit pour sortir.")

    def _count_bases(self) -> int:
        kb = getattr(self.runtime, "kb", None)
        if kb is None:
            return 0
        try:
            return len(list(kb.list_bases()))
        except Exception:  # noqa: BLE001
            return 0

    def _sync_session_meta(self) -> None:
        if self.session is None:
            return
        self.session.meta["backend"] = str(getattr(self.runtime.backend, "name", "") or "")
        self.session.meta["model"] = str(getattr(self.runtime.backend, "model", "") or "")
        try:
            self.session.save()
        except OSError as e:
            warn(f"session non enregistrée : {e}")

    # ------------------------------------------------------------ commandes
    def rebuild(self, *, backend_name: str | None, model: str | None) -> None:
        try:
            new_runtime = build_runtime(
                self.settings,
                backend_name=backend_name,
                model=model,
                confirm=self.opts["confirm"],
                tools=bool(self.opts["tools"]),
                session=self.session,
            )
        except Exception as e:  # noqa: BLE001
            warn(f"changement impossible : {e}")
            return
        try:
            self.runtime.close()
        except Exception:  # noqa: BLE001
            pass
        self.runtime = new_runtime
        self.backend_name, self.model = backend_name, model
        self._sync_session_meta()
        success(
            f"backend {getattr(new_runtime.backend, 'name', backend_name)} · "
            f"modèle {getattr(new_runtime.backend, 'model', '') or '(défaut)'}"
        )

    def handle_command(self, line: str) -> bool:
        """Traite une commande ``/…`` ; renvoie ``False`` pour quitter."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd in ("/quit", "/exit"):
            return False
        if cmd == "/help":
            table = ui.make_table("Commande", "Effet")
            for name, effect in self.COMMANDS:
                table.add_row(name, effect)
            console.print(table)
        elif cmd == "/reset":
            self.runtime.agent.reset()
            success("Historique effacé.")
        elif cmd == "/backend":
            if arg.lower() not in BACKEND_NAMES:
                warn(f"backend inconnu : {arg or '(vide)'} (attendu : {', '.join(BACKEND_NAMES)})")
            else:
                self.rebuild(backend_name=arg.lower(), model=None)
        elif cmd == "/model":
            if not arg:
                warn("usage : /model NOM")
            else:
                self.rebuild(backend_name=self.backend_name, model=arg)
        elif cmd == "/kb":
            kb = getattr(self.runtime, "kb", None)
            if kb is None:
                note("Bases de savoir désactivées.")
            else:
                try:
                    ui.print_bases_table(kb.list_bases())
                except Exception as e:  # noqa: BLE001
                    warn(f"bases de savoir illisibles : {e}")
        elif cmd == "/tools":
            names = list(getattr(self.runtime.registry, "names", []) or [])
            console.print(escape(", ".join(names)) if names else "[dim]Aucun outil.[/]")
        elif cmd == "/session":
            console.print(escape(self.session.id) if self.session is not None else "[dim]aucune session[/]")
        else:
            warn(f"commande inconnue : {cmd} (/help pour la liste)")
        return True

    def turn(self, text: str) -> None:
        try:
            result = _run_turn(self.runtime, text)
        except KeyboardInterrupt:
            console.print("\n[yellow]tour interrompu[/]")
            return
        if getattr(result, "error", None):
            console.print(f"[bold red]Erreur du backend :[/] {escape(str(result.error))}")
        _print_usage(result)

    # ------------------------------------------------------------- boucle
    def run(self) -> None:
        self.header()
        try:
            while True:
                try:
                    line = console.input(PROMPT)
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    if not self.handle_command(line):
                        break
                    continue
                self.turn(line)
        finally:
            try:
                self.runtime.close()
            except Exception:  # noqa: BLE001
                pass
            _drop_if_empty(self.store, self.session)
        note("Au revoir.")


@app.command()
def chat(
    ctx: typer.Context,
    session_id: str | None = typer.Option(None, "--session", help="Identifiant d'une session à reprendre."),
) -> None:
    """Conversation interactive avec l'agent (REPL)."""
    opts = _opts(ctx)
    settings = _settings(ctx)
    store = _session_store(settings)
    if session_id:
        try:
            session = store.get(session_id)
        except KeyError as e:
            fail(str(e))
            return
    else:
        session = store.create(
            backend=opts["backend_name"] or settings.backends.default, model=opts["model"] or ""
        )
    try:
        runtime = _make_runtime(opts, backend_name=opts["backend_name"], model=opts["model"], session=session)
    except typer.Exit:
        if not session_id:
            _drop_if_empty(store, session)
        raise
    loop = ChatLoop(opts, runtime, session, store)
    loop._sync_session_meta()
    loop.run()


# ========================================================================= ask
@app.command()
def ask(
    ctx: typer.Context,
    text: list[str] | None = typer.Argument(None, help="Question ou instruction (sinon lue sur stdin)."),
    as_json: bool = typer.Option(False, "--json", help="Imprimer le résultat en JSON."),
) -> None:
    """Un seul tour d'agent, réponse diffusée en flux."""
    opts = _opts(ctx)
    settings = _settings(ctx)
    prompt = " ".join(text).strip() if text else ""
    if not prompt and not ui.stdin_is_interactive():
        try:
            prompt = sys.stdin.read().strip()
        except (OSError, ValueError):
            prompt = ""
    if not prompt:
        fail("aucun texte fourni (argument ou stdin)")
        return
    store = _session_store(settings)
    session = store.create(backend=opts["backend_name"] or settings.backends.default, model=opts["model"] or "")
    try:
        runtime = _make_runtime(opts, backend_name=opts["backend_name"], model=opts["model"], session=session)
    except typer.Exit:
        _drop_if_empty(store, session)
        raise
    try:
        result = _run_turn(runtime, prompt, stream=not as_json)
    except KeyboardInterrupt:
        fail("interrompu", 130)
        return
    finally:
        try:
            runtime.close()
        except Exception:  # noqa: BLE001
            pass
        _drop_if_empty(store, session)
    error = getattr(result, "error", None)
    if as_json:
        usage = getattr(result, "usage", None)
        ui.print_json(
            {
                "text": getattr(result, "text", ""),
                "usage": {
                    "input_tokens": getattr(usage, "input_tokens", 0),
                    "output_tokens": getattr(usage, "output_tokens", 0),
                },
                "stop_reason": getattr(result, "stop_reason", ""),
                "iterations": getattr(result, "iterations", 0),
                "tool_calls": getattr(result, "tool_calls", 0),
                "error": error,
            }
        )
    elif error:
        console.print(f"[bold red]Erreur du backend :[/] {escape(str(error))}")
    if error:
        raise typer.Exit(1)


# ========================================================================== kb
class _KB:
    """Ouvre le gestionnaire de bases, traduit ``KnowledgeError`` en sortie 1."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.manager: Any = None

    def __enter__(self) -> Any:
        from ..kb.manager import KnowledgeError, KnowledgeManager

        try:
            self.manager = KnowledgeManager(self.settings)
        except KnowledgeError as e:
            fail(str(e))
        except Exception as e:  # noqa: BLE001
            fail(f"impossible d'ouvrir les bases de savoir : {type(e).__name__}: {e}")
        return self.manager

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> bool:
        from ..kb.manager import KnowledgeError

        if self.manager is not None:
            try:
                self.manager.close()
            except Exception:  # noqa: BLE001
                pass
        if exc is not None and isinstance(exc, KnowledgeError):
            fail(str(exc))
        return False


def _base_line(info: Any) -> str:
    return (
        f"base '{getattr(info, 'name', '')}' — {getattr(info, 'n_docs', 0)} document(s), "
        f"{getattr(info, 'n_chunks', 0)} chunk(s), embedder {getattr(info, 'embedder', '') or '?'}"
    )


@kb_app.command("create")
def kb_create(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Nom de la base (minuscules, a-z 0-9 . _ -)."),
    description: str = typer.Option("", "--description", "-d", help="Description de la catégorie."),
) -> None:
    """Créer une base de savoir (catégorie)."""
    with _KB(_settings(ctx)) as kb:
        info = kb.create_base(name, description)
        success(f"Base créée : {_base_line(info)}")


@kb_app.command("list")
def kb_list(ctx: typer.Context) -> None:
    """Lister les bases de savoir."""
    with _KB(_settings(ctx)) as kb:
        ui.print_bases_table(kb.list_bases())


@kb_app.command("show")
def kb_show(ctx: typer.Context, name: str = typer.Argument(..., help="Nom de la base.")) -> None:
    """Afficher les documents d'une base."""
    from ..utils import human_size

    with _KB(_settings(ctx)) as kb:
        info = kb.get_base(name)
        if info is not None:
            console.print(f"[bold]{escape(info.name)}[/] — {escape(info.description or '(sans description)')}")
        docs = kb.list_documents(name)
        if not docs:
            note("Aucun document.")
            return
        table = ui.make_table("Id", "Source", "Titre", "Chunks", "Taille", "Ajouté le")
        for d in docs:
            table.add_row(
                str(d.id), escape(str(d.source)), escape(ui.shorten(d.title, 60)),
                str(d.n_chunks), human_size(int(d.size or 0)), escape(str(d.added_at)),
            )
        console.print(table)


@kb_app.command("rename")
def kb_rename(
    ctx: typer.Context,
    old: str = typer.Argument(..., help="Nom actuel."),
    new: str = typer.Argument(..., help="Nouveau nom."),
) -> None:
    """Renommer une base."""
    with _KB(_settings(ctx)) as kb:
        info = kb.rename_base(old, new)
        success(f"Base renommée : {old} → {info.name}")


@kb_app.command("describe")
def kb_describe(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Nom de la base."),
    text: str = typer.Argument(..., help="Nouvelle description."),
) -> None:
    """Modifier la description d'une base."""
    with _KB(_settings(ctx)) as kb:
        info = kb.set_description(name, text)
        success(f"Description de '{info.name}' mise à jour.")


@kb_app.command("delete")
def kb_delete(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Nom de la base."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Ne pas demander de confirmation."),
) -> None:
    """Supprimer une base et tous ses documents."""
    opts = _opts(ctx)
    with _KB(_settings(ctx)) as kb:
        info = kb.get_base(name)
        if info is None:
            fail(f"base inconnue : {name}")
            return
        if not (yes or opts.get("yes")):
            if not ui.ask_confirm(f"Supprimer la base '{info.name}' ({info.n_docs} document(s)) ?"):
                note("Suppression annulée.")
                raise typer.Exit(1)
        kb.delete_base(name)
        success(f"Base '{info.name}' supprimée.")


@kb_app.command("add")
def kb_add(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Nom de la base."),
    sources: list[str] = typer.Argument(..., help="Fichiers, dossiers ou URLs http(s)."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Parcourir les dossiers récursivement."),
) -> None:
    """Ingérer des fichiers, dossiers ou URLs dans une base."""
    with _KB(_settings(ctx)) as kb:
        report = kb.add(
            name, list(sources), recursive=recursive,
            on_progress=lambda line: console.print(escape(str(line)), highlight=False),
        )
        console.print(escape(report.summary()))
        for err in report.errors:
            warn(f"  - {err}")
        if report.failed and not (report.added or report.updated or report.skipped):
            raise typer.Exit(1)


@kb_app.command("remove")
def kb_remove(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Nom de la base."),
    source: str = typer.Argument(..., help="Source du document (chemin, URL ou note:<titre>)."),
) -> None:
    """Retirer un document d'une base."""
    with _KB(_settings(ctx)) as kb:
        if kb.remove_document(name, source):
            success(f"Document retiré : {source}")
        else:
            fail(f"document introuvable dans '{name}' : {source}")


@kb_app.command("search")
def kb_search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Requête."),
    bases: list[str] | None = typer.Option(None, "--base", "-b", help="Base(s) à interroger (défaut : toutes)."),
    top_k: int | None = typer.Option(None, "--top-k", "-k", min=1, help="Nombre de résultats."),
    mode: SearchMode = typer.Option(SearchMode.hybrid, "--mode", help="hybrid, vector ou keyword."),
) -> None:
    """Rechercher dans les bases de savoir."""
    with _KB(_settings(ctx)) as kb:
        hits = kb.search(query, bases=list(bases) if bases else None, top_k=top_k, mode=mode.value)
        if not hits:
            note("Aucun résultat.")
            return
        for i, hit in enumerate(hits, 1):
            console.print(
                f"[bold]{i}.[/] [cyan]{escape(str(hit.base))}[/] · {escape(str(hit.source))}"
                f"{' · ' + escape(ui.shorten(hit.title, 60)) if hit.title else ''} · score {float(hit.score):.3f}",
                highlight=False,
            )
            console.print(f"   {escape(ui.shorten(hit.text, 300))}", highlight=False)


@kb_app.command("stats")
def kb_stats(ctx: typer.Context, name: str | None = typer.Argument(None, help="Base (défaut : toutes).")) -> None:
    """Statistiques d'une base ou de l'ensemble."""
    with _KB(_settings(ctx)) as kb:
        ui.print_mapping(kb.stats(name), title=f"Statistiques{' : ' + name if name else ''}")


@kb_app.command("reindex")
def kb_reindex(ctx: typer.Context, name: str = typer.Argument(..., help="Nom de la base.")) -> None:
    """Recalculer les embeddings d'une base avec l'embedder courant."""
    with _KB(_settings(ctx)) as kb:
        n = kb.reindex(name)
        success(f"Base '{name}' réindexée : {n} chunk(s).")


@kb_app.command("export")
def kb_export(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Nom de la base."),
    output: Path = typer.Argument(..., help="Fichier JSONL de sortie.", dir_okay=False, resolve_path=True),
) -> None:
    """Exporter les documents d'une base en JSONL."""
    with _KB(_settings(ctx)) as kb:
        try:
            n = kb.export(name, output)
        except OSError as e:
            fail(f"export impossible vers {output} : {e}")
            return
        success(f"{n} document(s) exporté(s) vers {output}")


# ====================================================================== config
def _masked(settings: Settings) -> Settings:
    masked = settings
    for key in SECRET_KEYS:
        section, _, field_name = key.partition(".")
        if getattr(getattr(settings, section), field_name, None):
            masked = masked.with_override(key, MASK)
    return masked


def _config_path(settings: Settings) -> Path:
    return Path(settings.source_path or default_config_path())


def _has_key(data: Any, dotted: str) -> bool:
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _get_key(data: Any, dotted: str) -> Any:
    node = data
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def _put_key(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Écrit ``value`` sous la clé pointée ; ``None`` retire la clé (et les
    sections devenues vides)."""
    parts = dotted.split(".")
    if value is None:
        chain: list[dict[str, Any]] = [data]
        for part in parts[:-1]:
            child = chain[-1].get(part)
            if not isinstance(child, dict):
                return
            chain.append(child)
        chain[-1].pop(parts[-1], None)
        for parent, part in zip(reversed(chain[:-1]), reversed(parts[:-1])):
            if not parent.get(part):
                parent.pop(part, None)
        return
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = node[part] = {}
        node = child
    node[parts[-1]] = value


def _env_name(dotted: str) -> str:
    return ENV_PREFIX + "__".join(part.upper() for part in dotted.split("."))


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Afficher la configuration effective (secrets masqués)."""
    settings = _settings(ctx)
    # Écrit directement sur stdout (pas via la console rich, qui replierait la
    # ligne hors TTY) pour que la sortie redirigée reste du TOML valide.
    sys.stdout.write(f"# fichier : {_config_path(settings)}\n")
    sys.stdout.write(_masked(settings).to_toml())
    sys.stdout.flush()


@config_app.command("path")
def config_path(ctx: typer.Context) -> None:
    """Afficher le chemin du fichier de configuration."""
    console.print(escape(str(_config_path(_settings(ctx)))), highlight=False)


@config_app.command("init")
def config_init(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", help="Écraser un fichier existant."),
) -> None:
    """Écrire un fichier de configuration par défaut."""
    path = _config_path(_settings(ctx))
    if path.exists() and not force:
        fail(f"le fichier existe déjà : {path} (utilisez --force pour l'écraser)")
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CONFIG_HEADER + "\n" + Settings().to_toml(), encoding="utf-8")
    except OSError as e:
        fail(f"écriture impossible : {e}")
        return
    success(f"Configuration écrite : {path}")


@config_app.command("set")
def config_set(
    ctx: typer.Context,
    key: str = typer.Argument(..., help="Clé pointée, ex. backends.default"),
    value: str = typer.Argument(..., help="Valeur (JSON si possible, sinon chaîne)."),
) -> None:
    """Modifier une clé et enregistrer le fichier.

    Seul le contenu du fichier (plus la clé modifiée) est réécrit : les
    surcharges d'environnement (``DHAOS__…``, y compris les secrets),
    ``--project`` et les chemins calculés (XDG) ne sont jamais figés dedans.
    """
    key = key.strip()
    path = _config_path(_settings(ctx))
    try:
        stored: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        base = Settings.model_validate(stored)
    except (OSError, ValueError, ValidationError) as e:
        fail(f"configuration illisible : {path} : {e}")
        return
    if not _has_key(base.model_dump(mode="json", exclude={"source_path"}), key):
        fail(f"clé inconnue : {key}")
        return
    try:
        new = base.with_override(key, value)
    except ValidationError as e:
        first = e.errors()[0] if e.errors() else {}
        fail(f"valeur invalide pour {key} : {first.get('msg', e)}")
        return
    node = _get_key(new.model_dump(mode="json", exclude_none=True, exclude={"source_path"}), key)
    _put_key(stored, key, node)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomli_w.dumps(stored), encoding="utf-8")
    except OSError as e:
        fail(f"enregistrement impossible : {e}")
        return
    shown = MASK if key in SECRET_KEYS and node else ui.render_scalar(node)
    success(f"{key} = {shown} (enregistré dans {path})")
    if _has_key(env_overrides(), key):
        warn(f"la variable d'environnement {_env_name(key)} surcharge cette clé : la valeur enregistrée "
             "ne sera effective qu'une fois cette variable retirée.")


# ==================================================================== backends
@app.command()
def backends(ctx: typer.Context) -> None:
    """État des backends (Ollama, Claude) : joignabilité et modèles."""
    settings = _settings(ctx)
    table = ui.make_table("Backend", "État", "Modèle", "Détail", "Modèles disponibles")
    for name in BACKEND_NAMES:
        try:
            backend = get_backend(settings, name)
            health = backend.healthcheck()
        except Exception as e:  # noqa: BLE001 — un backend en échec n'empêche pas l'autre
            health = {"ok": False, "backend": name, "model": "", "detail": f"{type(e).__name__}: {e}", "models": []}
        if not isinstance(health, dict):
            health = {"ok": False, "detail": "réponse de healthcheck invalide"}
        ok = bool(health.get("ok"))
        default_mark = " (défaut)" if name == settings.backends.default else ""
        models = health.get("models") or []
        table.add_row(
            f"{name}{default_mark}",
            "[green]ok[/]" if ok else "[red]indisponible[/]",
            escape(str(health.get("model") or "")),
            escape(ui.shorten(health.get("detail") or "", 120)),
            escape(ui.shorten(", ".join(str(m) for m in models), 120)),
        )
    console.print(table)


# ==================================================================== sessions
@sessions_app.command("list")
def sessions_list(ctx: typer.Context) -> None:
    """Lister les sessions (plus récentes d'abord)."""
    store = _session_store(_settings(ctx))
    infos = store.list()
    if not infos:
        note("Aucune session.")
        return
    table = ui.make_table("Id", "Titre", "Backend", "Modèle", "Messages", "Mis à jour")
    for s in infos:
        table.add_row(
            escape(s.id), escape(ui.shorten(s.title, 50)), escape(s.backend), escape(s.model),
            str(s.n_messages), escape(s.updated_at),
        )
    console.print(table)


@sessions_app.command("show")
def sessions_show(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Identifiant de la session."),
    limit: int = typer.Option(160, "--width", min=20, help="Longueur max de chaque message affiché."),
) -> None:
    """Afficher une session (messages abrégés)."""
    store = _session_store(_settings(ctx))
    try:
        session = store.get(session_id)
    except KeyError as e:
        fail(str(e))
        return
    meta = {k: v for k, v in session.meta.items() if k != "id"}
    console.print(f"[bold]session {escape(session.id)}[/]")
    ui.print_mapping(meta)
    if not session.messages:
        note("Aucun message.")
        return
    styles = {"user": "cyan", "assistant": "green", "tool": "magenta"}
    for m in session.messages:
        label = m.role if m.role != "tool" else f"tool:{m.name or '?'}{' (erreur)' if m.is_error else ''}"
        body = ui.shorten(m.content, limit)
        console.print(f"[{styles.get(m.role, 'white')}]{escape(label)}>[/] {escape(body)}", highlight=False)
        for call in m.tool_calls:
            console.print(f"   [dim]⚙ {escape(call.name)}({escape(ui.format_args(call.arguments))})[/]", highlight=False)


@sessions_app.command("delete")
def sessions_delete(ctx: typer.Context, session_id: str = typer.Argument(..., help="Identifiant.")) -> None:
    """Supprimer une session."""
    store = _session_store(_settings(ctx))
    try:
        deleted = store.delete(session_id)
    except KeyError as e:
        fail(str(e))
        return
    if not deleted:
        fail(f"session introuvable : {session_id}")
        return
    success(f"Session {session_id} supprimée.")


# ===================================================================== journal
@app.command()
def journal(
    ctx: typer.Context,
    n: int = typer.Option(50, "--n", "-n", min=1, help="Nombre d'entrées à afficher."),
) -> None:
    """Dernières entrées du journal des actions (écritures, commandes)."""
    settings = _settings(ctx)
    entries = Journal(settings.journal_path).tail(n)
    if not entries:
        note(f"Journal vide ({settings.journal_path}).")
        return
    table = ui.make_table("Date", "Type", "Détails")
    for entry in entries:
        details = ", ".join(
            f"{k}={ui.shorten(ui.render_scalar(v), 80)}" for k, v in entry.items() if k not in ("ts", "iso", "kind")
        )
        table.add_row(escape(str(entry.get("iso", ""))), escape(str(entry.get("kind", ""))), escape(details))
    console.print(table)


# ======================================================================= serve
@app.command()
def serve(
    ctx: typer.Context,
    host: str | None = typer.Option(None, "--host", help="Adresse d'écoute (défaut : api.host)."),
    port: int | None = typer.Option(None, "--port", min=1, max=65535, help="Port (défaut : api.port)."),
) -> None:
    """Lancer l'API HTTP (uvicorn)."""
    settings = _settings(ctx)
    host = host or settings.api.host
    port = port or settings.api.port
    # L'application dérive ses hôtes acceptés de api.host : refléter l'option.
    settings.api.host, settings.api.port = host, port
    try:
        import uvicorn

        from ..api import server as api_server

        application = api_server.create_app(settings)
    except (ImportError, AttributeError, NotImplementedError) as e:
        fail(f"API indisponible : {e}")
        return
    note(f"dhaos API sur http://{host}:{port}")
    state = getattr(application, "state", None)
    if getattr(state, "token_generated", False):
        note(f"jeton d'accès (api.token absent) : {state.token}")
    uvicorn.run(application, host=host, port=port)


# ======================================================================= train
_TRAIN_ERRORS = (ImportError, AttributeError, NotImplementedError)
# Erreurs « métier » signalées par le module d'entraînement (corpus vide,
# paramètre invalide, modèle incomplet, dépendance absente, écriture impossible).
_TRAIN_USER_ERRORS = (ValueError, OSError, RuntimeError)


def _train_log(line: Any) -> None:
    console.print(escape(str(line)), highlight=False)


@train_app.command("dataset")
def train_dataset(
    ctx: typer.Context,
    out: Path | None = typer.Option(None, "--out", help="Fichier JSONL de sortie.", dir_okay=False, resolve_path=True),
    bases: list[str] | None = typer.Option(
        None, "--base", "-b", hidden=True,
        help="Sans effet (conservé pour compatibilité) : les bases alimentent `train corpus`.",
    ),
) -> None:
    """Construire un jeu de données SFT (JSONL) à partir des sessions."""
    settings = _settings(ctx)
    if bases:
        warn("--base est sans effet pour le jeu SFT (les bases de savoir alimentent `train corpus`).")
    try:
        from ..train import dataset as dataset_mod

        report = dataset_mod.build_sft_dataset(settings, out_path=out)
    except _TRAIN_ERRORS as e:
        fail(f"module d'entraînement indisponible : {e}")
        return
    except _TRAIN_USER_ERRORS as e:
        fail(f"jeu de données impossible : {e}")
        return
    console.print(escape(str(report.summary())), highlight=False)
    success(f"Jeu de données : {report.path} ({report.n_examples} exemple(s), {report.n_sessions_skipped} session(s) ignorée(s))")


@train_app.command("corpus")
def train_corpus(
    ctx: typer.Context,
    out: Path | None = typer.Option(None, "--out", help="Fichier texte de sortie.", dir_okay=False, resolve_path=True),
    bases: list[str] | None = typer.Option(None, "--base", "-b", help="Bases de savoir à inclure."),
) -> None:
    """Assembler un corpus texte depuis les bases de savoir."""
    settings = _settings(ctx)
    try:
        from ..kb.manager import KnowledgeError
        from ..train import dataset as dataset_mod

        path = dataset_mod.build_corpus(settings, out_path=out, bases=list(bases) if bases else None)
    except _TRAIN_ERRORS as e:
        fail(f"module d'entraînement indisponible : {e}")
        return
    except (*_TRAIN_USER_ERRORS, KnowledgeError) as e:
        fail(f"corpus impossible : {e}")
        return
    success(f"Corpus écrit : {path}")


@train_app.command("nano")
def train_nano_cmd(
    ctx: typer.Context,
    corpus: Path = typer.Argument(..., help="Corpus texte.", exists=True, dir_okay=False, resolve_path=True),
    name: str = typer.Option("nano", "--name", help="Nom du modèle (dossier de sortie)."),
    steps: int | None = typer.Option(None, "--steps", min=1),
    n_layer: int | None = typer.Option(None, "--n-layer", min=1),
    n_head: int | None = typer.Option(None, "--n-head", min=1),
    n_embd: int | None = typer.Option(None, "--n-embd", min=1),
    block_size: int | None = typer.Option(None, "--block-size", min=1),
    batch_size: int | None = typer.Option(None, "--batch-size", min=1),
    lr: float | None = typer.Option(None, "--lr", help="Taux d'apprentissage."),
    force: bool = typer.Option(False, "--force", help="Écraser un modèle existant du même nom."),
) -> None:
    """Entraîner le modèle nano (GPT minimal, pédagogique) sur un corpus."""
    settings = _settings(ctx)
    overrides = {
        k: v
        for k, v in {
            "steps": steps, "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "block_size": block_size, "batch_size": batch_size, "learning_rate": lr,
        }.items()
        if v is not None
    }
    try:
        from ..train.nano import train as nano_train

        result = nano_train.train_nano(
            settings, corpus, name=name, overrides=overrides, on_log=_train_log, overwrite=force
        )
    except _TRAIN_ERRORS as e:
        fail(f"module nano indisponible : {e}")
        return
    except _TRAIN_USER_ERRORS as e:
        fail(f"entraînement impossible : {e}")
        return
    console.print(escape(str(result.summary())), highlight=False)
    success(f"Modèle enregistré dans {result.out_dir} (perte finale {result.final_loss:.4f})")


@train_app.command("sample")
def train_sample(
    ctx: typer.Context,
    model_dir: Path = typer.Argument(..., help="Dossier du modèle nano.", exists=True, file_okay=False, resolve_path=True),
    prompt: str = typer.Argument(..., help="Amorce."),
    max_new_tokens: int = typer.Option(200, "--max-new-tokens", min=1),
    temperature: float = typer.Option(0.8, "--temperature", min=0.0),
) -> None:
    """Générer du texte avec un modèle nano entraîné."""
    try:
        from ..train.nano import train as nano_train

        text = nano_train.sample(model_dir, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    except _TRAIN_ERRORS as e:
        fail(f"module nano indisponible : {e}")
        return
    except _TRAIN_USER_ERRORS as e:
        fail(f"génération impossible : {e}")
        return
    sys.stdout.write(str(text) + "\n")
    sys.stdout.flush()


@train_app.command("lora")
def train_lora(
    ctx: typer.Context,
    dataset: Path = typer.Argument(..., help="Jeu de données SFT (JSONL).", exists=True, dir_okay=False, resolve_path=True),
    out: Path | None = typer.Option(None, "--out", help="Dossier de sortie.", file_okay=False, resolve_path=True),
    base_model: str | None = typer.Option(None, "--base-model", help="Modèle de base HF."),
    epochs: float | None = typer.Option(None, "--epochs", min=0.0, help="Nombre d'époques (> 0)."),
) -> None:
    """Fine-tuning LoRA d'un modèle ouvert sur les traces (dépendances optionnelles)."""
    settings = _settings(ctx)
    if epochs is not None and epochs <= 0:
        fail("--epochs doit être strictement positif")
        return
    overrides = {k: v for k, v in {"base_model": base_model, "epochs": epochs}.items() if v is not None}
    try:
        from ..train import finetune

        available, reason = finetune.finetune_available()
        if not available:
            fail(f"fine-tuning indisponible : {reason}")
            return
        path = finetune.run_finetune(settings, dataset, out_dir=out, overrides=overrides, on_log=_train_log)
    except _TRAIN_ERRORS as e:
        fail(f"module de fine-tuning indisponible : {e}")
        return
    except _TRAIN_USER_ERRORS as e:
        fail(f"fine-tuning impossible : {e}")
        return
    success(f"Adaptateur LoRA enregistré dans {path}")


# ===================================================================== version
@app.command()
def version() -> None:
    """Afficher la version."""
    console.print(f"dhaos {__version__}", highlight=False)


def run() -> None:
    """Point d'entrée programmatique."""
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
