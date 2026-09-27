"""Tests de la CLI (typer.testing.CliRunner).

- Aucun réseau, aucun Ollama, aucune clé API : ``ask`` / ``chat`` remplacent
  ``dhaos.cli.main.build_runtime`` par une exécution factice.
- Les tests ``kb`` s'appuient sur ``dhaos.kb.manager`` (module d'un autre
  agent) : ils sont ignorés tant que ``KnowledgeManager`` lève
  ``NotImplementedError``.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from typer.testing import CliRunner

from dhaos.agent.loop import AgentResult
from dhaos.agent.session import SessionStore
from dhaos.cli import main, ui
from dhaos.cli.main import app
from dhaos.config import Settings
from dhaos.policy import Journal, auto_confirm
from dhaos.tools.base import ToolResult
from dhaos.types import Message, ToolCall, Usage

runner = CliRunner()


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Console large et sans couleur pour des sorties stables."""
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("TTY_COMPATIBLE", raising=False)


@pytest.fixture
def cfg(settings: Settings, tmp_path: Path) -> Path:
    """Fichier de configuration isolé (données et projet sous tmp_path)."""
    return settings.save(tmp_path / "config.toml")


Invoke = Callable[..., Any]


@pytest.fixture
def invoke(cfg: Path) -> Invoke:
    def _invoke(*args: str, input: str | None = None) -> Any:
        return runner.invoke(app, ["--config", str(cfg), *args], input=input)

    return _invoke


def _skip_if_unimplemented(result: Any) -> None:
    if isinstance(result.exception, NotImplementedError):
        pytest.skip("dhaos.kb.manager non implémenté (module d'un autre agent)")


@pytest.fixture
def kb_ready(settings: Settings) -> None:
    from dhaos.kb.manager import KnowledgeManager

    try:
        manager = KnowledgeManager(settings)
    except NotImplementedError:
        pytest.skip("KnowledgeManager non implémenté (module kb d'un autre agent)")
    try:
        manager.list_bases()
    except NotImplementedError:
        pytest.skip("KnowledgeManager.list_bases non implémenté")
    finally:
        try:
            manager.close()
        except Exception:  # noqa: BLE001
            pass


def fake_runtime_factory(
    result: AgentResult, calls: list[dict[str, Any]], *, with_tool: bool = True
) -> Callable[..., Any]:
    """Fabrique un ``build_runtime`` factice : enregistre les kwargs, simule un
    appel d'outil, diffuse le texte, ajoute les messages à la session."""

    def build(settings: Settings, **kwargs: Any) -> Any:
        calls.append(kwargs)
        session = kwargs.get("session")

        def run(text: str, *, on_text=None, on_thinking=None, on_tool_call=None, on_tool_result=None) -> AgentResult:
            if session is not None:
                session.append(Message(role="user", content=text))
            if with_tool:
                call = ToolCall(id="c1", name="read_file", arguments={"path": "a.py", "start_line": 1})
                if on_tool_call is not None:
                    on_tool_call(call)
                if on_tool_result is not None:
                    on_tool_result(call, ToolResult("contenu", is_error=False))
            if on_text is not None and result.text:
                on_text(result.text)
            if session is not None:
                session.append(Message(role="assistant", content=result.text))
                session.save()
            return result

        agent = SimpleNamespace(run=run, reset=lambda: None, session=session)
        backend = SimpleNamespace(name=kwargs.get("backend_name") or "fake", model=kwargs.get("model") or "fake-model")
        return SimpleNamespace(
            settings=settings,
            agent=agent,
            backend=backend,
            kb=None,
            registry=SimpleNamespace(names=["read_file", "grep"]),
            project_root=settings.resolve_project_root(),
            close=lambda: None,
        )

    return build


# ------------------------------------------------------------------- aide
@pytest.mark.parametrize(
    "args",
    [[], ["kb"], ["config"], ["sessions"], ["train"], ["chat"], ["ask"], ["kb", "search"], ["train", "nano"]],
)
def test_help_for_each_group(args: list[str]) -> None:
    result = runner.invoke(app, [*args, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
    assert "chat" in result.output and "kb" in result.output


def test_version(invoke: Invoke) -> None:
    from dhaos import __version__

    result = invoke("version")
    assert result.exit_code == 0
    assert f"dhaos {__version__}" in result.output


def test_project_option_must_exist(cfg: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(cfg), "--project", str(tmp_path / "absent"), "version"])
    assert result.exit_code != 0


# ------------------------------------------------------------------ config
def test_config_path(invoke: Invoke, cfg: Path) -> None:
    result = invoke("config", "path")
    assert result.exit_code == 0
    assert str(cfg) in result.output


def test_config_init_refuses_overwrite_without_force(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "fresh" / "config.toml"
    result = runner.invoke(app, ["--config", str(path), "config", "init"])
    assert result.exit_code == 0, result.output
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert "[backends]" in text and "[kb]" in text
    # Le fichier écrit est rechargeable tel quel.
    assert Settings.load(path, use_env=False).backends.default == "ollama"

    again = runner.invoke(app, ["--config", str(path), "config", "init"])
    assert again.exit_code == 1
    assert "existe déjà" in again.output

    forced = runner.invoke(app, ["--config", str(path), "config", "init", "--force"])
    assert forced.exit_code == 0, forced.output


def test_config_show_masks_secrets(settings: Settings, tmp_path: Path) -> None:
    settings.api.token = "secret-token-123"
    settings.web.brave_api_key = "brave-key-xyz"
    path = settings.save(tmp_path / "secret.toml")
    result = runner.invoke(app, ["--config", str(path), "config", "show"])
    assert result.exit_code == 0, result.output
    assert "[backends]" in result.output
    assert "secret-token-123" not in result.output
    assert "brave-key-xyz" not in result.output
    assert main.MASK in result.output
    # Le fichier n'est pas modifié par l'affichage.
    assert "secret-token-123" in path.read_text(encoding="utf-8")


def test_config_set_persists_value(invoke: Invoke, cfg: Path) -> None:
    result = invoke("config", "set", "backends.default", "claude")
    assert result.exit_code == 0, result.output
    assert "backends.default = claude" in result.output
    assert Settings.load(cfg, use_env=False).backends.default == "claude"

    result = invoke("config", "set", "agent.max_iterations", "12")
    assert result.exit_code == 0, result.output
    assert Settings.load(cfg, use_env=False).agent.max_iterations == 12


def test_config_set_rejects_unknown_key_and_bad_value(invoke: Invoke, cfg: Path) -> None:
    before = cfg.read_text(encoding="utf-8")
    result = invoke("config", "set", "backends.nope", "x")
    assert result.exit_code == 1
    assert "clé inconnue" in result.output
    result = invoke("config", "set", "backends.default", "gpt")
    assert result.exit_code == 1
    assert "valeur invalide" in result.output
    assert cfg.read_text(encoding="utf-8") == before


def test_config_set_rejects_max_iterations_below_one(invoke: Invoke, cfg: Path) -> None:
    """Régression : ``agent.max_iterations`` ≤ 0 exécutait quand même un tour d'outils."""
    before = cfg.read_text(encoding="utf-8")
    previous = Settings.load(cfg, use_env=False).agent.max_iterations
    result = invoke("config", "set", "agent.max_iterations", "0")
    assert result.exit_code == 1
    assert "valeur invalide pour agent.max_iterations" in result.output
    assert cfg.read_text(encoding="utf-8") == before
    assert Settings.load(cfg, use_env=False).agent.max_iterations == previous >= 1


def test_config_set_masks_secret_in_output(invoke: Invoke, cfg: Path) -> None:
    result = invoke("config", "set", "api.token", "tres-secret")
    assert result.exit_code == 0, result.output
    assert "tres-secret" not in result.output
    assert Settings.load(cfg, use_env=False).api.token == "tres-secret"


# ---------------------------------------------------------------------- kb
def test_kb_create_list_add_search_rename_delete(invoke: Invoke, kb_ready: None, tmp_path: Path) -> None:
    result = invoke("kb", "create", "devx", "--description", "Base développeur")
    _skip_if_unimplemented(result)
    assert result.exit_code == 0, result.output
    assert "devx" in result.output

    result = invoke("kb", "list")
    _skip_if_unimplemented(result)
    assert result.exit_code == 0, result.output
    assert "devx" in result.output and "Base développeur" in result.output

    doc = tmp_path / "django.txt"
    doc.write_text(
        "Le framework Django repose sur des vues, des modèles et des gabarits.\n"
        "Les migrations Django décrivent l'évolution du schéma de la base.\n",
        encoding="utf-8",
    )
    result = invoke("kb", "add", "devx", str(doc))
    _skip_if_unimplemented(result)
    assert result.exit_code == 0, result.output
    assert "ajouté" in result.output

    result = invoke("kb", "show", "devx")
    _skip_if_unimplemented(result)
    assert result.exit_code == 0, result.output
    assert "django.txt" in result.output

    result = invoke("kb", "search", "migrations Django", "--base", "devx", "--top-k", "3", "--mode", "hybrid")
    _skip_if_unimplemented(result)
    assert result.exit_code == 0, result.output
    assert "devx" in result.output and "score" in result.output

    result = invoke("kb", "stats", "devx")
    _skip_if_unimplemented(result)
    assert result.exit_code == 0, result.output

    result = invoke("kb", "describe", "devx", "Nouvelle description")
    _skip_if_unimplemented(result)
    assert result.exit_code == 0, result.output

    result = invoke("kb", "rename", "devx", "dev")
    _skip_if_unimplemented(result)
    assert result.exit_code == 0, result.output
    assert "dev" in invoke("kb", "list").output

    out = tmp_path / "export.jsonl"
    result = invoke("kb", "export", "dev", str(out))
    _skip_if_unimplemented(result)
    assert result.exit_code == 0, result.output
    assert out.is_file()

    # Refus de suppression : réponse « n » à la confirmation.
    result = invoke("kb", "delete", "dev", input="n\n")
    _skip_if_unimplemented(result)
    assert result.exit_code == 1
    assert "annulée" in result.output
    assert "dev" in invoke("kb", "list").output

    result = invoke("kb", "delete", "dev", "--yes")
    _skip_if_unimplemented(result)
    assert result.exit_code == 0, result.output
    listing = invoke("kb", "list")
    assert "Aucune base" in listing.output


def test_kb_errors_exit_1(invoke: Invoke, kb_ready: None) -> None:
    result = invoke("kb", "show", "inexistante")
    _skip_if_unimplemented(result)
    assert result.exit_code == 1
    assert "Erreur" in result.output

    result = invoke("kb", "delete", "inexistante", "--yes")
    _skip_if_unimplemented(result)
    assert result.exit_code == 1

    result = invoke("kb", "remove", "inexistante", "/nulle/part")
    _skip_if_unimplemented(result)
    assert result.exit_code == 1


def test_kb_list_when_manager_unavailable(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    """Un gestionnaire qui ne s'ouvre pas donne un message rouge, pas une trace."""
    from dhaos.kb import manager as manager_mod

    class Broken:
        def __init__(self, settings: Settings, **kwargs: Any) -> None:
            raise manager_mod.KnowledgeError("base corrompue")

    monkeypatch.setattr(manager_mod, "KnowledgeManager", Broken)
    result = invoke("kb", "list")
    assert result.exit_code == 1
    assert "base corrompue" in result.output


# --------------------------------------------------------------------- ask
def test_ask_streams_text_and_tool_lines(invoke: Invoke, monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    calls: list[dict[str, Any]] = []
    result_obj = AgentResult(text="Salut !", usage=Usage(12, 5), iterations=1, tool_calls=1)
    monkeypatch.setattr(main, "build_runtime", fake_runtime_factory(result_obj, calls))
    result = invoke("ask", "bonjour", "toi")
    assert result.exit_code == 0, result.output
    assert "Salut !" in result.output
    assert "⚙ read_file(path='a.py', start_line=1)" in result.output
    assert "↳ ok (7 car.)" in result.output
    assert len(calls) == 1
    assert calls[0]["backend_name"] is None and calls[0]["model"] is None
    assert calls[0]["tools"] is True
    assert calls[0]["confirm"] is ui.interactive_confirm
    # La session est persistée (le double y ajoute deux messages).
    infos = SessionStore(settings).list()
    assert len(infos) == 1 and infos[0].n_messages == 2


def test_ask_global_options_reach_runtime(cfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(main, "build_runtime", fake_runtime_factory(AgentResult(text="ok"), calls, with_tool=False))
    result = runner.invoke(
        app, ["--config", str(cfg), "--backend", "claude", "--model", "m1", "--yes", "--no-tools", "ask", "x"]
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["backend_name"] == "claude"
    assert calls[0]["model"] == "m1"
    assert calls[0]["tools"] is False
    assert calls[0]["confirm"] is auto_confirm


def test_ask_error_exits_1(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    failed = AgentResult(text="", stop_reason="error", error="Ollama injoignable")
    monkeypatch.setattr(main, "build_runtime", fake_runtime_factory(failed, calls, with_tool=False))
    result = invoke("ask", "bonjour")
    assert result.exit_code == 1
    assert "Ollama injoignable" in result.output


def test_ask_json(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    ok = AgentResult(text="réponse", usage=Usage(3, 4), iterations=2, tool_calls=1, stop_reason="end_turn")
    monkeypatch.setattr(main, "build_runtime", fake_runtime_factory(ok, calls))
    result = invoke("ask", "--json", "bonjour")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["text"] == "réponse"
    assert data["usage"] == {"input_tokens": 3, "output_tokens": 4}
    assert data["stop_reason"] == "end_turn"
    assert data["iterations"] == 2
    assert data["error"] is None


def test_ask_json_error(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    failed = AgentResult(text="", stop_reason="error", error="boum")
    monkeypatch.setattr(main, "build_runtime", fake_runtime_factory(failed, calls, with_tool=False))
    result = invoke("ask", "--json", "bonjour")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"] == "boum"


def test_ask_reads_stdin_when_no_text(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    calls: list[dict[str, Any]] = []
    factory = fake_runtime_factory(AgentResult(text="ok"), calls, with_tool=False)

    def build(settings: Settings, **kwargs: Any) -> Any:
        rt = factory(settings, **kwargs)
        inner = rt.agent.run

        def run(text: str, **kw: Any) -> AgentResult:
            seen.append(text)
            return inner(text, **kw)

        rt.agent.run = run
        return rt

    monkeypatch.setattr(main, "build_runtime", build)
    result = invoke("ask", input="question depuis stdin\n")
    assert result.exit_code == 0, result.output
    assert seen == ["question depuis stdin"]

    empty = invoke("ask", input="")
    assert empty.exit_code == 1
    assert "aucun texte" in empty.output


def test_ask_runtime_failure_is_reported(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(settings: Settings, **kwargs: Any) -> Any:
        raise RuntimeError("backend inconnu")

    monkeypatch.setattr(main, "build_runtime", boom)
    result = invoke("ask", "x")
    assert result.exit_code == 1
    assert "impossible de démarrer" in result.output and "backend inconnu" in result.output


# -------------------------------------------------------------------- chat
def test_chat_repl_commands_and_turn(invoke: Invoke, monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    calls: list[dict[str, Any]] = []
    reply = AgentResult(text="Bonjour, je suis dhaos.", usage=Usage(7, 9), iterations=1, tool_calls=1)
    monkeypatch.setattr(main, "build_runtime", fake_runtime_factory(reply, calls))
    script = "\n".join(
        ["bonjour", "/tools", "/kb", "/session", "/help", "/inconnue", "/backend gpt", "/backend claude", "/model m2", "/reset", "/quit"]
    ) + "\n"
    result = invoke("chat", input=script)
    assert result.exit_code == 0, result.output
    out = result.output
    assert "backend fake" in out and "0 base(s) de savoir" in out
    assert "vous>" in out
    assert "Bonjour, je suis dhaos." in out
    assert "⚙ read_file(" in out and "↳ ok" in out
    assert "usage : 7 jetons" in out
    assert "read_file, grep" in out
    assert "Bases de savoir désactivées" in out
    assert "commande inconnue" in out
    assert "backend inconnu : gpt" in out
    assert "Historique effacé" in out
    assert "Au revoir" in out
    # /backend claude puis /model m2 reconstruisent l'exécution (3 appels au total).
    assert [c["backend_name"] for c in calls] == [None, "claude", "claude"]
    assert calls[2]["model"] == "m2"
    # La session a été créée, affichée par /session et persistée avec ses messages.
    infos = SessionStore(settings).list()
    assert len(infos) == 1
    assert infos[0].id in out
    assert infos[0].n_messages == 2
    assert infos[0].backend == "claude" and infos[0].model == "m2"


def test_chat_eof_exits_cleanly_and_drops_empty_session(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(main, "build_runtime", fake_runtime_factory(AgentResult(text="x"), calls))
    result = invoke("chat", input="")
    assert result.exit_code == 0, result.output
    assert "Au revoir" in result.output
    assert SessionStore(settings).list() == []


def test_chat_resume_session(invoke: Invoke, monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    store = SessionStore(settings)
    session = store.create(title="ancienne", backend="ollama", model="q")
    session.append(Message(role="user", content="salut"))
    session.save()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(main, "build_runtime", fake_runtime_factory(AgentResult(text="x"), calls))
    result = invoke("chat", "--session", session.id, input="/session\n/quit\n")
    assert result.exit_code == 0, result.output
    assert session.id in result.output
    assert calls[0]["session"].id == session.id

    missing = invoke("chat", "--session", "inexistante-0000", input="/quit\n")
    assert missing.exit_code == 1
    assert "introuvable" in missing.output


def test_chat_backend_error_is_shown(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    failed = AgentResult(text="", stop_reason="error", error="modèle absent")
    monkeypatch.setattr(main, "build_runtime", fake_runtime_factory(failed, calls, with_tool=False))
    result = invoke("chat", input="bonjour\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "modèle absent" in result.output


# ----------------------------------------------------------------- journal
def test_journal_empty_then_entries(invoke: Invoke, settings: Settings) -> None:
    result = invoke("journal")
    assert result.exit_code == 0, result.output
    assert "Journal vide" in result.output

    journal = Journal(settings.journal_path)
    journal.record("write", path="/tmp/a.py", bytes=12)
    journal.record("command", command="pytest -q")
    journal.record("command", command="ls")
    result = invoke("journal", "--n", "2")
    assert result.exit_code == 0, result.output
    assert "command" in result.output and "pytest -q" in result.output
    assert "/tmp/a.py" not in result.output  # seules les 2 dernières entrées


# ---------------------------------------------------------------- sessions
def test_sessions_list_show_delete(invoke: Invoke, settings: Settings) -> None:
    result = invoke("sessions", "list")
    assert result.exit_code == 0, result.output
    assert "Aucune session" in result.output

    store = SessionStore(settings)
    session = store.create(title="Corriger le bug", backend="ollama", model="qwen")
    session.append(Message(role="user", content="corrige le bug [urgent]"))
    session.append(
        Message(role="assistant", content="je regarde", tool_calls=[ToolCall("c1", "grep", {"pattern": "bug"})])
    )
    session.append(Message(role="tool", content="a.py:3: bug", tool_call_id="c1", name="grep"))
    session.save()

    result = invoke("sessions", "list")
    assert result.exit_code == 0, result.output
    assert session.id in result.output and "Corriger le bug" in result.output

    result = invoke("sessions", "show", session.id)
    assert result.exit_code == 0, result.output
    assert "corrige le bug [urgent]" in result.output
    assert "⚙ grep(pattern='bug')" in result.output
    assert "tool:grep" in result.output

    result = invoke("sessions", "show", "../evil")
    assert result.exit_code == 1

    result = invoke("sessions", "delete", session.id)
    assert result.exit_code == 0, result.output
    assert not session.path.exists()
    result = invoke("sessions", "delete", session.id)
    assert result.exit_code == 1
    assert "introuvable" in result.output


# ---------------------------------------------------------------- backends
def test_backends_table_with_one_failing(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    class Good:
        name = "ollama"
        model = "qwen2.5-coder:7b"

        def healthcheck(self) -> dict[str, Any]:
            return {"ok": True, "backend": "ollama", "model": self.model, "detail": "joignable", "models": ["qwen2.5-coder:7b", "llama3"]}

    def fake_get_backend(settings: Settings, name: str | None = None, *, model: str | None = None) -> Any:
        if name == "ollama":
            return Good()
        raise RuntimeError("clé API absente")

    monkeypatch.setattr(main, "get_backend", fake_get_backend)
    result = invoke("backends")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "ollama (défaut)" in out and "ok" in out and "llama3" in out
    assert "claude" in out and "indisponible" in out and "clé API absente" in out


# ------------------------------------------------------------------- serve
def test_serve_calls_uvicorn(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    from dhaos.api import server

    seen: dict[str, Any] = {}
    monkeypatch.setattr(server, "create_app", lambda settings: "APPLICATION", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda application, host, port: seen.update(app=application, host=host, port=port))
    result = invoke("serve", "--port", "9999")
    assert result.exit_code == 0, result.output
    assert seen == {"app": "APPLICATION", "host": "127.0.0.1", "port": 9999}
    assert "http://127.0.0.1:9999" in result.output


def test_serve_reports_missing_api(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    from dhaos.api import server

    def not_ready(settings: Settings) -> Any:
        raise NotImplementedError("API à implémenter")

    monkeypatch.setattr(server, "create_app", not_ready, raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("uvicorn ne doit pas démarrer"))
    result = invoke("serve")
    assert result.exit_code == 1
    assert "API indisponible" in result.output


# ------------------------------------------------------------------- train
def test_train_dataset_and_corpus(invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dhaos.train import dataset as dataset_mod

    seen: dict[str, Any] = {}

    def build_sft_dataset(settings: Settings, out_path: Path | None = None, bases: list[str] | None = None) -> Any:
        seen["dataset"] = (out_path, bases)
        return SimpleNamespace(path=tmp_path / "sft.jsonl", n_examples=3, n_sessions_skipped=1, summary=lambda: "3 exemples SFT")

    def build_corpus(settings: Settings, out_path: Path | None = None, bases: list[str] | None = None) -> Path:
        seen["corpus"] = (out_path, bases)
        return tmp_path / "corpus.txt"

    monkeypatch.setattr(dataset_mod, "build_sft_dataset", build_sft_dataset, raising=False)
    monkeypatch.setattr(dataset_mod, "build_corpus", build_corpus, raising=False)

    result = invoke("train", "dataset")
    assert result.exit_code == 0, result.output
    assert "3 exemples SFT" in result.output and "sft.jsonl" in result.output
    assert seen["dataset"] == (None, None)

    out = tmp_path / "c.txt"
    result = invoke("train", "corpus", "--out", str(out))
    assert result.exit_code == 0, result.output
    assert seen["corpus"] == (out, None)
    assert "corpus.txt" in result.output


def test_train_unavailable_module(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    from dhaos.train import dataset as dataset_mod

    def not_ready(settings: Settings, out_path: Any = None, bases: Any = None) -> Any:
        raise NotImplementedError("en cours")

    monkeypatch.setattr(dataset_mod, "build_sft_dataset", not_ready, raising=False)
    result = invoke("train", "dataset")
    assert result.exit_code == 1
    assert "indisponible" in result.output


def test_train_nano_and_sample(invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dhaos.train.nano import train as nano_train

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("def f():\n    return 1\n", encoding="utf-8")
    seen: dict[str, Any] = {}

    def train_nano(
        settings: Settings, corpus_path: Path, *, name: str = "nano", overrides: dict, on_log=print, overwrite: bool = False
    ) -> Any:
        seen["nano"] = (corpus_path, name, overrides)
        on_log("step 1 loss 2.0")
        return SimpleNamespace(out_dir=tmp_path / "models" / name, final_loss=1.2345, summary=lambda: "entraînement terminé")

    def sample(model_dir: Path, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8) -> str:
        seen["sample"] = (model_dir, prompt, max_new_tokens, temperature)
        return "texte généré"

    monkeypatch.setattr(nano_train, "train_nano", train_nano, raising=False)
    monkeypatch.setattr(nano_train, "sample", sample, raising=False)

    result = invoke("train", "nano", str(corpus), "--name", "mini", "--steps", "5", "--n-layer", "2", "--lr", "0.001")
    assert result.exit_code == 0, result.output
    assert seen["nano"] == (corpus, "mini", {"steps": 5, "n_layer": 2, "learning_rate": 0.001})
    assert "step 1 loss 2.0" in result.output and "entraînement terminé" in result.output
    assert "1.2345" in result.output

    model_dir = tmp_path / "models" / "mini"
    model_dir.mkdir(parents=True)
    result = invoke("train", "sample", str(model_dir), "def g", "--max-new-tokens", "10", "--temperature", "0.5")
    assert result.exit_code == 0, result.output
    assert seen["sample"] == (model_dir, "def g", 10, 0.5)
    assert "texte généré" in result.output

    missing = invoke("train", "nano", str(tmp_path / "absent.txt"))
    assert missing.exit_code != 0


def test_train_lora(invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dhaos.train import finetune

    dataset = tmp_path / "sft.jsonl"
    dataset.write_text('{"messages": []}\n', encoding="utf-8")
    monkeypatch.setattr(finetune, "finetune_available", lambda: (False, "peft manquant"), raising=False)
    result = invoke("train", "lora", str(dataset))
    assert result.exit_code == 1
    assert "peft manquant" in result.output

    seen: dict[str, Any] = {}

    def run_finetune(settings: Settings, dataset_path: Path, *, out_dir: Path | None = None, overrides: dict, on_log=print) -> Path:
        seen["lora"] = (dataset_path, out_dir, overrides)
        return tmp_path / "lora-out"

    monkeypatch.setattr(finetune, "finetune_available", lambda: (True, ""), raising=False)
    monkeypatch.setattr(finetune, "run_finetune", run_finetune, raising=False)
    result = invoke("train", "lora", str(dataset), "--base-model", "Qwen/x", "--epochs", "2")
    assert result.exit_code == 0, result.output
    assert seen["lora"] == (dataset, None, {"base_model": "Qwen/x", "epochs": 2.0})
    assert "lora-out" in result.output


# ---------------------------------------------------------------------- ui
def test_ui_helpers() -> None:
    assert ui.shorten("  a   b\nc  ", 10) == "a b c"
    assert ui.shorten("x" * 20, 8).endswith("…") and len(ui.shorten("x" * 20, 8)) == 8
    assert ui.format_args({"path": "a.py", "n": 3, "l": [1, 2]}) == "path='a.py', n=3, l=[1, 2]"
    assert ui.format_args({}) == "" and ui.format_args("pas un dict") == ""
    assert ui.render_scalar({"a": 1}) == '{"a": 1}' and ui.render_scalar(None) == ""


def test_interactive_confirm_refuses_without_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    assert ui.stdin_is_interactive() is False
    assert ui.interactive_confirm("Écraser ?") is False


def test_ask_confirm_handles_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert ui.ask_confirm("Supprimer ?") is False
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    assert ui.ask_confirm("Supprimer ?") is True


# ------------------------------------------------------------- régressions
def test_config_set_ignores_env_and_cli_overrides(
    invoke: Invoke, cfg: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Régression : ``config set`` figeait dans le fichier les surcharges
    d'environnement (dont les secrets), ``--project`` et le ``data_dir``."""
    monkeypatch.setenv("DHAOS__API__TOKEN", "SUPERSECRET")
    monkeypatch.setenv("DHAOS__BACKENDS__DEFAULT", "claude")
    project = tmp_path / "proj"
    project.mkdir()
    result = invoke("--project", str(project), "config", "set", "kb.top_k", "5")
    assert result.exit_code == 0, result.output
    assert "kb.top_k = 5" in result.output
    saved = Settings.load(cfg, use_env=False)
    assert saved.kb.top_k == 5
    assert saved.backends.default == "ollama"
    assert saved.api.token is None
    assert saved.paths.project_root != project
    assert "SUPERSECRET" not in cfg.read_text(encoding="utf-8")

    # La clé modifiée reste surchargée par l'environnement : l'utilisateur est prévenu.
    result = invoke("config", "set", "backends.default", "ollama")
    assert result.exit_code == 0, result.output
    assert "DHAOS__BACKENDS__DEFAULT" in result.output

    # Sans fichier préexistant : seule la clé demandée est écrite (ni secret, ni data_dir absolu).
    monkeypatch.setenv("DHAOS_DATA_DIR", str(tmp_path / "otherdata"))
    fresh = tmp_path / "fresh" / "config.toml"
    result = runner.invoke(app, ["--config", str(fresh), "config", "set", "kb.top_k", "7"])
    assert result.exit_code == 0, result.output
    text = fresh.read_text(encoding="utf-8")
    assert "token =" not in text and "data_dir" not in text and "project_root" not in text
    assert Settings.load(fresh, use_env=False).kb.top_k == 7

    # ``null`` retire la clé du fichier au lieu d'y écrire une valeur vide.
    result = runner.invoke(app, ["--config", str(fresh), "config", "set", "kb.top_k", "null"])
    assert result.exit_code == 1  # top_k n'accepte pas null : validation pydantic
    result = invoke("config", "set", "api.token", "abc")
    assert result.exit_code == 0, result.output
    assert Settings.load(cfg, use_env=False).api.token == "abc"
    result = invoke("config", "set", "api.token", "null")
    assert result.exit_code == 0, result.output
    assert Settings.load(cfg, use_env=False).api.token is None
    assert not any(line.startswith("token =") for line in cfg.read_text(encoding="utf-8").splitlines())


def test_config_show_output_is_valid_toml_with_long_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Régression : la ligne ``# fichier : …`` était repliée par rich hors TTY
    (80 colonnes), rendant la sortie redirigée non analysable."""
    import tomllib

    monkeypatch.setenv("COLUMNS", "80")
    path = tmp_path / ("x" * 90) / "config.toml"
    assert len(str(path)) > 80
    result = runner.invoke(app, ["--config", str(path), "config", "init"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["--config", str(path), "config", "show"])
    assert result.exit_code == 0, result.output
    first = result.stdout.splitlines()[0]
    assert first == f"# fichier : {path}"
    data = tomllib.loads(result.stdout)
    assert data["backends"]["default"] == "ollama"


def test_kb_export_unwritable_parent(invoke: Invoke, kb_ready: None, tmp_path: Path) -> None:
    """Régression : une ``OSError`` à l'export (parent non créable) sortait en trace."""
    result = invoke("kb", "create", "t1")
    assert result.exit_code == 0, result.output
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    result = invoke("kb", "export", "t1", str(blocker / "out.jsonl"))
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Erreur :" in result.output and "export impossible" in result.output
    assert "Traceback" not in result.output


def _assert_clean_failure(result: Any, *fragments: str) -> None:
    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Erreur :" in result.output and "Traceback" not in result.output
    for fragment in fragments:
        assert fragment in result.output


def test_train_nano_and_sample_user_errors_are_clean(
    invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Régression : ``FileNotFoundError`` / ``ValueError`` du module nano sortaient en trace."""
    pytest.importorskip("torch")
    empty_dir = tmp_path / "empty-model"
    empty_dir.mkdir()
    _assert_clean_failure(invoke("train", "sample", str(empty_dir), "salut"), "config.json manquant")

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    _assert_clean_failure(invoke("train", "nano", str(empty), "--name", "t"), "corpus vide")

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("def f():\n    return 1\n" * 20, encoding="utf-8")
    _assert_clean_failure(invoke("train", "nano", str(corpus), "--n-embd", "6", "--n-head", "4"), "divisible par n_head")
    _assert_clean_failure(invoke("train", "nano", str(corpus), "--lr", "-1"), "learning_rate")
    _assert_clean_failure(invoke("train", "nano", str(corpus), "--name", "../x"), "nom de modèle invalide")

    monkeypatch.setenv("DHAOS__TRAIN__NANO__BPE_VOCAB_SIZE", "100")
    _assert_clean_failure(invoke("train", "nano", str(corpus), "--steps", "1"), "vocab_size")


def test_train_dataset_and_corpus_user_errors_are_clean(
    invoke: Invoke, kb_ready: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dhaos.train import dataset as dataset_mod

    def broken(settings: Settings, out_path: Any = None) -> Any:
        raise OSError("disque plein")

    monkeypatch.setattr(dataset_mod, "build_sft_dataset", broken, raising=False)
    _assert_clean_failure(invoke("train", "dataset"), "disque plein")
    _assert_clean_failure(invoke("train", "corpus", "--base", "base_inconnue"), "base inconnue")


def test_train_dataset_base_option_warns(invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Régression : ``--base`` était documenté mais ignoré en silence par ``train dataset``."""
    from dhaos.train import dataset as dataset_mod

    def build_sft_dataset(settings: Settings, out_path: Path | None = None, bases: list[str] | None = None) -> Any:
        return SimpleNamespace(path=tmp_path / "sft.jsonl", n_examples=0, n_sessions_skipped=0, summary=lambda: "vide")

    monkeypatch.setattr(dataset_mod, "build_sft_dataset", build_sft_dataset, raising=False)
    result = invoke("train", "dataset", "--base", "devx")
    assert result.exit_code == 0, result.output
    assert "sans effet" in result.output
    result = invoke("train", "dataset")
    assert result.exit_code == 0, result.output
    assert "sans effet" not in result.output
    result = invoke("train", "dataset", "--help")
    assert "--base" not in result.output


def test_train_lora_user_errors_are_clean(invoke: Invoke, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Régression : ``ValueError`` / ``RuntimeError`` de ``run_finetune`` sortaient en
    trace, et ``--epochs 0`` passait la validation Typer."""
    from dhaos.train import finetune

    monkeypatch.setattr(finetune, "finetune_available", lambda: (True, ""), raising=False)
    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"messages": []}\n', encoding="utf-8")
    _assert_clean_failure(invoke("train", "lora", str(empty)), "aucun exemple avec une réponse assistant")

    dataset = tmp_path / "ok.jsonl"
    dataset.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "r"}]}) + "\n",
        encoding="utf-8",
    )
    result = invoke("train", "lora", str(dataset), "--epochs", "0")
    assert result.exit_code != 0
    assert not isinstance(result.exception, ValueError)
    assert "--epochs doit être strictement positif" in result.output

    def boom(settings: Settings, dataset_path: Path, *, out_dir: Path | None = None, overrides: dict, on_log=print) -> Path:
        raise RuntimeError("fine-tuning indisponible : x")

    monkeypatch.setattr(finetune, "run_finetune", boom, raising=False)
    result = invoke("train", "lora", str(dataset))
    _assert_clean_failure(result, "indisponible : x")
    assert not isinstance(result.exception, RuntimeError)


# ============================================================================ ui
def _read_cfg(cfg: Path) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(cfg.read_text(encoding="utf-8"))


def test_ui_generates_token_and_opens_browser(invoke: Invoke, cfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serveur déjà en place et jeton accepté : le jeton est enregistré dans la
    configuration et le navigateur reçoit l'URL avec ce jeton."""
    import webbrowser

    from dhaos.cli import main as cli

    opened: list[str] = []
    monkeypatch.setattr(cli, "_api_reachable", lambda host, port, timeout=1.5: True)
    monkeypatch.setattr(cli, "_token_accepted", lambda host, port, token, timeout=3.0: True)
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    assert "token" not in _read_cfg(cfg).get("api", {})
    r = invoke("ui")
    assert r.exit_code == 0, r.output
    token = _read_cfg(cfg)["api"]["token"]
    assert len(token) > 20 and opened == [f"http://127.0.0.1:8642/?token={token}"]
    assert "serveur déjà en place" in r.output
    # second appel : même jeton, pas de régénération
    invoke("ui")
    assert _read_cfg(cfg)["api"]["token"] == token and opened[-1].endswith(token)


def test_ui_starts_server_when_down(invoke: Invoke, cfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from dhaos.cli import main as cli

    calls = iter([False, True])
    launched: list[list[str]] = []
    monkeypatch.setattr(cli, "_api_reachable", lambda host, port, timeout=1.5: next(calls, True))
    monkeypatch.setattr(cli, "_token_accepted", lambda host, port, token, timeout=3.0: True)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: launched.append(list(cmd)))
    r = invoke("ui", "--no-browser", "--port", "8700")
    assert r.exit_code == 0, r.output
    assert len(launched) == 1
    cmd = launched[0]
    assert cmd[1:3] == ["-m", "dhaos.cli"] and "serve" in cmd and cmd[cmd.index("--port") + 1] == "8700"
    assert cmd[cmd.index("--config") + 1] == str(cfg)
    assert "lancé en arrière-plan" in r.output and "http://127.0.0.1:8700/?token=" in r.output


def test_ui_refuses_server_with_other_token(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    from dhaos.cli import main as cli

    monkeypatch.setattr(cli, "_api_reachable", lambda host, port, timeout=1.5: True)
    monkeypatch.setattr(cli, "_token_accepted", lambda host, port, token, timeout=3.0: False)
    r = invoke("ui", "--no-browser")
    assert r.exit_code == 1 and "autre jeton" in r.output


# ========================================================================= model
class _FakeOllama:
    def __init__(self, models: list[str]) -> None:
        self.models = models
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def list_models(self) -> list[str]:
        return list(self.models)

    def create_model(self, name: str, base: str, *, system: str = "", parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        self.created.append({"name": name, "base": base, "system": system, "parameters": parameters or {}})
        self.models.append(f"{name}:latest")
        return {"status": "success"}

    def show_model(self, name: str) -> dict[str, Any]:
        return {"details": {"family": "qwen2", "parameter_size": "7.6B"}, "parameters": "num_ctx 16384", "system": "Tu es dhaos"}

    def delete_model(self, name: str) -> None:
        self.deleted.append(name)


def test_model_create_sets_default(invoke: Invoke, cfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dhaos.cli import main as cli

    fake = _FakeOllama(["qwen2.5-coder:7b"])
    monkeypatch.setattr(cli, "_ollama_backend", lambda ctx: (cli._settings(ctx), fake))
    r = invoke("model", "create")
    assert r.exit_code == 0, r.output
    assert fake.created[0]["base"] == "qwen2.5-coder:7b" and fake.created[0]["name"] == "dhaos"
    assert "Tu es dhaos" in fake.created[0]["system"] and fake.created[0]["parameters"]["num_ctx"] == 16384
    stored = _read_cfg(cfg)
    assert stored["backends"]["ollama"]["model"] == "dhaos" and stored["backends"]["ollama"]["base_model"] == "qwen2.5-coder:7b"
    r = invoke("model", "list")
    assert r.exit_code == 0 and "dhaos:latest (défaut dhaos)" in r.output
    assert invoke("model", "show").exit_code == 0
    assert invoke("model", "remove", "dhaos").exit_code == 0 and fake.deleted == ["dhaos"]


def test_model_create_requires_base(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    from dhaos.cli import main as cli

    fake = _FakeOllama([])
    monkeypatch.setattr(cli, "_ollama_backend", lambda ctx: (cli._settings(ctx), fake))
    r = invoke("model", "create", "--base", "qwen2.5-coder:3b")
    assert r.exit_code == 1 and "ollama pull qwen2.5-coder:3b" in r.output and not fake.created


def test_model_import_command(invoke: Invoke, cfg: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dhaos.cli import main as cli

    class FakeImporter(_FakeOllama):
        def __init__(self) -> None:
            super().__init__([])
            self.imported: list[tuple[Path, str]] = []

        def import_modelfile(self, modelfile: Path, name: str = "dhaos", *, on_log: Any = None) -> dict[str, Any]:
            if on_log:
                on_log("téléversement…")
            self.imported.append((modelfile, name))
            return {"status": "success"}

    fake = FakeImporter()
    monkeypatch.setattr(cli, "_ollama_backend", lambda ctx: (cli._settings(ctx), fake))
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM dhaos.gguf\n", encoding="utf-8")
    r = invoke("model", "import", "--modelfile", str(modelfile))
    assert r.exit_code == 0, r.output
    assert fake.imported == [(modelfile, "dhaos")] and _read_cfg(cfg)["backends"]["ollama"]["model"] == "dhaos"
    r = invoke("model", "import", "--modelfile", str(tmp_path / "absent"))
    assert r.exit_code == 1 and "introuvable" in r.output
