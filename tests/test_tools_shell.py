"""Tests de l'outil run_command via ToolRegistry.execute."""
from __future__ import annotations

from pathlib import Path

import pytest

from dhaos.config import Settings
from dhaos.policy import AccessPolicy, Journal, auto_confirm
from dhaos.tools import shell
from dhaos.tools.base import ToolContext, ToolRegistry, ToolResult

from .fakes import tool_call


class Confirm:
    def __init__(self, answer: bool):
        self.answer = answer
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.answer


def make_ctx(settings: Settings, confirm=auto_confirm) -> ToolContext:
    return ToolContext(
        settings=settings,
        policy=AccessPolicy(settings),
        journal=Journal(settings.journal_path),
        confirm=confirm,
    )


@pytest.fixture
def registry(settings: Settings) -> ToolRegistry:
    return ToolRegistry(shell.tools(settings))


def run(registry: ToolRegistry, ctx: ToolContext, **args) -> ToolResult:
    return registry.execute(tool_call("run_command", **args), ctx)


def test_tool_spec(settings: Settings) -> None:
    tools = shell.tools(settings)
    assert [t.name for t in tools] == ["run_command"]
    assert tools[0].may_require_confirmation and tools[0].description
    assert tools[0].spec().parameters["required"] == ["command"]


@pytest.mark.parametrize("args", [{}, {"command": ""}, {"command": "ls", "timeout": 0}, {"command": "ls", "timeout": "5"}])
def test_schema_validation(registry: ToolRegistry, ctx: ToolContext, args: dict) -> None:
    r = registry.execute(tool_call("run_command", **args), ctx)
    assert r.is_error and "INVALID_JSON" in r.content


def test_echo(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    r = run(registry, ctx, command="echo bonjour; echo erreur >&2")
    assert not r.is_error
    assert r.content.startswith("exit code 0\n--- stdout ---\nbonjour\n")
    assert "--- stderr ---\nerreur" in r.content
    assert r.data["exit"] == 0
    entries = ctx.journal.tail()
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "run_command" and e["command"] == "echo bonjour; echo erreur >&2"
    assert e["cwd"] == str(project_root) and e["exit"] == 0 and e["confirmed"] is False
    assert isinstance(e["duration"], float)


def test_nonzero_exit(registry: ToolRegistry, ctx: ToolContext) -> None:
    r = run(registry, ctx, command="exit 3")
    assert not r.is_error and r.content.startswith("exit code 3\n")
    assert "(vide)" in r.content
    assert ctx.journal.tail()[0]["exit"] == 3


def test_default_cwd_is_project(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    r = run(registry, ctx, command="pwd")
    assert r.content.splitlines()[2] == str(project_root)


def test_cwd_argument(registry: ToolRegistry, ctx: ToolContext, project_root: Path, tmp_path: Path) -> None:
    (project_root / "sub").mkdir()
    r = run(registry, ctx, command="pwd", cwd="sub")
    assert r.content.splitlines()[2] == str(project_root / "sub")
    r = run(registry, ctx, command="pwd", cwd=str(tmp_path))
    assert r.content.splitlines()[2] == str(tmp_path)
    assert ctx.journal.tail(1)[0]["cwd"] == str(tmp_path)


def test_cwd_missing_or_file(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    r = run(registry, ctx, command="pwd", cwd="absent")
    assert r.is_error and "introuvable" in r.content
    (project_root / "f.txt").write_text("x", encoding="utf-8")
    r = run(registry, ctx, command="pwd", cwd="f.txt")
    assert r.is_error and "introuvable" in r.content
    assert ctx.journal.tail() == []


def test_cwd_denied(settings: Settings, registry: ToolRegistry, tmp_path: Path) -> None:
    settings.tools.read_roots = [settings.resolve_project_root()]
    ctx = make_ctx(settings)
    r = run(registry, ctx, command="pwd", cwd=str(tmp_path))
    assert r.is_error and "commande refusée" in r.content and "hors des racines" in r.content
    r = run(registry, ctx, command="pwd", cwd="/proc/self")
    assert r.is_error and "protégé" in r.content


def test_stdin_closed(registry: ToolRegistry, ctx: ToolContext) -> None:
    r = run(registry, ctx, command="cat; echo fin")
    assert r.content.startswith("exit code 0\n--- stdout ---\nfin\n")


def test_timeout(settings: Settings, registry: ToolRegistry, ctx: ToolContext) -> None:
    r = run(registry, ctx, command="echo partiel; sleep 5", timeout=0.3)
    assert r.is_error and "délai dépassé (0.3s)" in r.content
    assert "partiel" in r.content
    e = ctx.journal.tail()[0]
    assert e["kind"] == "run_command" and e["exit"] is None and e["timeout"] == 0.3


def test_timeout_capped_by_settings(settings: Settings, registry: ToolRegistry, ctx: ToolContext) -> None:
    settings.tools.command_timeout = 0.3
    r = run(registry, ctx, command="sleep 5", timeout=60)
    assert r.is_error and "délai dépassé (0.3s)" in r.content
    r = run(registry, ctx, command="sleep 5")
    assert r.is_error and "délai dépassé" in r.content


def test_timeout_kills_process_group(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    import time as _time

    marker = project_root / "marker"
    r = run(registry, ctx, command=f"(sleep 1; touch {marker}) | cat", timeout=0.3)
    assert r.is_error
    _time.sleep(1.2)
    assert not marker.exists()


def test_env_scrubbed(monkeypatch: pytest.MonkeyPatch, registry: ToolRegistry, ctx: ToolContext) -> None:
    monkeypatch.setenv("MY_API_KEY", "k")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("APP_SECRET", "s")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.setenv("DHAOS_PLAIN_VAR", "visible")
    r = run(registry, ctx, command="env")
    assert "DHAOS_PLAIN_VAR=visible" in r.content
    for name in ("MY_API_KEY", "GITHUB_TOKEN", "APP_SECRET", "DB_PASSWORD"):
        assert name not in r.content
    assert "PATH=" in r.content


def test_scrub_env_function() -> None:
    env = shell.scrub_env({"HOME": "/h", "PATH": "/bin", "AWS_SECRET_ACCESS_KEY": "x", "api_token": "y", "passwd_file": "z"})
    assert env == {"HOME": "/h", "PATH": "/bin"}


def test_output_truncated(settings: Settings, registry: ToolRegistry, ctx: ToolContext) -> None:
    settings.tools.max_output_chars = 4000
    r = run(registry, ctx, command="yes a | head -n 20000; yes b | head -n 20000 >&2")
    assert not r.is_error
    assert r.content.count("tronqué") >= 2  # stdout et stderr tronqués séparément
    assert "--- stderr ---" in r.content and "b\nb\n" in r.content
    assert len(r.content) <= 4100


def test_shell_policy_deny(settings: Settings, registry: ToolRegistry, ctx: ToolContext) -> None:
    settings.tools.shell_policy = "deny"
    r = run(registry, ctx, command="ls")
    assert r.is_error and r.content.startswith("commande refusée") and "deny" in r.content
    assert ctx.journal.tail() == []


def test_shell_policy_ask_auto_allowed(settings: Settings, registry: ToolRegistry, project_root: Path) -> None:
    settings.tools.shell_policy = "ask"
    confirm = Confirm(False)  # ne doit pas être sollicité
    ctx = make_ctx(settings, confirm)
    (project_root / "a.txt").write_text("x", encoding="utf-8")
    r = run(registry, ctx, command="ls")
    assert not r.is_error and "a.txt" in r.content
    assert confirm.prompts == []
    assert ctx.journal.tail()[0]["confirmed"] is False


def test_shell_policy_ask_refused(settings: Settings, registry: ToolRegistry, project_root: Path) -> None:
    settings.tools.shell_policy = "ask"
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    r = run(registry, ctx, command="touch created.txt")
    assert r.is_error and r.content.startswith("commande refusée")
    assert confirm.prompts == ["Exécuter : touch created.txt ? "]
    assert not (project_root / "created.txt").exists()
    assert ctx.journal.tail() == []


def test_shell_policy_ask_accepted(settings: Settings, registry: ToolRegistry, project_root: Path) -> None:
    settings.tools.shell_policy = "ask"
    confirm = Confirm(True)
    ctx = make_ctx(settings, confirm)
    r = run(registry, ctx, command="touch created.txt")
    assert not r.is_error and (project_root / "created.txt").exists()
    assert len(confirm.prompts) == 1
    assert ctx.journal.tail()[0]["confirmed"] is True


def test_shell_policy_ask_metacharacters_need_confirmation(settings: Settings, registry: ToolRegistry) -> None:
    settings.tools.shell_policy = "ask"
    confirm = Confirm(True)
    ctx = make_ctx(settings, confirm)
    r = run(registry, ctx, command="ls | wc -l")
    assert not r.is_error and len(confirm.prompts) == 1


def test_confirm_exception_is_refusal(settings: Settings, registry: ToolRegistry) -> None:
    settings.tools.shell_policy = "ask"

    def boom(prompt: str) -> bool:
        raise EOFError

    ctx = make_ctx(settings, boom)
    r = run(registry, ctx, command="touch x")
    assert r.is_error and "commande refusée" in r.content


def test_invalid_utf8_output_replaced(registry: ToolRegistry, ctx: ToolContext) -> None:
    r = run(registry, ctx, command="printf 'ok\\xff\\n'")
    assert not r.is_error and "ok�" in r.content
