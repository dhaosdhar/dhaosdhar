"""Tests de l'outil run_command via ToolRegistry.execute."""
from __future__ import annotations

import os
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


# Régression : `find -delete` / `find -exec` en liste blanche supprimaient ou
# lisaient des fichiers (même protégés par deny_patterns) sans confirmation.
def test_find_delete_needs_confirmation(settings: Settings, registry: ToolRegistry, project_root: Path) -> None:
    settings.tools.shell_policy = "ask"
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    victim = project_root / "victim.txt"
    victim.write_text("x", encoding="utf-8")
    r = run(registry, ctx, command="find . -name victim.txt -delete")
    assert r.is_error and "confirmation refusée" in r.content
    assert len(confirm.prompts) == 1
    assert victim.exists()


def test_find_exec_on_protected_file_needs_confirmation(
    settings: Settings, registry: ToolRegistry, tmp_path: Path
) -> None:
    settings.tools.shell_policy = "ask"
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    home = tmp_path / "home"
    conf_dir = home / ".config" / "dhaos"
    conf_dir.mkdir(parents=True)
    (conf_dir / "config.toml").write_text('brave_api_key = "SENTINEL-B2"\n', encoding="utf-8")
    assert not ctx.policy.check_read(conf_dir / "config.toml").allowed
    r = run(registry, ctx, command=f"find {conf_dir} -maxdepth 1 -name config.toml -exec cat {{}} +")
    assert r.is_error and "confirmation refusée" in r.content
    assert "SENTINEL-B2" not in r.content
    assert len(confirm.prompts) == 1


# Régression : les options d'écriture des commandes en liste blanche (sort -o,
# git log --output, find -fprint) écrasaient des fichiers hors projet sans
# confirmation, contournant write_policy = project.
def test_sort_output_outside_project_needs_confirmation(
    settings: Settings, registry: ToolRegistry, project_root: Path, tmp_path: Path
) -> None:
    settings.tools.shell_policy = "ask"
    settings.tools.write_policy = "project"
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    target = tmp_path / "home" / "notes.txt"
    target.write_text("ORIGINAL", encoding="utf-8")
    src = project_root / "in.txt"
    src.write_text("b\na\n", encoding="utf-8")
    assert ctx.policy.check_write(target).needs_confirmation
    r = run(registry, ctx, command=f"sort -o {target} {src}")
    assert r.is_error and "confirmation refusée" in r.content
    assert len(confirm.prompts) == 1
    assert target.read_text(encoding="utf-8") == "ORIGINAL"
    # Sans option d'écriture, sort reste auto-autorisé.
    r = run(registry, ctx, command=f"sort {src}")
    assert not r.is_error and "a\nb\n" in r.content
    assert len(confirm.prompts) == 1


def test_git_log_output_outside_project_needs_confirmation(
    settings: Settings, registry: ToolRegistry, project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git absent")
    for key in list(os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key, raising=False)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(tmp_path / "home")}
    subprocess.run(["git", "init", "-q"], cwd=project_root, env=env, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "SUBJECT-SENTINEL"], cwd=project_root, env=env, check=True)
    settings.tools.shell_policy = "ask"
    settings.tools.write_policy = "project"
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    target = tmp_path / "home" / "notes.txt"
    target.write_text("ORIGINAL", encoding="utf-8")
    r = run(registry, ctx, command=f"git log --output={target} --format=%s")
    assert r.is_error and "confirmation refusée" in r.content
    assert len(confirm.prompts) == 1
    assert target.read_text(encoding="utf-8") == "ORIGINAL"
    r = run(registry, ctx, command="git log --oneline")
    assert not r.is_error and "SUBJECT-SENTINEL" in r.content
    assert len(confirm.prompts) == 1


def test_find_fprint_outside_project_needs_confirmation(
    settings: Settings, registry: ToolRegistry, project_root: Path, tmp_path: Path
) -> None:
    settings.tools.shell_policy = "ask"
    settings.tools.write_policy = "project"
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    target = tmp_path / "home" / "created_by_find.txt"
    r = run(registry, ctx, command=f"find {project_root} -maxdepth 0 -fprint {target}")
    assert r.is_error and "confirmation refusée" in r.content
    assert len(confirm.prompts) == 1
    assert not target.exists()


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


# Régression : les commandes en liste blanche (cat, head, tail, grep) lisaient
# les fichiers protégés (deny_patterns) que read_file / grep refusent, sans
# confirmation — y compris via l'API (never_confirm).
@pytest.fixture
def secrets(tmp_path: Path, project_root: Path) -> dict[str, Path]:
    home = tmp_path / "home"
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("PRIVATE-KEY-SENTINEL-A1\n", encoding="utf-8")
    (home / ".config" / "dhaos").mkdir(parents=True)
    (home / ".config" / "dhaos" / "config.toml").write_text('brave_api_key = "SENTINEL-B2"\n', encoding="utf-8")
    (project_root / ".env").write_text("SECRET=SENTINEL-D4\n", encoding="utf-8")
    return {"home": home, "project": project_root}


@pytest.mark.parametrize(
    "template",
    ["cat ~/.ssh/id_rsa", "head -c 40 ~/.ssh/id_rsa", "grep -r SENTINEL {home}/.config/dhaos",
     "cat /proc/self/status", "tail {project}/.env", "tail .env"],
)
def test_whitelisted_commands_cannot_read_protected_paths(
    settings: Settings, registry: ToolRegistry, secrets: dict[str, Path], template: str
) -> None:
    settings.tools.shell_policy = "ask"
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    command = template.format(**secrets)
    r = run(registry, ctx, command=command)
    assert r.is_error and r.content.startswith("commande refusée") and "protégé" in r.content
    assert confirm.prompts == [f"Exécuter : {command} ? "]
    for sentinel in ("SENTINEL-A1", "SENTINEL-B2", "SENTINEL-D4"):
        assert sentinel not in r.content
    assert ctx.journal.tail() == []


def test_whitelisted_read_via_api_never_confirm_is_refused(
    settings: Settings, registry: ToolRegistry, secrets: dict[str, Path]
) -> None:
    from dhaos.policy import never_confirm

    settings.tools.shell_policy = "ask"
    ctx = make_ctx(settings, never_confirm)
    r = run(registry, ctx, command="cat ~/.ssh/id_rsa")
    assert r.is_error and "SENTINEL-A1" not in r.content
    r = run(registry, ctx, command="cat README.md")
    assert not r.is_error  # lecture ordinaire du projet toujours auto-autorisée


def test_relative_protected_path_resolved_from_cwd(
    settings: Settings, registry: ToolRegistry, secrets: dict[str, Path]
) -> None:
    settings.tools.shell_policy = "ask"
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    r = run(registry, ctx, command="cat id_rsa", cwd=str(secrets["home"] / ".ssh"))
    assert r.is_error and "protégé" in r.content and "SENTINEL-A1" not in r.content
    assert len(confirm.prompts) == 1


# Régression : scrub_env ne filtrait que par fragment de nom ; DATABASE_URL
# (user:mdp@hôte), SSH_AUTH_SOCK, GPG_AGENT_INFO passaient et `printenv`
# (liste blanche) les livrait au modèle sans confirmation.
def test_scrub_env_hides_urls_with_credentials_and_agent_sockets() -> None:
    env = shell.scrub_env({
        "DATABASE_URL": "postgres://u:p@h/db", "REDIS_URL": "redis://:p@h", "PG_DSN": "postgresql://user:pw@db:5432/x",
        "SSH_AUTH_SOCK": "/s", "SSH_AGENT_PID": "12", "GPG_AGENT_INFO": "/g", "ssh_auth_sock": "/s2",
        "PATH": "/bin", "HOME": "/h", "API_URL": "https://example.org/api", "PUBLIC_URL": "https://u@h/",
    })
    assert env == {"PATH": "/bin", "HOME": "/h", "API_URL": "https://example.org/api", "PUBLIC_URL": "https://u@h/"}
    assert shell.is_sensitive_env("DATABASE_URL", "mysql://root:root@localhost/app")
    assert not shell.is_sensitive_env("DATABASE_URL", "sqlite:///tmp/app.db")


def test_printenv_needs_confirmation_and_env_secrets_hidden(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, registry: ToolRegistry
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgres://u:s3cr3t@h/db")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/user/1000/agent.sock")
    monkeypatch.setenv("DHAOS_PLAIN_VAR", "visible")
    settings.tools.shell_policy = "ask"
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    for command in ("printenv", "printenv DATABASE_URL", "env"):
        r = run(registry, ctx, command=command)
        assert r.is_error and r.content.startswith("commande refusée")
        assert "s3cr3t" not in r.content and "agent.sock" not in r.content
    assert len(confirm.prompts) == 3
    # Même confirmée (ou en shell_policy = auto), l'environnement reste expurgé.
    settings.tools.shell_policy = "auto"
    r = run(registry, make_ctx(settings), command="printenv")
    assert not r.is_error and "DHAOS_PLAIN_VAR=visible" in r.content
    assert "s3cr3t" not in r.content and "DATABASE_URL" not in r.content and "SSH_AUTH_SOCK" not in r.content


# Régression : le schéma de run_command acceptait des propriétés inconnues.
def test_run_command_rejects_extra_args(registry: ToolRegistry, ctx: ToolContext) -> None:
    r = registry.execute(tool_call("run_command", command="echo ok", cmd="rm -rf /"), ctx)
    assert r.is_error and "INVALID_JSON" in r.content and "Additional properties" in r.content
    assert ctx.journal.tail() == []
    assert shell.tools(ctx.settings)[0].parameters["additionalProperties"] is False
