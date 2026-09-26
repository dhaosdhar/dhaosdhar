"""Tests de la politique d'accès (AccessPolicy) et du journal (Journal)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dhaos.config import Settings
from dhaos.policy import AccessPolicy, Journal, auto_confirm, glob_to_regex, never_confirm


# ---------------------------------------------------------------- resolve


def test_resolve_relative_is_relative_to_project(policy: AccessPolicy, project_root: Path) -> None:
    assert policy.resolve("src/main.py") == (project_root / "src" / "main.py").resolve()
    assert policy.resolve(".") == project_root
    assert policy.resolve("sub/../x.txt") == project_root / "x.txt"


def test_resolve_absolute_and_home(policy: AccessPolicy, tmp_path: Path) -> None:
    absolute = tmp_path / "elsewhere" / "f.txt"
    assert policy.resolve(str(absolute)) == absolute.resolve()
    home = Path(os.environ["HOME"])
    assert policy.resolve("~/notes.md") == (home / "notes.md").resolve()
    assert policy.resolve(Path("~")) == home.resolve()


def test_is_inside_project(policy: AccessPolicy, project_root: Path, tmp_path: Path) -> None:
    assert policy.is_inside_project("a/b.txt")
    assert policy.is_inside_project(project_root)
    assert not policy.is_inside_project(tmp_path / "other.txt")
    # Un "../" qui sort du projet est bien détecté après résolution.
    assert not policy.is_inside_project("../escape.txt")


# ------------------------------------------------------------ deny (name)


@pytest.mark.parametrize(
    "rel",
    [".env", ".env.local", "config/.env.production", "certs/server.pem", "keys/private.key",
     "id_rsa", ".ssh/id_ed25519", "gcp/credentials.json", "service-account-prod.json", "vault.kdbx"],
)
def test_denied_by_name_pattern(policy: AccessPolicy, rel: str) -> None:
    reason = policy.denied_reason(rel)
    assert reason and "nom protégé" in reason
    assert not policy.check_read(rel).allowed
    assert not policy.check_write(rel).allowed


def test_name_pattern_applies_to_directory_components(policy: AccessPolicy) -> None:
    # Le motif ".env" protège aussi tout ce qui se trouve sous un dossier ".env".
    assert policy.denied_reason(".env/settings.toml")


@pytest.mark.parametrize("rel", ["main.py", "README.md", "environment.txt", "keys.txt", "envfile", "docs/.envrc-example"])
def test_not_denied_ordinary_names(policy: AccessPolicy, rel: str) -> None:
    assert policy.denied_reason(rel) is None
    assert policy.check_read(rel).allowed


# ------------------------------------------------------------ deny (path)


def test_denied_by_home_path_pattern(policy: AccessPolicy) -> None:
    home = Path(os.environ["HOME"])
    reason = policy.denied_reason(home / ".ssh" / "config")
    assert reason and "~/.ssh/**" in reason
    reason = policy.denied_reason(home / ".config" / "dhaos" / "config.toml")
    assert reason and "~/.config/dhaos/**" in reason
    # Motif avec ** au milieu.
    assert policy.denied_reason(home / ".mozilla" / "firefox" / "abc.default" / "logins.json")
    assert policy.denied_reason(home / ".config" / "google-chrome" / "Default" / "Login Data")
    # Motif exact sans **.
    assert policy.denied_reason(home / ".netrc")
    assert policy.denied_reason(home / ".docker" / "config.json")


def test_denied_by_absolute_path_pattern(policy: AccessPolicy) -> None:
    assert policy.denied_reason("/etc/shadow")
    assert policy.denied_reason("/etc/sudoers.d/90-admin")
    assert policy.denied_reason("/proc/self/environ")
    assert policy.denied_reason("/sys/kernel/foo")
    assert policy.denied_reason("/etc/hostname") is None


def test_home_pattern_is_not_applied_to_other_users(policy: AccessPolicy, tmp_path: Path) -> None:
    other = tmp_path / "other-home" / ".ssh" / "config"
    # "~/.ssh/**" ne vise que le HOME courant ; "config" n'est pas un nom protégé.
    assert policy.denied_reason(other) is None


def test_custom_deny_patterns(settings: Settings, project_root: Path) -> None:
    settings.tools.deny_patterns = ["**/secret/**", "*.bak", "~/private/**"]
    policy = AccessPolicy(settings)
    assert policy.denied_reason("a/secret/x.txt")
    assert policy.denied_reason("notes.bak")
    assert policy.denied_reason(Path(os.environ["HOME"]) / "private" / "doc.txt")
    assert policy.denied_reason("a/public/x.txt") is None


def test_glob_to_regex() -> None:
    assert glob_to_regex("/a/**/b.txt").match("/a/b.txt")
    assert glob_to_regex("/a/**/b.txt").match("/a/x/y/b.txt")
    assert glob_to_regex("/a/*.txt").match("/a/f.txt")
    assert not glob_to_regex("/a/*.txt").match("/a/sub/f.txt")
    assert glob_to_regex("/a/?.txt").match("/a/f.txt")
    assert not glob_to_regex("/a/?.txt").match("/a/ff.txt")
    assert glob_to_regex("/a/**").match("/a/anything/deep")


# ------------------------------------------------------------- read roots


def test_read_outside_roots(settings: Settings, project_root: Path, tmp_path: Path) -> None:
    settings.tools.read_roots = [project_root]
    policy = AccessPolicy(settings)
    assert policy.check_read("x.txt").allowed
    assert policy.check_read(project_root).allowed
    d = policy.check_read(tmp_path / "outside.txt")
    assert not d.allowed and "hors des racines" in d.reason
    d = policy.check_read("../outside.txt")
    assert not d.allowed


def test_read_roots_default_is_whole_disk(policy: AccessPolicy, tmp_path: Path) -> None:
    assert policy.check_read(tmp_path / "anything.txt").allowed
    assert policy.check_read("/etc/hostname").allowed


def test_read_roots_expand_home(settings: Settings) -> None:
    settings.tools.read_roots = [Path("~/work")]
    policy = AccessPolicy(settings)
    home = Path(os.environ["HOME"])
    assert policy.check_read(home / "work" / "a.txt").allowed
    assert not policy.check_read(home / "elsewhere.txt").allowed


# ---------------------------------------------------------- write policy


def test_write_policy_project(policy: AccessPolicy, tmp_path: Path) -> None:
    inside = policy.check_write("src/new.py")
    assert inside.allowed and not inside.needs_confirmation
    outside = policy.check_write(tmp_path / "elsewhere.txt")
    assert outside.allowed and outside.needs_confirmation
    assert "confirmation" in outside.reason


def test_write_policy_ask(settings: Settings, policy: AccessPolicy, tmp_path: Path) -> None:
    settings.tools.write_policy = "ask"
    inside = policy.check_write("src/new.py")
    assert inside.allowed and inside.needs_confirmation
    outside = policy.check_write(tmp_path / "elsewhere.txt")
    assert outside.allowed and outside.needs_confirmation


def test_write_policy_all(settings: Settings, policy: AccessPolicy, tmp_path: Path) -> None:
    settings.tools.write_policy = "all"
    outside = policy.check_write(tmp_path / "elsewhere.txt")
    assert outside.allowed and not outside.needs_confirmation


def test_write_policy_deny(settings: Settings, policy: AccessPolicy) -> None:
    settings.tools.write_policy = "deny"
    d = policy.check_write("src/new.py")
    assert not d.allowed and "deny" in d.reason


@pytest.mark.parametrize("mode", ["project", "ask", "all"])
def test_write_denied_pattern_wins_over_policy(settings: Settings, policy: AccessPolicy, mode: str) -> None:
    settings.tools.write_policy = mode  # type: ignore[assignment]
    d = policy.check_write(".env")
    assert not d.allowed and "protégé" in d.reason


# ------------------------------------------------------------- commands


@pytest.mark.parametrize(
    "command",
    ["ls", "ls -la src", "git status", "git diff --stat HEAD~1", "pytest tests/ -q",
     "python -m pytest -k foo", "  cat README.md  ", "grep -rn 'def main' .", "rg TODO"],
)
def test_auto_allowed_commands(policy: AccessPolicy, command: str) -> None:
    assert policy.is_auto_allowed(command)


@pytest.mark.parametrize(
    "command",
    ["git push", "rm -rf /", "pip install requests", "python setup.py", "lsblk", "gitk", "cats"],
)
def test_not_auto_allowed_commands(policy: AccessPolicy, command: str) -> None:
    assert not policy.is_auto_allowed(command)


@pytest.mark.parametrize(
    "command",
    ["ls; rm -rf /", "cat x | grep y", "ls && rm x", "echo $HOME", "cat < in.txt", "ls > out.txt",
     "echo `id`", "ls\nrm x", "ls \\\n -la", "env | grep FOO"],
)
def test_metacharacters_refused(policy: AccessPolicy, command: str) -> None:
    assert not policy.is_auto_allowed(command)


# Régression : `find` en liste blanche exécutait / supprimait sans confirmation.
@pytest.mark.parametrize(
    "command",
    ["find . -exec cat {} +", "find . -execdir sh -c 'id' sh {} +", "find . -name x -delete",
     "find . -fprintf /tmp/out '%p'", "find . -ok rm {} +", "find . -okdir rm {} +",
     "find . -fprint /tmp/x", "find . -fprint0 /tmp/x", "find . -fls /tmp/x",
     "find /home -maxdepth 0 -exec sh -c 'touch /home/pwned' sh {} +"],
)
def test_find_dangerous_actions_need_confirmation(settings: Settings, policy: AccessPolicy, command: str) -> None:
    settings.tools.shell_policy = "ask"
    assert policy.is_auto_allowed(command) is False
    assert policy.check_command(command).needs_confirmation is True


# Régression : options d'écriture des commandes en liste blanche (sort -o, git log
# --output, tree -o, uniq IN OUT, pytest --basetemp) contournaient write_policy.
@pytest.mark.parametrize(
    "command",
    ["sort -o /tmp/x /etc/hostname", "sort --output=/tmp/x /etc/hostname", "sort --output /tmp/x a",
     "sort --out=/tmp/x a", "sort -o/tmp/x a", "sort -ro /tmp/x a", "sort -T /tmp/d a",
     "git log --output=/tmp/x", "git log --output /tmp/x", "git log --out=/tmp/x",
     "git show --output=/tmp/x HEAD", "git diff --output=/tmp/x", "tree -o /tmp/x",
     "uniq in.txt /tmp/out.txt", "pytest --basetemp=/tmp/x", "python -m pytest --basetemp /tmp/x",
     "python3 -m pytest --junitxml=/tmp/x.xml", "pytest -o cache_dir=/tmp/x", "pytest --log-file=/tmp/x"],
)
def test_whitelisted_commands_with_write_options_need_confirmation(
    settings: Settings, policy: AccessPolicy, command: str
) -> None:
    settings.tools.shell_policy = "ask"
    assert policy.is_auto_allowed(command) is False
    assert policy.check_command(command).needs_confirmation is True


@pytest.mark.parametrize(
    "command",
    ["find . -name '*.py' -maxdepth 2", "find src -type f -newer x -print", "sort fichier.txt",
     "sort -n -r -k2 fichier.txt", "sort --reverse fichier.txt", "git log --oneline -n 5 --stat",
     "git show --stat HEAD", "git diff --stat HEAD~1", "tree -L 2 -a", "uniq -c fichier.txt",
     "pytest tests/ -q -x", "python -m pytest -k foo --collect-only"],
)
def test_whitelisted_commands_without_write_options_stay_allowed(policy: AccessPolicy, command: str) -> None:
    assert policy.is_auto_allowed(command) is True


def test_invalid_shlex_and_empty(policy: AccessPolicy) -> None:
    assert not policy.is_auto_allowed("")
    assert not policy.is_auto_allowed("   ")
    assert not policy.is_auto_allowed("echo 'non terminé")
    assert not policy.is_auto_allowed('cat "x')


def test_custom_auto_allow_list(settings: Settings) -> None:
    settings.tools.shell_auto_allow = ["make test", "cargo check"]
    policy = AccessPolicy(settings)
    assert policy.is_auto_allowed("make test -j4")
    assert not policy.is_auto_allowed("make")
    assert not policy.is_auto_allowed("ls")


def test_check_command_policies(settings: Settings, policy: AccessPolicy) -> None:
    settings.tools.shell_policy = "deny"
    d = policy.check_command("ls")
    assert not d.allowed and "deny" in d.reason

    settings.tools.shell_policy = "auto"
    d = policy.check_command("rm -rf build")
    assert d.allowed and not d.needs_confirmation

    settings.tools.shell_policy = "ask"
    d = policy.check_command("git status")
    assert d.allowed and not d.needs_confirmation
    d = policy.check_command("git push")
    assert d.allowed and d.needs_confirmation and "confirmation" in d.reason


def test_confirm_helpers() -> None:
    assert auto_confirm("?") is True
    assert never_confirm("?") is False


# -------------------------------------------------------------- journal


def test_journal_record_and_tail(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "nested" / "journal.jsonl")
    entry = journal.record("write_file", path="/x/y.txt", bytes=12, backup=None, confirmed=False)
    assert entry["kind"] == "write_file" and entry["bytes"] == 12
    assert "ts" in entry and "iso" in entry
    journal.record("run_command", command="ls", cwd="/x", exit=0, duration=0.01, confirmed=False)
    assert journal.path.is_file()

    lines = journal.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["path"] == "/x/y.txt"

    entries = journal.tail()
    assert [e["kind"] for e in entries] == ["write_file", "run_command"]
    assert journal.tail(1)[0]["kind"] == "run_command"


def test_journal_tail_missing_file(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "absent.jsonl")
    assert journal.tail() == []


def test_journal_tail_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_text('{"kind": "a"}\nnot json\n\n{"kind": "b"}\n', encoding="utf-8")
    journal = Journal(path)
    assert [e["kind"] for e in journal.tail()] == ["a", "b"]


def test_journal_serializes_paths(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    journal.record("write_file", path=tmp_path / "f.txt", backup=None)
    assert journal.tail()[0]["path"] == str(tmp_path / "f.txt")


# ------------------------------------------------- commands : chemins protégés
# Régression : les commandes en liste blanche (cat, head, grep…) lisaient les
# chemins refusés par check_read (deny_patterns) sans confirmation.


@pytest.fixture
def secrets_home(tmp_path: Path, project_root: Path) -> Path:
    home = Path(os.environ["HOME"])
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_rsa").write_text("PRIVATE-KEY-SENTINEL-A1\n", encoding="utf-8")
    (home / ".aws").mkdir()
    (home / ".aws" / "credentials").write_text("aws_secret_access_key = SENTINEL-C3\n", encoding="utf-8")
    (project_root / ".env").write_text("SECRET=SENTINEL-D4\n", encoding="utf-8")
    (project_root / "README.md").write_text("# lisible\n", encoding="utf-8")
    return home


@pytest.mark.parametrize(
    "template",
    ["cat ~/.ssh/id_rsa", "cat {home}/.ssh/id_rsa", "head -c 40 ~/.ssh/id_rsa", "tail -n 5 {home}/.ssh/id_rsa",
     "grep -r SENTINEL {home}/.aws", "grep -rl x {home}/.config/dhaos", "cat /proc/self/status",
     "wc -l /etc/shadow", "tail .env", "cat {project}/.env", "grep --file=~/.ssh/id_rsa README.md",
     "cat id_rsa", "ls -la ~/.ssh", "find ~/.ssh -type f"],
)
def test_auto_allow_respects_deny_patterns(
    settings: Settings, policy: AccessPolicy, secrets_home: Path, project_root: Path, template: str
) -> None:
    settings.tools.shell_policy = "ask"
    (project_root / "id_rsa").write_text("x", encoding="utf-8")
    command = template.format(home=secrets_home, project=project_root)
    assert not policy.is_auto_allowed(command)
    d = policy.check_command(command)
    assert d.allowed and d.needs_confirmation
    assert "protégé" in d.reason and "confirmation" in d.reason


@pytest.mark.parametrize(
    "command",
    ["cat README.md", "grep -rn 'def main' .", "ls -la src", "git diff --stat HEAD~1", "head -n 3 /etc/hostname",
     "grep -rn SENTINEL .", "cat absent.txt", "wc -l README.md"],
)
def test_auto_allow_keeps_ordinary_reads(
    settings: Settings, policy: AccessPolicy, secrets_home: Path, command: str
) -> None:
    settings.tools.shell_policy = "ask"
    assert policy.is_auto_allowed(command)
    d = policy.check_command(command)
    assert d.allowed and not d.needs_confirmation


def test_auto_allow_respects_read_roots(settings: Settings, project_root: Path, tmp_path: Path) -> None:
    settings.tools.read_roots = [project_root]
    policy = AccessPolicy(settings)
    assert policy.is_auto_allowed("cat README.md")
    assert not policy.is_auto_allowed(f"cat {tmp_path}/outside.txt")
    assert not policy.is_auto_allowed("ls /etc")


def test_check_command_resolves_relative_paths_from_cwd(
    settings: Settings, policy: AccessPolicy, secrets_home: Path
) -> None:
    settings.tools.shell_policy = "ask"
    # Sans cwd, « id_rsa » est relatif au projet (inexistant ⇒ nom ordinaire) ;
    # depuis ~/.ssh, il désigne la clé privée.
    assert policy.is_auto_allowed("cat id_rsa")
    d = policy.check_command("cat id_rsa", cwd=secrets_home / ".ssh")
    assert d.needs_confirmation and "protégé" in d.reason
    assert not policy.is_auto_allowed("cat id_rsa", cwd=secrets_home / ".ssh")
    assert policy.check_command("cat README.md", cwd=policy.project_root).needs_confirmation is False


def test_auto_allow_refusal_reasons(policy: AccessPolicy) -> None:
    assert policy.auto_allow_refusal("git status") is None
    assert "liste blanche" in (policy.auto_allow_refusal("git push") or "")
    assert "métacaractère" in (policy.auto_allow_refusal("ls | wc") or "")
    assert "vide" in (policy.auto_allow_refusal("   ") or "")
    assert "mal formée" in (policy.auto_allow_refusal("cat 'x") or "")
    assert "option" in (policy.auto_allow_refusal("find . -delete") or "")


def test_env_dumps_are_not_auto_allowed(policy: AccessPolicy) -> None:
    # L'environnement (même expurgé) n'est pas livré au modèle sans confirmation.
    assert not policy.is_auto_allowed("printenv")
    assert not policy.is_auto_allowed("printenv DATABASE_URL")
    assert not policy.is_auto_allowed("env")
