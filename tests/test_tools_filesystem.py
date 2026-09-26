"""Tests des outils disque via ToolRegistry.execute (validation de schéma incluse)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from dhaos.config import Settings
from dhaos.policy import AccessPolicy, Journal, auto_confirm
from dhaos.tools import filesystem
from dhaos.tools.base import ToolContext, ToolRegistry, ToolResult

from .fakes import tool_call


class Confirm:
    """Callback de confirmation scripté qui mémorise les questions posées."""

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
    return ToolRegistry(filesystem.tools(settings))


@pytest.fixture
def tree(project_root: Path) -> Path:
    """Petit projet : sources, dossiers ignorés, secrets, binaire, lien."""
    (project_root / "src" / "pkg").mkdir(parents=True)
    (project_root / "src" / "main.py").write_text("import os\n\ndef main():\n    print('hello')\n", encoding="utf-8")
    (project_root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "src" / "pkg" / "util.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    (project_root / "README.md").write_text("# Projet\n\nHello World\n", encoding="utf-8")
    (project_root / "node_modules" / "lib").mkdir(parents=True)
    (project_root / "node_modules" / "lib" / "index.js").write_text("hello = 1\n", encoding="utf-8")
    (project_root / ".git").mkdir()
    (project_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (project_root / ".env").write_text("SECRET=hello\n", encoding="utf-8")
    (project_root / "server.pem").write_text("-----BEGIN KEY-----\nhello\n", encoding="utf-8")
    (project_root / "blob.bin").write_bytes(b"\x00\x01\x02\xff" * 64 + b"hello")
    (project_root / "yarn.lock").write_text("hello lock\n", encoding="utf-8")
    os.symlink(project_root / "README.md", project_root / "readme_link")
    return project_root


def run(registry: ToolRegistry, ctx: ToolContext, name: str, **args) -> ToolResult:
    return registry.execute(tool_call(name, **args), ctx)


# ---------------------------------------------------------------- général


def test_tools_names(settings: Settings) -> None:
    names = [t.name for t in filesystem.tools(settings)]
    assert names == ["read_file", "list_dir", "find_files", "grep", "write_file", "edit_file"]
    for t in filesystem.tools(settings):
        assert t.description and "type" in t.parameters
        assert t.spec().name == t.name


def test_unknown_tool(registry: ToolRegistry, ctx: ToolContext) -> None:
    r = run(registry, ctx, "nope")
    assert r.is_error and "inconnu" in r.content


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_file", {}),
        ("read_file", {"path": "x", "start_line": 0}),
        ("read_file", {"path": "x", "start_line": "1"}),
        ("list_dir", {}),
        ("list_dir", {"path": ".", "depth": 5}),
        ("find_files", {}),
        ("find_files", {"pattern": "*", "max_results": 0}),
        ("grep", {"root": "."}),
        ("write_file", {"path": "x"}),
        ("write_file", {"path": "x", "content": 42}),
        ("edit_file", {"path": "x", "old_string": "a"}),
        ("edit_file", {"path": "x", "old_string": "", "new_string": "b"}),
    ],
)
def test_schema_validation(registry: ToolRegistry, ctx: ToolContext, name: str, args: dict) -> None:
    r = registry.execute(tool_call(name, **args), ctx)
    assert r.is_error and "INVALID_JSON" in r.content


# -------------------------------------------------------------- read_file


def test_read_file_full(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "read_file", path="src/main.py")
    assert not r.is_error
    assert r.content.splitlines()[0] == f"{tree / 'src' / 'main.py'} (lignes 1–4 sur 4)"
    assert "1| import os" in r.content
    assert "4|     print('hello')" in r.content
    assert r.data["total_lines"] == 4


def test_read_file_range(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "read_file", path=str(tree / "src" / "main.py"), start_line=3, end_line=10)
    assert not r.is_error
    assert "(lignes 3–4 sur 4)" in r.content
    assert "3| def main():" in r.content
    assert "1| import os" not in r.content

    r = run(registry, ctx, "read_file", path="src/main.py", start_line=9)
    assert r.is_error and "dépasse" in r.content
    r = run(registry, ctx, "read_file", path="src/main.py", start_line=3, end_line=2)
    assert r.is_error


def test_read_file_empty(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "read_file", path="src/pkg/__init__.py")
    assert not r.is_error and "vide" in r.content


def test_read_file_missing_and_directory(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "read_file", path="nope.txt")
    assert r.is_error and "introuvable" in r.content
    r = run(registry, ctx, "read_file", path="src")
    assert r.is_error and "répertoire" in r.content


def test_read_file_binary_refused(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "read_file", path="blob.bin")
    assert r.is_error and "binaire" in r.content
    assert "hello" not in r.content


def test_read_file_protected_refused(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    for name in (".env", "server.pem", str(tree / ".env")):
        r = run(registry, ctx, "read_file", path=name)
        assert r.is_error and "protégé" in r.content
        assert "hello" not in r.content


def test_read_file_outside_roots(settings: Settings, registry: ToolRegistry, tmp_path: Path) -> None:
    settings.tools.read_roots = [settings.resolve_project_root()]
    ctx = make_ctx(settings)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    r = run(registry, ctx, "read_file", path=str(outside))
    assert r.is_error and "hors des racines" in r.content


def test_read_file_truncation(settings: Settings, registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    (tree / "big.txt").write_text("\n".join(f"ligne {i}" for i in range(500)), encoding="utf-8")
    settings.tools.max_file_chars = 300
    r = run(registry, ctx, "read_file", path="big.txt")
    assert not r.is_error
    assert "tronqué" in r.content
    assert len(r.content) < 400


def test_read_file_invalid_utf8_replaced(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    (tree / "latin.txt").write_bytes("caf\xe9 cr\xe8me\n".encode("latin-1"))
    r = run(registry, ctx, "read_file", path="latin.txt")
    assert not r.is_error and "caf�" in r.content


def test_read_file_symlink_followed(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "read_file", path="readme_link")
    assert not r.is_error and "Hello World" in r.content


def test_registry_output_truncation(settings: Settings, registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    (tree / "big.txt").write_text("x" * 5000, encoding="utf-8")
    settings.tools.max_output_chars = 1000
    r = run(registry, ctx, "read_file", path="big.txt")
    assert "tronqué" in r.content and len(r.content) < 1100


# --------------------------------------------------------------- list_dir


def test_list_dir_depth1(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "list_dir", path=".")
    assert not r.is_error
    lines = r.content.splitlines()
    assert lines[0].startswith(str(tree))
    assert "d src/" in lines
    assert any(line.startswith("f README.md  ") and line.endswith(" o") for line in lines)
    assert any(line.startswith("l readme_link -> ") for line in lines)
    assert "main.py" not in r.content  # profondeur 1
    # cachés / ignorés masqués par défaut
    assert ".git" not in r.content and ".env" not in r.content and "node_modules" not in r.content
    assert "yarn.lock" not in r.content
    assert "masquée" in r.content


def test_list_dir_depth2_and_hidden(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "list_dir", path=str(tree), depth=2, show_hidden=True)
    assert not r.is_error
    assert "  f main.py" in r.content
    assert "  d pkg/" in r.content
    assert "util.py" not in r.content  # profondeur 3
    assert "d .git/" in r.content and "f .env" in r.content and "d node_modules/" in r.content

    r = run(registry, ctx, "list_dir", path=".", depth=3)
    assert "    f util.py" in r.content


def test_list_dir_errors(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "list_dir", path="absent")
    assert r.is_error and "introuvable" in r.content
    r = run(registry, ctx, "list_dir", path="README.md")
    assert r.is_error and "répertoire" in r.content
    r = run(registry, ctx, "list_dir", path="/proc")
    assert r.is_error and "protégé" in r.content


def test_list_dir_empty_and_limit(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    (project_root / "empty").mkdir()
    r = run(registry, ctx, "list_dir", path="empty")
    assert not r.is_error and "vide" in r.content

    many = project_root / "many"
    many.mkdir()
    for i in range(600):
        (many / f"f{i:04d}.txt").write_text("x", encoding="utf-8")
    r = run(registry, ctx, "list_dir", path="many")
    assert "tronquée" in r.content
    assert r.data["entries"] == 500


def test_list_dir_does_not_descend_into_protected_dirs(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    (project_root / "keys" / ".env" ).mkdir(parents=True)
    (project_root / "keys" / ".env" / "prod.txt").write_text("x", encoding="utf-8")
    r = run(registry, ctx, "list_dir", path="keys", depth=3, show_hidden=True)
    assert "d .env/" in r.content and "protégé" in r.content
    assert "prod.txt" not in r.content


# ------------------------------------------------------------- find_files


def test_find_files_basic(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "find_files", pattern="*.py")
    assert not r.is_error
    lines = r.content.splitlines()
    assert lines[0].startswith("3 fichier(s)")
    assert lines[1:] == ["src/main.py", "src/pkg/__init__.py", "src/pkg/util.py"]


def test_find_files_ignores_and_root(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "find_files", pattern="*.js")
    assert "index.js" not in r.content and r.data["count"] == 0
    r = run(registry, ctx, "find_files", pattern="*.lock")
    assert r.data["count"] == 0
    r = run(registry, ctx, "find_files", pattern="*", root="src/pkg")
    assert r.data["count"] == 2 and "src/pkg/util.py" in r.content
    r = run(registry, ctx, "find_files", pattern="src/**/*.py")
    assert "src/pkg/util.py" in r.content
    r = run(registry, ctx, "find_files", pattern="**/util.py")
    assert r.data["count"] == 1


def test_find_files_symlinks_not_followed(registry: ToolRegistry, ctx: ToolContext, tree: Path, tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "deep.py").write_text("x", encoding="utf-8")
    os.symlink(ext, tree / "linkdir")
    r = run(registry, ctx, "find_files", pattern="deep.py")
    assert r.data["count"] == 0


def test_find_files_truncation_and_errors(registry: ToolRegistry, ctx: ToolContext, tree: Path, tmp_path: Path) -> None:
    r = run(registry, ctx, "find_files", pattern="*.py", max_results=2)
    assert r.data["truncated"] is True
    assert "tronqué" in r.content and "src/pkg/util.py" not in r.content
    r = run(registry, ctx, "find_files", pattern="*", root="absent")
    assert r.is_error and "introuvable" in r.content
    r = run(registry, ctx, "find_files", pattern="*", root="/proc")
    assert r.is_error and "protégé" in r.content


def test_find_files_outside_project_shows_absolute(registry: ToolRegistry, ctx: ToolContext, tmp_path: Path) -> None:
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "a.txt").write_text("x", encoding="utf-8")
    r = run(registry, ctx, "find_files", pattern="*.txt", root=str(ext))
    assert str(ext / "a.txt") in r.content


# ------------------------------------------------------------------- grep


def test_grep_basic(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "grep", pattern=r"def \w+\(")
    assert not r.is_error
    lines = r.content.splitlines()
    assert lines[0].startswith("2 correspondance(s)")
    assert "src/main.py:3: def main():" in lines
    assert "src/pkg/util.py:1: def helper():" in lines


def test_grep_skips_protected_binary_ignored(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "grep", pattern="hello")
    assert "src/main.py:4:" in r.content
    assert "README.md" not in r.content  # « Hello » majuscule
    assert ".env" not in r.content and "server.pem" not in r.content
    assert "blob.bin" not in r.content and "index.js" not in r.content and "yarn.lock" not in r.content


def test_grep_case_insensitive_and_glob(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "grep", pattern="hello", case_insensitive=True)
    assert "README.md:3: Hello World" in r.content
    r = run(registry, ctx, "grep", pattern="hello", case_insensitive=True, glob="*.md")
    assert "README.md:3:" in r.content and "main.py" not in r.content
    r = run(registry, ctx, "grep", pattern="hello", glob=None)
    assert "main.py" in r.content


def test_grep_invalid_regex_and_no_match(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "grep", pattern="(unclosed")
    assert r.is_error and "invalide" in r.content
    r = run(registry, ctx, "grep", pattern="zzz_nothing")
    assert not r.is_error and "aucune correspondance" in r.content


def test_grep_line_truncation_and_max_results(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    (tree / "long.txt").write_text("needle " + "y" * 1000 + "\n" + "needle\n" * 5, encoding="utf-8")
    r = run(registry, ctx, "grep", pattern="needle", root="long.txt")
    first = r.content.splitlines()[1]
    assert first.startswith("long.txt:1: needle") and first.endswith("…")
    assert len(first) < 330
    r = run(registry, ctx, "grep", pattern="needle", max_results=3)
    assert r.data["count"] == 3 and r.data["truncated"] is True and "tronqué" in r.content


def test_grep_too_large_file_skipped(settings: Settings, registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    settings.kb.max_file_bytes = 10
    r = run(registry, ctx, "grep", pattern="hello")
    assert "main.py" not in r.content and "aucune correspondance" in r.content


def test_grep_protected_root(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "grep", pattern="SECRET", root=".env")
    assert r.is_error and "protégé" in r.content


# ------------------------------------------------------------- write_file


def test_write_file_creates_and_journals(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    r = run(registry, ctx, "write_file", path="out/new.txt", content="bonjour\n")
    assert not r.is_error
    assert r.content.startswith(f"écrit {project_root / 'out' / 'new.txt'} (8 octets)")
    assert (project_root / "out" / "new.txt").read_text(encoding="utf-8") == "bonjour\n"
    entries = ctx.journal.tail()
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "write_file" and e["bytes"] == 8 and e["backup"] is None and e["confirmed"] is False
    assert e["path"] == str(project_root / "out" / "new.txt")
    assert not list((project_root / "out").glob(".*.tmp"))


def test_write_file_no_create_dirs(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    r = run(registry, ctx, "write_file", path="missing/x.txt", content="x", create_dirs=False)
    assert r.is_error and "create_dirs" in r.content
    assert not (project_root / "missing").exists()


def test_write_file_backup(settings: Settings, registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    target = project_root / "a.txt"
    target.write_text("ancien\n", encoding="utf-8")
    r = run(registry, ctx, "write_file", path="a.txt", content="nouveau\n")
    assert not r.is_error and "sauvegarde" in r.content
    assert target.read_text(encoding="utf-8") == "nouveau\n"
    backups = list(settings.backups_dir.iterdir())
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "ancien\n"
    assert "__" in backups[0].name and backups[0].name.endswith("a.txt")
    assert ctx.journal.tail()[0]["backup"] == str(backups[0])

    settings.tools.backup_before_write = False
    run(registry, ctx, "write_file", path="a.txt", content="encore\n")
    assert len(list(settings.backups_dir.iterdir())) == 1


def test_write_file_preserves_mode(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    target = project_root / "script.sh"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    run(registry, ctx, "write_file", path="script.sh", content="#!/bin/sh\necho ok\n")
    assert target.stat().st_mode & 0o777 == 0o755


def test_write_file_outside_project_confirmed(settings: Settings, registry: ToolRegistry, tmp_path: Path) -> None:
    confirm = Confirm(True)
    ctx = make_ctx(settings, confirm)
    target = tmp_path / "outside" / "x.txt"
    r = run(registry, ctx, "write_file", path=str(target), content="abc")
    assert not r.is_error and target.read_text(encoding="utf-8") == "abc"
    assert len(confirm.prompts) == 1
    assert confirm.prompts[0].startswith(f"Écrire dans {target} (3 caractères)")
    assert ctx.journal.tail()[0]["confirmed"] is True


def test_write_file_outside_project_refused(settings: Settings, registry: ToolRegistry, tmp_path: Path) -> None:
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    target = tmp_path / "outside" / "x.txt"
    r = run(registry, ctx, "write_file", path=str(target), content="abc")
    assert r.is_error and r.content.startswith("écriture refusée")
    assert not target.exists() and confirm.prompts
    assert ctx.journal.tail() == []


def test_write_file_inside_project_no_confirmation(settings: Settings, registry: ToolRegistry) -> None:
    confirm = Confirm(False)  # ne doit jamais être appelé
    ctx = make_ctx(settings, confirm)
    r = run(registry, ctx, "write_file", path="free.txt", content="x")
    assert not r.is_error and confirm.prompts == []


def test_write_file_policy_ask_and_deny(settings: Settings, registry: ToolRegistry, project_root: Path) -> None:
    settings.tools.write_policy = "ask"
    confirm = Confirm(True)
    ctx = make_ctx(settings, confirm)
    r = run(registry, ctx, "write_file", path="asked.txt", content="x")
    assert not r.is_error and len(confirm.prompts) == 1

    settings.tools.write_policy = "deny"
    r = run(registry, ctx, "write_file", path="denied.txt", content="x")
    assert r.is_error and "écriture refusée" in r.content and "deny" in r.content
    assert not (project_root / "denied.txt").exists()


def test_write_file_protected_refused(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    for name in (".env", "keys/id_rsa", "cert.key"):
        r = run(registry, ctx, "write_file", path=name, content="x")
        assert r.is_error and "écriture refusée" in r.content and "protégé" in r.content
        assert not (project_root / name).exists()
    assert ctx.journal.tail() == []


def test_write_file_on_directory(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    (project_root / "adir").mkdir()
    r = run(registry, ctx, "write_file", path="adir", content="x")
    assert r.is_error and "répertoire" in r.content


def test_write_file_confirm_exception_is_refusal(settings: Settings, registry: ToolRegistry, tmp_path: Path) -> None:
    def boom(prompt: str) -> bool:
        raise EOFError

    ctx = make_ctx(settings, boom)
    r = run(registry, ctx, "write_file", path=str(tmp_path / "o.txt"), content="x")
    assert r.is_error and "refusée" in r.content


# -------------------------------------------------------------- edit_file


def test_edit_file_single(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "edit_file", path="src/main.py", old_string="print('hello')", new_string="print('bonjour')")
    assert not r.is_error
    assert r.content.startswith(f"modifié {tree / 'src' / 'main.py'} (1 remplacement(s)")
    assert "-    print('hello')" in r.content and "+    print('bonjour')" in r.content
    assert "--- a/src/main.py" in r.content
    assert (tree / "src" / "main.py").read_text(encoding="utf-8") == "import os\n\ndef main():\n    print('bonjour')\n"
    e = ctx.journal.tail()[0]
    assert e["kind"] == "edit_file" and e["replacements"] == 1 and e["backup"]
    assert Path(e["backup"]).read_text(encoding="utf-8").endswith("print('hello')\n")


def test_edit_file_not_found_and_ambiguous(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    f = project_root / "t.txt"
    f.write_text("a b a\n", encoding="utf-8")
    r = run(registry, ctx, "edit_file", path="t.txt", old_string="zzz", new_string="y")
    assert r.is_error and "introuvable" in r.content
    r = run(registry, ctx, "edit_file", path="t.txt", old_string="a", new_string="y")
    assert r.is_error and "2 fois" in r.content and "replace_all" in r.content
    assert f.read_text(encoding="utf-8") == "a b a\n"
    assert ctx.journal.tail() == []


def test_edit_file_replace_all(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    f = project_root / "t.txt"
    f.write_text("a b a\n", encoding="utf-8")
    r = run(registry, ctx, "edit_file", path="t.txt", old_string="a", new_string="y", replace_all=True)
    assert not r.is_error and "2 remplacement(s)" in r.content
    assert f.read_text(encoding="utf-8") == "y b y\n"
    assert ctx.journal.tail()[0]["replacements"] == 2


def test_edit_file_identical_and_missing(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    (project_root / "t.txt").write_text("abc\n", encoding="utf-8")
    r = run(registry, ctx, "edit_file", path="t.txt", old_string="abc", new_string="abc")
    assert r.is_error and "identiques" in r.content
    r = run(registry, ctx, "edit_file", path="nope.txt", old_string="a", new_string="b")
    assert r.is_error and "introuvable" in r.content


def test_edit_file_protected_and_binary(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "edit_file", path=".env", old_string="hello", new_string="x")
    assert r.is_error and "protégé" in r.content
    assert (tree / ".env").read_text(encoding="utf-8") == "SECRET=hello\n"
    r = run(registry, ctx, "edit_file", path="blob.bin", old_string="hello", new_string="x")
    assert r.is_error and "binaire" in r.content


def test_edit_file_confirmation_refused(settings: Settings, registry: ToolRegistry, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("hello\n", encoding="utf-8")
    confirm = Confirm(False)
    ctx = make_ctx(settings, confirm)
    r = run(registry, ctx, "edit_file", path=str(outside), old_string="hello", new_string="bye")
    assert r.is_error and "écriture refusée" in r.content
    assert outside.read_text(encoding="utf-8") == "hello\n" and confirm.prompts

    confirm.answer = True
    r = run(registry, ctx, "edit_file", path=str(outside), old_string="hello", new_string="bye")
    assert not r.is_error and outside.read_text(encoding="utf-8") == "bye\n"
    assert ctx.journal.tail()[0]["confirmed"] is True


def test_edit_file_diff_truncated(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    f = project_root / "big.txt"
    f.write_text("\n".join("x" for _ in range(300)) + "\n", encoding="utf-8")
    r = run(registry, ctx, "edit_file", path="big.txt", old_string="x", new_string="yy", replace_all=True)
    assert not r.is_error and "300 remplacement(s)" in r.content
    assert "diff tronqué" in r.content
    assert len(r.content.splitlines()) <= 63


def test_edit_file_invalid_utf8_refused(registry: ToolRegistry, ctx: ToolContext, project_root: Path) -> None:
    f = project_root / "latin.txt"
    f.write_bytes("caf\xe9\n".encode("latin-1"))
    r = run(registry, ctx, "edit_file", path="latin.txt", old_string="caf", new_string="the")
    assert r.is_error and "UTF-8" in r.content
    assert f.read_bytes() == "caf\xe9\n".encode("latin-1")


# ------------------------------------------------- filtre des chemins protégés


def test_find_files_hides_protected_files(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "find_files", pattern="*")
    assert not r.is_error
    assert ".env" not in r.content and "server.pem" not in r.content
    assert "README.md" in r.content


def test_walk_respects_home_path_patterns(registry: ToolRegistry, ctx: ToolContext, tmp_path: Path) -> None:
    home = Path(os.environ["HOME"])
    (home / ".ssh").mkdir()
    (home / ".ssh" / "config").write_text("Host secret\n", encoding="utf-8")
    (home / "notes.txt").write_text("Host public\n", encoding="utf-8")
    r = run(registry, ctx, "find_files", pattern="*", root=str(home))
    assert "notes.txt" in r.content and "config" not in r.content
    r = run(registry, ctx, "grep", pattern="Host", root=str(home))
    assert "notes.txt:1:" in r.content and "secret" not in r.content
    r = run(registry, ctx, "find_files", pattern="*", root=str(home / ".ssh"))
    assert r.is_error and "protégé" in r.content and "~/.ssh/**" in r.content
    r = run(registry, ctx, "list_dir", path=str(home / ".ssh"))
    assert r.is_error and "protégé" in r.content
    r = run(registry, ctx, "list_dir", path=str(home), show_hidden=True)
    assert "d .ssh/  (chemin protégé" in r.content and "config" not in r.content


def test_list_dir_marks_protected_files(registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    r = run(registry, ctx, "list_dir", path=".", show_hidden=True)
    lines = r.content.splitlines()
    assert any(line.startswith("f .env  ") and "protégé" in line for line in lines)
    assert any(line.startswith("f server.pem  ") and "protégé" in line for line in lines)
    assert any(line.startswith("f README.md  ") and "protégé" not in line for line in lines)


# Régression : les schémas des outils disque acceptaient des propriétés
# inconnues (typo d'argument masquée, ex. startLine au lieu de start_line).
def test_tools_reject_extra_args(settings: Settings, registry: ToolRegistry, ctx: ToolContext, tree: Path) -> None:
    for t in filesystem.tools(settings):
        assert t.parameters.get("additionalProperties") is False, t.name
    r = run(registry, ctx, "read_file", path="README.md", bogus=1)
    assert r.is_error and "INVALID_JSON" in r.content and "Additional properties" in r.content
    r = run(registry, ctx, "read_file", path="src/main.py", startLine=2, endLine=3)
    assert r.is_error and "startLine" in r.content
    r = run(registry, ctx, "write_file", path="new.txt", content="x", mode="append")
    assert r.is_error and not (tree / "new.txt").exists()
