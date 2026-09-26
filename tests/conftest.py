from __future__ import annotations

import os
from pathlib import Path

import pytest

from dhaos.config import Settings
from dhaos.policy import AccessPolicy, Journal, auto_confirm
from dhaos.tools.base import ToolContext


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings isolés : HOME, données et projet sous tmp_path ; embedder hash ;
    écritures libres dans le projet ; shell sans confirmation."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for key in list(os.environ):
        if key.startswith("DHAOS"):
            monkeypatch.delenv(key, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    s = Settings.load(path=tmp_path / "config.toml", use_env=False)
    s.paths.data_dir = tmp_path / "data"
    s.paths.project_root = project
    s.tools.write_policy = "project"
    s.tools.shell_policy = "auto"
    s.kb.embedder = "hash"
    s.agent.max_iterations = 8
    s.ensure_dirs()
    return s


@pytest.fixture
def project_root(settings: Settings) -> Path:
    return settings.resolve_project_root()


@pytest.fixture
def policy(settings: Settings) -> AccessPolicy:
    return AccessPolicy(settings)


@pytest.fixture
def ctx(settings: Settings, policy: AccessPolicy) -> ToolContext:
    return ToolContext(
        settings=settings,
        policy=policy,
        journal=Journal(settings.journal_path),
        confirm=auto_confirm,
    )
