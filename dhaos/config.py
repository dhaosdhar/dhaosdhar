"""Configuration de dhaos.

Priorité : valeurs par défaut < fichier TOML < variables d'environnement.

- Fichier : ``$DHAOS_CONFIG``, sinon ``$XDG_CONFIG_HOME/dhaos/config.toml``
  (``~/.config/dhaos/config.toml``).
- Données : ``$DHAOS_DATA_DIR``, sinon ``$XDG_DATA_HOME/dhaos``
  (``~/.local/share/dhaos``) : base de savoir, sessions, journal, sauvegardes,
  jeux de données, modèles.
- Environnement : ``DHAOS__SECTION__CLE=valeur`` (ex. ``DHAOS__BACKENDS__DEFAULT=claude``,
  ``DHAOS__BACKENDS__OLLAMA__MODEL=qwen2.5-coder:7b``). La valeur est lue comme
  JSON quand c'est possible (``true``, ``42``, ``["a","b"]``), sinon comme chaîne.
"""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, Field

ENV_PREFIX = "DHAOS__"


def _home() -> Path:
    return Path(os.environ.get("HOME") or Path.home())


def default_config_path() -> Path:
    if os.environ.get("DHAOS_CONFIG"):
        return Path(os.environ["DHAOS_CONFIG"]).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else _home() / ".config"
    return root / "dhaos" / "config.toml"


def default_data_dir() -> Path:
    if os.environ.get("DHAOS_DATA_DIR"):
        return Path(os.environ["DHAOS_DATA_DIR"]).expanduser()
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else _home() / ".local" / "share"
    return root / "dhaos"


# Fichiers et dossiers jamais lus ni écrits par l'agent (secrets, trousseaux).
# Motif avec "/" : comparé au chemin absolu résolu (``**`` accepté, ``~`` développé).
# Motif sans "/" : comparé au nom de chaque composant du chemin.
DEFAULT_DENY_PATTERNS: list[str] = [
    "~/.ssh/**",
    "~/.gnupg/**",
    "~/.aws/**",
    "~/.config/gcloud/**",
    "~/.kube/**",
    "~/.docker/config.json",
    "~/.netrc",
    "~/.local/share/keyrings/**",
    "~/.mozilla/**/logins.json",
    "~/.mozilla/**/key4.db",
    "~/.config/**/Login Data",
    "~/.anthropic/**",
    "~/.config/anthropic/**",
    "~/.config/dhaos/**",
    "/etc/shadow",
    "/etc/gshadow",
    "/etc/sudoers",
    "/etc/sudoers.d/**",
    "/proc/**",
    "/sys/**",
    "/dev/**",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.kdbx",
    "*.keychain",
    "*.keychain-db",
    "id_rsa",
    "id_rsa.pub",
    "id_ed25519",
    "id_ed25519.pub",
    "id_ecdsa",
    "credentials.json",
    "service-account*.json",
]

# Commandes exécutées sans confirmation quand shell_policy = "ask"
# (comparaison sur les premiers mots ; aucun métacaractère shell accepté ;
# les options listées dans DEFAULT_SHELL_UNSAFE_OPTIONS retirent la commande
# de la liste blanche, ainsi qu'un argument-chemin refusé par check_read).
# La liste blanche ne remplace ni check_read ni check_write : une commande qui
# lit un chemin protégé, écrit ou exécute via un argument reste soumise à
# confirmation. `env` / `printenv` n'y figurent pas : l'environnement, même
# expurgé, ne doit pas être livré au modèle sans confirmation.
DEFAULT_SHELL_AUTO_ALLOW: list[str] = [
    "ls", "cat", "head", "tail", "wc", "stat", "file", "which", "pwd", "echo",
    "tree", "du", "df", "find", "grep", "rg", "diff", "sort", "uniq",
    "git status", "git diff", "git log", "git show", "git branch", "git remote -v",
    "git ls-files", "git blame",
    "python --version", "python3 --version", "pip list", "pip show",
    "pytest", "python -m pytest", "python3 -m pytest",
    "node --version", "npm ls", "cargo --version", "go version", "make -n",
    "uname", "id", "date",
]

# Options qui font écrire, supprimer ou exécuter un programme à une commande
# de la liste blanche : leur présence impose la confirmation (shell_policy =
# "ask"). Clé : nom du programme (premier mot ; ``python -m X`` compte comme
# ``X``). Une option longue ``--xxx`` est aussi reconnue abrégée (``--out=``,
# GNU/argparse) ; une option courte ``-o`` aussi collée ou groupée (``-ofichier``,
# ``-ro``) ; un mot ``-exec`` (style find) seulement à l'identique.
DEFAULT_SHELL_UNSAFE_OPTIONS: dict[str, tuple[str, ...]] = {
    "find": ("-exec", "-execdir", "-ok", "-okdir", "-delete",
             "-fprint", "-fprint0", "-fprintf", "-fls"),
    "sort": ("-o", "--output", "-T", "--temporary-directory"),
    "tree": ("-o",),
    "git": ("--output",),
    "pytest": ("--basetemp", "--junitxml", "--junit-xml", "--log-file", "-o", "--override-ini"),
}

# Nombre maximal d'arguments positionnels (hors options) toléré sans
# confirmation : ``uniq ENTRÉE SORTIE`` écrit dans son second argument.
DEFAULT_SHELL_MAX_POSITIONALS: dict[str, int] = {"uniq": 1}


class PathsConfig(BaseModel):
    data_dir: Path = Field(default_factory=default_data_dir)
    project_root: Path | None = None  # défaut : répertoire courant au lancement


class OllamaConfig(BaseModel):
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5-coder:7b"
    embed_model: str = "nomic-embed-text"
    timeout: float = 300.0
    num_ctx: int | None = 16384  # fenêtre de contexte demandée (options.num_ctx)
    keep_alive: str | None = None  # ex. "30m"


class ClaudeConfig(BaseModel):
    model: str = "claude-opus-5"
    max_tokens: int = 64000  # requêtes en streaming : on laisse de la marge
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "xhigh"
    thinking_display: Literal["omitted", "summarized"] = "omitted"
    fallbacks: bool = True  # bascule serveur en cas de refus (fallbacks="default")
    timeout: float = 600.0
    base_url: str | None = None  # défaut : ANTHROPIC_BASE_URL / API publique


class BackendsConfig(BaseModel):
    default: Literal["ollama", "claude"] = "ollama"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)


class ToolsConfig(BaseModel):
    read_roots: list[Path] = Field(default_factory=lambda: [Path("/")])
    deny_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_DENY_PATTERNS))
    # project : libre dans le projet, confirmation ailleurs ; ask : toujours
    # confirmer ; all : jamais de confirmation (dangereux) ; deny : aucune écriture.
    write_policy: Literal["project", "ask", "all", "deny"] = "project"
    # ask : confirmation sauf commandes listées ; auto : jamais ; deny : pas de shell.
    shell_policy: Literal["ask", "auto", "deny"] = "ask"
    shell_auto_allow: list[str] = Field(default_factory=lambda: list(DEFAULT_SHELL_AUTO_ALLOW))
    command_timeout: float = 120.0
    max_output_chars: int = 40_000  # troncature des résultats d'outils
    max_file_chars: int = 200_000  # troncature de read_file
    backup_before_write: bool = True  # copie de l'ancien contenu dans backups/


class WebConfig(BaseModel):
    provider: Literal["duckduckgo", "searxng", "brave"] = "duckduckgo"
    searxng_url: str | None = None  # ex. "http://searx.local:8080"
    brave_api_key: str | None = None
    max_results: int = 8
    fetch_max_chars: int = 40_000
    timeout: float = 20.0
    user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) dhaos/0.1"


class KBConfig(BaseModel):
    # auto : Ollama si joignable et modèle présent, sinon hash (toujours disponible)
    embedder: Literal["auto", "ollama", "sentence-transformers", "hash"] = "auto"
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hash_dim: int = 512
    chunk_chars: int = 2400
    chunk_overlap: int = 300
    top_k: int = 8  # résultats de `dhaos kb search`
    tool_top_k: int = 5  # passages renvoyés à l'agent par kb_search (si top_k non précisé)
    tool_snippet_chars: int = 700  # longueur max d'un passage renvoyé à l'agent (modèles locaux : contexte et débit limités)
    max_file_bytes: int = 5_000_000
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
            ".mypy_cache", ".pytest_cache", ".ruff_cache", "*.min.js", "*.map",
            "*.lock", "package-lock.json", "*.pyc", "*.so", "*.o", "*.class", "*.egg-info",
        ]
    )


class AgentConfig(BaseModel):
    max_iterations: int = Field(default=40, ge=1)  # nombre max de tours outil par requête (≥ 1)
    collect_traces: bool = True  # sessions conservées pour l'entraînement
    auto_kb_search: bool = True  # inciter le modèle à consulter les bases de savoir
    extra_system_prompt: str = ""
    language: str = "fr"


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8642
    # Jeton Bearer exigé sur toutes les routes (sauf l'état minimal de /health).
    # Absent => un jeton aléatoire est généré au démarrage et affiché par
    # `dhaos serve` : l'API n'est jamais accessible sans jeton.
    token: str | None = None
    # En-têtes Host acceptés (sans port) ; vide => localhost, 127.0.0.1, ::1 et
    # api.host. "*" désactive le contrôle (derrière un reverse proxy de confiance).
    allowed_hosts: list[str] = Field(default_factory=list)
    # Origines navigateur autorisées (schéma://hôte[:port]) ; vide => toute
    # requête portant un en-tête Origin est refusée (aucun client navigateur
    # attendu, protection contre le DNS rebinding).
    allowed_origins: list[str] = Field(default_factory=list)
    # L'API n'a personne à qui demander : False => les actions qui exigent une
    # confirmation sont refusées ; True => confirmées automatiquement (dangereux).
    auto_confirm: bool = False


class NanoConfig(BaseModel):
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 256
    batch_size: int = 16
    learning_rate: float = 3e-4
    steps: int = 500
    tokenizer: Literal["bytes", "bpe"] = "bpe"
    bpe_vocab_size: int = 2048
    dropout: float = 0.0


class FinetuneConfig(BaseModel):
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: float = 1.0
    learning_rate: float = 2e-4
    max_seq_len: int = 2048
    batch_size: int = 2
    gradient_accumulation: int = 8
    load_in_4bit: bool = True


class TrainConfig(BaseModel):
    nano: NanoConfig = Field(default_factory=NanoConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _parse_env_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Construit un dict imbriqué depuis les variables ``DHAOS__A__B=val``."""
    environ = os.environ if environ is None else environ
    out: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                break
        else:
            node[parts[-1]] = _parse_env_value(raw)
    return out


def _nested_set(d: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


class Settings(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    backends: BackendsConfig = Field(default_factory=BackendsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    kb: KBConfig = Field(default_factory=KBConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    source_path: Path | None = Field(default=None, exclude=True)

    # ------------------------------------------------------------------ chargement
    @classmethod
    def load(cls, path: str | Path | None = None, *, use_env: bool = True) -> "Settings":
        p = Path(path).expanduser() if path else default_config_path()
        data: dict[str, Any] = {}
        if p.is_file():
            data = tomllib.loads(p.read_text(encoding="utf-8"))
        if use_env:
            _deep_merge(data, env_overrides())
        settings = cls.model_validate(data)
        settings.source_path = p
        return settings

    def to_toml(self) -> str:
        data = self.model_dump(mode="json", exclude_none=True, exclude={"source_path"})
        return tomli_w.dumps(data)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path).expanduser() if path else (self.source_path or default_config_path())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_toml(), encoding="utf-8")
        self.source_path = p
        return p

    def with_override(self, dotted_key: str, value: Any) -> "Settings":
        """Retourne une copie avec ``section.cle`` remplacé (valeur JSON ou chaîne)."""
        data = self.model_dump(mode="json", exclude={"source_path"})
        _nested_set(data, dotted_key, _parse_env_value(value) if isinstance(value, str) else value)
        new = Settings.model_validate(data)
        new.source_path = self.source_path
        return new

    # ------------------------------------------------------------------ chemins
    def resolve_project_root(self) -> Path:
        root = self.paths.project_root or Path.cwd()
        return Path(os.path.expanduser(str(root))).resolve()

    @property
    def data_dir(self) -> Path:
        return Path(os.path.expanduser(str(self.paths.data_dir)))

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def backups_dir(self) -> Path:
        return self.data_dir / "backups"

    @property
    def journal_path(self) -> Path:
        return self.data_dir / "journal.jsonl"

    @property
    def kb_db_path(self) -> Path:
        return self.data_dir / "knowledge.db"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.sessions_dir, self.backups_dir, self.datasets_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)
