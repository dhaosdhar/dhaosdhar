"""Jeux de données pour l'entraînement.

- ``build_sft_dataset`` : sessions JSONL (``settings.sessions_dir``) →
  exemples SFT au format « messages » (un exemple par session). Les appels
  d'outils de l'assistant sont rendus comme blocs
  ``<tool_call>{"name": …, "arguments": {…}}</tool_call>`` ajoutés au texte ;
  les résultats d'outils deviennent des tours ``{"role": "tool", "name",
  "content"}``. Le champ ``raw`` (charge brute du fournisseur) est retiré.
- ``build_corpus`` : texte brut des bases de savoir (``kb.corpus_text``) et,
  par défaut, des sessions (tours utilisateur / assistant), documents séparés
  par ``\\n\\n<|doc|>\\n\\n`` — matière première du tokenizer et du modèle nano.

Le JSONL des sessions est parsé ici sans dépendre de ``dhaos.agent.session`` :
ligne ``{"type": "meta", …}``, lignes ``{"type": "message", …}`` (format
``Message.to_dict()``) et lignes ``{"type": "event", …}`` (ignorées). Ces
fichiers sont des données **non fiables** : toute ligne ou message invalide
est ignoré silencieusement.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..config import Settings
from .tokenizer import DOC_TOKEN

DOC_SEPARATOR = f"\n\n{DOC_TOKEN}\n\n"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
SESSION_SUFFIX = ".jsonl"
CORPUS_FILENAME = "corpus.txt"
_VALID_ROLES = ("user", "assistant", "tool")
_USER_LABEL = "### Utilisateur"
_ASSISTANT_LABEL = "### Assistant"


@dataclass
class DatasetReport:
    path: Path
    n_examples: int = 0
    n_sessions: int = 0  # fichiers de session lus
    n_sessions_skipped: int = 0  # sans traces, sans réponse, trop courtes, dupliquées, illisibles
    n_tool_turns: int = 0  # tours ``tool`` dans les exemples produits

    def summary(self) -> str:
        return (
            f"{self.n_examples} exemple(s) depuis {self.n_sessions} session(s) "
            f"({self.n_sessions_skipped} ignorée(s), {self.n_tool_turns} tour(s) d'outil) → {self.path}"
        )


@dataclass
class ParsedSession:
    id: str
    meta: dict[str, Any]
    messages: list[dict[str, Any]]

    @property
    def traces_enabled(self) -> bool:
        return self.meta.get("traces") is not False

    def has_assistant(self) -> bool:
        return any(m["role"] == "assistant" for m in self.messages)

    def assistant_chars(self) -> int:
        return sum(len(m["content"].strip()) for m in self.messages if m["role"] == "assistant")


# ------------------------------------------------------------------ parsing
def _clean_tool_calls(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return calls
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = item.get("arguments")
        calls.append({"name": name.strip(), "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


def _clean_message(body: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise un message (rôle valide, contenu texte, appels d'outils propres) ;
    ``None`` si inutilisable. Le champ ``raw`` n'est jamais conservé."""
    role = body.get("role")
    if role not in _VALID_ROLES:
        return None
    content = body.get("content")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    msg: dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
        msg["tool_calls"] = _clean_tool_calls(body.get("tool_calls"))
    elif role == "tool":
        name = body.get("name")
        msg["name"] = name.strip() if isinstance(name, str) and name.strip() else "outil"
        msg["is_error"] = bool(body.get("is_error", False))
    return msg


def parse_session_jsonl(text: str, *, fallback_id: str = "") -> ParsedSession:
    """Parse le contenu JSONL d'une session (lignes invalides ignorées)."""
    meta: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type")
        body = {k: v for k, v in obj.items() if k != "type"}
        if kind == "meta":
            meta.update(body)
        elif kind == "message":
            msg = _clean_message(body)
            if msg is not None:
                messages.append(msg)
    session_id = meta.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        session_id = fallback_id
    return ParsedSession(id=str(session_id), meta=meta, messages=messages)


def parse_session_file(path: Path) -> ParsedSession | None:
    """Lit un fichier de session ; ``None`` s'il est illisible."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return parse_session_jsonl(text, fallback_id=path.stem)


def iter_session_files(settings: Settings) -> list[Path]:
    """Fichiers ``*.jsonl`` de ``settings.sessions_dir``, triés par nom."""
    directory = Path(settings.sessions_dir)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob(f"*{SESSION_SUFFIX}") if p.is_file() and not p.name.startswith("."))


# ------------------------------------------------------------------ rendu SFT
def render_tool_calls(content: str, tool_calls: Iterable[dict[str, Any]]) -> str:
    """Texte de l'assistant suivi d'un bloc ``<tool_call>{json}</tool_call>`` par appel."""
    parts: list[str] = []
    if content and content.strip():
        parts.append(content.rstrip())
    for call in tool_calls:
        payload = json.dumps(
            {"name": call.get("name", ""), "arguments": call.get("arguments") or {}},
            ensure_ascii=False,
            default=str,
        )
        parts.append(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")
    return "\n".join(parts)


def session_to_example(session: ParsedSession) -> dict[str, Any]:
    """Un exemple SFT ``{"id", "messages", "meta"}`` pour une session."""
    messages: list[dict[str, Any]] = []
    system_prompt = session.meta.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    for msg in session.messages:
        role = msg["role"]
        if role == "assistant":
            content = render_tool_calls(msg["content"], msg["tool_calls"]) if msg["tool_calls"] else msg["content"]
            messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            messages.append({"role": "tool", "name": msg["name"], "content": msg["content"]})
        else:
            messages.append({"role": "user", "content": msg["content"]})
    meta = session.meta
    return {
        "id": session.id,
        "messages": messages,
        "meta": {
            "backend": str(meta.get("backend") or ""),
            "model": str(meta.get("model") or ""),
            "created_at": str(meta.get("created_at") or ""),
        },
    }


def example_hash(example: dict[str, Any]) -> str:
    """Empreinte du contenu (messages seulement) pour le dédoublonnage."""
    payload = json.dumps(example.get("messages", []), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_sft_dataset(
    settings: Settings,
    out_path: str | Path | None = None,
    bases: list[str] | None = None,
    *,
    min_assistant_chars: int = 20,
) -> DatasetReport:
    """Construit le jeu de données SFT (JSONL, un exemple par session).

    Sessions ignorées : ``meta.traces == false``, sans message assistant,
    dont le texte assistant cumulé fait moins de ``min_assistant_chars``
    caractères, ou dont le contenu est identique à une session déjà retenue.

    ``bases`` est accepté pour la symétrie avec la CLI mais **n'est pas
    utilisé** ici : les bases de savoir alimentent ``build_corpus``, pas le
    jeu SFT (aucun exemple synthétique question / réponse n'est fabriqué).
    Sortie : ``out_path`` ou ``settings.datasets_dir/sft-<AAAAMMJJ>.jsonl``
    (écrasé s'il existe).
    """
    del bases  # voir docstring
    min_chars = max(0, int(min_assistant_chars))
    if out_path is not None:
        target = Path(out_path)
    else:
        target = Path(settings.datasets_dir) / f"sft-{_dt.date.today():%Y%m%d}{SESSION_SUFFIX}"
    target.parent.mkdir(parents=True, exist_ok=True)

    report = DatasetReport(path=target)
    seen: set[str] = set()
    with open(target, "w", encoding="utf-8") as f:
        for path in iter_session_files(settings):
            report.n_sessions += 1
            session = parse_session_file(path)
            if (
                session is None
                or not session.traces_enabled
                or not session.has_assistant()
                or session.assistant_chars() < min_chars
            ):
                report.n_sessions_skipped += 1
                continue
            example = session_to_example(session)
            digest = example_hash(example)
            if digest in seen:
                report.n_sessions_skipped += 1
                continue
            seen.add(digest)
            f.write(json.dumps(example, ensure_ascii=False, default=str) + "\n")
            report.n_examples += 1
            report.n_tool_turns += sum(1 for m in example["messages"] if m["role"] == "tool")
    return report


# --------------------------------------------------------------------- corpus
def session_to_text(session: ParsedSession) -> str:
    """Texte d'une session pour le corpus : tours utilisateur / assistant seulement
    (le texte de l'assistant, sans les appels d'outils) ; vide si aucun tour utile."""
    blocks: list[tuple[str, str]] = []  # (étiquette, contenu)
    for msg in session.messages:
        content = msg["content"].strip()
        if not content or msg["role"] == "tool":
            continue
        label = _ASSISTANT_LABEL if msg["role"] == "assistant" else _USER_LABEL
        if blocks and blocks[-1][0] == label:
            # Tours consécutifs de même rôle (ex. autour d'un appel d'outil retiré).
            blocks[-1] = (label, blocks[-1][1] + "\n\n" + content)
        else:
            blocks.append((label, content))
    return "\n\n".join(f"{label}\n{content}" for label, content in blocks)


def iter_session_texts(settings: Settings) -> Iterator[str]:
    """Textes des sessions exploitables (traces activées, au moins une réponse)."""
    for path in iter_session_files(settings):
        session = parse_session_file(path)
        if session is None or not session.traces_enabled or not session.has_assistant():
            continue
        text = session_to_text(session)
        if text.strip():
            yield text


def build_corpus(
    settings: Settings,
    out_path: str | Path | None = None,
    bases: list[str] | None = None,
    *,
    include_sessions: bool = True,
    kb: Any = None,
) -> Path:
    """Assemble le corpus texte : documents des bases de savoir (``bases`` :
    liste de noms, ``None`` = toutes) puis, si ``include_sessions``, les
    sessions. Documents séparés par ``DOC_SEPARATOR``.

    ``kb`` : objet exposant ``corpus_text(bases)`` et ``close()`` ; par défaut
    un ``KnowledgeManager(settings)`` est ouvert puis fermé.
    Sortie : ``out_path`` ou ``settings.datasets_dir/corpus.txt``.
    """
    target = Path(out_path) if out_path is not None else Path(settings.datasets_dir) / CORPUS_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    own_kb = False
    if kb is None:
        from ..kb.manager import KnowledgeManager

        kb = KnowledgeManager(settings)
        own_kb = True
    try:
        with open(target, "w", encoding="utf-8") as f:
            n_docs = 0

            def write_doc(text: str) -> None:
                nonlocal n_docs
                body = str(text).strip()
                if not body:
                    return
                if n_docs:
                    f.write(DOC_SEPARATOR)
                f.write(body)
                n_docs += 1

            for doc in kb.corpus_text(list(bases) if bases else None):
                write_doc(doc)
            if include_sessions:
                for text in iter_session_texts(settings):
                    write_doc(text)
            if n_docs:
                f.write("\n")
    finally:
        if own_kb:
            try:
                kb.close()
            except Exception:  # noqa: BLE001 — fermeture best-effort
                pass
    return target


__all__ = [
    "CORPUS_FILENAME",
    "DOC_SEPARATOR",
    "DatasetReport",
    "ParsedSession",
    "build_corpus",
    "build_sft_dataset",
    "example_hash",
    "iter_session_files",
    "iter_session_texts",
    "parse_session_file",
    "parse_session_jsonl",
    "render_tool_calls",
    "session_to_example",
    "session_to_text",
]
