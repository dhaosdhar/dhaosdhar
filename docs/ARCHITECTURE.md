# Architecture de dhaos

Assistant de codage agentique, **cerveau commutable** (Ollama en local ou Claude via
API), **outils** disque / shell / web, **bases de savoir locales par catégorie**,
CLI + API HTTP, module d'entraînement. Python ≥ 3.11, Linux en priorité.

```
dhaos/
  config.py        Settings (pydantic) : TOML + env DHAOS__SECTION__CLE ; chemins XDG
  types.py         Message / ToolCall / ToolSpec / ChatResponse / Usage (format neutre)
  policy.py        AccessPolicy (lecture / écriture / commandes), Journal JSONL, Confirmer
  utils.py         sha256, détection binaire, troncature
  backends/        base.py (contrat Backend), ollama.py, claude.py, __init__ (get_backend)
  tools/           base.py (Tool, ToolRegistry, ToolContext), filesystem.py, shell.py,
                   web.py, knowledge.py, __init__ (default_registry)
  kb/              manager.py (KnowledgeManager = façade), store.py (SQLite + FTS5),
                   embeddings.py, chunking.py, ingest.py
  agent/           loop.py (Agent), prompts.py (prompt système), session.py (persistance)
  cli/main.py      Typer : chat, ask, kb, config, backends, sessions, journal, serve, train
  api/server.py    FastAPI : create_app(settings)
  train/           dataset.py (traces → SFT JSONL, corpus), tokenizer.py (BPE),
                   nano/model.py + nano/train.py (GPT from scratch), finetune.py (LoRA)
tests/             pytest ; conftest.py (fixture `settings` isolée), fakes.py (FakeBackend)
systemd/           unité de service pour l'API
```

## Conventions

- Identifiants en anglais, docstrings / messages utilisateur en **français**.
- Aucun accès réseau ni disque hors `tmp_path` dans les tests ; les backends et
  le web sont testés avec des doubles (`httpx.MockTransport`, `FakeBackend`,
  monkeypatch du client Anthropic). Les tests doivent passer sans Ollama,
  sans clé API, sans Internet et sans torch (tests torch marqués
  `pytest.importorskip("torch")`).
- Les **entrées du modèle sont non fiables** : chemins, commandes, URLs et
  arguments passent par `AccessPolicy` / validation de schéma avant tout effet.
- Les résultats d'outils sont du texte ; longue sortie ⇒ troncature
  (`tools.max_output_chars`), jamais d'exception hors du registre.
- Un module = un propriétaire lors de l'implémentation parallèle ; on ne
  modifie pas les fichiers de contrat (`types.py`, `config.py`, `policy.py`,
  `backends/base.py`, `tools/base.py`, signatures publiques de `kb/manager.py`,
  `agent/loop.py`, `agent/session.py`) sans le signaler.

## Flux d'une requête

1. CLI/API construit `Settings`, `AccessPolicy`, `Journal`, `KnowledgeManager`,
   `Backend`, `ToolRegistry` (`default_registry`), `ToolContext`, `Agent`.
2. `Agent.run(texte)` : prompt système (`build_system_prompt`, avec la liste
   des bases de savoir) + historique → `backend.chat(...)` (texte streamé via
   `on_text`).
3. Si `tool_calls` : chaque appel est validé et exécuté par `ToolRegistry.execute`
   (politique d'accès, confirmation via `ctx.confirm`, journal), tous les
   résultats du tour sont ajoutés comme messages `tool`, on rappelle le modèle.
4. Fin quand plus d'appels d'outils, ou `agent.max_iterations`. La session est
   sauvée (JSONL) — c'est la matière première de `train/dataset.py`.

## Outils (noms stables)

| Outil | Paramètres | Politique |
|---|---|---|
| `read_file` | `path`, `start_line?`, `end_line?` | `check_read` ; binaire refusé ; tronqué à `max_file_chars` |
| `list_dir` | `path`, `depth?` (1..3), `show_hidden?` | `check_read` |
| `find_files` | `pattern` (glob), `root?`, `max_results?` | `check_read` sur root ; respecte `kb.ignore_patterns` |
| `grep` | `pattern` (regex), `root?`, `glob?`, `case_insensitive?`, `max_results?` | `check_read` |
| `write_file` | `path`, `content`, `create_dirs?` | `check_write` (+ confirmation) ; sauvegarde ; journal |
| `edit_file` | `path`, `old_string`, `new_string`, `replace_all?` | idem ; `old_string` doit être unique sauf `replace_all` |
| `run_command` | `command`, `cwd?`, `timeout?` | `check_command` (+ confirmation) ; journal ; sortie tronquée. La liste blanche (`shell_auto_allow`) ne remplace pas `check_write` : une option qui fait écrire, supprimer ou exécuter (`find -exec/-delete/-fprint`, `sort -o`, `git log --output`, `tree -o`, `uniq IN OUT`, `pytest --basetemp`… cf. `DEFAULT_SHELL_UNSAFE_OPTIONS`) impose la confirmation, de même qu'un argument-chemin refusé par `check_read` (`cat ~/.ssh/id_rsa`, résolu depuis `cwd`) |
| `web_search` | `query`, `max_results?` | provider `web.provider` |
| `fetch_url` | `url`, `max_chars?` | http/https seulement ; pas d'adresses privées/localhost |
| `kb_list` | — | — |
| `kb_search` | `query`, `bases?` (liste), `top_k?` | — |
| `kb_add_note` | `base`, `text`, `title?` | crée la base si absente |

## Backends

- Conversion neutre → provider. `Message.raw` (assistant) contient la charge
  brute du provider pour la rejouer telle quelle (Claude : blocs de contenu, y
  compris `thinking`, **obligatoires** lors d'un enchaînement d'outils).
- Messages `tool` consécutifs ⇒ Claude : un seul message `user` avec tous les
  `tool_result` ; Ollama : un message `role: tool` par résultat.
- Claude : SDK `anthropic` (≥ 1.8, basé sur httpx2 — ne jamais lui passer
  d'objets `httpx`), `client.beta.messages.stream(...)` avec
  `thinking={"type": "adaptive", "display": ...}`, `output_config={"effort": ...}`,
  `betas=["server-side-fallback-2026-07-01"]` + `fallbacks="default"` si
  `claude.fallbacks`, outils `{"name","description","input_schema","eager_input_streaming": True}`.
  Valider les entrées d'outils (le parseur tolérant peut tronquer) ; `stop_reason`
  `max_tokens` avec un `tool_use` ⇒ ne pas exécuter ; `refusal` ⇒ ne pas exécuter.
- Ollama : `POST /api/chat` (stream NDJSON, `tools` au format function),
  `POST /api/embed`, `GET /api/tags`. Format tool-call Ollama :
  `message.tool_calls[].function.{name, arguments}`.

## Bases de savoir

SQLite : `bases(id, name UNIQUE, description, embedder, created_at)`,
`documents(id, base_id, source, title, hash, size, added_at)`,
`chunks(id, doc_id, ord, text, embedding BLOB float32)`, `chunks_fts` (FTS5,
contenu externe sur `chunks`). Recherche : cosinus (numpy) sur les chunks des
bases ciblées + BM25 FTS5, fusion RRF. Ingestion incrémentale (hash inchangé ⇒
skip). Extracteurs : texte/code (détection binaire), Markdown, HTML (bs4),
PDF (pypdf), URL (fetch web). Une base mémorise son embedder ; changer
d'embedder ⇒ `reindex`.

## Données locales

`~/.local/share/dhaos/` : `knowledge.db`, `sessions/*.jsonl`, `journal.jsonl`,
`backups/`, `datasets/`, `models/`. Config : `~/.config/dhaos/config.toml`.

## Entraînement

- `dataset.py` : sessions → exemples SFT (format messages JSONL, outils inclus
  comme texte structuré) ; bases de savoir → corpus texte.
- `tokenizer.py` : BPE octet-niveau pur Python (entraînement, encode/decode,
  save/load JSON) ; mode `bytes` sans apprentissage.
- `nano/` : GPT décodeur minimal en PyTorch (couches, têtes, dimension, contexte
  configurables), boucle d'entraînement CPU/GPU, checkpoints, échantillonnage.
  **Honnêteté** : c'est un modèle jouet, pédagogique ; la voie vers un modèle
  utile est `finetune.py` (LoRA/QLoRA d'un modèle ouvert sur les traces).
