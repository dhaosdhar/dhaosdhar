# dhaos

**dhaos** est un assistant de codage agentique qui tourne sur votre machine Linux. Son « cerveau » est
commutable : un modèle **Ollama** en local (par défaut `qwen2.5-coder:7b`) ou **Claude** via l'API
Anthropic. Il lit votre disque, modifie vos fichiers, lance des commandes, cherche sur le web et consulte
des **bases de savoir locales** organisées par catégorie (SQLite + FTS5, recherche hybride vecteurs +
mots-clés). Il s'utilise en **ligne de commande** (`dhaos chat`, `dhaos ask`) ou par une **API HTTP**
(`dhaos serve`), et embarque un module d'**entraînement** qui transforme vos conversations en jeux de
données. Python ≥ 3.11, Linux en priorité. Version 0.1.0.

## Un mot d'honnêteté sur « notre propre LLM »

dhaos contient un modèle « nano » entraîné *from scratch* (`dhaos train nano`). C'est un GPT minimal,
**pédagogique** : il fait comprendre tokenizer, attention, boucle d'entraînement et échantillonnage, il ne
sait pas coder. Entraîner un modèle réellement utile demande des **billions de tokens** (10¹²) et au
minimum des **centaines de milliers d'euros** de calcul (des dizaines de millions pour les grands
modèles) : ce n'est pas un projet de bureau.

La voie réaliste vers un modèle **qui vous appartient** : partir d'un **modèle ouvert** (Qwen2.5-Coder,
par exemple) et le spécialiser par **fine-tuning LoRA** sur vos propres traces d'usage. dhaos collecte
ces traces à chaque session (`agent.collect_traces = true`), en fait un jeu de données SFT et lance le
LoRA. Voir [Entraînement](#entraînement).

## Installation

Prérequis : Python ≥ 3.11, Linux, `git` ; un GPU aide Ollama sans être obligatoire (un 7B tourne sur CPU, lentement).

```sh
git clone https://github.com/dhaosdhar/dhaosdhar.git dhaosdhar && cd dhaosdhar
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # cœur : CLI, API, bases de savoir, backends
pip install -e '.[dev]'          # + pytest
pip install -e '.[embeddings]'   # + sentence-transformers (embedder local sans Ollama)
pip install -e '.[train]'        # + torch (modèle nano)
pip install -e '.[finetune]'     # + torch, transformers, peft, datasets, accelerate (LoRA)
```

**Ollama** (cerveau local, recommandé pour démarrer) :

```sh
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b     # modèle de conversation par défaut
ollama pull nomic-embed-text     # modèle d'embeddings des bases de savoir
```

**Claude** (cerveau API) — la clé est lue par le SDK Anthropic :

```sh
export ANTHROPIC_API_KEY=sk-ant-…          # ou ANTHROPIC_AUTH_TOKEN, ou un profil `ant auth login`
dhaos config set backends.default claude   # facultatif : Claude par défaut
```

**Vérification** : `dhaos backends` affiche l'état de chaque backend (`ok` / `indisponible`), le modèle
configuré, un détail (« Ollama joignable sur … », « aucune clé API détectée… ») et les modèles disponibles.

## Démarrage rapide

```sh
dhaos config init          # écrit ~/.config/dhaos/config.toml avec les valeurs par défaut
cd ~/mon-projet
dhaos chat                 # conversation interactive (REPL), projet = répertoire courant
dhaos ask "Explique la structure de ce dépôt et propose un plan de tests"
echo "Résume ce fichier" | dhaos ask          # le texte peut venir de stdin
dhaos ask --json "Liste les TODO du projet"   # résultat JSON (texte, usage, arrêt…)
```

Le REPL affiche `dhaos · backend ollama · modèle qwen2.5-coder:7b · projet /… · 2 base(s) de savoir`
puis attend vos messages ; chaque appel d'outil est affiché (`⚙ read_file(path=…)`), suivi d'un
résumé (jetons, itérations, appels d'outils, raison d'arrêt). Ctrl-C interrompt le tour en cours.

| Commande du REPL | Effet |
|---|---|
| `/help` | aide |
| `/quit`, `/exit` | quitter (ou Ctrl-D) |
| `/reset` | vider l'historique de la conversation |
| `/backend NOM` | changer de backend (`ollama`, `claude`) |
| `/model NOM` | changer de modèle |
| `/kb` | lister les bases de savoir |
| `/tools` | lister les outils disponibles |
| `/session` | afficher l'identifiant de la session (`dhaos chat --session ID` pour la reprendre) |

**Options globales** (à placer *avant* la sous-commande) :

| Option | Rôle |
|---|---|
| `--backend ollama\|claude` | backend pour cette exécution |
| `--model NOM` | modèle pour cette exécution |
| `--project DIR` | racine du projet (défaut : répertoire courant) : zone d'écriture libre |
| `--yes`, `-y` | confirmer automatiquement les actions sensibles (scripts maîtrisés seulement) |
| `--no-tools` | désactiver tous les outils : le modèle répond sans agir |
| `--config FICHIER` | fichier de configuration TOML |

Exemple : `dhaos --backend claude --project ~/api ask "Ajoute des tests à users.py"`.

## Ce que l'agent peut faire, et les garde-fous

Les douze outils exposés au modèle :

| Outil | Ce qu'il fait | Politique |
|---|---|---|
| `read_file` | lit un fichier texte (plage de lignes optionnelle) | lecture ; binaire refusé ; tronqué à `tools.max_file_chars` |
| `list_dir` | liste un dossier (profondeur 1 à 3, fichiers cachés en option) | lecture |
| `find_files` | cherche des fichiers par motif glob | lecture ; respecte `kb.ignore_patterns` |
| `grep` | cherche une expression régulière dans les fichiers | lecture |
| `write_file` | crée ou remplace un fichier | écriture (+ confirmation), sauvegarde, journal |
| `edit_file` | remplace un passage exact (unique, sauf `replace_all`) | idem |
| `run_command` | exécute `bash -c COMMANDE` (non interactive, délai maximal) | commande (+ confirmation), journal |
| `web_search` | recherche web (DuckDuckGo, SearXNG ou Brave) | selon `web.provider` |
| `fetch_url` | télécharge une page (HTML nettoyé, texte, JSON, Markdown, PDF) | http/https seulement ; adresses internes refusées |
| `kb_list` | liste les bases de savoir | — |
| `kb_search` | recherche hybride dans une, plusieurs ou toutes les bases | — |
| `kb_add_note` | ajoute une note à une base (créée si absente) | — |

Les arguments viennent du modèle et sont **non fiables** : validés (schéma JSON) puis passés
à la politique d'accès (`dhaos/policy.py`) avant tout effet. Le prompt système rappelle au
modèle que résultats d'outils et pages web sont des *données*, jamais des instructions.

**Politique d'accès** (section `[tools]`) :

| Clé | Défaut | Effet |
|---|---|---|
| `read_roots` | `["/"]` | racines lisibles : tout le disque par défaut |
| `deny_patterns` | secrets usuels | jamais lus ni écrits : `~/.ssh/**`, `~/.gnupg/**`, `~/.aws/**`, `~/.kube/**`, `~/.config/dhaos/**`, `/etc/shadow`, `/proc/**`, `.env`, `*.pem`, `*.key`, `id_rsa`, `credentials.json`… |
| `write_policy` | `project` | `project` : libre dans le projet, confirmation ailleurs ; `ask` : toujours confirmer ; `all` : jamais (dangereux) ; `deny` : aucune écriture |
| `shell_policy` | `ask` | `ask` : confirmation sauf liste blanche ; `auto` : jamais ; `deny` : pas de shell |
| `shell_auto_allow` | `ls`, `cat`, `git status`, `git diff`, `pytest`, … | commandes exécutées sans confirmation (préfixe exact ; aucun métacaractère shell — `;`, `&`, `\|`, `<`, `>`, `$`, accent grave — accepté ; une option qui écrit, supprime ou exécute — `find -exec`/`-delete`, `sort -o`, `git log --output`… — impose la confirmation, de même qu'un argument désignant un chemin refusé par `deny_patterns` / `read_roots` : `cat ~/.ssh/id_rsa`, `grep -r x ~/.aws`, `tail .env`) |
| `command_timeout` | `120` | délai maximal d'une commande (secondes) |
| `backup_before_write` | `true` | l'ancien contenu est copié dans `backups/` avant écrasement |
| `max_output_chars` / `max_file_chars` | `40000` / `200000` | troncature des résultats d'outils / de `read_file` |

Autres garde-fous : les commandes reçoivent un environnement expurgé des variables dont le nom
contient `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD` ou `CREDENTIAL`, des variables d'agent
(`SSH_AUTH_SOCK`, `SSH_AGENT_PID`, `GPG_AGENT_INFO`) et de toute variable dont la valeur est une URL
avec identifiants (`postgres://user:mdp@hôte/…`) ; `env` / `printenv` ne sont pas en liste blanche ; `fetch_url` refuse
localhost et les réseaux privés, à chaque redirection ; l'agent s'arrête après `agent.max_iterations` tours (40). Risque résiduel assumé : `pytest` (et `python -m pytest`) figure en liste blanche alors qu'il exécute par construction le code du projet (`conftest.py`, plugins, `-p`) ; retirez-le de `shell_auto_allow` si le projet n'est pas de confiance.

**Confirmations.** En CLI, une question oui/non apparaît (`Exécuter : make install ?`). Si l'entrée
standard n'est pas interactive (tube, script, service), la confirmation est **refusée** et le modèle
reçoit un résultat en erreur ; `--yes` confirme tout. **Sauvegardes** :
`~/.local/share/dhaos/backups/<horodatage>__<chemin-aplati>`, chemin renvoyé au modèle et journalisé.

**Journal.** Chaque écriture et chaque commande est ajoutée à `~/.local/share/dhaos/journal.jsonl`
(`kind` : `write_file`, `edit_file`, `run_command` ; champs `path`, `bytes`, `backup`,
`confirmed`, `replacements` ou `command`, `cwd`, `exit`, `duration`, `confirmed`).

```sh
dhaos journal -n 20                                   # tableau des 20 dernières entrées
jq -c 'select(.kind=="run_command")' ~/.local/share/dhaos/journal.jsonl   # ou le fichier brut
```

**Durcir** (config.toml) : `read_roots = ["~/projets"]`, `write_policy = "ask"`, `shell_policy = "deny"`,
`deny_patterns` étendu (`"~/Documents/**"`). **Assouplir** (à vos risques) : `shell_policy = "auto"`,
`write_policy = "all"`, ou plus finement `shell_auto_allow = ["ls", "cat", "git status", "npm test", "make"]`.
Une ligne suffit : `dhaos config set tools.shell_policy deny`.

## Bases de savoir

Une **base** est une catégorie de savoir nommée (`[a-z0-9._-]`, 64 caractères max) :
`developpeur` (conventions, outils), `infra` (serveurs, réseau), `projet-x`
(spécifications)… Sa description est montrée au modèle dans le prompt système.

```sh
dhaos kb create developpeur -d "Conventions, outils et bonnes pratiques"
dhaos kb create infra -d "Serveurs, systemd, réseau"
dhaos kb add developpeur docs/ README.md ~/notes/python.md          # fichiers et dossiers (récursif)
dhaos kb add infra https://www.freedesktop.org/software/systemd/man/systemd.unit.html
dhaos kb add projet-x ./specs --no-recursive
dhaos kb search "politique d'écriture et confirmation"               # toutes les bases
dhaos kb search "unité utilisateur" --base infra --base developpeur -k 5 --mode keyword
dhaos kb list                       # nom, description, docs, chunks, embedder
dhaos kb show infra                 # documents d'une base (id, source, titre, chunks, taille)
dhaos kb rename projet-x projet-y
dhaos kb describe infra "Serveurs, systemd, réseau, sauvegardes"
dhaos kb remove infra /home/moi/notes/obsolete.md      # chemin, URL ou note:<titre>
dhaos kb stats                      # global (ou `dhaos kb stats infra`)
dhaos kb reindex infra              # recalcule les embeddings avec l'embedder courant
dhaos kb export infra infra.jsonl   # {source, title, text, added_at} par ligne
dhaos kb delete projet-y -y         # supprime la base et ses documents
```

`kb add` affiche `ajouté : /…/docs/ARCHITECTURE.md (5 chunk(s))` puis un bilan (`2 ajouté(s), 0 mis à jour,
0 ignoré(s), 0 échec(s), 8 chunk(s)`). L'ingestion est **incrémentale** (contenu inchangé ⇒ `inchangé`).
Formats : texte et code (détection binaire), Markdown, HTML, PDF, URLs ; fichiers au-delà de `kb.max_file_bytes`
(5 Mo) et motifs `kb.ignore_patterns` (`.git`, `node_modules`, `.venv`, `*.lock`…) ignorés ; découpage en
chunks de `kb.chunk_chars` (2400) caractères avec chevauchement (300).

**Recherche** (`--mode`) : `hybrid` (défaut) fusionne cosinus sur les embeddings et BM25 FTS5 (fusion
RRF) ; `vector` et `keyword` isolent l'un ou l'autre ; `kb.top_k` (8) résultats par défaut.

**Embedders** (`kb.embedder`) :

| Valeur | Comportement |
|---|---|
| `auto` (défaut) | Ollama si joignable et `backends.ollama.embed_model` présent, sinon `hash` |
| `ollama` | `POST /api/embed` avec `nomic-embed-text` (bonne qualité, local) |
| `sentence-transformers` | modèle `kb.st_model` (`all-MiniLM-L6-v2`), extra `[embeddings]`, sans Ollama |
| `hash` | hachage de n-grammes (`kb.hash_dim` = 512) : toujours disponible, qualité modeste, quasi lexical |

Chaque base **mémorise son embedder** (`hash:512`, `ollama:nomic-embed-text:768`…). Si l'embedder
courant diffère, l'ajout est refusé avec un message explicite : lancez `dhaos kb reindex NOM`.
Tant qu'une base n'est pas réindexée, seule la recherche par mots-clés y fonctionne.

**Comment l'agent s'en sert** : le prompt système liste les bases et, avec `agent.auto_kb_search = true`
(défaut), demande au modèle d'appeler `kb_search` avant de répondre sur un sujet couvert ; il mémorise
ses apprentissages avec `kb_add_note`. Vous pouvez aussi lui dire « cherche dans la base infra ».
**Données** : `~/.local/share/dhaos/knowledge.db` (une seule base SQLite pour toutes les catégories),
déplaçable via `DHAOS_DATA_DIR` ou `paths.data_dir`.

## Configuration

| Source | Emplacement |
|---|---|
| fichier | `$DHAOS_CONFIG`, sinon `$XDG_CONFIG_HOME/dhaos/config.toml` (`~/.config/dhaos/config.toml`) |
| données | `$DHAOS_DATA_DIR`, sinon `$XDG_DATA_HOME/dhaos` (`~/.local/share/dhaos`) : `knowledge.db`, `sessions/`, `journal.jsonl`, `backups/`, `datasets/`, `models/` |
| environnement | `DHAOS__SECTION__CLE=valeur` ; valeur lue comme JSON si possible (`true`, `42`, `["a","b"]`), sinon comme chaîne |

Priorité : valeurs par défaut < fichier TOML < environnement. Exemples :
`DHAOS__BACKENDS__DEFAULT=claude`, `DHAOS__BACKENDS__OLLAMA__MODEL=qwen2.5-coder:14b`,
`DHAOS__TOOLS__SHELL_POLICY=deny`, `DHAOS__TOOLS__READ_ROOTS='["/home/moi/projets"]'`.

| Section | Clés principales (défaut) |
|---|---|
| `paths` | `data_dir` (XDG), `project_root` (répertoire courant) |
| `backends` | `default` (`ollama`) |
| `backends.ollama` | `host` (`http://127.0.0.1:11434`), `model` (`qwen2.5-coder:7b`), `embed_model` (`nomic-embed-text`), `timeout` (300 s), `num_ctx` (16384), `keep_alive` (non défini, ex. `"30m"`) |
| `backends.claude` | `model` (`claude-opus-5`), `max_tokens` (64000), `effort` (`xhigh` ; `low`…`max`), `thinking_display` (`omitted`/`summarized`), `fallbacks` (true), `timeout` (600 s), `base_url` |
| `tools` | voir [garde-fous](#ce-que-lagent-peut-faire-et-les-garde-fous) |
| `web` | `provider` (`duckduckgo` / `searxng` + `searxng_url` / `brave` + `brave_api_key`), `max_results` (8), `fetch_max_chars` (40000), `timeout` (20 s), `user_agent` |
| `kb` | `embedder` (`auto`), `st_model`, `hash_dim` (512), `chunk_chars` (2400), `chunk_overlap` (300), `top_k` (8), `max_file_bytes` (5 000 000), `ignore_patterns` |
| `agent` | `max_iterations` (40), `collect_traces` (true), `auto_kb_search` (true), `extra_system_prompt` (""), `language` (`fr`) |
| `api` | `host` (`127.0.0.1`), `port` (8642), `token` (aucun), `auto_confirm` (false) |
| `train.nano` | `n_layer` 4, `n_head` 4, `n_embd` 128, `block_size` 256, `batch_size` 16, `learning_rate` 3e-4, `steps` 500, `tokenizer` (`bpe`/`bytes`), `bpe_vocab_size` 2048, `dropout` 0 |
| `train.finetune` | `base_model` (`Qwen/Qwen2.5-Coder-1.5B-Instruct`), `lora_r` 16, `lora_alpha` 32, `lora_dropout` 0.05, `epochs` 1, `learning_rate` 2e-4, `max_seq_len` 2048, `batch_size` 2, `gradient_accumulation` 8, `load_in_4bit` true |

```sh
dhaos config path                         # où est le fichier
dhaos config init [--force]               # écrire les valeurs par défaut (--force écrase)
dhaos config show                         # configuration effective, secrets masqués (api.token, web.brave_api_key)
dhaos config set backends.ollama.model qwen2.5-coder:14b
dhaos config set agent.max_iterations 60  # valeur validée (clé inconnue ou type invalide ⇒ erreur)
```

`config.example.toml` à la racine du dépôt documente chaque section.

## Interface web

`dhaos serve` sert aussi une **interface web** sur la même adresse (`http://127.0.0.1:8642/`) :
conversation en flux avec les appels d'outils affichés au fil de l'eau, **confirmations interactives**
(« Exécuter : make install ? » — Oui / Non, dans la page), choix du cerveau et du modèle, gestion
complète des bases de savoir (créer, ingérer fichiers/dossiers/URLs, notes, recherche, renommer,
réindexer, supprimer), sessions, journal des actions et réglages courants. Sans dépendance externe,
utilisable hors ligne, thème clair/sombre.

```sh
dhaos serve --open          # lance l'API et ouvre l'interface dans le navigateur
```

Au démarrage, `dhaos serve` affiche l'URL de l'interface **avec le jeton** (`/?token=…`) : la page le
mémorise dans le navigateur et le retire de l'adresse. Avec `api.token` défini, la page le demande
une fois. Les confirmations attendent `api.confirm_timeout` secondes (300) avant de refuser l'action.

## API HTTP

```sh
dhaos serve                        # http://127.0.0.1:8642 (api.host / api.port)
dhaos serve --host 0.0.0.0 --port 9000
```

L'API donne le même accès disque que la CLI : **toutes les routes exigent** `Authorization: Bearer <jeton>`.
Le jeton est `api.token` ou, s'il n'est pas défini, un jeton aléatoire généré à chaque démarrage et
affiché par `dhaos serve`. `/health` reste joignable sans jeton mais ne renvoie alors qu'un état minimal
(sans détail du backend ni appel sortant). Un en-tête `Host` inattendu est refusé (400 ; défaut :
`localhost`, `127.0.0.1`, `::1` et `api.host`, ajustable avec `api.allowed_hosts`, `"*"` pour un reverse
proxy de confiance) et toute requête portant un `Origin` absent de `api.allowed_origins` (vide par défaut)
reçoit 403 : aucun client navigateur n'est attendu, ce qui neutralise le DNS rebinding. FastAPI génère une
documentation interactive sur `/docs`.

| Méthode et chemin | Rôle |
|---|---|
| `GET /health` | état du service, du backend par défaut et des bases (sans jeton : état minimal) |
| `GET /config` | configuration effective, secrets masqués |
| `GET /kb` · `POST /kb` | lister les bases · créer (`{"name", "description"}`) |
| `POST /kb/search` | `{"query", "bases"?, "top_k"?, "mode"?}` → passages avec base, source, score |
| `GET /kb/stats` | statistiques globales |
| `GET /kb/{name}` · `PATCH /kb/{name}` · `DELETE /kb/{name}` | détail + documents · `{"new_name"?, "description"?}` · suppression (204) |
| `GET /kb/{name}/stats` | statistiques d'une base |
| `POST /kb/{name}/documents` | ingérer `{"sources": [chemins ou URLs], "recursive"?}` (chaque fichier, y compris dans un dossier, est soumis à la politique de lecture : `tools.deny_patterns`) |
| `DELETE /kb/{name}/documents` | retirer un document : corps `{"source"}` ou `?source=` |
| `POST /kb/{name}/notes` | ajouter une note `{"text", "title"?}` |
| `GET /` · `GET /ui/*` | interface web (sans jeton : aucun secret dans la page) |
| `GET /models?backend=` | modèles disponibles (liste Ollama, modèles Claude connus) |
| `PATCH /config` | `{"key", "value"}` : modifie un réglage courant (liste blanche), fichier + application à chaud |
| `POST /kb/{name}/reindex` | recalcule les embeddings de la base |
| `POST /chat/confirm` | `{"id", "answer"}` : réponse à un événement SSE `confirm` |
| `POST /chat` | un tour d'agent : `{"message", "session_id"?, "backend"?, "model"?, "stream"? (true), "no_tools"?}` |
| `GET /sessions` · `GET /sessions/{id}` · `DELETE /sessions/{id}` | sessions persistées |
| `GET /journal?n=50` | dernières entrées du journal (1 à 1000) |

```sh
TOKEN=monjeton   # dhaos config set api.token monjeton, puis relancer le serveur (ou jeton affiché par dhaos serve)
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  http://127.0.0.1:8642/kb/search -d '{"query": "unité systemd utilisateur", "bases": ["infra"], "top_k": 3}'

# chat, réponse JSON complète
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  http://127.0.0.1:8642/chat -d '{"message": "Quels fichiers de config lit dhaos ?", "stream": false}'
# → {"text": "…", "session_id": "20260926-162649-1c59", "usage": {…}, "stop_reason": "end_turn",
#    "iterations": 1, "tool_calls": 0, "error": null}

# chat en flux SSE (défaut) : événements text, thinking, tool_call, tool_result, confirm, done, error
curl -sN -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  http://127.0.0.1:8642/chat -d '{"message": "Résume docs/ARCHITECTURE.md", "session_id": "20260926-162649-1c59"}'
# event: text
# data: {"text": "Le dépôt…"}
# event: done
# data: {"session_id": "…", "usage": {…}, "stop_reason": "end_turn", "iterations": 2, "tool_calls": 1, "error": null}
```

L'identifiant de session est aussi renvoyé dans l'en-tête `X-Session-Id` ; un commentaire `: keep-alive`
est émis toutes les 15 s sans événement ; une session déjà occupée par un tour renvoie **409**.

**systemd** (unité utilisateur, voir `systemd/dhaos-api.service`) :

```sh
mkdir -p ~/.config/systemd/user && cp systemd/dhaos-api.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now dhaos-api
journalctl --user -u dhaos-api -f
loginctl enable-linger $USER      # démarrer sans session ouverte
```

```ini
[Service]
# Adaptez ExecStart au chemin de votre .venv ; EnvironmentFile (0600) pour ANTHROPIC_API_KEY.
ExecStart=%h/dhaosdhar/.venv/bin/dhaos serve
Restart=on-failure
# EnvironmentFile=%h/.config/dhaos/env
NoNewPrivileges=true
```

**Avertissement.** Le serveur écoute sur `127.0.0.1` par défaut : gardez-le ainsi, ou placez-le derrière
un reverse proxy TLS (nginx, Caddy) avec `api.token` fixé et `api.allowed_hosts` renseigné si vous
l'exposez ; il a le même accès disque que la CLI. L'API n'a personne à qui demander une confirmation : avec `api.auto_confirm = false` (défaut),
les écritures hors projet et les commandes hors liste blanche sont **refusées** ; `api.auto_confirm = true`
les confirme toutes — dangereux sur un serveur accessible.

## Entraînement

| Commande | Produit | Où |
|---|---|---|
| `dhaos train dataset [--out F]` | jeu SFT JSONL, un exemple par session (`{"id", "messages", "meta"}`), appels d'outils rendus en `<tool_call>{…}</tool_call>`, tours `tool` conservés | `datasets/sft-AAAAMMJJ.jsonl` |
| `dhaos train corpus [--out F] [--base B]` | corpus texte : documents des bases (toutes ou `--base`) + sessions, séparés par `<\|doc\|>` | `datasets/corpus.txt` |
| `dhaos train nano CORPUS [--name N] [--steps] [--n-layer] [--n-head] [--n-embd] [--block-size] [--batch-size] [--lr]` | GPT minimal PyTorch + tokenizer BPE (extra `[train]`) | `models/<name>/` : `model.pt`, `config.json`, `tokenizer.json` |
| `dhaos train sample MODEL_DIR "amorce" [--max-new-tokens 200] [--temperature 0.8]` | texte généré par un modèle nano (CPU) | stdout |
| `dhaos train lora DATASET [--out D] [--base-model HF] [--epochs E]` | adaptateur LoRA PEFT d'un modèle ouvert (extra `[finetune]`) | `models/lora-<base>/` : `adapter/`, `Modelfile`, `README.md`, `training.json` |

Sessions ignorées par `train dataset` : traces désactivées, sans réponse de l'assistant, texte assistant
< 20 caractères, doublons. `--base` y est accepté pour compatibilité mais sans effet sur le jeu SFT (un avertissement le rappelle) : les bases alimentent `train corpus`.

**Attentes réalistes.** Le modèle nano apprend la *forme* d'un corpus : avec quelques dizaines de Ko il
produit du charabia, avec quelques Mo des phrases plausibles, jamais du code correct. Extrait réel
(corpus de 12 Ko, 2 couches, 64 dimensions, 100 étapes, 4 s sur CPU) :

```
4436 token(s) (bpe, vocabulaire 1293) : 3992 entraînement / 444 validation
modèle : 186944 paramètre(s), 2 couche(s), 2 tête(s), dim 64
étape 10/100 : perte 7.0263, validation 7.0885, lr 2.99e-04
étape 100/100 : perte 6.3417, validation 6.7363, lr 3.01e-05
modèle nano : 186 944 paramètres, 100 étape(s), perte finale 6.3417, validation 6.7363, tokenizer bpe, 3.8 s → …/models/demo
```

Le LoRA, lui, donne un modèle utilisable : il conserve les compétences du modèle de base et adopte
votre style, vos conventions, vos outils. Comptez des **centaines** de sessions de qualité pour un effet net.

**Pipeline recommandé vers un modèle personnel** :

```sh
dhaos sessions list                    # 1. utiliser dhaos normalement : chaque session est une trace
dhaos train dataset --out ~/sft.jsonl  # 2. construire le jeu (élaguer avant : dhaos sessions delete ID)
pip install -e '.[finetune]' bitsandbytes
dhaos train lora ~/sft.jsonl --epochs 2   # 3. défaut : Qwen/Qwen2.5-Coder-1.5B-Instruct, 4 bits si GPU
cd ~/.local/share/dhaos/models/lora-qwen2.5-coder-1.5b-instruct
ollama pull qwen2.5-coder:1.5b-instruct   # 4. importer dans Ollama avec le Modelfile généré
ollama create dhaos-qwen2.5-coder-1.5b-instruct -f Modelfile
dhaos config set backends.ollama.model dhaos-qwen2.5-coder-1.5b-instruct   # 5. en faire le cerveau
```

Le `README.md` généré à côté de l'adaptateur explique aussi la fusion des poids (`merge_and_unload`)
pour obtenir un modèle Hugging Face autonome, convertible en GGUF. **Matériel** : nano, n'importe quel
CPU ; LoRA 4 bits d'un 1.5B, un GPU NVIDIA avec ~8 Go de VRAM ; un 7B, ~16–24 Go ; sans GPU, pleine
précision sur CPU (« très lent », 4 bits désactivé).

## Sessions et journal

Chaque `chat` ou `ask` crée une session : `~/.local/share/dhaos/sessions/<id>.jsonl`
(`{"type": "meta"}`, puis un message par ligne, puis des événements d'usage). Les
sessions sans réponse de l'assistant sont supprimées à la sortie de la CLI.

```sh
dhaos sessions list                 # id, titre, backend, modèle, messages, mise à jour
dhaos sessions show 20260926-162649-8a80 --width 300   # messages abrégés, appels d'outils
dhaos sessions delete 20260926-162649-8a80
dhaos chat --session 20260926-162649-8a80              # reprendre une conversation
dhaos journal -n 50                 # écritures et commandes (voir garde-fous)
```

`agent.collect_traces = false` marque les nouvelles sessions `traces: false` :
conservées, mais exclues de `train dataset` et `train corpus`.

## Performances sur CPU (sans GPU)

Le temps de réponse dépend presque entièrement d'Ollama, pas de dhaos. Deux mesures suffisent :

```sh
ollama run qwen2.5-coder:7b --verbose "Dis bonjour"   # prompt eval rate / eval rate en tokens/s
free -h                                                 # RAM disponible et swap utilisé
```

| Symptôme | Cause | Remède |
|---|---|---|
| `prompt eval rate` < 20 tok/s, swap utilisé, RAM « available » < taille du modèle (`ollama ps`) | le modèle est en partie en mémoire d'échange | fermer ce qui consomme (machines virtuelles, navigateurs), modèle plus petit (`qwen2.5-coder:3b`, 1,9 Go), `backends.ollama.num_ctx` à 8192 (KV cache divisé par deux) |
| `prompt eval rate` 50–150 tok/s mais réponse longue | débit normal d'un 7B sur CPU (4–10 tok/s en génération) | modèle plus petit pour les tâches simples, Claude (`--backend claude`) pour les tâches lourdes, GPU |
| 30–60 s avant le premier appel d'outil | lecture du prompt système et des schémas d'outils (~2 000 tokens) ; ensuite Ollama réutilise le cache du préfixe | normal ; `--no-tools` pour une question sans action |

Un changement de modèle ne décharge pas l'ancien (`ollama ps` les liste ; `keep_alive` les garde) :
`ollama stop qwen2.5-coder:7b` libère sa mémoire. Les passages renvoyés à l'agent par `kb_search`
sont tronqués (`kb.tool_snippet_chars`, 700 par défaut, `kb.tool_top_k` passages) : huit chunks entiers
représentent 4 000 tokens qu'un modèle local met plusieurs minutes à lire.

Ordre de grandeur mémoire : poids du modèle (7B Q4 ≈ 4,7 Go ; 3B ≈ 1,9 Go ; 1,5B ≈ 1 Go) + KV cache
(≈ 0,9 Go pour 16 k de contexte sur un 7B) + le reste de la machine. `keep_alive = "30m"` évite de recharger
le modèle (≈ 20 s) entre deux requêtes.

## Dépannage

| Symptôme | Cause et remède |
|---|---|
| `Ollama injoignable sur http://127.0.0.1:11434 — lancez ollama serve` | Ollama arrêté ou autre hôte : `ollama serve`, ou `dhaos config set backends.ollama.host http://…` |
| `modèle absent : ollama pull qwen2.5-coder:7b` | `ollama pull <modèle>` ; `dhaos backends` liste les modèles présents |
| `Ollama n'a pas répondu dans le délai (300 s)` | modèle trop lourd : `backends.ollama.timeout` plus grand ou modèle plus léger (`qwen2.5-coder:1.5b`) |
| `aucune clé API détectée` (`dhaos backends`), `Could not resolve authentication method` (à l'appel) | `export ANTHROPIC_API_KEY=…` (ou `ANTHROPIC_AUTH_TOKEN`, ou `ant auth login`) dans le shell qui lance dhaos ou le service |
| `clé API absente ou invalide`, `modèle inconnu : …` | clé révoquée, ou `backends.claude.model` erroné |
| `recherche indisponible : …` | DuckDuckGo via `ddgs` (installé avec dhaos) injoignable : réseau, proxy, quota ; alternative `web.provider = "searxng"` + `searxng_url`, ou `"brave"` + `brave_api_key` |
| `la base 'x' a été indexée avec l'embedder hash:512 alors que l'embedder courant est ollama:…` | l'embedder a changé (Ollama démarré depuis, `kb.embedder` modifié) : `dhaos kb reindex x`, ou revenir à l'ancien embedder |
| `embedder indisponible : Ollama injoignable` | `kb.embedder = "ollama"` sans Ollama : démarrez-le ou passez à `auto`, `hash` ou `sentence-transformers` |
| `base inconnue : x` (CLI, `404` en API) | `dhaos kb list` (noms normalisés en minuscules) ; `dhaos kb create x` |
| `écriture refusée : confirmation refusée par l'utilisateur (hors du projet … : confirmation requise)`, `commande refusée : …` | l'API (ou une CLI non interactive) ne peut pas poser de question : `--project` adapté, `shell_auto_allow` élargi, `--yes` en CLI, ou `api.auto_confirm = true` en connaissance de cause |
| `session occupée` (HTTP 409) | un tour est en cours sur cette session : attendez ou ouvrez-en une nouvelle |
| `fine-tuning indisponible : dépendances manquantes : transformers, peft, datasets` / `torch est requis pour le modèle nano` | `pip install -e '.[finetune]'` / `pip install -e '.[train]'` |

## Architecture, tests, licence

- **Architecture** : `docs/ARCHITECTURE.md` (modules, flux d'une requête, contrat des outils, schéma SQLite, backends).
- **Tests** : `pip install -e '.[dev]'` puis `.venv/bin/pytest` — 745 tests, sans réseau, sans Ollama, sans clé API (tests torch ignorés si torch est absent).
- **Licence** : à définir.
