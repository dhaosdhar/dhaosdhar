# Livraison — dhaos 0.1.1 (interface de bureau, modèle dhaos, paquet avec modèle)

**Base :** 0.1.0 (premier paquet, interface web seule) — **ce paquet la contient et la remplace** ; `apt` le traite comme une mise à jour.
**Déploiement :** `sudo apt install ./dhaos_0.1.1_all.deb -y`, puis `dhaos-setup` (utilisateur), puis `dhaos gui`.

**Depuis 0.1.0** : interface de bureau Tkinter (`dhaos gui`, lanceur du menu), modèle Ollama `dhaos` par défaut (`dhaos model create/import/list/show/remove`), prompt système qui affirme l'accès au disque et à Internet, schémas d'outils compacts, paquet `--with-model` (poids embarqués), dépendance `python3-tk`.

## 1 · Ce que le paquet installe

| Chemin | Rôle |
|---|---|
| `/opt/dhaos/venv` | environnement Python créé par `postinst` (dépendances épinglées, `/opt/dhaos/requirements.txt`) |
| `/opt/dhaos/wheels/dhaos-0.1.1-py3-none-any.whl` | l'application (CLI, API, interface web) |
| `/usr/bin/dhaos` | commande principale (`chat`, `ask`, `kb`, `serve`, `ui`, `train`…) |
| `/usr/bin/dhaos-setup` | installe Ollama si absent, télécharge les modèles selon la RAM, prépare la configuration |
| `/usr/lib/systemd/user/dhaos-api.service` | service utilisateur : `systemctl --user enable --now dhaos-api` |
| `/lib/systemd/system/dhaos-api@.service` | service système par compte : `systemctl enable --now dhaos-api@js` |
| `/usr/share/applications/dhaos.desktop` | lanceur de bureau → `dhaos ui` |
| `/usr/share/doc/dhaos/` | README, `config.example.toml`, copyright |

Variante hors ligne : `packaging/build-deb.sh --offline 3.14` embarque les roues des dépendances (paquet `amd64`).

## 2 · Interface de bureau et modèle `dhaos`

- `dhaos gui` (lanceur « dhaos » du menu) : application Python/Tkinter — onglets = conversations simultanées, fil
  complet, confirmations en ligne, bases de savoir, journal, réglages. Dépendance `python3-tk` déclarée.
- Le modèle par défaut est `dhaos`, un modèle Ollama construit sur `qwen2.5-coder` (identité + paramètres) ;
  `dhaos-setup` le crée (`dhaos model create`) ou l'importe s'il est livré dans le paquet.
- **Non exécuté ici** : l'interface Tkinter (le conteneur de construction n'a pas Tk) — code vérifié par analyse
  statique et contrôleur testé sans affichage (8 tests) ; à essayer en premier chez l'utilisateur.

## 3 · Les modèles

Deux variantes de paquet :

- `dhaos_0.1.1_all.deb` (~210 Ko) : sans poids ; `dhaos-setup` installe Ollama (installeur officiel, via sudo),
  choisit `qwen2.5-coder:7b` si ≥ 9 Go de mémoire sont disponibles, sinon `:3b`, télécharge les poids et
  `nomic-embed-text`, puis crée le modèle `dhaos`.
- `dhaos_0.1.1_amd64.deb` construit avec `--with-model` : poids + Modelfile dans `/opt/dhaos/models` ;
  `postinst` importe le modèle si Ollama répond, sinon `dhaos-setup` le fait ; aucun téléchargement de
  poids. Chaîne validée ici avec un dépôt Ollama factice (export, construction, installation, importation
  différée, purge).

## À TESTER

1. `sudo apt install ./dhaos_0.1.1_all.deb -y` → message final « dhaos est installé », `dhaos version` affiche `dhaos 0.1.1`.
2. `dhaos-setup` (sans sudo) → Ollama présent, `ollama list` montre le modèle choisi et `nomic-embed-text`, `dhaos backends` affiche « ok ».
3. `dhaos gui` → la fenêtre s'ouvre (onglet « Nouvelle conversation ») ; `dhaos ui` → le navigateur s'ouvre sur `http://127.0.0.1:8642/`, point vert « prêt » en haut à droite ; poser une question, voir les appels d'outils (`⚙`).
4. Dans l'interface, demander « crée un fichier essai.txt dans /tmp » → une carte **Confirmation demandée** apparaît (écriture hors projet) ; refuser → l'agent reçoit le refus.
5. Onglet **Bases de savoir** : créer `developpeur`, ingérer `/usr/share/doc/dhaos/README.md`, chercher « politique d'accès » → passages affichés ; poser la question dans la conversation → l'agent appelle `kb_search`.
6. Menu des applications → « dhaos » ouvre l'interface de bureau ; deux onglets peuvent travailler en même temps.
7. `systemctl --user enable --now dhaos-api` puis fermer le terminal → l'interface reste accessible.
8. `sudo apt remove dhaos` → `/opt/dhaos/venv` supprimé ; `sudo apt purge dhaos` → `/opt/dhaos` supprimé ; `~/.config/dhaos` et `~/.local/share/dhaos` (données) conservés.

### Notes techniques

- Construction : `packaging/build-deb.sh` (roue via `pip wheel`, `dpkg-deb --build --root-owner-group`), contrôles sur le paquet **ré-extrait** : `md5sum -c` (11 fichiers), syntaxe des scripts de maintenance et des lanceurs, version dans `control`, interface web présente dans la roue.
- Vérifié dans un conteneur Ubuntu 24.04 : installation (`postinst` : venv + PyPI en 19 s), `dhaos version`, `dhaos-setup --help`, unités systemd, désinstallation et purge.
- Le `postinst` a besoin de PyPI (sauf variante hors ligne) ; en cas d'échec : `sudo /opt/dhaos/install-venv.sh` relance l'installation de l'environnement.
- Python ≥ 3.11 requis ; Ubuntu 22.04 : `PYTHON=python3.11 sudo /opt/dhaos/install-venv.sh` après installation de `python3.11-venv` (deadsnakes).
- **Rollback** : `sudo apt remove dhaos` ; les données utilisateur ne sont jamais touchées par le paquet.
- Non testé ici : `dhaos-setup` de bout en bout (nécessite l'installeur Ollama et le réseau vers ollama.com, bloqués dans le conteneur de construction) ; le script est validé syntaxiquement et par lecture.
