#!/bin/bash
# Crée (ou met à jour) l'environnement Python de dhaos dans /opt/dhaos/venv.
# Hors ligne si des roues de dépendances sont présentes dans /opt/dhaos/wheels,
# sinon depuis PyPI. Relançable : sudo /opt/dhaos/install-venv.sh
set -e
BASE=/opt/dhaos
VENV=$BASE/venv
PY=${PYTHON:-python3}
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "dhaos : Python >= 3.11 requis ($("$PY" --version 2>&1))." >&2
  echo "        Ubuntu 22.04 : sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv" >&2
  echo "        puis : sudo PYTHON=python3.11 /opt/dhaos/install-venv.sh" >&2
  exit 1
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "dhaos : création de l'environnement Python ($("$PY" --version 2>&1))…"
  "$PY" -m venv "$VENV"
fi
PIP="$VENV/bin/python -m pip"
$PIP install --quiet --upgrade pip >/dev/null
WHEEL=$(ls -1 "$BASE"/wheels/dhaos-*.whl | tail -n 1)
if ls "$BASE"/wheels/*.whl 2>/dev/null | grep -v '/dhaos-' >/dev/null; then
  echo "dhaos : installation hors ligne depuis $BASE/wheels…"
  $PIP install --quiet --no-index --find-links "$BASE/wheels" -r "$BASE/requirements.txt" "$WHEEL"
else
  echo "dhaos : installation des dépendances depuis PyPI…"
  $PIP install --quiet -r "$BASE/requirements.txt" "$WHEEL"
fi
"$VENV/bin/dhaos" version >/dev/null
echo "dhaos : environnement prêt ($("$VENV/bin/dhaos" version))."
