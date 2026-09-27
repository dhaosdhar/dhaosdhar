#!/bin/bash
# Construit le paquet Debian de dhaos et le contrôle (paquet ré-extrait).
#
#   packaging/build-deb.sh [--offline PYVER] [--with-model BASE] [--num-ctx N] [--out DIR]
#
# Sans option : paquet « all » léger, dépendances Python installées depuis
# PyPI par postinst. --offline 3.14 : roues des dépendances embarquées
# (Python 3.14 x86_64), installation sans réseau. --with-model BASE : les
# poids du modèle Ollama BASE (présent localement) sont embarqués avec un
# Modelfile ; à l'installation, `dhaos model import` crée le modèle « dhaos »
# dans Ollama sans téléchargement. Le paquet pèse alors la taille des poids.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/dist"; OFFLINE_PY=""; WITH_MODEL=""; NUM_CTX=""
while [ $# -gt 0 ]; do
  case "$1" in
    --offline) OFFLINE_PY="$2"; shift 2 ;;
    --with-model) WITH_MODEL="$2"; shift 2 ;;
    --num-ctx) NUM_CTX="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "option inconnue : $1" >&2; exit 2 ;;
  esac
done
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
VERSION=$(grep -m1 '^version' "$ROOT/pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')
[ -n "$VERSION" ] || { echo "version introuvable dans pyproject.toml" >&2; exit 1; }
ARCH=all; { [ -n "$OFFLINE_PY" ] || [ -n "$WITH_MODEL" ]; } && ARCH=amd64
BUILD="$ROOT/build/deb"; STAGE="$BUILD/dhaos"; CHECK="$BUILD/check"
rm -rf "$BUILD"; mkdir -p "$STAGE/DEBIAN" "$OUT"
echo "== dhaos $VERSION ($ARCH) =="

# --- roue de l'application
"$PY" -m pip wheel "$ROOT" --no-deps --quiet -w "$BUILD/wheels"
install -d "$STAGE/opt/dhaos/wheels"
cp "$BUILD"/wheels/dhaos-*.whl "$STAGE/opt/dhaos/wheels/"
cp "$ROOT/packaging/requirements.txt" "$STAGE/opt/dhaos/requirements.txt"
if [ -n "$OFFLINE_PY" ]; then
  echo "-- téléchargement des dépendances (Python $OFFLINE_PY, x86_64)…"
  "$PY" -m pip download --quiet -r "$ROOT/packaging/requirements.txt" -d "$STAGE/opt/dhaos/wheels" \
    --python-version "$OFFLINE_PY" --only-binary=:all: --implementation cp \
    --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_26_x86_64 --platform manylinux_2_27_x86_64
fi

# --- modèle embarqué (poids + Modelfile) depuis le dépôt Ollama local
if [ -n "$WITH_MODEL" ]; then
  echo "-- export du modèle $WITH_MODEL depuis le dépôt Ollama local…"
  "$PY" "$ROOT/packaging/export_model.py" --base "$WITH_MODEL" --out "$STAGE/opt/dhaos/models" ${NUM_CTX:+--num-ctx "$NUM_CTX"}
fi

# --- fichiers
install -m 0755 "$ROOT/packaging/bin/install-venv.sh" "$STAGE/opt/dhaos/install-venv.sh"
install -D -m 0755 "$ROOT/packaging/bin/dhaos" "$STAGE/usr/bin/dhaos"
install -D -m 0755 "$ROOT/packaging/bin/dhaos-setup" "$STAGE/usr/bin/dhaos-setup"
install -D -m 0644 "$ROOT/packaging/systemd/dhaos-api.service" "$STAGE/usr/lib/systemd/user/dhaos-api.service"
install -D -m 0644 "$ROOT/packaging/systemd/dhaos-api@.service" "$STAGE/lib/systemd/system/dhaos-api@.service"
install -D -m 0644 "$ROOT/packaging/desktop/dhaos.desktop" "$STAGE/usr/share/applications/dhaos.desktop"
install -D -m 0644 "$ROOT/packaging/desktop/dhaos.svg" "$STAGE/usr/share/icons/hicolor/scalable/apps/dhaos.svg"
install -D -m 0644 "$ROOT/README.md" "$STAGE/usr/share/doc/dhaos/README.md"
install -D -m 0644 "$ROOT/config.example.toml" "$STAGE/usr/share/doc/dhaos/config.example.toml"
install -D -m 0644 "$ROOT/packaging/debian/copyright" "$STAGE/usr/share/doc/dhaos/copyright"
for f in postinst prerm postrm; do install -m 0755 "$ROOT/packaging/debian/$f" "$STAGE/DEBIAN/$f"; done

# --- control + md5sums
SIZE=$(du -sk --exclude=DEBIAN "$STAGE" | cut -f1)
sed -e "s/@VERSION@/$VERSION/" -e "s/@ARCH@/$ARCH/" -e "s/@SIZE@/$SIZE/" "$ROOT/packaging/debian/control.in" > "$STAGE/DEBIAN/control"
( cd "$STAGE" && find . -type f -not -path './DEBIAN/*' -printf '%P\n' | sort | xargs md5sum > DEBIAN/md5sums )
find "$STAGE" -type d -exec chmod 0755 {} +
DEB="$OUT/dhaos_${VERSION}_${ARCH}.deb"
# Les poids d'un modèle ne se compressent pas : gzip rapide plutôt que xz (minutes vs heures).
COMPRESS=(); [ -n "$WITH_MODEL" ] && COMPRESS=(-Zgzip -z1)
dpkg-deb --build --root-owner-group "${COMPRESS[@]}" "$STAGE" "$DEB" >/dev/null

# --- contrôles sur le paquet construit (ré-extrait)
mkdir -p "$CHECK"; dpkg-deb -x "$DEB" "$CHECK"; dpkg-deb -e "$DEB" "$CHECK/DEBIAN"
( cd "$CHECK" && md5sum -c --quiet DEBIAN/md5sums ) && echo "md5sums : OK ($(wc -l < "$CHECK/DEBIAN/md5sums") fichiers)"
for f in postinst prerm postrm; do bash -n "$CHECK/DEBIAN/$f"; done && echo "scripts de maintenance : syntaxe OK"
bash -n "$CHECK/opt/dhaos/install-venv.sh" && bash -n "$CHECK/usr/bin/dhaos-setup" && echo "scripts : syntaxe OK"
grep -q "^Version: $VERSION$" "$CHECK/DEBIAN/control" && echo "control : version $VERSION"
"$PY" - "$CHECK/opt/dhaos/wheels" <<'PYEOF'
import sys, zipfile, pathlib
w = next(pathlib.Path(sys.argv[1]).glob("dhaos-*.whl"))
names = zipfile.ZipFile(w).namelist()
assert any(n.endswith("ui/index.html") for n in names), "interface web absente de la roue"
print(f"roue : {w.name} ({len(names)} fichiers, interface web incluse)")
PYEOF
if [ -n "$WITH_MODEL" ]; then
  [ -s "$CHECK/opt/dhaos/models/dhaos.gguf" ] && [ -s "$CHECK/opt/dhaos/models/Modelfile" ] && echo "modèle embarqué : $(du -h "$CHECK/opt/dhaos/models/dhaos.gguf" | cut -f1) (base $WITH_MODEL)"
fi
echo "paquet : $DEB ($(du -h "$DEB" | cut -f1))"
dpkg-deb --info "$DEB" | sed -n '/Package:/,/Depends:/p' | sed 's/^/  /'
