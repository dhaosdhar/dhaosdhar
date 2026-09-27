#!/usr/bin/env python3
"""Exporte un modèle du dépôt Ollama local vers le paquet : poids GGUF +
Modelfile (gabarit et paramètres du modèle de base, identité dhaos).

    python3 packaging/export_model.py --base qwen2.5-coder:7b --out build/deb/dhaos/opt/dhaos/models

Lecture seule du dépôt (``$OLLAMA_MODELS``, ``~/.ollama/models``,
``/usr/share/ollama/.ollama/models``) ; aucune dépendance à la ligne de
commande ``ollama``. Le blob de poids est copié sous ``dhaos.gguf`` ; le
Modelfile généré est importé à l'installation par ``dhaos model import``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

MODEL_MEDIA = "application/vnd.ollama.image.model"
TEMPLATE_MEDIA = "application/vnd.ollama.image.template"
PARAMS_MEDIA = "application/vnd.ollama.image.params"
SYSTEM_MEDIA = "application/vnd.ollama.image.system"
DEFAULT_IDENTITY = (
    "Tu es dhaos, un assistant de codage expert et rigoureux, exécuté localement. "
    "Tu réponds en français, avec précision et sans bavardage."
)


def candidate_stores() -> list[Path]:
    out: list[Path] = []
    if os.environ.get("OLLAMA_MODELS"):
        out.append(Path(os.environ["OLLAMA_MODELS"]))
    out.append(Path.home() / ".ollama" / "models")
    out.append(Path("/usr/share/ollama/.ollama/models"))
    return out


def split_name(name: str) -> tuple[str, str, str]:
    """``qwen2.5-coder:7b`` → (registry.ollama.ai, library/qwen2.5-coder, 7b)."""
    repo, _, tag = name.partition(":")
    tag = tag or "latest"
    parts = repo.split("/")
    if len(parts) == 1:
        return "registry.ollama.ai", f"library/{parts[0]}", tag
    if len(parts) == 2:
        return "registry.ollama.ai", repo, tag
    return parts[0], "/".join(parts[1:]), tag


def find_manifest(name: str, stores: list[Path]) -> tuple[Path, Path]:
    registry, repo, tag = split_name(name)
    for store in stores:
        manifest = store / "manifests" / registry / repo / tag
        if manifest.is_file():
            return store, manifest
    tried = ", ".join(str(s) for s in stores)
    raise FileNotFoundError(f"modèle {name} introuvable dans le dépôt Ollama (cherché dans : {tried}) — ollama pull {name} ?")


def blob_path(store: Path, digest: str) -> Path:
    return store / "blobs" / digest.replace(":", "-")


def read_layer(store: Path, layer: dict[str, Any]) -> str:
    return blob_path(store, str(layer["digest"])).read_text(encoding="utf-8")


def build_modelfile(*, template: str, params: dict[str, Any], identity: str, num_ctx: int | None, gguf_name: str = "dhaos.gguf") -> str:
    lines = [f"FROM /opt/dhaos/models/{gguf_name}"]
    if template:
        lines.append('TEMPLATE """' + template + '"""')
    for key, value in sorted(params.items()):
        values = value if isinstance(value, list) else [value]
        for v in values:
            if key == "num_ctx" and num_ctx:
                continue
            lines.append(f"PARAMETER {key} {json.dumps(v) if isinstance(v, str) and ' ' in v else v}")
    if num_ctx:
        lines.append(f"PARAMETER num_ctx {int(num_ctx)}")
    lines.append('SYSTEM """' + identity + '"""')
    return "\n".join(lines) + "\n"


def export(base: str, out_dir: Path, *, identity: str = DEFAULT_IDENTITY, num_ctx: int | None = None, stores: list[Path] | None = None, link: bool = False) -> dict[str, Any]:
    store, manifest_path = find_manifest(base, stores or candidate_stores())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = manifest.get("layers") or []
    model_layers = [layer for layer in layers if layer.get("mediaType") == MODEL_MEDIA]
    if not model_layers:
        raise ValueError(f"le manifeste de {base} ne contient pas de couche de poids ({MODEL_MEDIA})")
    if len(model_layers) > 1:
        raise ValueError(f"{base} a plusieurs couches de poids : export non pris en charge")
    weights = blob_path(store, str(model_layers[0]["digest"]))
    if not weights.is_file():
        raise FileNotFoundError(f"blob de poids absent : {weights}")
    template = ""
    params: dict[str, Any] = {}
    for layer in layers:
        media = layer.get("mediaType")
        if media == TEMPLATE_MEDIA:
            template = read_layer(store, layer)
        elif media == PARAMS_MEDIA:
            params = json.loads(read_layer(store, layer) or "{}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "dhaos.gguf"
    if link:
        if target.exists():
            target.unlink()
        os.link(weights, target)
    else:
        shutil.copyfile(weights, target)
    modelfile = build_modelfile(template=template, params=params, identity=identity, num_ctx=num_ctx)
    (out_dir / "Modelfile").write_text(modelfile, encoding="utf-8")
    info = {
        "base": base,
        "digest": str(model_layers[0]["digest"]),
        "size": weights.stat().st_size,
        "store": str(store),
        "template": bool(template),
        "parameters": params,
    }
    (out_dir / "model.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="modèle Ollama présent localement (ex. qwen2.5-coder:7b)")
    parser.add_argument("--out", required=True, type=Path, help="dossier de sortie (dhaos.gguf, Modelfile, model.json)")
    parser.add_argument("--num-ctx", type=int, default=None, help="fenêtre de contexte imposée")
    parser.add_argument("--link", action="store_true", help="lien physique au lieu d'une copie (même système de fichiers)")
    args = parser.parse_args(argv)
    try:
        info = export(args.base, args.out, num_ctx=args.num_ctx, link=args.link)
    except PermissionError as e:
        print(f"accès refusé au dépôt Ollama ({e}) : relancez avec sudo -E", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as e:
        print(f"export impossible : {e}", file=sys.stderr)
        return 1
    print(f"poids : {info['size'] / 1e9:.2f} Go ({info['digest']}) — gabarit : {'oui' if info['template'] else 'non'} — paramètres : {info['parameters']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
