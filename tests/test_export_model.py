"""Export d'un modèle du dépôt Ollama local (fichiers factices)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("export_model", Path(__file__).resolve().parents[1] / "packaging" / "export_model.py")
export_model = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(export_model)


def fake_store(root: Path, name: str = "qwen2.5-coder:7b", *, weights: bytes = b"GGUF-fake-weights" * 100) -> Path:
    store = root / "models"
    blobs = store / "blobs"
    blobs.mkdir(parents=True)
    layers = []

    def add(media: str, data: bytes) -> None:
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        (blobs / digest.replace(":", "-")).write_bytes(data)
        layers.append({"mediaType": media, "digest": digest, "size": len(data)})

    add(export_model.MODEL_MEDIA, weights)
    add(export_model.TEMPLATE_MEDIA, b"{{ if .System }}{{ .System }}{{ end }}{{ .Prompt }}")
    add(export_model.PARAMS_MEDIA, json.dumps({"stop": ["<|im_end|>", "<|endoftext|>"], "temperature": 0.7}).encode())
    registry, repo, tag = export_model.split_name(name)
    manifest = store / "manifests" / registry / repo / tag
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schemaVersion": 2, "layers": layers}))
    return store


def test_split_name() -> None:
    assert export_model.split_name("qwen2.5-coder:7b") == ("registry.ollama.ai", "library/qwen2.5-coder", "7b")
    assert export_model.split_name("llama3") == ("registry.ollama.ai", "library/llama3", "latest")
    assert export_model.split_name("moi/perso:v1") == ("registry.ollama.ai", "moi/perso", "v1")
    assert export_model.split_name("hf.co/org/repo:q4") == ("hf.co", "org/repo", "q4")


def test_export_copies_weights_and_builds_modelfile(tmp_path: Path) -> None:
    store = fake_store(tmp_path)
    out = tmp_path / "out"
    info = export_model.export("qwen2.5-coder:7b", out, num_ctx=8192, stores=[tmp_path / "nulle-part", store])
    assert (out / "dhaos.gguf").read_bytes() == b"GGUF-fake-weights" * 100
    modelfile = (out / "Modelfile").read_text(encoding="utf-8")
    assert modelfile.startswith("FROM /opt/dhaos/models/dhaos.gguf\n")
    assert 'TEMPLATE """{{ if .System }}' in modelfile
    assert "PARAMETER stop <|im_end|>" in modelfile and "PARAMETER stop <|endoftext|>" in modelfile
    assert "PARAMETER temperature 0.7" in modelfile and "PARAMETER num_ctx 8192" in modelfile
    assert 'SYSTEM """Tu es dhaos' in modelfile
    meta = json.loads((out / "model.json").read_text())
    assert meta["base"] == "qwen2.5-coder:7b" and meta["size"] == 1700 and meta["template"] is True
    assert info["digest"].startswith("sha256:")


def test_export_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ollama pull"):
        export_model.export("absent:1b", tmp_path / "o", stores=[tmp_path])
    store = fake_store(tmp_path)
    manifest = next((store / "manifests").rglob("7b"))
    data = json.loads(manifest.read_text())
    data["layers"] = [layer for layer in data["layers"] if layer["mediaType"] != export_model.MODEL_MEDIA]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="couche de poids"):
        export_model.export("qwen2.5-coder:7b", tmp_path / "o", stores=[store])


def test_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    store = fake_store(tmp_path)
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    assert export_model.main(["--base", "qwen2.5-coder:7b", "--out", str(tmp_path / "o2")]) == 0
    assert "poids" in capsys.readouterr().out
    assert export_model.main(["--base", "rien:1", "--out", str(tmp_path / "o3")]) == 1
