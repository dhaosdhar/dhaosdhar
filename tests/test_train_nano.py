"""Tests du modèle nano (GPT minimal) et de sa boucle d'entraînement.

Tout le fichier est ignoré si torch n'est pas installé.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from dhaos.config import Settings  # noqa: E402
from dhaos.train.nano import train as nano_train  # noqa: E402
from dhaos.train.nano.model import GPT, GPTConfig  # noqa: E402
from dhaos.train.nano.train import (  # noqa: E402
    CONFIG_FILE,
    MODEL_FILE,
    TOKENIZER_FILE,
    TrainReport,
    cosine_lr,
    resolve_config,
    sample,
    train_nano,
    validate_name,
)
from dhaos.train.tokenizer import DOC_ID, DOC_TOKEN, EOT_ID, EOT_TOKEN, N_BASE  # noqa: E402

TINY: dict[str, Any] = {
    "n_layer": 1,
    "n_head": 2,
    "n_embd": 32,
    "block_size": 32,
    "steps": 10,
    "batch_size": 4,
    "tokenizer": "bytes",
}

CORPUS_TEXT = (
    "def add(a, b):\n    return a + b\n\n"
    "class Foo:\n    def __init__(self):\n        self.value = 0\n\n"
    "print('héllo 🎉 le monde')\n"
) * 30  # ≈ 3 Ko


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text(CORPUS_TEXT, encoding="utf-8")
    assert 2500 < len(CORPUS_TEXT.encode("utf-8")) < 4000
    return path


# ------------------------------------------------------------------ modèle
def test_gpt_config_validation() -> None:
    cfg = GPTConfig(vocab_size=300, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    assert cfg.to_dict() == {"vocab_size": 300, "block_size": 8, "n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0}
    assert GPTConfig.from_dict({**cfg.to_dict(), "inconnu": 1}) == cfg
    assert GPTConfig.from_dict({"vocab_size": "300", "n_embd": 16, "n_head": 2}).vocab_size == 300
    with pytest.raises(ValueError):
        GPTConfig(n_embd=30, n_head=4)
    with pytest.raises(ValueError):
        GPTConfig(n_layer=0)
    with pytest.raises(ValueError):
        GPTConfig(dropout=1.0)
    with pytest.raises(ValueError):
        GPTConfig(vocab_size="abc")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GPTConfig.from_dict("nope")  # type: ignore[arg-type]


def test_gpt_forward_generate_and_tied_weights() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=300, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg)
    assert model.wte.weight is model.lm_head.weight
    unique = {p.data_ptr(): p.numel() for p in model.parameters()}
    assert model.n_params() == sum(unique.values())
    assert model.n_params(non_embedding=True) == model.n_params() - cfg.block_size * cfg.n_embd

    idx = torch.randint(0, cfg.vocab_size, (3, 8))
    logits, loss = model(idx)
    assert logits.shape == (3, 8, cfg.vocab_size) and loss is None
    logits, loss = model(idx, idx)
    assert loss is not None and math.isfinite(loss.item()) and loss.item() > 0
    loss.backward()
    assert model.wte.weight.grad is not None

    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.zeros(9, dtype=torch.long))

    model.eval()
    out = model.generate(idx[:1, :3], 5, temperature=1.0, top_k=5)
    assert out.shape == (1, 8) and torch.equal(out[:, :3], idx[:1, :3])
    long_ctx = model.generate(torch.zeros(1, 20, dtype=torch.long), 3)  # entrée plus longue que block_size
    assert long_ctx.shape == (1, 23)
    greedy_a = model.generate(idx[:1, :2], 4, temperature=0.0)
    greedy_b = model.generate(idx[:1, :2], 4, temperature=0.0)
    assert torch.equal(greedy_a, greedy_b)
    assert int(out.max()) < cfg.vocab_size


# ------------------------------------------------------------- entraînement
def test_train_nano_tiny_model_and_sample(settings: Settings, corpus: Path) -> None:
    lines: list[str] = []
    report = train_nano(settings, corpus, name="tiny", overrides=dict(TINY), on_log=lines.append, device="cpu")
    assert isinstance(report, TrainReport)
    out_dir = settings.models_dir / "tiny"
    assert report.out_dir == out_dir
    assert {p.name for p in out_dir.iterdir()} >= {MODEL_FILE, CONFIG_FILE, TOKENIZER_FILE}
    assert report.steps == 10 and report.tokenizer == "bytes"
    assert math.isfinite(report.final_loss) and math.isfinite(report.val_loss)
    assert report.final_loss > 0 and report.n_params > 0 and report.duration_s >= 0
    assert "paramètres" in report.summary() and str(out_dir) in report.summary()
    assert any("étape 10/10" in line for line in lines)
    assert any("sauvegardé" in line for line in lines)

    info = json.loads((out_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    assert info["model"] == {"vocab_size": N_BASE, "block_size": 32, "n_layer": 1, "n_head": 2, "n_embd": 32, "dropout": 0.0}
    assert info["tokenizer"] == "bytes" and info["n_params"] == report.n_params
    assert info["train"]["steps"] == 10 and info["train"]["device"] == "cpu"
    assert info["train"]["n_tokens"] == len(CORPUS_TEXT.encode("utf-8"))
    assert info["train"]["n_train_tokens"] + info["train"]["n_val_tokens"] == info["train"]["n_tokens"]
    state = torch.load(out_dir / MODEL_FILE, map_location="cpu", weights_only=True)
    assert "wte.weight" in state and "blocks.0.attn.c_attn.weight" in state

    text = sample(out_dir, "def ", max_new_tokens=12, temperature=0.8, top_k=40)
    assert isinstance(text, str) and text.startswith("def ") and len(text) > len("def ")
    greedy = sample(out_dir, "class", max_new_tokens=5, temperature=0.0, top_k=None)
    assert greedy == sample(out_dir, "class", max_new_tokens=5, temperature=0.0, top_k=None)
    assert isinstance(sample(out_dir, "", max_new_tokens=3), str)
    # Signature utilisée par la CLI.
    assert isinstance(sample(str(out_dir), "x", max_new_tokens=2, temperature=1.0), str)


def test_train_nano_is_deterministic_for_a_seed(settings: Settings, corpus: Path) -> None:
    overrides = {**TINY, "steps": 3}
    a = train_nano(settings, corpus, name="a", overrides=overrides, device="cpu")
    b = train_nano(settings, corpus, name="b", overrides=overrides, device="cpu")
    c = train_nano(settings, corpus, name="c", overrides={**overrides, "seed": 7}, device="cpu")
    assert a.final_loss == pytest.approx(b.final_loss)
    assert a.val_loss == pytest.approx(b.val_loss)
    assert c.final_loss != pytest.approx(a.final_loss)


def test_train_nano_bpe_tokenizer(settings: Settings, corpus: Path) -> None:
    report = train_nano(
        settings, corpus, name="bpe", overrides={**TINY, "steps": 3, "tokenizer": "bpe", "bpe_vocab_size": 300}, device="cpu"
    )
    assert report.tokenizer == "bpe"
    tok = json.loads((report.out_dir / TOKENIZER_FILE).read_text(encoding="utf-8"))
    assert tok["kind"] == "bpe" and len(tok["merges"]) > 0
    info = json.loads((report.out_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    assert info["model"]["vocab_size"] == N_BASE + len(tok["merges"]) == info["vocab_size"]
    assert info["train"]["n_tokens"] < len(CORPUS_TEXT.encode("utf-8"))
    assert sample(report.out_dir, "def ", max_new_tokens=8).startswith("def ")


def test_train_nano_reduces_block_size_for_short_corpus(settings: Settings, tmp_path: Path) -> None:
    short = tmp_path / "short.txt"
    short.write_text("abcdefghij" * 4, encoding="utf-8")  # 40 tokens
    lines: list[str] = []
    report = train_nano(
        settings, short, name="short", overrides={**TINY, "block_size": 256, "steps": 2}, on_log=lines.append, device="cpu"
    )
    info = json.loads((report.out_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    assert info["model"]["block_size"] < 256
    assert info["train"]["block_size_effective"] == info["model"]["block_size"]
    assert any("block_size réduit" in line for line in lines)
    assert any("validation sur l'entraînement" in line for line in lines)
    assert math.isfinite(report.final_loss)
    assert isinstance(sample(report.out_dir, "abc", max_new_tokens=2), str)


def test_train_nano_input_validation(settings: Settings, corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        train_nano(settings, tmp_path / "absent.txt", overrides=TINY, device="cpu")
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        train_nano(settings, empty, overrides=TINY, device="cpu")
    with pytest.raises(ValueError):
        train_nano(settings, corpus, overrides={**TINY, "inconnu": 1}, device="cpu")
    with pytest.raises(ValueError):
        train_nano(settings, corpus, overrides={**TINY, "n_embd": 30, "n_head": 4}, device="cpu")
    with pytest.raises(ValueError):
        train_nano(settings, corpus, overrides={**TINY, "steps": "beaucoup"}, device="cpu")
    with pytest.raises(ValueError):
        train_nano(settings, corpus, overrides={**TINY, "steps": 0}, device="cpu")
    with pytest.raises(ValueError):
        train_nano(settings, corpus, name="../evasion", overrides=TINY, device="cpu")
    with pytest.raises(ValueError):
        train_nano(settings, corpus, name=".cache", overrides=TINY, device="cpu")
    with pytest.raises(ValueError):
        train_nano(settings, corpus, overrides=TINY, device="pas-un-appareil")
    assert not (settings.models_dir / "evasion").exists()


def test_resolve_config_and_validate_name(settings: Settings) -> None:
    cfg, seed = resolve_config(settings, {"steps": 3, "seed": 42})
    assert cfg.steps == 3 and seed == 42
    assert cfg.n_layer == settings.train.nano.n_layer
    default_cfg, default_seed = resolve_config(settings, None)
    assert default_cfg == settings.train.nano and default_seed == nano_train.DEFAULT_SEED
    with pytest.raises(ValueError):
        resolve_config(settings, {"seed": "x"})
    with pytest.raises(ValueError):
        resolve_config(settings, {"dropout": 1.5})
    assert validate_name("mon-modele_v1.2") == "mon-modele_v1.2"
    for bad in ("", "a/b", "..", ".x", "x" * 65, "a b"):
        with pytest.raises(ValueError):
            validate_name(bad)


def test_cosine_schedule() -> None:
    lr = 1e-3
    steps = 100
    values = [cosine_lr(s, steps, lr) for s in range(steps)]
    assert values[0] == pytest.approx(lr / 5)  # warmup de 5 étapes
    assert values[4] == pytest.approx(lr)
    assert max(values) == pytest.approx(lr)
    assert values[-1] == pytest.approx(lr * 0.1, rel=1e-2)
    assert all(values[i] >= values[i + 1] for i in range(5, steps - 1))
    assert cosine_lr(0, 1, lr) == pytest.approx(lr)


def test_sample_validation_and_missing_model(settings: Settings, tmp_path: Path, corpus: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sample(tmp_path / "absent", "x")
    report = train_nano(settings, corpus, name="v", overrides={**TINY, "steps": 1}, device="cpu")
    with pytest.raises(ValueError):
        sample(report.out_dir, "x", max_new_tokens=0)
    with pytest.raises(ValueError):
        sample(report.out_dir, "x", temperature=-1)
    with pytest.raises(ValueError):
        sample(report.out_dir, "x", top_k=0)
    (report.out_dir / MODEL_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        sample(report.out_dir, "x")


def test_sample_stops_at_end_of_text(settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = train_nano(settings, corpus, name="eot", overrides={**TINY, "steps": 1}, device="cpu")

    def fake_generate(self: Any, idx: Any, max_new_tokens: int, temperature: float = 1.0, top_k: Any = None) -> Any:
        extra = torch.tensor([[ord("o"), ord("k"), EOT_ID, ord("z")]], dtype=torch.long)
        return torch.cat([idx, extra], dim=1)

    monkeypatch.setattr(GPT, "generate", fake_generate)
    assert sample(report.out_dir, "dis ", max_new_tokens=4) == "dis ok"


# ---------------------------------------------------------------- régressions
def test_train_nano_refuses_to_overwrite_existing_model(settings: Settings, corpus: Path) -> None:
    """Un second entraînement sous le même nom ne doit pas écraser le modèle
    précédent sans ``overwrite=True``, et doit échouer avant tout calcul."""
    overrides = {**TINY, "steps": 1}
    first = train_nano(settings, corpus, name="mini", overrides=overrides, device="cpu")
    model_bytes = (first.out_dir / MODEL_FILE).read_bytes()
    config_text = (first.out_dir / CONFIG_FILE).read_text(encoding="utf-8")

    lines: list[str] = []
    with pytest.raises(FileExistsError, match="--force"):
        train_nano(settings, corpus, name="mini", overrides={**overrides, "n_embd": 16}, on_log=lines.append, device="cpu")
    assert lines == []  # refus avant l'entraînement
    assert (first.out_dir / MODEL_FILE).read_bytes() == model_bytes
    assert (first.out_dir / CONFIG_FILE).read_text(encoding="utf-8") == config_text

    second = train_nano(settings, corpus, name="mini", overrides={**overrides, "n_embd": 16}, device="cpu", overwrite=True)
    assert second.out_dir == first.out_dir
    info = json.loads((first.out_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    assert info["model"]["n_embd"] == 16


def test_cli_train_nano_requires_force_to_overwrite(settings: Settings, corpus: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from dhaos.cli.main import app

    cfg = settings.save(tmp_path / "config.toml")
    runner = CliRunner()
    args = [
        "--config", str(cfg), "train", "nano", str(corpus), "--name", "mini",
        "--steps", "1", "--n-layer", "1", "--n-head", "1", "--n-embd", "8", "--block-size", "8", "--batch-size", "2",
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    model_path = settings.models_dir / "mini" / MODEL_FILE
    model_bytes = model_path.read_bytes()

    again = runner.invoke(app, args)
    assert again.exit_code != 0
    assert "--force" in again.output
    assert model_path.read_bytes() == model_bytes

    forced = runner.invoke(app, [*args, "--force"])
    assert forced.exit_code == 0, forced.output


def test_sample_stops_at_document_separator(settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Le corpus sépare les documents par ``<|doc|>`` (jamais ``<|endoftext|>``) :
    la génération doit s'arrêter là aussi, sans restituer le séparateur."""
    report = train_nano(settings, corpus, name="doc", overrides={**TINY, "steps": 1}, device="cpu")

    def fake_generate(self: Any, idx: Any, max_new_tokens: int, temperature: float = 1.0, top_k: Any = None) -> Any:
        extra = torch.tensor([[ord("o"), ord("k"), DOC_ID, ord("z"), EOT_ID]], dtype=torch.long)
        return torch.cat([idx, extra], dim=1)

    monkeypatch.setattr(GPT, "generate", fake_generate)
    assert sample(report.out_dir, "dis ", max_new_tokens=5) == "dis ok"


def test_sample_never_emits_document_separator_end_to_end(settings: Settings, tmp_path: Path) -> None:
    """Corpus réel (``build_corpus`` sur des documents séparés par ``<|doc|>``)
    → entraînement → échantillonnage : le séparateur ne doit jamais apparaître."""
    from dhaos.train.dataset import build_corpus

    class FakeKB:
        def corpus_text(self, bases: Any) -> list[str]:
            return [f"doc {i} contenu court." for i in range(120)]

        def close(self) -> None:
            pass

    corpus_path = build_corpus(settings, out_path=tmp_path / "docs.txt", include_sessions=False, kb=FakeKB())
    text = corpus_path.read_text(encoding="utf-8")
    assert DOC_TOKEN in text and EOT_TOKEN not in text

    report = train_nano(
        settings, corpus_path, name="docs",
        overrides={**TINY, "steps": 30, "tokenizer": "bpe", "bpe_vocab_size": 300}, device="cpu",
    )
    model, tokenizer, _ = nano_train.load_model(report.out_dir)
    # Le modèle produit bien DOC_ID dans sa génération brute sur ce corpus...
    ids = tokenizer.encode("doc 1 contenu court.")
    raw = model.generate(torch.tensor([ids]), 40, temperature=1.0, top_k=None)[0].tolist()
    assert len(raw) > len(ids)
    # ... mais ``sample`` ne le restitue jamais, quel que soit le tirage.
    for temperature in (0.0, 0.8, 1.5):
        for _ in range(3):
            out = sample(report.out_dir, "doc 1 contenu court.", max_new_tokens=40, temperature=temperature)
            assert DOC_TOKEN not in out and EOT_TOKEN not in out
