"""Boucle d'entraînement et échantillonnage du modèle nano.

``train_nano`` lit un corpus texte, entraîne (ou non) un tokenizer, découpe
90 / 10 en entraînement / validation, optimise un ``GPT`` avec AdamW et un
taux d'apprentissage cosinus (warmup 5 %), évalue la validation dix fois au
cours de l'entraînement et sauvegarde dans ``settings.models_dir/<nom>/`` :
``model.pt`` (state_dict), ``config.json`` (configuration + informations
d'entraînement) et ``tokenizer.json``. Un modèle existant du même nom n'est jamais écrasé
sans ``overwrite=True`` (``FileExistsError``).

``sample`` recharge un modèle sauvegardé sur CPU et génère du texte.

**Honnêteté** : c'est un modèle jouet, utile pour comprendre et expérimenter ;
la voie vers un modèle réellement utile est ``dhaos.train.finetune`` (LoRA
d'un modèle ouvert sur les traces).

``torch`` est importé paresseusement : ce module reste importable sans lui.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from ...config import NanoConfig, Settings
from ..tokenizer import DOC_ID, EOT_ID, BPETokenizer, ByteTokenizer, get_tokenizer, load_tokenizer

LogCallback = Callable[[str], None]

MODEL_FILE = "model.pt"
CONFIG_FILE = "config.json"
TOKENIZER_FILE = "tokenizer.json"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DEFAULT_SEED = 1337
VAL_FRACTION = 0.1
EVAL_ITERS = 8
WARMUP_FRACTION = 0.05
MIN_LR_FACTOR = 0.1
GRAD_CLIP = 1.0
_POSITIVE_INT_FIELDS = ("n_layer", "n_head", "n_embd", "block_size", "batch_size", "steps", "bpe_vocab_size")


@dataclass
class TrainReport:
    out_dir: Path
    steps: int
    final_loss: float
    val_loss: float
    n_params: int
    tokenizer: str
    duration_s: float

    def summary(self) -> str:
        n_params = f"{self.n_params:,}".replace(",", " ")
        return (
            f"modèle nano : {n_params} paramètres, {self.steps} étape(s), "
            f"perte finale {self.final_loss:.4f}, validation {self.val_loss:.4f}, "
            f"tokenizer {self.tokenizer}, {self.duration_s:.1f} s → {self.out_dir}"
        )


def _log(cb: LogCallback | None, message: str) -> None:
    if cb is None:
        return
    try:
        cb(message)
    except Exception:  # noqa: BLE001 — un callback d'affichage ne doit rien casser
        pass


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover - dépend de l'environnement
        raise ImportError("torch est requis pour le modèle nano : pip install 'dhaos[train]'") from e
    return torch


def validate_name(name: str) -> str:
    """Nom de modèle sûr (dossier sous ``models_dir``) ; ``ValueError`` sinon."""
    key = str(name or "").strip()
    if not NAME_RE.match(key):
        raise ValueError(
            f"nom de modèle invalide : {name!r} (lettres, chiffres, '.', '_', '-', 64 caractères max, "
            "sans '.' initial)"
        )
    return key


def resolve_config(settings: Settings, overrides: dict[str, Any] | None = None) -> tuple[NanoConfig, int]:
    """``settings.train.nano`` + ``overrides`` validés ; renvoie ``(config, graine)``.

    ``overrides`` accepte en plus la clé ``seed`` ; toute autre clé inconnue
    lève ``ValueError``.
    """
    base = settings.train.nano.model_dump()
    extra = dict(overrides or {})
    seed_raw = extra.pop("seed", DEFAULT_SEED)
    try:
        seed = int(seed_raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"seed invalide : {seed_raw!r}") from e
    unknown = sorted(set(extra) - set(base))
    if unknown:
        raise ValueError(f"paramètre(s) inconnu(s) pour le modèle nano : {', '.join(unknown)}")
    base.update(extra)
    try:
        cfg = NanoConfig.model_validate(base)
    except ValidationError as e:
        raise ValueError(f"paramètres nano invalides : {e}") from e
    for field_name in _POSITIVE_INT_FIELDS:
        if getattr(cfg, field_name) <= 0:
            raise ValueError(f"train.nano.{field_name} doit être strictement positif")
    if cfg.learning_rate <= 0:
        raise ValueError("train.nano.learning_rate doit être strictement positif")
    if not 0.0 <= cfg.dropout < 1.0:
        raise ValueError("train.nano.dropout doit être dans [0, 1)")
    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError(f"train.nano.n_embd ({cfg.n_embd}) doit être divisible par n_head ({cfg.n_head})")
    return cfg, seed


def _select_device(torch: Any, device: str | None) -> Any:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        dev = torch.device(str(device))
    except (RuntimeError, TypeError, ValueError) as e:
        raise ValueError(f"appareil invalide : {device!r}") from e
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("appareil cuda demandé mais indisponible")
    return dev


def cosine_lr(step: int, steps: int, learning_rate: float) -> float:
    """Warmup linéaire (5 % des étapes, au moins 1) puis décroissance cosinus
    jusqu'à ``MIN_LR_FACTOR × learning_rate``."""
    warmup = max(1, int(round(steps * WARMUP_FRACTION)))
    if step < warmup:
        return learning_rate * (step + 1) / warmup
    min_lr = learning_rate * MIN_LR_FACTOR
    remaining = max(1, steps - warmup)
    progress = min(1.0, (step - warmup) / remaining)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (learning_rate - min_lr)


def _build_tokenizer(cfg: NanoConfig, text: str, on_log: LogCallback | None) -> BPETokenizer | ByteTokenizer:
    tokenizer = get_tokenizer(cfg.tokenizer)
    if isinstance(tokenizer, BPETokenizer):
        tokenizer.train(text, cfg.bpe_vocab_size, on_log=on_log)
    return tokenizer


def train_nano(
    settings: Settings,
    corpus_path: str | Path,
    *,
    name: str = "nano",
    overrides: dict[str, Any] | None = None,
    on_log: LogCallback | None = None,
    device: str | None = None,
    overwrite: bool = False,
) -> TrainReport:
    """Entraîne un modèle nano sur ``corpus_path`` et le sauvegarde dans
    ``settings.models_dir/<name>/``. ``device`` : ``None`` ⇒ cuda si
    disponible sinon cpu. Si un modèle existe déjà sous ce nom, lève
    ``FileExistsError`` avant tout calcul, sauf ``overwrite=True``."""
    torch = _import_torch()
    from .model import GPT, GPTConfig

    name = validate_name(name)
    cfg, seed = resolve_config(settings, overrides)
    out_dir = settings.models_dir / name
    if (out_dir / MODEL_FILE).exists() and not overwrite:
        raise FileExistsError(f"un modèle nano existe déjà dans {out_dir} (utilisez --force pour l'écraser)")
    corpus = Path(corpus_path)
    if not corpus.is_file():
        raise FileNotFoundError(f"corpus introuvable : {corpus}")
    text = corpus.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise ValueError(f"corpus vide : {corpus}")
    dev = _select_device(torch, device)
    t0 = time.perf_counter()
    _log(on_log, f"corpus {corpus} : {len(text)} caractère(s) ; appareil : {dev}")

    tokenizer = _build_tokenizer(cfg, text, on_log)
    ids = tokenizer.encode(text)
    n_tokens = len(ids)
    if n_tokens < 4:
        raise ValueError(f"corpus trop court : {n_tokens} token(s), au moins 4 requis")
    data = torch.tensor(ids, dtype=torch.long)
    n_train = max(2, int(n_tokens * (1.0 - VAL_FRACTION)))
    train_data = data[:n_train]
    val_data = data[n_train:]
    block_size = min(cfg.block_size, n_train - 1)
    if block_size < cfg.block_size:
        _log(on_log, f"block_size réduit de {cfg.block_size} à {block_size} (corpus court)")
    if len(val_data) < block_size + 1:
        _log(on_log, "corpus trop court pour une validation séparée : validation sur l'entraînement")
        val_data = train_data
    _log(on_log, f"{n_tokens} token(s) ({tokenizer.kind}, vocabulaire {tokenizer.vocab_size}) : "
                 f"{len(train_data)} entraînement / {len(val_data)} validation")

    torch.manual_seed(seed)
    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(dev)
    n_params = model.n_params()
    _log(on_log, f"modèle : {n_params} paramètre(s), {cfg.n_layer} couche(s), {cfg.n_head} tête(s), dim {cfg.n_embd}")

    generator = torch.Generator().manual_seed(seed)

    def get_batch(split: Any) -> tuple[Any, Any]:
        ix = torch.randint(0, len(split) - block_size, (cfg.batch_size,), generator=generator)
        x = torch.stack([split[i : i + block_size] for i in ix.tolist()])
        y = torch.stack([split[i + 1 : i + 1 + block_size] for i in ix.tolist()])
        return x.to(dev), y.to(dev)

    @torch.no_grad()
    def estimate_val() -> float:
        model.eval()
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(val_data)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        model.train()
        return sum(losses) / len(losses)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    steps = cfg.steps
    eval_interval = max(1, steps // 10)
    final_loss = float("nan")
    val_loss = float("nan")
    model.train()
    for step in range(steps):
        lr = cosine_lr(step, steps, cfg.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        final_loss = float(loss.item())
        if (step + 1) % eval_interval == 0 or step + 1 == steps:
            val_loss = estimate_val()
            _log(on_log, f"étape {step + 1}/{steps} : perte {final_loss:.4f}, validation {val_loss:.4f}, lr {lr:.2e}")
    duration = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir / MODEL_FILE)
    tokenizer.save(out_dir / TOKENIZER_FILE)
    info: dict[str, Any] = {
        "name": name,
        "model": model_cfg.to_dict(),
        "tokenizer": tokenizer.kind,
        "vocab_size": tokenizer.vocab_size,
        "n_params": n_params,
        "train": {
            **cfg.model_dump(),
            "block_size_effective": block_size,
            "seed": seed,
            "device": str(dev),
            "n_tokens": n_tokens,
            "n_train_tokens": int(len(train_data)),
            "n_val_tokens": int(len(val_data)),
            "final_loss": final_loss,
            "val_loss": val_loss,
            "duration_s": duration,
            "corpus": str(corpus),
            "created_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    }
    (out_dir / CONFIG_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(on_log, f"modèle sauvegardé dans {out_dir}")
    return TrainReport(
        out_dir=out_dir,
        steps=steps,
        final_loss=final_loss,
        val_loss=val_loss,
        n_params=n_params,
        tokenizer=tokenizer.kind,
        duration_s=duration,
    )


def load_model(model_dir: str | Path) -> tuple[Any, BPETokenizer | ByteTokenizer, dict[str, Any]]:
    """Recharge ``(modèle en mode eval sur CPU, tokenizer, config.json)``."""
    torch = _import_torch()
    from .model import GPT, GPTConfig

    directory = Path(model_dir)
    for filename in (CONFIG_FILE, TOKENIZER_FILE, MODEL_FILE):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"modèle nano incomplet dans {directory} : {filename} manquant")
    try:
        info = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"config.json illisible dans {directory} : {e}") from e
    if not isinstance(info, dict):
        raise ValueError(f"config.json invalide dans {directory}")
    model_cfg = GPTConfig.from_dict(info.get("model") or {})
    tokenizer = load_tokenizer(directory / TOKENIZER_FILE)
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError(
            f"tokenizer ({tokenizer.vocab_size}) et modèle ({model_cfg.vocab_size}) de vocabulaires différents"
        )
    model = GPT(model_cfg)
    state = torch.load(directory / MODEL_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, info


def sample(
    model_dir: str | Path,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    """Génère la suite de ``prompt`` avec un modèle nano sauvegardé (CPU).

    Renvoie ``prompt + continuation`` ; la génération s'arrête au premier
    token spécial (``<|endoftext|>`` ou ``<|doc|>``, séparateur de
    documents du corpus), qui n'est pas restitué.
    """
    torch = _import_torch()
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens doit valoir au moins 1")
    temperature = float(temperature)
    if temperature < 0:
        raise ValueError("temperature doit être positive ou nulle")
    if top_k is not None:
        top_k = int(top_k)
        if top_k < 1:
            raise ValueError("top_k doit valoir au moins 1 (ou None)")
    model, tokenizer, _info = load_model(model_dir)
    prompt = str(prompt if prompt is not None else "")
    ids = tokenizer.encode(prompt) or [EOT_ID]
    block_size = model.config.block_size
    ids = ids[-block_size:]
    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    generated = out[0].tolist()[len(ids) :]
    stops = [i for i, token_id in enumerate(generated) if token_id in (EOT_ID, DOC_ID)]
    if stops:
        generated = generated[: stops[0]]
    return prompt + tokenizer.decode(generated)


__all__ = [
    "CONFIG_FILE",
    "MODEL_FILE",
    "TOKENIZER_FILE",
    "TrainReport",
    "cosine_lr",
    "load_model",
    "resolve_config",
    "sample",
    "train_nano",
    "validate_name",
]
