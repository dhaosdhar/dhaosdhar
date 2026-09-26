"""Fine-tuning LoRA d'un modèle ouvert sur les traces (jeu SFT JSONL).

Dépendances **optionnelles** (``pip install 'dhaos[finetune]'``) : torch,
transformers, peft, datasets (+ bitsandbytes pour le 4 bits sur GPU). Elles
sont importées paresseusement : ``finetune_available()`` indique ce qui
manque et ``run_finetune`` lève ``RuntimeError`` si le fine-tuning est
indisponible.

Pipeline : JSONL SFT → ``render_example`` (les tours ``tool`` deviennent des
messages utilisateur « [résultat de l'outil X] » quand le gabarit de chat ne
connaît pas ce rôle) → ``apply_chat_template`` → LoRA (``q_proj``, ``k_proj``,
``v_proj``, ``o_proj``) → ``Trainer`` → adaptateur dans ``<out_dir>/adapter``,
plus un ``Modelfile`` Ollama et un ``README.md`` expliquant ``ollama create``
et la fusion (``merge_and_unload``).
"""
from __future__ import annotations

import datetime as _dt
import importlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from ..config import FinetuneConfig, Settings

LogCallback = Callable[[str], None]

REQUIRED_MODULES: tuple[str, ...] = ("torch", "transformers", "peft", "datasets")
LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
ADAPTER_DIRNAME = "adapter"
MODELFILE_NAME = "Modelfile"
README_NAME = "README.md"
INFO_NAME = "training.json"
TOOL_RESULT_PREFIX = "[résultat de l'outil {name}]"
_ROLES = ("system", "user", "assistant", "tool")
_MERGEABLE_ROLES = ("system", "user", "assistant")
_TOOL_PROBE = "DHAOS_TOOL_PROBE_7f3a"
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)[bB](?=[-_.]|$)")


def _log(cb: LogCallback | None, message: str) -> None:
    if cb is None:
        return
    try:
        cb(message)
    except Exception:  # noqa: BLE001 — un callback d'affichage ne doit rien casser
        pass


# --------------------------------------------------------------- disponibilité
def finetune_available() -> tuple[bool, str]:
    """``(True, détail des versions et du GPU)`` ou ``(False, ce qui manque)``."""
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        return False, (
            f"dépendances manquantes : {', '.join(missing)} — installez-les avec pip install 'dhaos[finetune]'"
        )
    parts: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            parts.append(f"{name} {getattr(module, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001 — module cassé ⇒ indisponible
            return False, f"import de {name} impossible : {type(e).__name__}: {e}"
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            parts.append(f"GPU : {torch.cuda.get_device_name(0)}")
        else:
            parts.append("GPU : aucun (CPU seulement : très lent, 4 bits désactivé)")
    except Exception as e:  # noqa: BLE001
        parts.append(f"GPU : inconnu ({e})")
    parts.append(
        "bitsandbytes disponible" if importlib.util.find_spec("bitsandbytes") is not None
        else "bitsandbytes absent (pas de chargement 4 bits)"
    )
    return True, " ; ".join(parts)


# ---------------------------------------------------------------- configuration
def resolve_config(settings: Settings, overrides: dict[str, Any] | None = None) -> FinetuneConfig:
    """``settings.train.finetune`` + ``overrides`` validés ; ``ValueError`` sinon."""
    base = settings.train.finetune.model_dump()
    extra = dict(overrides or {})
    unknown = sorted(set(extra) - set(base))
    if unknown:
        raise ValueError(f"paramètre(s) inconnu(s) pour le fine-tuning : {', '.join(unknown)}")
    base.update(extra)
    try:
        cfg = FinetuneConfig.model_validate(base)
    except ValidationError as e:
        raise ValueError(f"paramètres de fine-tuning invalides : {e}") from e
    if not str(cfg.base_model).strip():
        raise ValueError("train.finetune.base_model est vide")
    for name in ("lora_r", "lora_alpha", "max_seq_len", "batch_size", "gradient_accumulation"):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"train.finetune.{name} doit être strictement positif")
    if cfg.epochs <= 0 or cfg.learning_rate <= 0:
        raise ValueError("train.finetune.epochs et learning_rate doivent être strictement positifs")
    if not 0.0 <= cfg.lora_dropout < 1.0:
        raise ValueError("train.finetune.lora_dropout doit être dans [0, 1)")
    return cfg


def safe_model_name(base_model: str) -> str:
    """Nom de dossier dérivé d'un identifiant HF (``org/Nom-1B`` → ``nom-1b``)."""
    tail = str(base_model).strip().rsplit("/", 1)[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-.").lower()
    return cleaned or "modele"


# ------------------------------------------------------------- jeu de données
def _clean_messages(raw: Any) -> list[dict[str, str]]:
    """Messages valides (rôle connu, contenu texte) ; ``name`` conservé pour ``tool``."""
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in _ROLES:
            continue
        content = item.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        msg = {"role": str(role), "content": content}
        if role == "tool":
            name = item.get("name")
            msg["name"] = name.strip() if isinstance(name, str) and name.strip() else "outil"
        out.append(msg)
    return out


def load_sft_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Lit un JSONL SFT (``{"id", "messages": [...]}`` par ligne) ; lignes
    invalides ignorées ; ``ValueError`` si aucun exemple exploitable."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"jeu de données introuvable : {p}")
    examples: list[dict[str, Any]] = []
    for n, raw_line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        messages = _clean_messages(obj.get("messages"))
        if not any(m["role"] == "assistant" for m in messages):
            continue
        ident = obj.get("id")
        examples.append({"id": str(ident) if ident not in (None, "") else f"ligne-{n}", "messages": messages})
    if not examples:
        raise ValueError(f"jeu de données vide ou invalide : {p} (aucun exemple avec une réponse assistant)")
    return examples


def render_example(messages: list[dict[str, Any]], has_tool_role: bool) -> list[dict[str, str]]:
    """Prépare les messages d'un exemple pour ``apply_chat_template`` (fonction pure).

    - rôles inconnus et contenus vides ignorés ;
    - ``tool`` conservé (avec ``name``) si ``has_tool_role``, sinon rendu comme
      message ``user`` « [résultat de l'outil X]\\n… » ;
    - messages consécutifs de même rôle (system / user / assistant) fusionnés
      pour respecter l'alternance attendue par la plupart des gabarits.
    """
    out: list[dict[str, str]] = []
    for msg in _clean_messages(messages):
        role = msg["role"]
        content = msg["content"]
        if not content.strip():
            continue
        entry: dict[str, str]
        if role == "tool":
            if has_tool_role:
                entry = {"role": "tool", "content": content, "name": msg["name"]}
            else:
                role = "user"
                entry = {"role": "user", "content": f"{TOOL_RESULT_PREFIX.format(name=msg['name'])}\n{content}"}
        else:
            entry = {"role": role, "content": content}
        if out and role in _MERGEABLE_ROLES and out[-1]["role"] == role:
            out[-1]["content"] = out[-1]["content"].rstrip() + "\n\n" + entry["content"].lstrip()
        else:
            out.append(entry)
    return out


def template_supports_tool_role(tokenizer: Any) -> bool:
    """``True`` si le gabarit de chat du tokenizer rend les messages ``tool``."""
    probe = [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
        {"role": "tool", "name": "sonde", "content": _TOOL_PROBE},
    ]
    try:
        rendered = tokenizer.apply_chat_template(probe, tokenize=False)
    except Exception:  # noqa: BLE001 — gabarit qui refuse le rôle
        return False
    return isinstance(rendered, str) and _TOOL_PROBE in rendered


# ---------------------------------------------------------- fichiers de sortie
def ollama_base_hint(base_model: str) -> str:
    """Heuristique ``org/Nom-Famille-1.5B-Instruct`` → ``nom-famille:1.5b-instruct``."""
    tail = str(base_model).strip().rsplit("/", 1)[-1]
    match = _SIZE_RE.search(tail)
    if not match:
        return tail.lower() or "modele"
    family = tail[: match.start()].rstrip("-_.").lower() or tail.lower()
    size = f"{match.group(1)}b"
    suffix = tail[match.end():].strip("-_.").lower()
    return f"{family}:{size}-{suffix}" if suffix else f"{family}:{size}"


def _adapter_reference(out_dir: Path, adapter_dir: Path) -> str:
    if not adapter_dir.is_absolute():
        return "./" + adapter_dir.as_posix().lstrip("./")
    try:
        rel = adapter_dir.resolve().relative_to(out_dir.resolve())
    except (ValueError, OSError):
        return str(adapter_dir)
    return "./" + rel.as_posix()


def write_modelfile(out_dir: str | Path, base_model: str, adapter_dir: str | Path) -> Path:
    """Écrit ``<out_dir>/Modelfile`` : ``FROM <indice Ollama>`` + ``ADAPTER ./adapter``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hint = ollama_base_hint(base_model)
    adapter_ref = _adapter_reference(out, Path(adapter_dir))
    content = (
        f"# Modelfile généré par dhaos : adaptateur LoRA entraîné sur {base_model}\n"
        f"# Ajustez la ligne FROM si le nom du modèle Ollama diffère (ollama pull {hint}).\n"
        f"FROM {hint}\n"
        f"ADAPTER {adapter_ref}\n"
    )
    path = out / MODELFILE_NAME
    path.write_text(content, encoding="utf-8")
    return path


def write_readme(out_dir: str | Path, base_model: str, adapter_dir: str | Path) -> Path:
    """Écrit ``<out_dir>/README.md`` : usage avec Ollama et fusion des poids."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hint = ollama_base_hint(base_model)
    adapter_ref = _adapter_reference(out, Path(adapter_dir))
    model_name = f"dhaos-{safe_model_name(base_model)}"
    content = f"""# Adaptateur LoRA dhaos

Modèle de base : `{base_model}`
Adaptateur : `{adapter_ref}` (poids LoRA + tokenizer, format PEFT)

## Utiliser avec Ollama

```sh
ollama pull {hint}          # le modèle de base doit être présent localement
ollama create {model_name} -f Modelfile
ollama run {model_name}
```

Si le nom Ollama du modèle de base diffère, modifiez la ligne `FROM` du
`Modelfile` (les deux modèles doivent avoir exactement la même architecture).
Pour utiliser ce modèle dans dhaos : `dhaos config set backends.ollama.model {model_name}`.

## Fusionner les poids (modèle complet, sans PEFT)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{adapter_ref}")
merged = model.merge_and_unload()
merged.save_pretrained("merged")
AutoTokenizer.from_pretrained("{adapter_ref}").save_pretrained("merged")
```

Le dossier `merged/` est un modèle Hugging Face autonome (convertible en GGUF
avec `llama.cpp`, puis importable dans Ollama par un `Modelfile` `FROM ./modele.gguf`).
"""
    path = out / README_NAME
    path.write_text(content, encoding="utf-8")
    return path


# ------------------------------------------------------------------- pipeline
def run_finetune(
    settings: Settings,
    dataset_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    on_log: LogCallback | None = None,
) -> Path:
    """Entraîne un adaptateur LoRA sur ``dataset_path`` (JSONL SFT) ; renvoie
    le dossier de sortie (``out_dir`` ou ``settings.models_dir/lora-<base>/``).

    ``RuntimeError`` si les dépendances optionnelles manquent.
    """
    available, reason = finetune_available()
    if not available:
        raise RuntimeError(f"fine-tuning indisponible : {reason}")
    cfg = resolve_config(settings, overrides)
    examples = load_sft_dataset(dataset_path)
    target = Path(out_dir) if out_dir is not None else Path(settings.models_dir) / f"lora-{safe_model_name(cfg.base_model)}"
    target.mkdir(parents=True, exist_ok=True)
    adapter_dir = target / ADAPTER_DIRNAME

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        raise RuntimeError(f"fine-tuning indisponible : {e}") from e

    _log(on_log, f"fine-tuning LoRA de {cfg.base_model} sur {len(examples)} exemple(s) → {target}")
    _log(on_log, reason)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    has_tool_role = template_supports_tool_role(tokenizer)
    _log(on_log, "gabarit de chat : rôle tool " + ("pris en charge" if has_tool_role else "absent → rendu comme user"))

    texts: list[str] = []
    for example in examples:
        rendered = render_example(example["messages"], has_tool_role)
        if not rendered:
            continue
        texts.append(str(tokenizer.apply_chat_template(rendered, tokenize=False)))
    if not texts:
        raise ValueError("aucun exemple exploitable après rendu")

    cuda = bool(torch.cuda.is_available())
    use_4bit = bool(cfg.load_in_4bit and cuda and importlib.util.find_spec("bitsandbytes") is not None)
    model_kwargs: dict[str, Any] = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif cuda:
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
    _log(on_log, f"chargement du modèle ({'4 bits' if use_4bit else 'pleine précision'}, {'cuda' if cuda else 'cpu'})")
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    if use_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(LORA_TARGET_MODULES),
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    printer = getattr(model, "print_trainable_parameters", None)
    if callable(printer):
        try:
            printer()
        except Exception:  # noqa: BLE001 — purement informatif
            pass

    def tokenize(batch: dict[str, list[str]]) -> Any:
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_seq_len)

    dataset = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])
    bf16 = bool(cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    args = TrainingArguments(
        output_dir=str(target / "checkpoints"),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=bf16,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    _log(on_log, f"entraînement : {len(texts)} séquence(s), {cfg.epochs} époque(s), lr {cfg.learning_rate}")
    result = trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_modelfile(target, cfg.base_model, adapter_dir)
    write_readme(target, cfg.base_model, adapter_dir)
    metrics = getattr(result, "metrics", None)
    info = {
        "base_model": cfg.base_model,
        "config": cfg.model_dump(),
        "dataset": str(Path(dataset_path)),
        "n_examples": len(texts),
        "has_tool_role": has_tool_role,
        "load_in_4bit": use_4bit,
        "metrics": metrics if isinstance(metrics, dict) else {},
        "created_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (target / INFO_NAME).write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _log(on_log, f"adaptateur enregistré dans {adapter_dir} ; voir {target / README_NAME}")
    return target


__all__ = [
    "ADAPTER_DIRNAME",
    "LORA_TARGET_MODULES",
    "MODELFILE_NAME",
    "README_NAME",
    "finetune_available",
    "load_sft_dataset",
    "ollama_base_hint",
    "render_example",
    "resolve_config",
    "run_finetune",
    "safe_model_name",
    "template_supports_tool_role",
    "write_modelfile",
    "write_readme",
]
