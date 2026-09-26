"""Tests du module de fine-tuning LoRA (sans transformers/peft/datasets réels)."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from dhaos.config import Settings
from dhaos.train import finetune
from dhaos.train.finetune import (
    ADAPTER_DIRNAME,
    LORA_TARGET_MODULES,
    MODELFILE_NAME,
    README_NAME,
    finetune_available,
    load_sft_dataset,
    ollama_base_hint,
    render_example,
    resolve_config,
    run_finetune,
    safe_model_name,
    template_supports_tool_role,
    write_modelfile,
    write_readme,
)

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "lis a.py"},
    {"role": "assistant", "content": 'Je lis.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'},
    {"role": "tool", "name": "read_file", "content": "print(1)"},
    {"role": "tool", "name": "grep", "content": "aucun résultat"},
    {"role": "assistant", "content": "Le fichier affiche 1."},
]


def _write_dataset(path: Path, examples: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------- disponibilité
def test_finetune_available_reports_missing_transformers_here() -> None:
    if importlib.util.find_spec("transformers") is not None:  # pragma: no cover - environnement complet
        pytest.skip("transformers est installé dans cet environnement")
    available, message = finetune_available()
    assert available is False
    assert "transformers" in message
    assert "dhaos[finetune]" in message


def test_finetune_available_lists_every_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.util.find_spec

    def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("transformers", "peft", "datasets"):
            return None
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    available, message = finetune_available()
    assert available is False
    for name in ("transformers", "peft", "datasets"):
        assert name in message


def test_run_finetune_raises_clean_runtime_error_when_unavailable(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finetune, "finetune_available", lambda: (False, "dépendances manquantes : transformers"))
    dataset = _write_dataset(tmp_path / "sft.jsonl", [{"id": "1", "messages": MESSAGES}])
    with pytest.raises(RuntimeError) as exc:
        run_finetune(settings, dataset)
    assert "fine-tuning indisponible" in str(exc.value)
    assert "transformers" in str(exc.value)
    assert not any(settings.models_dir.iterdir())


def test_run_finetune_unavailable_in_this_environment(settings: Settings, tmp_path: Path) -> None:
    if importlib.util.find_spec("transformers") is not None:  # pragma: no cover
        pytest.skip("transformers est installé dans cet environnement")
    dataset = _write_dataset(tmp_path / "sft.jsonl", [{"id": "1", "messages": MESSAGES}])
    with pytest.raises(RuntimeError):
        run_finetune(settings, dataset)


# ------------------------------------------------------------ configuration
def test_resolve_config(settings: Settings) -> None:
    cfg = resolve_config(settings, {"epochs": 2, "base_model": "org/Modele-1B"})
    assert cfg.epochs == 2.0 and cfg.base_model == "org/Modele-1B"
    assert cfg.lora_r == settings.train.finetune.lora_r
    assert resolve_config(settings, None) == settings.train.finetune
    for bad in ({"inconnu": 1}, {"epochs": 0}, {"lora_r": -1}, {"base_model": " "}, {"batch_size": "deux"}, {"lora_dropout": 1.0}):
        with pytest.raises(ValueError):
            resolve_config(settings, bad)


def test_safe_model_name_and_ollama_hint() -> None:
    assert safe_model_name(BASE) == "qwen2.5-coder-1.5b-instruct"
    assert safe_model_name("../é/../x y") == "x-y"
    assert safe_model_name("///") == "modele"
    assert ollama_base_hint(BASE) == "qwen2.5-coder:1.5b-instruct"
    assert ollama_base_hint("meta-llama/Llama-3.1-8B-Instruct") == "llama-3.1:8b-instruct"
    assert ollama_base_hint("mistralai/Mistral-7B-v0.3") == "mistral:7b-v0.3"
    assert ollama_base_hint("org/Nom-2B") == "nom:2b"
    assert ollama_base_hint("gpt2") == "gpt2"
    assert ollama_base_hint("Qwen/Qwen3-0.6B") == "qwen3:0.6b"


# ------------------------------------------------------------ fichiers Ollama
def test_write_modelfile(tmp_path: Path) -> None:
    out = tmp_path / "lora"
    path = write_modelfile(out, BASE, out / ADAPTER_DIRNAME)
    assert path == out / MODELFILE_NAME and path.is_file()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert lines == ["FROM qwen2.5-coder:1.5b-instruct", "ADAPTER ./adapter"]
    assert BASE in path.read_text(encoding="utf-8")

    relative = write_modelfile(out, BASE, "adapter")
    assert "ADAPTER ./adapter" in relative.read_text(encoding="utf-8")
    nested = write_modelfile(out, BASE, out / "sous" / "dossier")
    assert "ADAPTER ./sous/dossier" in nested.read_text(encoding="utf-8")
    elsewhere = tmp_path / "ailleurs"
    outside = write_modelfile(out, BASE, elsewhere)
    assert f"ADAPTER {elsewhere}" in outside.read_text(encoding="utf-8")


def test_write_readme(tmp_path: Path) -> None:
    out = tmp_path / "lora"
    path = write_readme(out, BASE, out / ADAPTER_DIRNAME)
    assert path == out / README_NAME
    text = path.read_text(encoding="utf-8")
    assert "ollama create" in text and "merge_and_unload" in text
    assert "./adapter" in text and BASE in text and "qwen2.5-coder:1.5b-instruct" in text


# --------------------------------------------------------------- exemples
def test_render_example_without_tool_role() -> None:
    original = json.loads(json.dumps(MESSAGES))
    rendered = render_example(MESSAGES, has_tool_role=False)
    assert MESSAGES == original  # fonction pure : entrée non modifiée
    assert [m["role"] for m in rendered] == ["system", "user", "assistant", "user", "assistant"]
    assert rendered[0] == {"role": "system", "content": "SYS"}
    assert rendered[3]["content"] == "[résultat de l'outil read_file]\nprint(1)\n\n[résultat de l'outil grep]\naucun résultat"
    assert rendered[2]["content"].startswith("Je lis.")
    assert all(set(m) == {"role", "content"} for m in rendered)


def test_render_example_with_tool_role() -> None:
    rendered = render_example(MESSAGES, has_tool_role=True)
    assert [m["role"] for m in rendered] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert rendered[3] == {"role": "tool", "content": "print(1)", "name": "read_file"}
    assert rendered[4]["name"] == "grep"


def test_render_example_cleans_untrusted_input() -> None:
    messy: list[Any] = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "inconnu", "content": "x"},
        "pas un dict",
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": {"json": True}},
        {"role": "tool", "content": "r", "name": ""},
        {"role": "tool", "content": None},
    ]
    rendered = render_example(messy, has_tool_role=False)
    assert rendered == [
        {"role": "user", "content": "a\n\nb"},
        {"role": "assistant", "content": '{"json": true}'},
        {"role": "user", "content": "[résultat de l'outil outil]\nr"},
    ]
    assert render_example([], True) == []
    assert render_example("n'importe quoi", True) == []  # type: ignore[arg-type]


class _Tokenizer:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = False, **_: Any) -> str:
        if self.mode == "raise" and any(m["role"] == "tool" for m in messages):
            raise ValueError("rôle inconnu")
        if self.mode == "ignore":
            return "".join(m["content"] for m in messages if m["role"] != "tool")
        return "".join(f"<{m['role']}>{m['content']}" for m in messages)


def test_template_supports_tool_role() -> None:
    assert template_supports_tool_role(_Tokenizer("raise")) is False
    assert template_supports_tool_role(_Tokenizer("ignore")) is False
    assert template_supports_tool_role(_Tokenizer("ok")) is True
    assert template_supports_tool_role(object()) is False


def test_load_sft_dataset(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps({"id": "a", "messages": MESSAGES}) + "\n"
        + "{cassé\n"
        + json.dumps([1, 2]) + "\n"
        + json.dumps({"id": "sans-assistant", "messages": [{"role": "user", "content": "?"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "r"}]}) + "\n",
        encoding="utf-8",
    )
    examples = load_sft_dataset(path)
    assert [e["id"] for e in examples] == ["a", "ligne-5"]
    assert examples[0]["messages"][3]["name"] == "read_file"
    with pytest.raises(FileNotFoundError):
        load_sft_dataset(tmp_path / "absent.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n{\"messages\": []}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_sft_dataset(empty)


# ------------------------------------------------- pipeline avec doubles HF
def _module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__version__ = "0.0-test"
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Installe des doubles de transformers / peft / datasets ; renvoie l'état observé."""
    pytest.importorskip("torch")
    state: dict[str, Any] = {"tool_role": True}

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token: str | None = None
            self.eos_token = "</s>"
            self.rendered: list[list[dict[str, Any]]] = []

        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> "FakeTokenizer":
            state["tokenizer_name"] = name
            tok = cls()
            state["tokenizer"] = tok
            return tok

        def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = False, **_: Any) -> str:
            if not state["tool_role"] and any(m["role"] == "tool" for m in messages):
                raise ValueError("rôle tool inconnu du gabarit")
            self.rendered.append(messages)
            return "".join(f"<{m['role']}>{m['content']}" for m in messages) + self.eos_token

        def __call__(self, texts: list[str], truncation: bool = True, max_length: int | None = None) -> dict[str, list[list[int]]]:
            state["max_length"] = max_length
            ids = [list(range(min(len(t), max_length or len(t)))) for t in texts]
            return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}

        def save_pretrained(self, directory: str) -> None:
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class FakeModel:
        def __init__(self) -> None:
            self.lora: Any = None

        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> "FakeModel":
            state["model_name"] = name
            state["model_kwargs"] = kwargs
            return cls()

        def print_trainable_parameters(self) -> None:
            state["printed"] = True

        def save_pretrained(self, directory: str) -> None:
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / "adapter_config.json").write_text(json.dumps({"r": self.lora.kwargs["r"]}), encoding="utf-8")

    class TrainingArguments:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            state["training_args"] = kwargs

    class DataCollator:
        def __init__(self, tokenizer: Any, mlm: bool = True) -> None:
            state["mlm"] = mlm

    class Trainer:
        def __init__(self, *, model: Any, args: Any, train_dataset: Any, data_collator: Any) -> None:
            state["trainer"] = {"model": model, "args": args, "dataset": train_dataset, "collator": data_collator}

        def train(self) -> Any:
            state["trained"] = True
            return types.SimpleNamespace(metrics={"train_loss": 0.5})

    class LoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            state["lora"] = kwargs

    def get_peft_model(model: Any, config: Any) -> Any:
        model.lora = config
        return model

    class Dataset:
        def __init__(self, columns: dict[str, list[Any]]) -> None:
            self.columns = columns

        @classmethod
        def from_dict(cls, columns: dict[str, list[Any]]) -> "Dataset":
            return cls(dict(columns))

        def map(self, fn: Any, batched: bool = False, remove_columns: list[str] | None = None) -> "Dataset":
            result = dict(fn(self.columns))
            for col in remove_columns or []:
                result.pop(col, None)
            state["dataset_columns"] = result
            return Dataset(result)

        def __len__(self) -> int:
            return len(next(iter(self.columns.values()), []))

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _module(
            "transformers",
            AutoTokenizer=FakeTokenizer,
            AutoModelForCausalLM=FakeModel,
            TrainingArguments=TrainingArguments,
            Trainer=Trainer,
            DataCollatorForLanguageModeling=DataCollator,
        ),
    )
    monkeypatch.setitem(sys.modules, "peft", _module("peft", LoraConfig=LoraConfig, get_peft_model=get_peft_model))
    monkeypatch.setitem(sys.modules, "datasets", _module("datasets", Dataset=Dataset))
    return state


def test_finetune_available_with_fake_modules(fake_hf: dict[str, Any]) -> None:
    available, message = finetune_available()
    assert available is True
    assert "transformers 0.0-test" in message and "peft" in message and "GPU" in message


def test_run_finetune_pipeline_with_fakes(settings: Settings, tmp_path: Path, fake_hf: dict[str, Any]) -> None:
    dataset = _write_dataset(
        tmp_path / "sft.jsonl",
        [{"id": "1", "messages": MESSAGES}, {"id": "2", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "r"}]}],
    )
    lines: list[str] = []
    out = run_finetune(settings, dataset, overrides={"epochs": 2, "max_seq_len": 16}, on_log=lines.append)
    assert out == settings.models_dir / "lora-qwen2.5-coder-1.5b-instruct"
    adapter = out / ADAPTER_DIRNAME
    assert (adapter / "adapter_config.json").is_file() and (adapter / "tokenizer_config.json").is_file()
    assert json.loads((adapter / "adapter_config.json").read_text())["r"] == settings.train.finetune.lora_r
    assert "ADAPTER ./adapter" in (out / MODELFILE_NAME).read_text(encoding="utf-8")
    assert "merge_and_unload" in (out / README_NAME).read_text(encoding="utf-8")
    info = json.loads((out / "training.json").read_text(encoding="utf-8"))
    assert info["n_examples"] == 2 and info["has_tool_role"] is True and info["load_in_4bit"] is False
    assert info["metrics"] == {"train_loss": 0.5} and info["config"]["epochs"] == 2.0

    assert fake_hf["tokenizer_name"] == BASE and fake_hf["model_name"] == BASE
    assert fake_hf["tokenizer"].pad_token == "</s>"
    assert fake_hf["model_kwargs"] == {}  # pas de cuda ⇒ ni 4 bits ni bf16
    assert fake_hf["lora"]["r"] == settings.train.finetune.lora_r
    assert fake_hf["lora"]["lora_alpha"] == settings.train.finetune.lora_alpha
    assert fake_hf["lora"]["target_modules"] == list(LORA_TARGET_MODULES)
    assert fake_hf["lora"]["task_type"] == "CAUSAL_LM"
    args = fake_hf["training_args"]
    assert args["num_train_epochs"] == 2.0 and args["learning_rate"] == settings.train.finetune.learning_rate
    assert args["per_device_train_batch_size"] == settings.train.finetune.batch_size
    assert args["gradient_accumulation_steps"] == settings.train.finetune.gradient_accumulation
    assert args["bf16"] is False and args["report_to"] == "none"
    assert fake_hf["max_length"] == 16
    assert all(len(ids) <= 16 for ids in fake_hf["dataset_columns"]["input_ids"])
    assert "text" not in fake_hf["dataset_columns"]
    assert fake_hf["mlm"] is False and fake_hf["trained"] is True and fake_hf["printed"] is True
    # Les tours tool sont conservés quand le gabarit les connaît (rendu de l'exemple 1, après la sonde).
    rendered_first = fake_hf["tokenizer"].rendered[1]
    assert [m["role"] for m in rendered_first] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert any("fine-tuning LoRA" in line for line in lines) and any("adaptateur enregistré" in line for line in lines)


def test_run_finetune_renders_tool_turns_as_user_when_template_lacks_role(
    settings: Settings, tmp_path: Path, fake_hf: dict[str, Any]
) -> None:
    fake_hf["tool_role"] = False
    dataset = _write_dataset(tmp_path / "sft.jsonl", [{"id": "1", "messages": MESSAGES}])
    out = run_finetune(settings, dataset, out_dir=tmp_path / "sortie")
    assert out == tmp_path / "sortie" and (out / ADAPTER_DIRNAME / "adapter_config.json").is_file()
    rendered = fake_hf["tokenizer"].rendered[-1]
    assert [m["role"] for m in rendered] == ["system", "user", "assistant", "user", "assistant"]
    assert rendered[3]["content"].startswith("[résultat de l'outil read_file]\nprint(1)")
    info = json.loads((out / "training.json").read_text(encoding="utf-8"))
    assert info["has_tool_role"] is False


def test_run_finetune_validates_inputs_before_training(settings: Settings, tmp_path: Path, fake_hf: dict[str, Any]) -> None:
    with pytest.raises(FileNotFoundError):
        run_finetune(settings, tmp_path / "absent.jsonl")
    dataset = _write_dataset(tmp_path / "sft.jsonl", [{"id": "1", "messages": MESSAGES}])
    with pytest.raises(ValueError):
        run_finetune(settings, dataset, overrides={"inconnu": True})
    assert "trained" not in fake_hf
