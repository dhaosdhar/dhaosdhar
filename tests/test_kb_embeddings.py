"""Tests des embedders : hash déterministe, Ollama (MockTransport), fabrique."""
from __future__ import annotations

import json
import sys

import httpx
import numpy as np
import pytest

from dhaos.config import Settings
from dhaos.kb import embeddings as emb
from dhaos.kb.embeddings import (
    EmbeddingError,
    HashEmbedder,
    OllamaEmbedder,
    SentenceTransformersEmbedder,
    embedder_id,
    get_embedder,
    ollama_model_available,
)


# ---------------------------------------------------------------- HashEmbedder
def test_hash_embedder_shape_dtype_and_norm() -> None:
    e = HashEmbedder(dim=64)
    m = e.embed(["bonjour le monde", "def foo(): pass", "autre"])
    assert m.shape == (3, 64)
    assert m.dtype == np.float32
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    assert e.name == "hash"
    assert e.dim == 64


def test_hash_embedder_is_deterministic_across_instances() -> None:
    a = HashEmbedder(dim=128).embed(["Les kangourous sautent haut."])
    b = HashEmbedder(dim=128).embed(["Les kangourous sautent haut."])
    assert np.array_equal(a, b)


def test_hash_embedder_empty_text_gives_zero_vector() -> None:
    m = HashEmbedder(dim=32).embed(["", "   ", "!!! ??? ..."])
    assert m.shape == (3, 32)
    assert not m.any()


def test_hash_embedder_empty_batch() -> None:
    m = HashEmbedder(dim=32).embed([])
    assert m.shape == (0, 32)


def test_hash_embedder_similar_texts_are_closer() -> None:
    e = HashEmbedder(dim=256)
    m = e.embed([
        "installer pytest et lancer les tests unitaires",
        "lancer les tests unitaires avec pytest",
        "recette de la tarte aux pommes et à la cannelle",
    ])
    sim_close = float(m[0] @ m[1])
    sim_far = float(m[0] @ m[2])
    assert sim_close > sim_far
    assert sim_close > 0.3


def test_hash_embedder_case_insensitive_and_min_dim() -> None:
    e = HashEmbedder(dim=2)
    assert e.dim >= emb.MIN_HASH_DIM
    a = e.embed(["Python"])
    b = e.embed(["python"])
    assert np.array_equal(a, b)


def test_hash_features_include_words_bigrams_and_trigrams() -> None:
    feats = HashEmbedder.features("Hello World")
    assert feats["w:hello"] == 1 and feats["w:world"] == 1
    assert feats["b:hello world"] == 1
    assert feats["c:hel"] == 1


# -------------------------------------------------------------- OllamaEmbedder
def _ollama_client(dim: int = 4, *, fail: bool = False, tags: list[str] | None = None) -> tuple[httpx.Client, list[dict]]:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            models = [{"name": n} for n in (tags or [])]
            return httpx.Response(200, json={"models": models})
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        calls.append(payload)
        if fail:
            return httpx.Response(404, json={"error": "model 'x' not found"})
        inputs = payload["input"]
        vectors = [[float(len(t)) + i for i in range(dim)] for t in inputs]
        return httpx.Response(200, json={"model": payload["model"], "embeddings": vectors})

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_ollama_embedder_batches_and_normalizes(settings: Settings) -> None:
    client, calls = _ollama_client(dim=4)
    e = OllamaEmbedder(settings, client=client)
    assert e.name == f"ollama:{settings.backends.ollama.embed_model}"
    texts = [f"texte {i}" for i in range(70)]
    m = e.embed(texts)
    assert m.shape == (70, 4)
    assert m.dtype == np.float32
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    assert [len(c["input"]) for c in calls] == [32, 32, 6]
    assert all(c["model"] == settings.backends.ollama.embed_model for c in calls)
    assert e.dim == 4
    assert embedder_id(e) == f"ollama:{settings.backends.ollama.embed_model}:4"


def test_ollama_embedder_probes_dim_when_unknown(settings: Settings) -> None:
    client, calls = _ollama_client(dim=6)
    e = OllamaEmbedder(settings, client=client)
    assert e.dim == 6
    assert calls and calls[0]["input"] == ["dim"]
    assert e.embed([]).shape == (0, 6)


def test_ollama_embedder_replaces_empty_inputs(settings: Settings) -> None:
    client, calls = _ollama_client(dim=3)
    e = OllamaEmbedder(settings, client=client)
    e.embed(["", "ok"])
    assert calls[-1]["input"][0].strip() != ""


def test_ollama_embedder_http_error(settings: Settings) -> None:
    client, _ = _ollama_client(fail=True)
    e = OllamaEmbedder(settings, client=client)
    with pytest.raises(EmbeddingError) as exc:
        e.embed(["x"])
    assert "404" in str(exc.value)


def test_ollama_embedder_network_error(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refusé")

    e = OllamaEmbedder(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(EmbeddingError):
        e.embed(["x"])


def test_ollama_embedder_invalid_response(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0]]})

    e = OllamaEmbedder(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(EmbeddingError):
        e.embed(["a", "b"])  # 2 textes, 1 vecteur


def test_ollama_model_available_matches_latest_suffix(settings: Settings) -> None:
    settings.backends.ollama.embed_model = "nomic-embed-text"
    client, _ = _ollama_client(tags=["qwen2.5-coder:7b", "nomic-embed-text:latest"])
    assert ollama_model_available(settings, client=client)
    client, _ = _ollama_client(tags=["qwen2.5-coder:7b"])
    assert not ollama_model_available(settings, client=client)


def test_ollama_model_available_never_raises(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("hors ligne")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert ollama_model_available(settings, client=client) is False


# ------------------------------------------------------- SentenceTransformers
def test_sentence_transformers_missing_gives_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError) as exc:
        SentenceTransformersEmbedder("all-MiniLM-L6-v2")
    assert "pip install dhaos[embeddings]" in str(exc.value)


# ------------------------------------------------------------------ fabrique
def test_get_embedder_hash_uses_configured_dim(settings: Settings) -> None:
    settings.kb.hash_dim = 128
    e = get_embedder(settings)
    assert isinstance(e, HashEmbedder)
    assert e.dim == 128
    assert embedder_id(e) == "hash:128"


def test_get_embedder_explicit_kind_overrides_settings(settings: Settings) -> None:
    settings.kb.embedder = "hash"
    e = get_embedder(settings, "ollama")
    assert isinstance(e, OllamaEmbedder)


def test_get_embedder_auto_falls_back_to_hash(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings.kb.embedder = "auto"
    monkeypatch.setattr(emb, "ollama_model_available", lambda s, **kw: False)
    assert isinstance(get_embedder(settings), HashEmbedder)


def test_get_embedder_auto_prefers_ollama(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings.kb.embedder = "auto"
    monkeypatch.setattr(emb, "ollama_model_available", lambda s, **kw: True)
    assert isinstance(get_embedder(settings, "auto"), OllamaEmbedder)


def test_get_embedder_auto_never_raises(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(s: Settings, **kw: object) -> bool:
        raise RuntimeError("panne")

    monkeypatch.setattr(emb, "ollama_model_available", boom)
    assert isinstance(get_embedder(settings, "auto"), HashEmbedder)


def test_get_embedder_unknown_kind(settings: Settings) -> None:
    with pytest.raises(ValueError):
        get_embedder(settings, "magie")
