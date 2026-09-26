"""Embedders.

``Embedder`` : ``name`` (identifiant stable), ``dim``, ``embed(texts) -> np.ndarray``
de forme ``(n, dim)`` en float32, vecteurs normalisés L2.

- ``HashEmbedder``  : hachage de n-grammes de mots/caractères, sans modèle,
  toujours disponible (qualité modeste, recherche lexicale-ish).
- ``OllamaEmbedder`` : ``POST {host}/api/embed`` avec ``kb.embed_model``.
- ``SentenceTransformersEmbedder`` : optionnel (``pip install dhaos[embeddings]``).

``get_embedder(settings, kind=None)`` applique ``kb.embedder`` (``auto`` :
Ollama si joignable et modèle présent, sinon hash).
"""
from __future__ import annotations

import functools
import hashlib
import math
import re
from collections import Counter
from typing import Any, Protocol

import httpx
import numpy as np

from ..config import Settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingError(RuntimeError):
    """Échec du calcul d'embeddings (service injoignable, réponse invalide)."""


# --------------------------------------------------------------- utilitaires
_WORD_RE = re.compile(r"\w+", re.UNICODE)
MIN_HASH_DIM = 16
OLLAMA_BATCH_SIZE = 32
OLLAMA_PROBE_TIMEOUT = 3.0
_FEATURE_WEIGHTS = {"w": 1.0, "b": 0.6, "c": 0.4}  # mot, bigramme de mots, 3-gramme de caractères


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalise chaque ligne en norme L2 ; une ligne nulle reste nulle."""
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(m / norms, dtype=np.float32)


@functools.lru_cache(maxsize=1 << 18)
def _feature_hash(feature: str) -> int:
    """Hachage déterministe (blake2b, 64 bits) d'une caractéristique textuelle.

    Jamais ``hash()`` : il est randomisé d'un processus à l'autre.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def embedder_id(embedder: Any) -> str:
    """Identifiant ``nom:dim`` mémorisé par chaque base de savoir."""
    return f"{embedder.name}:{int(embedder.dim)}"


# ------------------------------------------------------------- HashEmbedder
class HashEmbedder:
    """Feature hashing déterministe : unigrammes et bigrammes de mots
    (minuscules) et 3-grammes de caractères, signe alterné selon un bit du
    hachage, pondération sublinéaire des fréquences, normalisation L2."""

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = max(MIN_HASH_DIM, int(dim))

    @staticmethod
    def features(text: str) -> Counter[str]:
        """Caractéristiques comptées d'un texte (préfixe w:/b:/c: selon le type)."""
        words = _WORD_RE.findall(str(text).lower())
        feats: Counter[str] = Counter()
        for w in words:
            feats["w:" + w] += 1
        for a, b in zip(words, words[1:]):
            feats["b:" + a + " " + b] += 1
        padded = " " + " ".join(words) + " "
        for i in range(len(padded) - 2):
            feats["c:" + padded[i : i + 3]] += 1
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            feats = self.features(text or "")
            if not feats:
                continue
            idx = np.empty(len(feats), dtype=np.int64)
            vals = np.empty(len(feats), dtype=np.float32)
            for i, (feat, count) in enumerate(feats.items()):
                h = _feature_hash(feat)
                idx[i] = h % self.dim
                sign = -1.0 if (h >> 60) & 1 else 1.0
                vals[i] = sign * _FEATURE_WEIGHTS[feat[0]] * (1.0 + math.log(count))
            np.add.at(out[row], idx, vals)
        return normalize_rows(out)


# ----------------------------------------------------------- OllamaEmbedder
class OllamaEmbedder:
    """Embeddings via ``POST {host}/api/embed`` (modèle ``backends.ollama.embed_model``).

    La dimension est découverte au premier appel (ou sondée avec le texte
    ``"dim"``). Un ``httpx.Client`` peut être injecté (tests : ``MockTransport``).
    """

    def __init__(self, settings: Settings, *, client: httpx.Client | None = None):
        cfg = settings.backends.ollama
        self.host = cfg.host.rstrip("/")
        self.model = cfg.embed_model
        self.name = f"ollama:{self.model}"
        self.timeout = float(cfg.timeout)
        self._client = client
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._request(["dim"])
        assert self._dim is not None
        return self._dim

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        url = f"{self.host}/api/embed"
        try:
            if self._client is not None:
                return self._client.post(url, json=payload, timeout=self.timeout)
            with httpx.Client(timeout=self.timeout) as client:
                return client.post(url, json=payload)
        except httpx.HTTPError as e:
            raise EmbeddingError(f"Ollama injoignable ({self.host}) : {e}") from e

    def _request(self, batch: list[str]) -> np.ndarray:
        inputs = [t if isinstance(t, str) and t.strip() else "(vide)" for t in batch]
        response = self._post({"model": self.model, "input": inputs})
        if response.status_code >= 400:
            detail = ""
            try:
                detail = str(response.json().get("error", ""))
            except Exception:  # noqa: BLE001
                detail = response.text[:200]
            raise EmbeddingError(
                f"Ollama /api/embed a répondu {response.status_code} pour le modèle "
                f"{self.model!r} : {detail or 'sans détail'}"
            )
        try:
            data = response.json()
            vectors = data["embeddings"]
            matrix = np.asarray(vectors, dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            raise EmbeddingError(f"réponse d'embedding Ollama invalide : {e}") from e
        if matrix.ndim != 2 or matrix.shape[0] != len(inputs) or matrix.shape[1] == 0:
            raise EmbeddingError(
                f"réponse d'embedding Ollama invalide : forme {matrix.shape} pour {len(inputs)} texte(s)"
            )
        if self._dim is None:
            self._dim = int(matrix.shape[1])
        elif int(matrix.shape[1]) != self._dim:
            raise EmbeddingError(
                f"dimension d'embedding incohérente : {matrix.shape[1]} (attendu {self._dim})"
            )
        return matrix

    def embed(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = [
            self._request(texts[i : i + OLLAMA_BATCH_SIZE])
            for i in range(0, len(texts), OLLAMA_BATCH_SIZE)
        ]
        return normalize_rows(np.vstack(rows))


def ollama_model_available(settings: Settings, *, client: httpx.Client | None = None) -> bool:
    """``True`` si Ollama répond sur ``GET /api/tags`` et propose ``embed_model``
    (comparaison aussi sans le suffixe ``:latest``). Ne lève jamais."""
    cfg = settings.backends.ollama
    url = f"{cfg.host.rstrip('/')}/api/tags"
    try:
        if client is not None:
            response = client.get(url, timeout=OLLAMA_PROBE_TIMEOUT)
        else:
            with httpx.Client(timeout=OLLAMA_PROBE_TIMEOUT) as own:
                response = own.get(url)
        if response.status_code >= 400:
            return False
        models = response.json().get("models") or []
        names = {str(m.get("name") or m.get("model") or "") for m in models if isinstance(m, dict)}
    except Exception:  # noqa: BLE001
        return False

    def bare(name: str) -> str:
        return name[: -len(":latest")] if name.endswith(":latest") else name

    wanted = bare(cfg.embed_model)
    return any(bare(n) == wanted for n in names)


# ----------------------------------------------- SentenceTransformersEmbedder
class SentenceTransformersEmbedder:
    """Embeddings locaux via ``sentence-transformers`` (import paresseux)."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "sentence-transformers n'est pas installé : pip install dhaos[embeddings]"
            ) from e
        self.model_name = model_name
        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return normalize_rows(np.asarray(vectors, dtype=np.float32))


# ------------------------------------------------------------------ fabrique
def get_embedder(settings: Settings, kind: str | None = None) -> Any:
    """Construit l'embedder demandé (``kind`` ou ``settings.kb.embedder``).

    ``auto`` : Ollama si joignable et modèle présent, sinon ``HashEmbedder`` ;
    ne lève jamais. ``ollama`` / ``sentence-transformers`` / ``hash`` : explicite.
    """
    kind = (kind or settings.kb.embedder or "auto").strip().lower()
    if kind == "hash":
        return HashEmbedder(settings.kb.hash_dim)
    if kind == "ollama":
        return OllamaEmbedder(settings)
    if kind in ("sentence-transformers", "st"):
        return SentenceTransformersEmbedder(settings.kb.st_model)
    if kind == "auto":
        try:
            if ollama_model_available(settings):
                return OllamaEmbedder(settings)
        except Exception:  # noqa: BLE001 — auto ne doit jamais échouer
            pass
        return HashEmbedder(settings.kb.hash_dim)
    raise ValueError(f"embedder inconnu : {kind!r} (attendu : auto, ollama, sentence-transformers, hash)")


__all__ = [
    "Embedder",
    "EmbeddingError",
    "HashEmbedder",
    "OllamaEmbedder",
    "SentenceTransformersEmbedder",
    "embedder_id",
    "get_embedder",
    "normalize_rows",
    "ollama_model_available",
]
