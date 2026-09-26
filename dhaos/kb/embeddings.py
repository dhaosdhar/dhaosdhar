"""Embedders — CONTRAT.

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

from typing import Any, Protocol

import numpy as np

from ..config import Settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def get_embedder(settings: Settings, kind: str | None = None) -> Any:
    raise NotImplementedError
