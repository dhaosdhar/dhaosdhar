"""GPT décodeur minimal en PyTorch (modèle « nano », pédagogique).

Architecture classique : embeddings de tokens (``wte``) et de positions
(``wpe``), blocs pré-LayerNorm (attention causale multi-têtes via
``scaled_dot_product_attention(is_causal=True)`` + MLP GELU), ``ln_f`` et
une tête ``lm_head`` dont les poids sont **liés** à ``wte``.

C'est le seul module de ``dhaos.train`` qui importe ``torch`` au niveau
module : les autres l'importent paresseusement pour rester importables sans
torch.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparamètres du modèle (validés à la construction)."""

    vocab_size: int = 258
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            try:
                value = int(getattr(self, name))
            except (TypeError, ValueError) as e:
                raise ValueError(f"GPTConfig.{name} doit être un entier") from e
            if value <= 0:
                raise ValueError(f"GPTConfig.{name} doit être strictement positif, reçu {value}")
            setattr(self, name, value)
        try:
            self.dropout = float(self.dropout)
        except (TypeError, ValueError) as e:
            raise ValueError("GPTConfig.dropout doit être un nombre") from e
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"GPTConfig.dropout doit être dans [0, 1), reçu {self.dropout}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) doit être divisible par n_head ({self.n_head})")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPTConfig":
        """Construit depuis un dict (clés inconnues ignorées, valeurs validées)."""
        if not isinstance(data, dict):
            raise ValueError("configuration du modèle invalide (objet attendu)")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class CausalSelfAttention(nn.Module):
    """Attention multi-têtes causale (masque géré par ``is_causal=True``)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Perceptron à deux couches (facteur 4) avec GELU."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Bloc transformeur pré-LayerNorm avec connexions résiduelles."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Modèle de langage causal complet."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # poids liés
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_params(self, *, non_embedding: bool = False) -> int:
        """Nombre de paramètres (les poids liés ne sont comptés qu'une fois)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return int(n)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``idx`` : (B, T) d'entiers ; renvoie ``(logits (B, T, V), perte ou None)``."""
        if idx.dim() != 2:
            raise ValueError(f"idx doit être de forme (B, T), reçu {tuple(idx.shape)}")
        _B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"séquence de longueur {T} > block_size {self.config.block_size}")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Complète ``idx`` (B, T) de ``max_new_tokens`` tokens échantillonnés.

        ``temperature <= 0`` ⇒ décodage glouton ; ``top_k`` restreint aux k
        logits les plus élevés.
        """
        max_new_tokens = int(max_new_tokens)
        temperature = float(temperature)
        if top_k is not None:
            top_k = max(1, int(top_k))
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


__all__ = ["Block", "CausalSelfAttention", "GPT", "GPTConfig", "MLP"]
