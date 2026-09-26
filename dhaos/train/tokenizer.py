"""Tokenizers pour l'entraînement du modèle nano.

- ``BPETokenizer``  : BPE *octet-niveau* pur Python. Base = 256 octets +
  tokens spéciaux ; fusions apprises sur les paires les plus fréquentes.
  L'apprentissage compte les paires sur la liste des *mots uniques* (pré-
  découpage regex sur espaces / lettres / chiffres / ponctuation) pondérés
  par leur fréquence, avec mise à jour incrémentale des comptes et un tas
  pour sélectionner la meilleure paire — pas de recomptage sur tout le texte
  à chaque fusion.
- ``ByteTokenizer`` : octets bruts (256 + spéciaux), aucun apprentissage.

Tokens spéciaux (identifiants réservés, communs aux deux tokenizers) :
``<|doc|>`` (séparateur de documents du corpus) et ``<|endoftext|>``.
Ils sont reconnus tels quels dans le texte à l'encodage et ne participent
jamais aux fusions.

Aller-retour exact : ``decode(encode(texte)) == texte`` pour tout texte
Unicode valide (accents, emoji, code). ``decode`` accepte des identifiants
inconnus ou des séquences d'octets invalides (``errors="replace"``).
"""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

LogCallback = Callable[[str], None]

DOC_TOKEN = "<|doc|>"
EOT_TOKEN = "<|endoftext|>"
SPECIAL_TOKENS: tuple[str, ...] = (DOC_TOKEN, EOT_TOKEN)
N_BYTES = 256
SPECIAL_IDS: dict[str, int] = {tok: N_BYTES + i for i, tok in enumerate(SPECIAL_TOKENS)}
DOC_ID = SPECIAL_IDS[DOC_TOKEN]
EOT_ID = SPECIAL_IDS[EOT_TOKEN]
N_BASE = N_BYTES + len(SPECIAL_TOKENS)  # premier identifiant disponible pour une fusion

TOKENIZER_KINDS = ("bytes", "bpe")
FORMAT_VERSION = 1
REPLACEMENT_BYTES = "�".encode("utf-8")
_ENCODE_CACHE_MAX = 200_000

# Découpe le texte autour des tokens spéciaux (groupe capturant ⇒ conservés).
_SPECIAL_RE = re.compile(
    "(" + "|".join(re.escape(t) for t in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + ")"
)
# Pré-découpage façon GPT-2 avec la syntaxe de ``re`` : espace optionnel +
# lettres/soulignés, espace optionnel + chiffres, espace optionnel +
# ponctuation, blancs. Toute chaîne est couverte intégralement.
PRETOKEN_RE = re.compile(r" ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")

Pair = tuple[int, int]


def _log(cb: LogCallback | None, message: str) -> None:
    if cb is None:
        return
    try:
        cb(message)
    except Exception:  # noqa: BLE001 — un callback d'affichage ne doit rien casser
        pass


def _merge_word(word: list[int], pair: Pair, new_id: int) -> list[int]:
    """Remplace chaque occurrence (de gauche à droite) de ``pair`` par ``new_id``."""
    a, b = pair
    out: list[int] = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def _split_specials(text: str) -> Iterator[tuple[bool, str]]:
    """Alterne ``(False, texte)`` et ``(True, token spécial)`` ; parties vides omises."""
    for i, part in enumerate(_SPECIAL_RE.split(text)):
        if part:
            yield (i % 2 == 1, part)


class _BaseTokenizer:
    """Comportement commun : tokens spéciaux, décodage tolérant, sauvegarde."""

    kind: str = "base"

    def __init__(self) -> None:
        self._vocab: list[bytes] = [bytes([i]) for i in range(N_BYTES)]
        self._vocab.extend(tok.encode("utf-8") for tok in SPECIAL_TOKENS)

    # ------------------------------------------------------------- infos
    @property
    def special_ids(self) -> dict[str, int]:
        return dict(SPECIAL_IDS)

    @property
    def doc_id(self) -> int:
        return DOC_ID

    @property
    def eot_id(self) -> int:
        return EOT_ID

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    # ---------------------------------------------------------- encodage
    def _encode_plain(self, text: str) -> list[int]:  # pragma: no cover - abstrait
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Texte → identifiants (les tokens spéciaux présents dans le texte sont reconnus)."""
        ids: list[int] = []
        for is_special, part in _split_specials(str(text if text is not None else "")):
            if is_special:
                ids.append(SPECIAL_IDS[part])
            else:
                ids.extend(self._encode_plain(part))
        return ids

    def decode(self, ids: Any) -> str:
        """Identifiants → texte ; identifiants inconnus et octets invalides ⇒ U+FFFD."""
        out = bytearray()
        vocab = self._vocab
        n = len(vocab)
        for raw in ids:
            try:
                i = int(raw)
            except (TypeError, ValueError):
                out += REPLACEMENT_BYTES
                continue
            if 0 <= i < n:
                out += vocab[i]
            else:
                out += REPLACEMENT_BYTES
        return out.decode("utf-8", errors="replace")

    # -------------------------------------------------------- persistance
    def train(self, text: str, vocab_size: int, *, min_frequency: int = 2, on_log: LogCallback | None = None):
        """Apprentissage (sans effet pour un tokenizer sans fusions)."""
        return self

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": FORMAT_VERSION,
            "specials": dict(SPECIAL_IDS),
            "vocab_size": self.vocab_size,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._payload(), ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    @staticmethod
    def _read_payload(path: str | Path, expected_kind: str) -> dict[str, Any]:
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"tokenizer illisible : {p} ({e})") from e
        if not isinstance(data, dict):
            raise ValueError(f"tokenizer invalide : {p} (objet JSON attendu)")
        kind = data.get("kind")
        if kind != expected_kind:
            raise ValueError(f"tokenizer de type {kind!r} dans {p}, attendu {expected_kind!r}")
        specials = data.get("specials")
        if isinstance(specials, dict) and specials != SPECIAL_IDS:
            raise ValueError(f"tokens spéciaux incompatibles dans {p} : {specials}")
        return data


class ByteTokenizer(_BaseTokenizer):
    """Octets UTF-8 bruts (identifiants 0..255) + tokens spéciaux ; sans apprentissage."""

    kind = "bytes"

    def _encode_plain(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        cls._read_payload(path, cls.kind)
        return cls()


class BPETokenizer(_BaseTokenizer):
    """BPE octet-niveau : 256 octets + tokens spéciaux + fusions apprises."""

    kind = "bpe"

    def __init__(self, merges: list[Pair] | None = None) -> None:
        super().__init__()
        self._merges: list[Pair] = []
        self._ranks: dict[Pair, int] = {}
        self._cache: dict[bytes, list[int]] = {}
        for pair in merges or []:
            self._add_merge(pair)

    # ---------------------------------------------------------- internes
    def _reset(self) -> None:
        del self._vocab[N_BASE:]
        self._merges = []
        self._ranks = {}
        self._cache = {}

    def _add_merge(self, pair: Pair) -> int:
        a, b = int(pair[0]), int(pair[1])
        limit = len(self._vocab)
        if not (0 <= a < limit and 0 <= b < limit) or a in SPECIAL_IDS.values() or b in SPECIAL_IDS.values():
            raise ValueError(f"fusion invalide : {pair!r}")
        if (a, b) in self._ranks:
            raise ValueError(f"fusion dupliquée : {pair!r}")
        new_id = limit
        self._merges.append((a, b))
        self._ranks[(a, b)] = len(self._merges) - 1
        self._vocab.append(self._vocab[a] + self._vocab[b])
        self._cache = {}
        return new_id

    def _encode_word(self, word_bytes: bytes) -> list[int]:
        cached = self._cache.get(word_bytes)
        if cached is not None:
            return cached
        word = list(word_bytes)
        ranks = self._ranks
        if ranks:
            while len(word) > 1:
                best: Pair | None = None
                best_rank = -1
                for pair in zip(word, word[1:]):
                    rank = ranks.get(pair)
                    if rank is not None and (best is None or rank < best_rank):
                        best, best_rank = pair, rank
                if best is None:
                    break
                word = _merge_word(word, best, N_BASE + best_rank)
        if len(self._cache) >= _ENCODE_CACHE_MAX:
            self._cache = {}
        self._cache[word_bytes] = word
        return word

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for m in PRETOKEN_RE.finditer(text):
            ids.extend(self._encode_word(m.group().encode("utf-8")))
        return ids

    # ------------------------------------------------------------- infos
    @property
    def merges(self) -> list[Pair]:
        return list(self._merges)

    def token_bytes(self, token_id: int) -> bytes | None:
        """Octets d'un identifiant (``None`` s'il est inconnu)."""
        i = int(token_id)
        return self._vocab[i] if 0 <= i < len(self._vocab) else None

    # ------------------------------------------------------ apprentissage
    def train(
        self,
        text: str,
        vocab_size: int,
        *,
        min_frequency: int = 2,
        on_log: LogCallback | None = None,
    ) -> "BPETokenizer":
        """Apprend les fusions sur ``text`` jusqu'à ``vocab_size`` tokens, ou
        jusqu'à ce qu'aucune paire n'atteigne ``min_frequency``. Remplace les
        fusions existantes. Renvoie ``self``."""
        text = str(text if text is not None else "")
        vocab_size = int(vocab_size)
        if vocab_size < N_BASE:
            raise ValueError(
                f"vocab_size doit valoir au moins {N_BASE} (256 octets + {len(SPECIAL_TOKENS)} tokens spéciaux), reçu {vocab_size}"
            )
        min_frequency = max(1, int(min_frequency))
        target = vocab_size - N_BASE
        self._reset()

        word_freqs: Counter[bytes] = Counter()
        for is_special, part in _split_specials(text):
            if not is_special:
                for m in PRETOKEN_RE.finditer(part):
                    word_freqs[m.group().encode("utf-8")] += 1
        words: list[list[int]] = [list(w) for w in word_freqs]
        freqs: list[int] = list(word_freqs.values())
        _log(on_log, f"tokenizer BPE : {len(words)} mot(s) unique(s), objectif {target} fusion(s)")

        pair_counts: dict[Pair, int] = defaultdict(int)
        pair_words: dict[Pair, set[int]] = defaultdict(set)
        for i, w in enumerate(words):
            f = freqs[i]
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                pair_words[p].add(i)
        heap: list[tuple[int, Pair]] = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        log_every = max(1, target // 10)

        while len(self._merges) < target and heap:
            neg, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count != -neg:
                continue  # entrée périmée
            if count < min_frequency:
                break
            new_id = self._add_merge(pair)
            touched: set[Pair] = set()
            for i in list(pair_words.get(pair, ())):
                w = words[i]
                f = freqs[i]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    pair_words[p].discard(i)
                    touched.add(p)
                nw = _merge_word(w, pair, new_id)
                words[i] = nw
                for p in zip(nw, nw[1:]):
                    pair_counts[p] += f
                    pair_words[p].add(i)
                    touched.add(p)
            pair_counts.pop(pair, None)
            pair_words.pop(pair, None)
            touched.discard(pair)
            for p in touched:
                c = pair_counts.get(p, 0)
                if c <= 0:
                    pair_counts.pop(p, None)
                    pair_words.pop(p, None)
                else:
                    heapq.heappush(heap, (-c, p))
            n_done = len(self._merges)
            if n_done % log_every == 0 or n_done == target:
                _log(on_log, f"tokenizer BPE : {n_done}/{target} fusion(s), dernière paire ×{count}")
        _log(on_log, f"tokenizer BPE : {len(self._merges)} fusion(s), vocabulaire {self.vocab_size}")
        return self

    # -------------------------------------------------------- persistance
    def _payload(self) -> dict[str, Any]:
        data = super()._payload()
        data["merges"] = [[a, b] for a, b in self._merges]
        data["vocab"] = {str(N_BASE + i): self._vocab[N_BASE + i].hex() for i in range(len(self._merges))}
        return data

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = cls._read_payload(path, cls.kind)
        merges_raw = data.get("merges", [])
        if not isinstance(merges_raw, list):
            raise ValueError(f"tokenizer invalide : {path} (liste de fusions attendue)")
        merges: list[Pair] = []
        for item in merges_raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"tokenizer invalide : {path} (fusion {item!r})")
            try:
                merges.append((int(item[0]), int(item[1])))
            except (TypeError, ValueError) as e:
                raise ValueError(f"tokenizer invalide : {path} (fusion {item!r})") from e
        tok = cls(merges)
        vocab = data.get("vocab")
        if isinstance(vocab, dict):
            for key, hexbytes in vocab.items():
                try:
                    idx = int(key)
                    expected = bytes.fromhex(str(hexbytes))
                except ValueError as e:
                    raise ValueError(f"tokenizer invalide : {path} (vocabulaire {key!r})") from e
                if tok.token_bytes(idx) != expected:
                    raise ValueError(f"tokenizer incohérent : {path} (token {idx} ne correspond pas aux fusions)")
        return tok


def get_tokenizer(kind: str) -> BPETokenizer | ByteTokenizer:
    """``"bpe"`` → ``BPETokenizer`` (à entraîner), ``"bytes"`` → ``ByteTokenizer``."""
    key = str(kind or "").strip().lower()
    if key == "bpe":
        return BPETokenizer()
    if key == "bytes":
        return ByteTokenizer()
    raise ValueError(f"tokenizer inconnu : {kind!r} (attendu : {', '.join(TOKENIZER_KINDS)})")


def load_tokenizer(path: str | Path) -> BPETokenizer | ByteTokenizer:
    """Recharge un tokenizer sauvegardé, quel que soit son type."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"tokenizer illisible : {p} ({e})") from e
    kind = data.get("kind") if isinstance(data, dict) else None
    if kind == "bpe":
        return BPETokenizer.load(p)
    if kind == "bytes":
        return ByteTokenizer.load(p)
    raise ValueError(f"tokenizer de type inconnu dans {p} : {kind!r}")


__all__ = [
    "BPETokenizer",
    "ByteTokenizer",
    "DOC_ID",
    "DOC_TOKEN",
    "EOT_ID",
    "EOT_TOKEN",
    "N_BASE",
    "PRETOKEN_RE",
    "SPECIAL_IDS",
    "SPECIAL_TOKENS",
    "TOKENIZER_KINDS",
    "get_tokenizer",
    "load_tokenizer",
]
