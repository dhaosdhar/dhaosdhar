"""Tests du tokenizer BPE octet-niveau et du tokenizer octets."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from dhaos.train.tokenizer import (
    DOC_ID,
    DOC_TOKEN,
    EOT_ID,
    EOT_TOKEN,
    N_BASE,
    PRETOKEN_RE,
    SPECIAL_IDS,
    BPETokenizer,
    ByteTokenizer,
    get_tokenizer,
    load_tokenizer,
)

SAMPLE = (
    "Héllo wörld — café ☕ 🎉 emoji ; naïve façade\n"
    "\tdef f(x):\n\t\treturn x * 2  # commentaire\n"
    "snake_case_1 = {'clé': [1, 2.5, None]}\n"
    f"fin de document {DOC_TOKEN} suite {EOT_TOKEN}\n"
    "中文 日本語 العربية\r\n"
)

CODE_CORPUS = (
    "def fonction(x):\n    return x + 1\n\n" * 60
    + "class Chose:\n    def __init__(self):\n        self.valeur = 0\n\n" * 40
    + "éèà ça marche bien 🎉 pour les accents " * 40
)


@pytest.mark.parametrize("kind", ["bytes", "bpe"])
def test_roundtrip_exact_untrained(kind: str) -> None:
    tok = get_tokenizer(kind)
    ids = tok.encode(SAMPLE)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == SAMPLE


def test_special_tokens_have_reserved_ids() -> None:
    assert SPECIAL_IDS == {DOC_TOKEN: 256, EOT_TOKEN: 257}
    assert (DOC_ID, EOT_ID, N_BASE) == (256, 257, 258)
    for tok in (ByteTokenizer(), BPETokenizer()):
        ids = tok.encode(f"a{DOC_TOKEN}b{EOT_TOKEN}")
        assert ids == [ord("a"), DOC_ID, ord("b"), EOT_ID]
        assert tok.decode([DOC_ID]) == DOC_TOKEN
        assert tok.decode([EOT_ID]) == EOT_TOKEN
        assert tok.special_ids == SPECIAL_IDS
        assert (tok.doc_id, tok.eot_id) == (DOC_ID, EOT_ID)


def test_byte_tokenizer_vocab_and_training_noop() -> None:
    tok = ByteTokenizer()
    assert tok.vocab_size == N_BASE
    assert tok.train("n'importe quoi " * 100, 1000) is tok
    assert tok.vocab_size == N_BASE
    assert tok.encode("é") == [0xC3, 0xA9]


def test_bpe_training_improves_compression_and_keeps_roundtrip() -> None:
    raw = len(ByteTokenizer().encode(CODE_CORPUS))
    tok = BPETokenizer()
    assert tok.train(CODE_CORPUS, 400) is tok
    assert N_BASE < tok.vocab_size <= 400
    assert len(tok.merges) == tok.vocab_size - N_BASE
    compressed = tok.encode(CODE_CORPUS)
    assert len(compressed) < raw / 2
    assert max(compressed) < tok.vocab_size
    assert tok.decode(compressed) == CODE_CORPUS
    # Texte inédit (accents, emoji, tokens spéciaux) : aller-retour toujours exact.
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE
    assert DOC_ID in tok.encode(SAMPLE)


def test_bpe_training_is_deterministic() -> None:
    a = BPETokenizer().train(CODE_CORPUS, 350)
    b = BPETokenizer().train(CODE_CORPUS, 350)
    assert a.merges == b.merges
    assert a.encode(SAMPLE) == b.encode(SAMPLE)


def test_bpe_training_stops_at_min_frequency() -> None:
    tok = BPETokenizer().train(CODE_CORPUS, 2000, min_frequency=10_000)
    assert tok.vocab_size == N_BASE
    assert tok.merges == []
    small = BPETokenizer().train("abc abc abc", 2000, min_frequency=2)
    assert 0 < len(small.merges) < 2000 - N_BASE


def test_bpe_training_rejects_too_small_vocab_and_logs() -> None:
    with pytest.raises(ValueError):
        BPETokenizer().train("abc", N_BASE - 1)
    lines: list[str] = []
    BPETokenizer().train(CODE_CORPUS, 300, on_log=lines.append)
    assert lines and any("fusion" in line for line in lines)

    def boom(_line: str) -> None:
        raise RuntimeError("le callback ne doit rien casser")

    BPETokenizer().train(CODE_CORPUS, 300, on_log=boom)


def test_retraining_replaces_previous_merges() -> None:
    tok = BPETokenizer().train(CODE_CORPUS, 400)
    first = tok.merges
    tok.train("xyz xyz xyz xyz", 300)
    assert tok.merges != first
    assert tok.decode(tok.encode(CODE_CORPUS)) == CODE_CORPUS


def test_save_load_bpe_identical(tmp_path: Path) -> None:
    tok = BPETokenizer().train(CODE_CORPUS, 400)
    path = tok.save(tmp_path / "sub" / "tok.json")
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["kind"] == "bpe"
    assert data["merges"] and all(len(m) == 2 for m in data["merges"])
    assert data["vocab"] and set(data["vocab"]) == {str(N_BASE + i) for i in range(len(data["merges"]))}
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(CODE_CORPUS) == tok.encode(CODE_CORPUS)
    assert loaded.encode(SAMPLE) == tok.encode(SAMPLE)
    generic = load_tokenizer(path)
    assert isinstance(generic, BPETokenizer) and generic.merges == tok.merges


def test_save_load_bytes(tmp_path: Path) -> None:
    path = ByteTokenizer().save(tmp_path / "bytes.json")
    loaded = load_tokenizer(path)
    assert isinstance(loaded, ByteTokenizer)
    assert loaded.encode(SAMPLE) == ByteTokenizer().encode(SAMPLE)
    assert isinstance(ByteTokenizer.load(path), ByteTokenizer)


def test_load_rejects_invalid_files(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{pas du json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_tokenizer(bad_json)
    with pytest.raises(ValueError):
        BPETokenizer.load(bad_json)
    with pytest.raises(ValueError):
        load_tokenizer(tmp_path / "absent.json")

    wrong_kind = tmp_path / "kind.json"
    wrong_kind.write_text(json.dumps({"kind": "inconnu"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_tokenizer(wrong_kind)
    with pytest.raises(ValueError):
        BPETokenizer.load(wrong_kind)

    bad_merge = tmp_path / "merge.json"
    bad_merge.write_text(json.dumps({"kind": "bpe", "merges": [[0, 9999]]}), encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(bad_merge)

    special_merge = tmp_path / "special.json"
    special_merge.write_text(json.dumps({"kind": "bpe", "merges": [[DOC_ID, 97]]}), encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(special_merge)

    inconsistent = tmp_path / "vocab.json"
    inconsistent.write_text(
        json.dumps({"kind": "bpe", "merges": [[97, 98]], "vocab": {str(N_BASE): "ffff"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        BPETokenizer.load(inconsistent)

    bad_specials = tmp_path / "specials.json"
    bad_specials.write_text(json.dumps({"kind": "bytes", "specials": {"<|x|>": 256}}), encoding="utf-8")
    with pytest.raises(ValueError):
        ByteTokenizer.load(bad_specials)


def test_decode_is_tolerant() -> None:
    tok = BPETokenizer().train(CODE_CORPUS, 300)
    assert tok.decode([]) == ""
    assert tok.decode([10_000, -1, "x", None]) == "�" * 4  # type: ignore[list-item]
    assert tok.decode([0xC3]) == "�"  # octet UTF-8 incomplet
    assert tok.decode([0xFF, ord("a")]) == "�a"
    assert tok.decode(iter([ord("o"), ord("k")])) == "ok"


def test_encode_accepts_empty_and_none() -> None:
    for tok in (ByteTokenizer(), BPETokenizer()):
        assert tok.encode("") == []
        assert tok.encode(None) == []  # type: ignore[arg-type]
        assert tok.decode(tok.encode("42")) == "42"


def test_get_tokenizer() -> None:
    assert isinstance(get_tokenizer("bpe"), BPETokenizer)
    assert isinstance(get_tokenizer("bytes"), ByteTokenizer)
    assert isinstance(get_tokenizer(" BPE "), BPETokenizer)
    with pytest.raises(ValueError):
        get_tokenizer("mots")


def test_pretokenizer_covers_every_character() -> None:
    rng = random.Random(7)
    text = "".join(chr(rng.randint(0, 0x2FF)) for _ in range(4000))
    text += "".join(chr(c) for c in range(0x4E00, 0x4E40)) + " a_b  c  \n\n\t x1y2 !? ... "
    assert "".join(m.group() for m in PRETOKEN_RE.finditer(text)) == text


def test_token_bytes_and_merged_tokens() -> None:
    tok = BPETokenizer().train("aaaa aaaa aaaa", 300)
    assert tok.token_bytes(ord("a")) == b"a"
    assert tok.token_bytes(N_BASE) is not None and len(tok.token_bytes(N_BASE) or b"") >= 2
    assert tok.token_bytes(tok.vocab_size) is None
    assert tok.encode("aaaa") != list(b"aaaa")
    assert tok.decode(tok.encode("aaaa")) == "aaaa"
