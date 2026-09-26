"""Petits utilitaires partagés."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def is_probably_binary(sample: bytes) -> bool:
    """Heuristique : octet nul ou trop de caractères de contrôle."""
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    text_chars = bytes(range(32, 127)) + b"\n\r\t\b\f\x1b"
    nontext = sum(1 for b in sample if b not in text_chars and b < 128)
    return nontext / max(1, len(sample)) > 0.30


def truncate(text: str, max_chars: int, marker: str = "\n… [tronqué : {n} caractères omis]") -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + marker.format(n=omitted)


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if size < 1024 or unit == "To":
            return f"{size:.0f} {unit}" if unit == "o" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} o"
