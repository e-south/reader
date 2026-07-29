"""Content provenance helpers for response-window records."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def stable_seed(base: int, *parts: str) -> int:
    """Derive a reproducible unsigned seed from an explicit identity tuple."""

    digest = hashlib.sha256((str(base) + ":" + ":".join(parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


__all__ = ["sha256_file", "stable_seed"]
