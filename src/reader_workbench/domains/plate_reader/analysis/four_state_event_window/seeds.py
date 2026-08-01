"""Deterministic seed derivation for four-state event-window resampling."""

from __future__ import annotations

import hashlib


def stable_seed(base: int, *parts: str) -> int:
    """Derive a reproducible unsigned seed from an explicit identity tuple."""

    digest = hashlib.sha256((str(base) + ":" + ":".join(parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


__all__ = ["stable_seed"]
