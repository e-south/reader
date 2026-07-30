from __future__ import annotations

import hashlib
import json

from .model import ReaderSpec


def reader_spec_digest(spec: ReaderSpec) -> str:
    """Digest the normalized, complete reader/v8 experiment configuration."""
    payload = spec.model_dump(mode="json", by_alias=True, exclude_none=False)
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()
