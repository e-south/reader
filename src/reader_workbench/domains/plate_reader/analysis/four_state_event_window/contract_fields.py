"""Validate scalar and mapping fields used by four-state event-window contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def exact_fields(
    value: object,
    *,
    context: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    payload = mapping(value, context=context)
    allowed = required | (optional or set())
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {unknown}.")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}.")
    return payload


def nonempty(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be a finite number.")
    return result


__all__ = ["exact_fields", "finite", "mapping", "nonempty"]
