"""Plate-reader ordering semantics for treatments, doses, and labels."""

from __future__ import annotations

import re

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_first_number(s: str) -> float | None:
    match = _NUM_RE.search(str(s))
    return float(match.group(0)) if match else None


def _unit_scale_to_uM(s: str) -> float:
    text = str(s).lower()
    if "nm" in text:
        return 1e-3
    if "µm" in text or "um" in text or "μm" in text:
        return 1.0
    if "mm" in text:
        return 1e3
    if re.search(r"\b(?<![a-z])m\b", text):
        return 1e6
    return 1.0


def smart_string_numeric_key(s: str) -> tuple[int, float, str]:
    value = str(s)
    num = _extract_first_number(value)
    if num is None:
        return (1, float("inf"), value.lower())
    return (0, float(num) * _unit_scale_to_uM(value), value.lower())


def _prefix_before_number(s: str) -> str:
    value = str(s)
    match = _NUM_RE.search(value)
    prefix = value[: match.start()].strip() if match else value.strip()
    return re.sub(r"\s+", " ", prefix).strip().lower()


def smart_grouped_dose_key(s: str) -> tuple[str, int, float, str]:
    value = str(s)
    prefix = _prefix_before_number(value)
    num = _extract_first_number(value)
    has_num = 0 if num is not None else 1
    scaled = float(num) * _unit_scale_to_uM(value) if num is not None else float("inf")
    return (prefix, has_num, scaled, value.lower())


def order_levels(levels: list[str]) -> list[str]:
    try:
        return sorted(levels, key=smart_grouped_dose_key)
    except Exception:
        return sorted(levels, key=smart_string_numeric_key)


__all__ = ["order_levels", "smart_grouped_dose_key", "smart_string_numeric_key"]
