"""Shared retron review normalization and sponge ordering semantics."""

from __future__ import annotations

from typing import Any

import pandas as pd

TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "relevant", "on"})
FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "irrelevant", "off"})
FAMILY_ORDER = {"mono": 0, "bi": 1, "tri": 2, "quad": 3, "control": 4}


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def coerce_optional_bool_series(series: pd.Series, *, label: str) -> pd.Series:
    return series.map(lambda value: coerce_optional_bool(value, label=label))


def coerce_optional_bool(value: Any, *, label: str) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().casefold()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"retron_review: {label} contains unsupported boolean value {value!r}")


def split_motifs(sponge: str) -> list[str]:
    return [part for part in str(sponge).split("-") if part]


def motif_count(sponge: str) -> int:
    return len(split_motifs(sponge))


def family_label(sponge: str) -> str:
    count = motif_count(sponge)
    if str(sponge) == "tetO":
        return "control"
    if count <= 1:
        return "mono"
    if count == 2:
        return "bi"
    if count == 3:
        return "tri"
    if count == 4:
        return "quad"
    return f"{count}-site"


def sponge_sort_key(value: str) -> tuple[int, int, str]:
    count = motif_count(value)
    family = family_label(value)
    return (FAMILY_ORDER.get(family, 99), count, str(value))


def slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value)).strip("_").lower()
