"""Plate-reader plot grouping helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from reader_workbench.domains.plate_reader.ordering import order_levels

GroupMatch = Literal["exact", "contains", "startswith", "endswith", "regex"]


def _match_value(val: str, needle: str, mode: GroupMatch) -> bool:
    v = str(val)
    n = str(needle)
    if mode == "exact":
        return v == n
    if mode == "contains":
        return n in v
    if mode == "startswith":
        return v.startswith(n)
    if mode == "endswith":
        return v.endswith(n)
    if mode == "regex":
        return re.search(n, v) is not None
    raise ValueError(f"Unknown group_match: {mode}")


def resolve_groups(
    universe: Iterable[str],
    groups: list[dict[str, list[str]]] | None,
    *,
    match: GroupMatch,
) -> list[tuple[str, list[str]]]:
    values = list(map(str, universe))
    if not groups:
        return [("all", values)]
    resolved: list[tuple[str, list[str]]] = []
    for item in groups:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("groups must be list of single-key dicts")
        label, needles = next(iter(item.items()))
        members = [v for v in values if any(_match_value(v, n, match) for n in needles)]
        resolved.append((str(label), members))
    return resolved


def ordered_groups(universe: Iterable[str]) -> list[str]:
    return order_levels(list(map(str, universe)))


__all__ = ["GroupMatch", "ordered_groups", "resolve_groups"]
