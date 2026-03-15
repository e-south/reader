"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/presets/registry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from reader.core.errors import ConfigError

from .plate_reader import PLATE_READER_PRESETS
from .plots import PLOT_PRESETS
from .sfxi import SFXI_PRESETS


def _build_preset_registry(*sources: tuple[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    owners: dict[str, str] = {}
    for owner, source in sources:
        for name, info in source.items():
            previous = owners.get(name)
            if previous is not None:
                raise ConfigError(f"Duplicate preset {name!r} declared in both {previous} and {owner}.")
            registry[name] = info
            owners[name] = owner
    return registry


PRESETS: dict[str, dict[str, Any]] = _build_preset_registry(
    ("plate_reader", PLATE_READER_PRESETS),
    ("sfxi", SFXI_PRESETS),
    ("plots", PLOT_PRESETS),
)


def infer_category(steps: list[dict[str, Any]]) -> str:
    cats = {str(step.get("uses", "")).split("/", 1)[0] for step in steps if isinstance(step, dict)}
    cats.discard("")
    if not cats:
        return "pipeline"
    if cats == {"plot"}:
        return "plot"
    if cats == {"export"}:
        return "export"
    return "pipeline"


def list_presets(category: str | None = None) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for name, info in PRESETS.items():
        steps = info.get("steps", [])
        cat = infer_category(steps)
        if category and cat != category:
            continue
        items.append((name, info["description"]))
    return sorted(items)


def resolve_preset(name: str, *, with_args: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if name not in PRESETS:
        opts = ", ".join(sorted(PRESETS))
        raise ConfigError(f"Unknown preset {name!r}. Available presets: {opts}")
    info = PRESETS[name]
    build: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = info.get("build")
    args = dict(with_args or {})
    if build is not None:
        steps = build(args)
        if not isinstance(steps, list):
            raise ConfigError(f"Preset {name!r} builder must return a list of steps.")
        return copy.deepcopy(steps)
    return copy.deepcopy(info["steps"])


def describe_preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        opts = ", ".join(sorted(PRESETS))
        raise ConfigError(f"Unknown preset {name!r}. Available presets: {opts}")
    info = PRESETS[name]
    steps = copy.deepcopy(info["steps"])
    return {
        "name": name,
        "description": info.get("description", ""),
        "category": infer_category(steps),
        "steps": steps,
    }
