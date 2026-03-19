from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from reader.tests.support import REPO_ROOT


@dataclass(frozen=True, slots=True)
class RepoConfigEntry:
    path: Path
    rel: str
    data: Any
    lifecycle: str


def _experiment_lifecycle(data: Any) -> str:
    experiment = data.get("experiment") or {} if isinstance(data, dict) else {}
    if not isinstance(experiment, dict):
        return "active"
    lifecycle = experiment.get("lifecycle", "active")
    return str(lifecycle).strip().lower() or "active"


@lru_cache(maxsize=1)
def repo_config_inventory() -> tuple[RepoConfigEntry, ...]:
    entries: list[RepoConfigEntry] = []
    for path in sorted(REPO_ROOT.glob("experiments/**/config.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries.append(
            RepoConfigEntry(
                path=path,
                rel=str(path.relative_to(REPO_ROOT)),
                data=data,
                lifecycle=_experiment_lifecycle(data),
            )
        )
    return tuple(entries)
