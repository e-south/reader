from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from reader_workbench.tests.support import REPO_ROOT


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


def _tracked_config_paths() -> tuple[Path, ...]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "experiments/**/config.yaml"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return tuple(sorted(REPO_ROOT.glob("experiments/**/config.yaml")))
    paths: list[Path] = []
    for raw in result.stdout.splitlines():
        rel = raw.strip()
        if not rel:
            continue
        path = (REPO_ROOT / rel).resolve()
        if path.exists():
            paths.append(path)
    return tuple(sorted(paths))


@lru_cache(maxsize=1)
def repo_config_inventory() -> tuple[RepoConfigEntry, ...]:
    entries: list[RepoConfigEntry] = []
    for path in _tracked_config_paths():
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
