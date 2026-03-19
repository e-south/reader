from __future__ import annotations

from pathlib import Path

import yaml

from reader.tests.support import REPO_ROOT

EXPERIMENT_CONFIGS = sorted(REPO_ROOT.glob("experiments/**/config.yaml"))


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _experiment_lifecycle(path: Path) -> str:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    experiment = data.get("experiment") or {}
    if not isinstance(experiment, dict):
        return "active"
    lifecycle = experiment.get("lifecycle", "active")
    return str(lifecycle).strip().lower() or "active"


EXPERIMENT_LIFECYCLES = {repo_rel(path): _experiment_lifecycle(path) for path in EXPERIMENT_CONFIGS}

NON_ACTIVE_LIFECYCLE_CONFIGS = {
    rel: lifecycle for rel, lifecycle in EXPERIMENT_LIFECYCLES.items() if lifecycle != "active"
}

OPTIONAL_DEPENDENCY_BLOCKERS = {}

END_TO_END_RUNNABLE_CONFIGS = [
    config_path
    for config_path in EXPERIMENT_CONFIGS
    if repo_rel(config_path) not in NON_ACTIVE_LIFECYCLE_CONFIGS
    and repo_rel(config_path) not in OPTIONAL_DEPENDENCY_BLOCKERS
]
