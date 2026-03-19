from __future__ import annotations

from pathlib import Path

from reader.tests.repo.config_inventory import repo_config_inventory
from reader.tests.support import REPO_ROOT

REPO_CONFIGS = repo_config_inventory()
EXPERIMENT_CONFIGS = tuple(entry.path for entry in REPO_CONFIGS)


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


EXPERIMENT_LIFECYCLES = {entry.rel: entry.lifecycle for entry in REPO_CONFIGS}

NON_ACTIVE_LIFECYCLE_CONFIGS = {
    rel: lifecycle for rel, lifecycle in EXPERIMENT_LIFECYCLES.items() if lifecycle != "active"
}

OPTIONAL_DEPENDENCY_BLOCKERS = {}

END_TO_END_RUNNABLE_CONFIGS = tuple(
    config_path
    for config_path in EXPERIMENT_CONFIGS
    if repo_rel(config_path) not in NON_ACTIVE_LIFECYCLE_CONFIGS
    and repo_rel(config_path) not in OPTIONAL_DEPENDENCY_BLOCKERS
)
