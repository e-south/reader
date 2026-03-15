from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reader.tests.support import REPO_ROOT

pytestmark = pytest.mark.integration

EXPERIMENT_CONFIGS = sorted(REPO_ROOT.glob("experiments/**/config.yaml"))

DEFAULT_PATHS = {
    "outputs": "./outputs",
    "plots": "plots",
    "exports": "exports",
    "notebooks": "notebooks",
}


@pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_repo_experiment_configs_omit_redundant_defaults(config_path: Path) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    assert data.get("paths") != DEFAULT_PATHS, f"{config_path}: omit default paths block"
    assert data.get("plotting") != {"palette": "colorblind"}, f"{config_path}: omit default plotting.palette"

    experiment = data.get("experiment") or {}
    if isinstance(experiment, dict):
        if "id" in experiment:
            assert experiment["id"] != config_path.parent.name, (
                f"{config_path}: omit experiment.id when it matches the experiment directory name"
            )
        if "title" in experiment and "id" in experiment:
            assert experiment["title"] != experiment["id"], f"{config_path}: omit experiment.title when it matches id"
