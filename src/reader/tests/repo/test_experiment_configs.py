from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from reader.core.config import ReaderSpec
from reader.core.engine import validate as validate_job
from reader.core.workbench import resolve_workbench
from reader.tests.support import REPO_ROOT

pytestmark = pytest.mark.integration

EXPERIMENT_CONFIGS = sorted(REPO_ROOT.glob("experiments/**/config.yaml"))


@pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_repo_experiment_configs_load_and_validate(config_path: Path) -> None:
    spec = ReaderSpec.load(config_path)
    workbench = resolve_workbench(spec)

    assert workbench.pipeline is not None
    validate_job(spec, console=Console(), check_files=False)
