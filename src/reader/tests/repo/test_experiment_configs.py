from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from reader.tests.repo.experiment_matrix import (
    EXPECTED_FILE_PREFLIGHT_BLOCKERS,
    EXPERIMENT_CONFIGS,
    OPTIONAL_DEPENDENCY_BLOCKERS,
    repo_rel,
)
from reader.tests.support import REPO_ROOT, load_decl
from reader.workbench import resolve_workbench
from reader.workbench.engine import validate as validate_job
from reader.workbench.engine.validation import validation_summary

pytestmark = pytest.mark.integration

RETRON_SPONGE_CONFIGS = (
    "experiments/2026/20260313_mono_functional_sponges/config.yaml",
    "experiments/2026/20260314_bi_functional_lexA_cpxR_baeR_family_sponges/config.yaml",
    "experiments/2026/20260315_bi_functional_sox_family_sponges/config.yaml",
    "experiments/2026/202603XX_tri_functional_sponges/config.yaml",
    "experiments/2026/202603XX_tetra_functional_sponges/config.yaml",
)


@pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_repo_experiment_configs_load_and_validate(config_path: Path) -> None:
    decl = load_decl(config_path)
    workbench = resolve_workbench(decl)

    assert workbench.pipeline is not None
    validate_job(decl, console=Console(), check_files=False)


@pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_repo_experiment_configs_file_preflight_matches_known_repo_state(config_path: Path) -> None:
    decl = load_decl(config_path)
    summary = validation_summary(decl, check_files=True, exp_root=decl.experiment.root)
    rel = repo_rel(config_path)

    if rel in EXPECTED_FILE_PREFLIGHT_BLOCKERS:
        assert summary["status"] == "error", rel
        assert any(EXPECTED_FILE_PREFLIGHT_BLOCKERS[rel] in item for item in summary["errors"]), rel
        return

    if rel in OPTIONAL_DEPENDENCY_BLOCKERS:
        assert not any("auto_roots" in item for item in summary["errors"])
        if summary["status"] == "error":
            assert any(OPTIONAL_DEPENDENCY_BLOCKERS[rel] in item for item in summary["errors"])
        return

    assert summary["status"] == "ok", f"{rel}: {summary['errors']}"


@pytest.mark.parametrize("relative_path", RETRON_SPONGE_CONFIGS)
def test_retron_sponge_repo_configs_use_dedicated_protocol(relative_path: str) -> None:
    config_path = REPO_ROOT / relative_path
    decl = load_decl(config_path)

    assert decl.experiment_semantics.protocol.id == "plate_reader/retron_sponge_screen"
