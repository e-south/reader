from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from reader.runtime import builtin_runtime
from reader.tests.repo.experiment_matrix import (
    EXPERIMENT_CONFIGS,
    NON_ACTIVE_LIFECYCLE_CONFIGS,
    OPTIONAL_DEPENDENCY_BLOCKERS,
    repo_rel,
)
from reader.tests.support import REPO_ROOT, load_decl
from reader.workbench import resolve_workbench
from reader.workbench.cli import app
from reader.workbench.engine import validate as validate_job
from reader.workbench.engine.validation import validation_summary
from reader.workbench.experiments import discover_experiment_configs

pytestmark = [pytest.mark.integration, pytest.mark.repo_matrix]


@pytest.fixture(scope="module")
def runtime():
    return builtin_runtime()


RETRON_SPONGE_CONFIGS = (
    "experiments/2026/20260313_mono_functional_sponges/config.yaml",
    "experiments/2026/20260314_bi_functional_lexA_cpxR_baeR_family_sponges/config.yaml",
    "experiments/2026/20260315_bi_functional_sox_family_sponges/config.yaml",
    "experiments/2026/20260316_tri_functional_sponges/config.yaml",
    "experiments/2026/20260317_tetra_functional_sponges/config.yaml",
)
RETRON_SPONGE_FULL_PLOT_IDS = [
    "raw_kinetics",
    "support_kinetics",
    "control_burden_panel",
    "baseline_shifted_kinetics",
    "matched_control_kinetics",
    "induced_effect_kinetics",
    "interaction_summary",
    "library_heatmaps",
    "stress_modulation_scores",
    "pareto_ranking",
]
CROSSTALK_CONFIG = "experiments/2025/20250620_sensor_panel_crosstalk/config.yaml"


@pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_repo_experiment_configs_load_and_validate(config_path: Path, runtime) -> None:
    decl = load_decl(config_path)
    workbench = resolve_workbench(decl)

    assert workbench.pipeline is not None
    validate_job(decl, console=Console(), check_files=False, runtime=runtime)


@pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_repo_experiment_configs_file_preflight_matches_known_repo_state(config_path: Path, runtime) -> None:
    decl = load_decl(config_path)
    summary = validation_summary(decl, check_files=True, exp_root=decl.experiment.root, runtime=runtime)
    rel = repo_rel(config_path)

    if rel in NON_ACTIVE_LIFECYCLE_CONFIGS:
        return

    if rel in OPTIONAL_DEPENDENCY_BLOCKERS:
        assert not any("auto_roots" in item for item in summary["errors"])
        if summary["status"] == "error":
            assert any(OPTIONAL_DEPENDENCY_BLOCKERS[rel] in item for item in summary["errors"])
        return

    assert summary["status"] == "ok", f"{rel}: {summary['errors']}"


def test_repo_non_active_configs_declare_explicit_lifecycle() -> None:
    assert NON_ACTIVE_LIFECYCLE_CONFIGS.get("experiments/template/config.yaml") == "template"
    assert all(lifecycle != "active" for lifecycle in NON_ACTIVE_LIFECYCLE_CONFIGS.values())


def test_repo_cli_inventory_details_matches_experiment_discovery() -> None:
    experiments_root = REPO_ROOT / "experiments"
    runner = CliRunner()

    result = runner.invoke(app, ["ls", "--root", str(experiments_root), "--details", "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    expected_configs = discover_experiment_configs(experiments_root)
    expected_decls = {str(path.resolve()): load_decl(path) for path in expected_configs}
    expected_protocols = Counter(decl.experiment_semantics.protocol.id for decl in expected_decls.values())
    expected_lifecycles = Counter(decl.experiment.lifecycle for decl in expected_decls.values())
    by_config = {item["config"]: item for item in payload["experiments"]}

    assert payload["catalog"] == {"kind": "experiments", "root": str(experiments_root.resolve())}
    assert payload["selection"] == {
        "details": True,
        "include_scaffolds": False,
        "lifecycle": None,
        "protocol": None,
        "readiness": False,
        "status": None,
    }
    assert payload["summary"]["experiments"] == len(expected_configs)
    assert payload["summary"]["by_status"] == {"ok": len(expected_configs)}
    assert payload["summary"]["by_protocol"] == dict(sorted(expected_protocols.items()))
    assert payload["summary"]["by_lifecycle"] == dict(sorted(expected_lifecycles.items()))
    assert set(by_config) == set(expected_decls)

    for config_path, decl in expected_decls.items():
        entry = by_config[config_path]
        assert entry["status"] == "ok"
        assert entry["protocol"] == decl.experiment_semantics.protocol.id
        assert entry["lifecycle"] == decl.experiment.lifecycle
        assert entry["selected"]["pipeline"]["ids"] == [step.id for step in decl.pipeline.steps]
        assert entry["selected"]["plots"]["ids"] == [spec.id for spec in decl.plots.specs]
        assert entry["selected"]["exports"]["ids"] == [spec.id for spec in decl.exports.specs]
        if decl.notebooks.specs:
            assert entry["selected"]["notebook_template"] == decl.notebooks.specs[0].template
        else:
            assert entry["selected"]["notebook_template"] is None


def test_crosstalk_repo_config_surfaces_both_heatmaps_via_cli() -> None:
    config_path = REPO_ROOT / CROSSTALK_CONFIG
    runner = CliRunner()

    result = runner.invoke(app, ["plot", str(config_path), "--list", "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [item["id"] for item in payload["plots"]] == [
        "ratio_heatmap",
        "support_heatmap",
        "raw_kinetics",
        "value_distributions",
    ]
    assert payload["summary"]["by_plugin"] == {
        "plot/distributions": 1,
        "plot/snapshot_heatmap": 2,
        "plot/time_series": 1,
    }


@pytest.mark.parametrize("relative_path", RETRON_SPONGE_CONFIGS)
def test_retron_sponge_repo_configs_surface_full_plot_portfolio_and_review_notebook(relative_path: str) -> None:
    config_path = REPO_ROOT / relative_path
    decl = load_decl(config_path)
    runner = CliRunner()

    assert decl.experiment_semantics.protocol.id == "plate_reader/retron_sponge_screen"
    plot_result = runner.invoke(app, ["plot", str(config_path), "--list", "--format", "json"])
    inspect_result = runner.invoke(app, ["inspect", str(config_path), "--format", "json"])

    assert plot_result.exit_code == 0
    assert inspect_result.exit_code == 0
    plot_payload = json.loads(plot_result.output)
    inspect_payload = json.loads(inspect_result.output)
    plot_ids = [item["id"] for item in plot_payload["plots"]]
    assert plot_payload["summary"]["plots"] == len(RETRON_SPONGE_FULL_PLOT_IDS)
    assert set(plot_ids) == set(RETRON_SPONGE_FULL_PLOT_IDS)
    assert "baseline_shifted_kinetics" in plot_ids
    assert plot_payload["summary"]["by_plugin"] == {
        "plot/retron_summary": 4,
        "plot/retron_trace": 4,
        "plot/time_series": 2,
    }
    assert inspect_payload["experiment"]["notebook_template"] == "notebook/retron_sponge"
