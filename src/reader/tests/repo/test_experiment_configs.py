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
from reader.tests.support import REPO_ROOT, cli_success_data, load_decl
from reader.workbench import resolve_workbench
from reader.workbench.cli import app
from reader.workbench.config import ReaderSpec
from reader.workbench.engine import validate as validate_job
from reader.workbench.engine.validation import validation_summary
from reader.workbench.experiments import discover_experiment_configs

pytestmark = [pytest.mark.integration, pytest.mark.repo_matrix]


@pytest.fixture(scope="module")
def runtime():
    return builtin_runtime()


RETRON_SPONGE_CONFIGS = tuple(
    relative_path
    for relative_path in (
        "experiments/2026/20260313_mono_functional_sponges/config.yaml",
        "experiments/2026/20260314_bi_functional_lexA_cpxR_baeR_family_sponges/config.yaml",
        "experiments/2026/20260315_bi_functional_sox_family_sponges/config.yaml",
        "experiments/2026/20260316_tri_functional_sponges/config.yaml",
        "experiments/2026/20260317_tetra_functional_sponges/config.yaml",
    )
    if (REPO_ROOT / relative_path).exists()
)
RETRON_SPONGE_FULL_PLOT_IDS = [
    "raw_kinetics",
    "support_kinetics",
    "control_burden_panel",
    "baseline_shifted_kinetics",
    "control_anchored_decomposition",
    "absolute_effect_kinetics",
    "induced_effect_kinetics",
    "library_heatmaps",
    "pareto_ranking",
]
CROSSTALK_CONFIG = "experiments/2025/20250620_sensor_panel_crosstalk/config.yaml"
SFXI_2026_VEC8_CONFIGS = (
    "experiments/2026/20260117_sfxi_ref-pDual10/config.yaml",
    "experiments/2026/20260119_sfxi_ref-pDual10/config.yaml",
    "experiments/2026/20260121_sfxi_ref-pDual10/config.yaml",
    "experiments/2026/20260619_sfxi_sensor-panel-m9-glu-1-10/config.yaml",
    "experiments/2026/20260620_sfxi_sensor-panel-m9-glu-12-19/config.yaml",
    "experiments/2026/20260621_sfxi_sensors-opal-20-28/config.yaml",
    "experiments/2026/20260622_sfxi_sensor-panel-m9-glu-29-30-sulAp-spyp/config.yaml",
    "experiments/2026/20260706_sfxi_sensor-panel-m9-glu-secg/config.yaml",
    "experiments/2026/20260707_sfxi_sensor-panel-m9-glu-secg/config.yaml",
)


def _repo_inputs_available(config_path: Path) -> bool:
    return (config_path.parent / "inputs").exists()


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

    if not _repo_inputs_available(config_path):
        pytest.skip(f"{rel}: repo checkout does not include experiment inputs")

    assert summary["status"] == "ok", f"{rel}: {summary['errors']}"


def test_repo_non_active_configs_declare_explicit_lifecycle() -> None:
    assert NON_ACTIVE_LIFECYCLE_CONFIGS.get("experiments/template/config.yaml") == "template"
    assert all(lifecycle != "active" for lifecycle in NON_ACTIVE_LIFECYCLE_CONFIGS.values())


@pytest.mark.parametrize("relative_path", SFXI_2026_VEC8_CONFIGS)
def test_2026_sfxi_vec8_configs_pin_j23105_anchor_and_12h_snapshot(relative_path: str) -> None:
    config_path = REPO_ROOT / relative_path
    if not config_path.is_file():
        pytest.skip(f"{relative_path} is a local-workbench config not present in this checkout")
    spec = ReaderSpec.load(config_path)
    inputs = spec.protocol.inputs

    assert spec.protocol.id == "logic/sfxi_screen"
    assert inputs["reference"]["design_id"] == "J23105"
    assert inputs["target_time_h"] == 12.0
    assert inputs["time_mode"] == "nearest"
    assert inputs["time_tolerance_h"] == 0.51


def test_repo_cli_inventory_details_matches_experiment_discovery() -> None:
    experiments_root = REPO_ROOT / "experiments"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ls",
            "--root",
            str(experiments_root),
            "--details",
            "--limit",
            "100",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    envelope = json.loads(result.output)
    assert envelope["meta"]["truncated"] is False
    assert envelope["meta"]["continuation"] is None
    payload = cli_success_data(result.output)
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
    if not config_path.exists():
        pytest.skip(f"{CROSSTALK_CONFIG} is not present in this checkout")
    runner = CliRunner()

    result = runner.invoke(app, ["plot", str(config_path), "--list", "--format", "json"])

    assert result.exit_code == 0
    payload = cli_success_data(result.output)
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
    plot_payload = cli_success_data(plot_result.output)
    inspect_payload = cli_success_data(inspect_result.output)
    plot_ids = [item["id"] for item in plot_payload["plots"]]
    assert plot_payload["summary"]["plots"] == len(RETRON_SPONGE_FULL_PLOT_IDS)
    assert set(plot_ids) == set(RETRON_SPONGE_FULL_PLOT_IDS)
    assert "baseline_shifted_kinetics" in plot_ids
    assert plot_payload["summary"]["by_plugin"] == {
        "plot/retron_summary": 3,
        "plot/retron_trace": 4,
        "plot/time_series": 2,
    }
    assert inspect_payload["experiment"]["notebook_template"] == "notebook/retron_sponge"
