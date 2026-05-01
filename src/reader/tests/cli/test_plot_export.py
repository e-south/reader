"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/cli/test_plot_export.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reader.tests.support import base_reader_config, default_notebook_name, load_decl, write_config
from reader.workbench import FileRef, RecordRef, resolve_workbench
from reader.workbench.cli import app
from reader.workbench.experiment import ResourceCatalog
from reader.workbench.spec_overrides import apply_step_overrides, parse_input_overrides


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def _base_config() -> dict:
    return base_reader_config(
        experiment_id="exp_cli",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [14.0]}},
        protocol_analysis={"crosstalk_pairs": {"enabled": True, "export": True}},
        protocol_outputs={
            "plots": {"profile": "none", "include": ["raw_kinetics", "endpoint_by_condition"]},
            "exports": {"include": ["crosstalk_pairs_table"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )


def _logic_plot_config() -> dict:
    return base_reader_config(
        experiment_id="exp_logic",
        protocol_id="logic/sfxi_screen",
        protocol_inputs={"logic_map_ref": "induction_logic"},
        protocol_analysis={"include_vec8": False, "include_fold_change": False},
        protocol_outputs={"plots": {"profile": "none", "include": ["logic_symmetry"]}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "logic_maps": {
                "induction_logic": {
                    "column": "treatment",
                    "corners": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )


def _logic_sfxi_scatter_config() -> dict:
    cfg = _logic_plot_config()
    cfg["protocol"]["analysis"] = {
        "include_vec8": True,
        "include_fold_change": False,
        "sfxi_objective": {
            "setpoints": {"and": [0.0, 0.0, 0.0, 1.0]},
            "scaling": {"percentile": 95, "min_n": 1, "eps": 1e-8},
            "exponents": {"logic_exponent_beta": 1.0, "intensity_exponent_gamma": 1.0},
        },
    }
    cfg["protocol"]["outputs"]["plots"] = {"profile": "none", "include": ["sfxi_setpoint_scatter"]}
    return cfg


def _retron_config() -> dict:
    return base_reader_config(
        experiment_id="exp_retron",
        protocol_id="plate_reader/retron_sponge_screen",
        protocol_analysis={
            "semantic_metrics": {
                "relevant_stress_map": {
                    "spyP": "3% EtOH",
                    "sulAp": "100 nM ciprofloxacin",
                    "soxSp": "15 uM PMS",
                },
                "sensor_target_map": {
                    "spyP": ["CpxR", "BaeR"],
                    "sulAp": ["LexA"],
                    "soxSp": ["SoxR", "SoxS"],
                },
            }
        },
        protocol_outputs={
            "plots": {
                "profile": "none",
                "include": ["matched_control_kinetics", "library_heatmaps"],
            },
            "exports": {"include": ["semantic_summary_table"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )


def _dual_reporter_profile_config(profile: str) -> dict:
    return base_reader_config(
        experiment_id=f"exp_{profile}",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [14.0]}},
        protocol_outputs={"plots": {"profile": profile}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )


def test_plot_list_filters(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "raw_kinetics" in result.output
    assert "2 total" in result.output
    assert "df <- ratio_yfp_od600/df" in result.output

    result = runner.invoke(app, ["plot", str(cfg), "--list", "--only", "raw_kinetics"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "raw_kinetics" in result.output
    assert "endpoint_by_condition" not in result.output

    result = runner.invoke(app, ["plot", str(cfg), "--list", "--exclude", "raw_kinetics"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "raw_kinetics" not in result.output
    assert "endpoint_by_condition" in result.output


def test_plot_list_empty(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["protocol"]["outputs"]["plots"] = {"profile": "none"}
    cfg_path = write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg_path), "--list"])
    assert result.exit_code == 0
    assert "No plots configured" in result.output


def test_plot_list_json(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--list", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["catalog"] == {"kind": "plot", "protocol": "plate_reader/dual_reporter_screen"}
    assert payload["selection"] == {"only": [], "exclude": []}
    assert payload["summary"]["plots"] == 2
    assert payload["summary"]["by_domain"] == {"plate_reader": 2}
    assert payload["plots"][0]["id"] == "raw_kinetics"
    assert payload["plots"][0]["semantics"]["category"] == "plot"
    assert "count" not in payload
    assert "filters" not in payload


def test_plot_list_json_empty(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["protocol"]["outputs"]["plots"] = {"profile": "none"}
    cfg_path = write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg_path), "--list", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["selection"] == {"only": [], "exclude": []}
    assert payload["summary"]["plots"] == 0
    assert payload["summary"]["by_plugin"] == {}
    assert payload["plots"] == []


def test_plot_list_json_surfaces_source_contract_metadata(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _logic_plot_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--list", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["plots"] == 1
    assert payload["summary"]["by_family"] == {"geometry_plot": 1}
    read = payload["plots"][0]["reads"][0]
    assert read["contract"] == "plate_reader.annotated.v1"
    assert read["source"]["producer"]["id"] == "promote_to_tidy_plus_map"
    assert read["source"]["surface"]["runtime_mode"] == "fixed"
    assert read["source"]["surface"]["rendered"] == "plate_reader.annotated.v1"


def test_logic_sfxi_plot_list_surfaces_setpoint_scatter(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _logic_sfxi_scatter_config())
    runner = CliRunner()

    result = runner.invoke(app, ["plot", str(cfg), "--list", "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["plots"] == 1
    assert payload["summary"]["by_plugin"] == {"plot/sfxi_setpoint_scatter": 1}
    assert payload["plots"][0]["id"] == "sfxi_setpoint_scatter"
    read = payload["plots"][0]["reads"][0]
    assert read["ref"] == {"record": "sfxi_vec8/vec8"}
    assert read["contract"] == "sfxi.vec8.v2"


def test_logic_sfxi_plot_dry_run_reports_missing_dnadesign_public_api(tmp_path: Path, monkeypatch) -> None:
    real_import_module = importlib.import_module

    def _fake_import_module(name: str, package: str | None = None):
        if name == "dnadesign.opal.api.sfxi":
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)
    cfg = write_config(tmp_path, _logic_sfxi_scatter_config())
    runner = CliRunner()

    result = runner.invoke(app, ["plot", str(cfg), "--dry-run"])

    assert result.exit_code != 0
    assert "reader[dnadesign]" in _plain(result.output)


def test_logic_sfxi_validate_reports_missing_dnadesign_public_api(tmp_path: Path, monkeypatch) -> None:
    real_import_module = importlib.import_module

    def _fake_import_module(name: str, package: str | None = None):
        if name == "dnadesign.opal.api.sfxi":
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)
    cfg = write_config(tmp_path, _logic_sfxi_scatter_config())
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "metadata.xlsx").write_text("stub", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["validate", str(cfg), "--format", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["summary"]["status"] == "error"
    assert any("reader[dnadesign]" in message for message in payload["validation"]["errors"])


def test_retron_plot_list_json(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _retron_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--list", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    plot_ids = [item["id"] for item in payload["plots"]]
    assert payload["experiment"]["protocol"] == "plate_reader/retron_sponge_screen"
    assert plot_ids == ["matched_control_kinetics", "library_heatmaps"]
    assert payload["summary"]["by_family"] == {"matched_control_kinetics": 1, "matched_control_summary": 1}
    reads = {item["id"]: item["reads"] for item in payload["plots"]}
    assert reads["matched_control_kinetics"][0]["contract"] == "plate_reader.sponge_trace.v1"
    assert reads["library_heatmaps"][0]["contract"] == "plate_reader.sponge_summary.v1"


@pytest.mark.parametrize(
    ("profile", "expected_ids", "expected_plugins"),
    [
        (
            "ratio_screen",
            ["raw_kinetics", "state_summary", "ratio_overview"],
            {
                "plot/snapshot_barplot": 1,
                "plot/time_series": 1,
                "plot/ts_and_snap": 1,
            },
        ),
        (
            "heatmap_review",
            ["ratio_heatmap", "support_heatmap"],
            {"plot/snapshot_heatmap": 2},
        ),
    ],
)
def test_dual_reporter_nondefault_plot_profiles_surface_live_cli_specs(
    tmp_path: Path,
    profile: str,
    expected_ids: list[str],
    expected_plugins: dict[str, int],
) -> None:
    cfg = write_config(tmp_path, _dual_reporter_profile_config(profile))
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--list", "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["summary"]["plots"] == len(expected_ids)
    assert payload["summary"]["by_plugin"] == expected_plugins
    assert [item["id"] for item in payload["plots"]] == expected_ids


def test_plot_json_requires_list(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--format", "json"])
    assert result.exit_code != 0
    assert "only supported with --list" in _plain(result.output)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--list", "--dry-run"], "--dry-run cannot be combined with --list"),
        (["--list", "--input", "df={record: ratio_yfp_od600/df}"], "--input cannot be combined with --list"),
        (["--list", "--set", "with.time=6.0"], "--set cannot be combined with --list"),
    ],
)
def test_plot_list_rejects_ignored_execution_flags(tmp_path: Path, args: list[str], expected: str) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), *args])
    assert result.exit_code != 0
    assert expected in _plain(result.output)


def test_plot_rejects_empty_selection_after_filters(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["plot", str(cfg), "--exclude", "raw_kinetics", "--exclude", "endpoint_by_condition", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "No plots selected" in _plain(result.output)


def test_plot_requires_records(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg)])
    assert result.exit_code != 0
    assert "outputs/manifests/records.json" in result.output
    assert "uv run reader run" in result.output


def test_plot_dry_run_does_not_require_records(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--dry-run"])

    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert "raw_kinetics" in result.output


def test_plot_dry_run_allows_non_active_lifecycle(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, {**_base_config(), "experiment": {"id": "exp_cli", "lifecycle": "draft"}})
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--dry-run"])
    assert result.exit_code == 0
    assert "raw_kinetics" in result.output


def test_export_requires_records(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg)])
    assert result.exit_code != 0
    assert "outputs/manifests/records.json" in result.output
    assert "uv run reader run" in result.output


def test_export_dry_run_does_not_require_records(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg), "--dry-run"])

    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert "crosstalk_pairs_table" in result.output


def test_export_dry_run_allows_non_active_lifecycle(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, {**_base_config(), "experiment": {"id": "exp_cli", "lifecycle": "draft"}})
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg), "--dry-run"])
    assert result.exit_code == 0
    assert "crosstalk_pairs_table" in result.output


@pytest.mark.parametrize("command", ["plot", "export"])
def test_plot_export_surfaces_corrupt_record_catalog_error(tmp_path: Path, command: str) -> None:
    cfg = write_config(tmp_path, _base_config())
    records_path = tmp_path / "outputs" / "manifests" / "records.json"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text("{not-json", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, [command, str(cfg)])

    assert result.exit_code != 0
    text = _plain(result.output)
    assert "Could not read record catalog" in text
    assert "records.json is not valid JSON" in text
    assert "Run 'uv run reader run" not in text


def test_plot_year_list(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    year_dir = tmp_path / "experiments" / "2025"
    exp_a = year_dir / "exp_a"
    exp_b = year_dir / "exp_b"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)
    write_config(exp_a, _base_config())
    write_config(exp_b, _base_config())
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["plot", "--year", "2025", "--list"])
    assert result.exit_code == 0
    assert "exp_a" in result.output
    assert "exp_b" in result.output


def test_plot_year_json_requires_single_experiment_listing(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    experiments_root = tmp_path / "experiments"
    year_dir = experiments_root / "2025" / "exp_a"
    year_dir.mkdir(parents=True)
    write_config(year_dir, _base_config())

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["plot", "--year", "2025", "--root", str(experiments_root), "--format", "json"])

    assert result.exit_code != 0
    assert "single-experiment plot" in result.output


def test_plot_year_dry_run_preflights_batch_before_execution(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    year_dir = tmp_path / "experiments" / "2025"
    exp_a = year_dir / "exp_a"
    exp_b = year_dir / "exp_b"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)
    write_config(exp_a, _base_config())
    cfg_b = _base_config()
    cfg_b["protocol"]["outputs"]["plots"] = {"profile": "none"}
    write_config(exp_b, cfg_b)

    calls: list[str] = []

    def _fake_run_plot_job(job_path: Path, **kwargs) -> None:
        calls.append(job_path.parent.name)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("reader.workbench.cli.surfaces._run_plot_job", _fake_run_plot_job)
    result = runner.invoke(app, ["plot", "--year", "2025", "--dry-run"])

    assert result.exit_code != 0
    assert "No plots configured in this experiment" in _plain(result.output)
    assert calls == []


def test_plot_year_run_preflights_batch_before_mutation(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    year_dir = tmp_path / "experiments" / "2025"
    exp_a = year_dir / "exp_a"
    exp_b = year_dir / "exp_b"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)
    write_config(exp_a, _base_config())
    cfg_b = _base_config()
    cfg_b["experiment"] = {"id": "exp_b", "lifecycle": "draft"}
    write_config(exp_b, cfg_b)

    calls: list[str] = []

    def _fake_run_plot_job(job_path: Path, **kwargs) -> None:
        calls.append(job_path.parent.name)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("reader.workbench.cli.surfaces.require_dataframe_records", lambda decl, job_path, runtime: None)
    monkeypatch.setattr("reader.workbench.cli.surfaces._run_plot_job", _fake_run_plot_job)
    result = runner.invoke(app, ["plot", "--year", "2025"])

    assert result.exit_code != 0
    assert "lifecycle 'draft'" in _plain(result.output)
    assert calls == []


def test_plot_year_run_preflights_override_errors_before_mutation(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    year_dir = tmp_path / "experiments" / "2025"
    exp_a = year_dir / "exp_a"
    exp_b = year_dir / "exp_b"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)
    write_config(exp_a, _base_config())
    write_config(exp_b, _base_config())

    calls: list[str] = []

    def _fake_run_plot_job(job_path: Path, **kwargs) -> None:
        calls.append(job_path.parent.name)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("reader.workbench.cli.surfaces.require_dataframe_records", lambda decl, job_path, runtime: None)
    monkeypatch.setattr("reader.workbench.cli.surfaces._run_plot_job", _fake_run_plot_job)
    result = runner.invoke(app, ["plot", "--year", "2025", "--set", "bad.path=1"])

    assert result.exit_code != 0
    assert "--set path must start with reads., with., or writes." in _plain(result.output)
    assert calls == []


def test_export_list_filters(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg), "--list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "crosstalk-safe" in result.output
    assert "from" in result.output
    assert "export/csv" in result.output

    result = runner.invoke(
        app,
        ["export", str(cfg), "--list", "--only", "crosstalk_pairs_table"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "crosstalk_pairs_table" in result.output


def test_export_list_empty(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["protocol"]["outputs"]["exports"] = {"exclude": ["crosstalk_pairs_table"]}
    cfg_path = write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg_path), "--list"])
    assert result.exit_code == 0
    assert "No exports configured" in result.output


def test_export_list_json(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg), "--list", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["experiment"]["protocol"] == "plate_reader/dual_reporter_screen"
    assert payload["catalog"] == {"kind": "export", "protocol": "plate_reader/dual_reporter_screen"}
    assert payload["selection"] == {"only": [], "exclude": []}
    assert payload["summary"]["exports"] == 1
    assert payload["summary"]["by_domain"] == {"generic": 1}
    assert payload["exports"][0]["id"] == "crosstalk_pairs_table"
    assert payload["exports"][0]["semantics"]["category"] == "export"
    assert "count" not in payload


def test_export_list_json_empty(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["protocol"]["outputs"]["exports"] = {"exclude": ["crosstalk_pairs_table"]}
    cfg_path = write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg_path), "--list", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["selection"] == {"only": [], "exclude": []}
    assert payload["summary"]["exports"] == 0
    assert payload["summary"]["by_family"] == {}
    assert payload["exports"] == []


def test_retron_export_list_json(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _retron_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg), "--list", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["experiment"]["protocol"] == "plate_reader/retron_sponge_screen"
    exports = {item["id"]: item for item in payload["exports"]}
    assert set(exports) == {"semantic_summary_table", "semantic_trace_table"}
    assert exports["semantic_summary_table"]["reads"][0]["source"]["contract"] == "plate_reader.sponge_summary.v1"
    assert exports["semantic_summary_table"]["reads"][0]["source"]["producer"]["id"] == "semantic_metrics"
    assert exports["semantic_trace_table"]["reads"][0]["source"]["contract"] == "plate_reader.sponge_trace.v1"


def test_export_json_requires_list(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg), "--format", "json"])
    assert result.exit_code != 0
    assert "only supported with --list" in _plain(result.output)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--list", "--dry-run"], "--dry-run cannot be combined with --list"),
        (["--list", "--input", "df={record: ratio_yfp_od600/df}"], "--input cannot be combined with --list"),
        (["--list", "--set", "with.path=exports/crosstalk_pairs.csv"], "--set cannot be combined with --list"),
    ],
)
def test_export_list_rejects_ignored_execution_flags(tmp_path: Path, args: list[str], expected: str) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg), *args])
    assert result.exit_code != 0
    assert expected in _plain(result.output)


def test_export_rejects_empty_selection_after_filters(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg), "--exclude", "crosstalk_pairs_table", "--dry-run"])
    assert result.exit_code != 0
    assert "No exports selected" in _plain(result.output)


def test_validate_checks_files_by_default(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    file_path = inputs_dir / "metadata.xlsx"
    file_path.write_text("stub", encoding="utf-8")
    raw_path = inputs_dir / "run1.xlsx"
    raw_path.write_text("stub", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(cfg_path)])
    assert result.exit_code == 0

    file_path.unlink()
    result = runner.invoke(app, ["validate", str(cfg_path)])
    assert result.exit_code == 1
    assert "inputs/metadata.xlsx" in result.output


def test_validate_no_files_skips_checks(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(cfg_path), "--no-files"])
    assert result.exit_code == 0


def test_plot_notebook_scaffold(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["notebook", str(cfg), "--template", "notebook/eda", "--only", "raw_kinetics", "--mode", "none"],
    )
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    assert nb_path.exists()
    content = nb_path.read_text(encoding="utf-8")
    assert "PLOT_SPECS" not in content
    assert 'label="Dataset (dataframe record)"' in content


def test_plot_override_parses_runtime_inputs_to_typed_refs(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, _base_config())
    decl = load_decl(cfg_path)
    plot_spec = next(spec for spec in resolve_workbench(decl).plots if spec.id == "raw_kinetics")

    overrides = parse_input_overrides(
        ["df=override/df", "sample_map={file: ./inputs/metadata.xlsx}"],
        root=tmp_path,
        resources=ResourceCatalog(),
    )
    updated = apply_step_overrides(
        [plot_spec],
        input_overrides=overrides,
        set_overrides=[],
        root=tmp_path,
        resources=ResourceCatalog(),
    )

    assert updated[0].reads["df"] == RecordRef(record_id="override/df")
    assert updated[0].reads["sample_map"] == FileRef(path=(tmp_path / "inputs" / "metadata.xlsx").resolve())
