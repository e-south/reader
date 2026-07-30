from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from reader_workbench.tests.support import (
    base_reader_config,
    cli_success_data,
    default_notebook_name,
    load_decl,
    write_config,
)
from reader_workbench.workbench import FileRef, RecordRef, resolve_workbench
from reader_workbench.workbench.cli import app
from reader_workbench.workbench.experiment import ResourceCatalog
from reader_workbench.workbench.spec_overrides import apply_step_overrides, parse_input_overrides


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def _write_openable_workbook(path: Path) -> None:
    pd.DataFrame({"value": [1]}).to_excel(path, index=False)


def _base_config() -> dict:
    return base_reader_config(
        experiment_id="exp_cli",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [14.0]}},
        protocol_analysis={"crosstalk_pairs": {"enabled": True, "export": True}},
        protocol_outputs={
            "plots": {
                "profile": "none",
                "include": ["raw_kinetics", "endpoint_by_condition"],
                "views": {"endpoint_by_condition": {"time": 14.0}},
            },
            "exports": {"include": ["crosstalk_pairs_table"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )


def _logic_plot_config() -> dict:
    return base_reader_config(
        experiment_id="exp_logic",
        protocol_id="logic/sfxi_screen",
        protocol_inputs={"state_map_ref": "induction_logic"},
        protocol_analysis={"include_vec8": False, "include_fold_change": False},
        protocol_outputs={"plots": {"profile": "none", "include": ["logic_symmetry"]}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "ordered_state_spaces": {
                "induction_logic": {
                    "column": "treatment",
                    "state_order": ["00", "10", "01", "11"],
                    "values": {"00": "A", "10": "B", "01": "C", "11": "D"},
                }
            }
        },
    )


def _dual_reporter_profile_config(profile: str) -> dict:
    views = {
        "ratio_screen": {
            "state_summary": {"time": 14.0},
            "ratio_overview": {"snap_time": 14.0},
        },
        "heatmap_review": {
            "ratio_heatmap": {"time": 14.0},
            "support_heatmap": {"time": 14.0},
        },
    }
    return base_reader_config(
        experiment_id=f"exp_{profile}",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [14.0]}},
        protocol_outputs={"plots": {"profile": profile, "views": views[profile]}},
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
    payload = cli_success_data(result.output)
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
    payload = cli_success_data(result.output)
    assert payload["selection"] == {"only": [], "exclude": []}
    assert payload["summary"]["plots"] == 0
    assert payload["summary"]["by_plugin"] == {}
    assert payload["plots"] == []


def test_plot_list_json_surfaces_source_contract_metadata(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _logic_plot_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--list", "--format", "json"])
    assert result.exit_code == 0
    payload = cli_success_data(result.output)
    assert payload["summary"]["plots"] == 1
    assert payload["summary"]["by_family"] == {"geometry_plot": 1}
    read = payload["plots"][0]["reads"][0]
    assert read["contract"] == "logic_symmetry.v1"
    assert read["source"]["producer"]["id"] == "logic_symmetry_summary"
    assert read["source"]["surface"]["runtime_mode"] == "fixed"
    assert read["source"]["surface"]["rendered"] == "logic_symmetry.v1"


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
    payload = cli_success_data(result.output)
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


def test_plot_dry_run_rejects_file_override_outside_experiment(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    cfg = write_config(experiment, _base_config())
    outside = tmp_path / "outside.csv"
    outside.write_text("value\n1\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["plot", str(cfg), "--only", "raw_kinetics", "--dry-run", "--input", f"df={{file: {outside}}}"],
    )

    assert result.exit_code != 0
    assert "stay under the experiment root" in _plain(result.output)


def test_plot_dry_run_rejects_file_override_for_dataframe_port(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    (inputs_dir / "override.csv").write_text("value\n1\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "plot",
            str(cfg),
            "--only",
            "raw_kinetics",
            "--dry-run",
            "--input",
            "df={file: ./inputs/override.csv}",
        ],
    )

    assert result.exit_code != 0
    assert "expects a dataframe record" in _plain(result.output)


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
    monkeypatch.setattr("reader_workbench.workbench.cli.surfaces._run_plot_job", _fake_run_plot_job)
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
    monkeypatch.setattr(
        "reader_workbench.workbench.cli.surfaces.require_dataframe_records", lambda decl, job_path, runtime: None
    )
    monkeypatch.setattr("reader_workbench.workbench.cli.surfaces._run_plot_job", _fake_run_plot_job)
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
    monkeypatch.setattr(
        "reader_workbench.workbench.cli.surfaces.require_dataframe_records", lambda decl, job_path, runtime: None
    )
    monkeypatch.setattr("reader_workbench.workbench.cli.surfaces._run_plot_job", _fake_run_plot_job)
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
    payload = cli_success_data(result.output)
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
    payload = cli_success_data(result.output)
    assert payload["selection"] == {"only": [], "exclude": []}
    assert payload["summary"]["exports"] == 0
    assert payload["summary"]["by_family"] == {}
    assert payload["exports"] == []


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
    _write_openable_workbook(raw_path)
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


def test_notebook_scaffold_has_no_parallel_plot_selection_surface(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    notebook_help = runner.invoke(app, ["notebook", "--help"])
    plot_help = runner.invoke(app, ["plot", "--help"])

    assert notebook_help.exit_code == 0
    assert "--only" not in _plain(notebook_help.output)
    assert "--exclude" not in _plain(notebook_help.output)
    assert plot_help.exit_code == 0
    assert "--only" in _plain(plot_help.output)
    assert "--exclude" in _plain(plot_help.output)

    removed_option = runner.invoke(
        app,
        ["notebook", str(cfg), "--only", "raw_kinetics", "--mode", "none"],
        env={"COLUMNS": "200"},
    )
    assert removed_option.exit_code != 0
    assert "No such option '--only'" in _plain(removed_option.output)
    assert not (tmp_path / "outputs").exists()

    result = runner.invoke(app, ["notebook", str(cfg), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    assert nb_path.exists()
    content = nb_path.read_text(encoding="utf-8")
    assert "records(experiment)" in content
    assert "revision=_revision" in content
    assert "revision_digest=_revision_digest" in content
    assert "read_artifact(" in content
    assert "verify(experiment)" in content
    assert content.count("build_notebook_deliverable_selector(mo, deliverables)") == 1
    assert content.count("render_notebook_deliverable_viewport(") == 1
    assert 'label="Dataset (dataframe record)"' not in content


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
