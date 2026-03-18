"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/cli/test_plot_export.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from reader.tests.support import base_reader_config, default_notebook_name, load_decl, write_config
from reader.workbench import FileRef, RecordRef, resolve_workbench
from reader.workbench.cli import app
from reader.workbench.experiment import ResourceCatalog
from reader.workbench.spec_overrides import apply_step_overrides, parse_input_overrides


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
    assert "No plot specs configured" in result.output


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


def test_plot_json_requires_list(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--format", "json"])
    assert result.exit_code != 0
    assert "only supported with --list" in result.output


def test_plot_requires_records(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg)])
    assert result.exit_code != 0
    assert "outputs/manifests/records.json" in result.output
    assert "reader run" in result.output


def test_export_requires_records(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg)])
    assert result.exit_code != 0
    assert "outputs/manifests/records.json" in result.output
    assert "reader run" in result.output


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
    assert "No export specs configured" in result.output


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


def test_export_json_requires_list(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg), "--format", "json"])
    assert result.exit_code != 0
    assert "only supported with --list" in result.output


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
