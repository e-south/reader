from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from typer.testing import CliRunner

from reader.core.cli import app


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _base_config() -> dict:
    return {
        "schema": "reader/v2",
        "experiment": {"id": "exp_cli"},
        "paths": {"outputs": "./outputs", "plots": "plots", "exports": "exports"},
        "pipeline": {"steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}]},
        "plots": {
            "specs": [
                {
                    "id": "plot_a",
                    "uses": "plot/time_series",
                    "reads": {"df": "ingest/df"},
                    "with": {"y": ["OD600"]},
                },
                {
                    "id": "plot_b",
                    "uses": "plot/snapshot_barplot",
                    "reads": {"df": "ingest/df"},
                    "with": {"x": "design_id", "y": ["OD600"], "time": 0.0},
                },
            ]
        },
        "exports": {
            "specs": [
                {"id": "export_a", "uses": "export/csv", "reads": {"df": "ingest/df"}, "with": {"path": "a.csv"}}
            ]
        },
    }


def test_plot_list_filters(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg), "--list"])
    assert result.exit_code == 0
    assert "plot_a" in result.output
    assert "plot_b" in result.output

    result = runner.invoke(app, ["plot", str(cfg), "--list", "--only", "plot_a"])
    assert result.exit_code == 0
    assert "plot_a" in result.output
    assert "plot_b" not in result.output

    result = runner.invoke(app, ["plot", str(cfg), "--list", "--exclude", "plot_a"])
    assert result.exit_code == 0
    assert "plot_a" not in result.output
    assert "plot_b" in result.output


def test_plot_list_empty(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["plots"]["specs"] = []
    cfg_path = _write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg_path), "--list"])
    assert result.exit_code == 0
    assert "No plot specs configured" in result.output


def test_plot_requires_artifacts(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["plot", str(cfg)])
    assert result.exit_code != 0
    assert "outputs/manifests/manifest.json" in result.output
    assert "reader run" in result.output


def test_plot_year_list(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    year_dir = tmp_path / "experiments" / "2025"
    exp_a = year_dir / "exp_a"
    exp_b = year_dir / "exp_b"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)
    _write_config(exp_a, _base_config())
    _write_config(exp_b, _base_config())
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["plot", "--year", "2025", "--list"])
    assert result.exit_code == 0
    assert "exp_a" in result.output
    assert "exp_b" in result.output


def test_export_list_filters(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg), "--list"])
    assert result.exit_code == 0
    assert "export_a" in result.output

    result = runner.invoke(app, ["export", str(cfg), "--list", "--only", "export_a"])
    assert result.exit_code == 0
    assert "export_a" in result.output


def test_export_list_empty(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["exports"]["specs"] = []
    cfg_path = _write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["export", str(cfg_path), "--list"])
    assert result.exit_code == 0
    assert "No export specs configured" in result.output


def test_validate_checks_files_by_default(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["pipeline"]["steps"] = [
        {"id": "ingest", "uses": "ingest/synergy_h1"},
        {
            "id": "merge_map",
            "uses": "merge/sample_map",
            "reads": {"df": "ingest/df", "sample_map": "file:./metadata.xlsx"},
        },
    ]
    cfg_path = _write_config(tmp_path, cfg)
    file_path = tmp_path / "metadata.xlsx"
    file_path.write_text("stub", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(cfg_path)])
    assert result.exit_code == 0

    file_path.unlink()
    result = runner.invoke(app, ["validate", str(cfg_path)])
    assert result.exit_code == 1
    assert "Missing input files" in result.output


def test_validate_no_files_skips_checks(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["pipeline"]["steps"] = [
        {"id": "ingest", "uses": "ingest/synergy_h1"},
        {
            "id": "merge_map",
            "uses": "merge/sample_map",
            "reads": {"df": "ingest/df", "sample_map": "file:./metadata.xlsx"},
        },
    ]
    cfg_path = _write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(cfg_path), "--no-files"])
    assert result.exit_code == 0


def test_plot_notebook_scaffold(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, _base_config())
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["notebook", str(cfg), "--preset", "notebook/eda", "--only", "plot_a", "--mode", "none"],
    )
    assert result.exit_code == 0
    date_stamp = datetime.now().strftime("%Y%m%d")
    nb_path = tmp_path / "outputs" / "notebooks" / f"EDA_{date_stamp}.py"
    assert nb_path.exists()
    content = nb_path.read_text(encoding="utf-8")
    assert "PLOT_SPECS" not in content
    assert "label=\"Dataset (artifact df.parquet)\"" in content
