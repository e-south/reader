from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from reader.core.cli import app


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_plot_notebook_scaffold_uses_specs(tmp_path: Path) -> None:
    cfg = {
        "schema": "reader/v2",
        "experiment": {"id": "exp_nb"},
        "paths": {"outputs": "./outputs", "plots": "plots", "exports": "exports"},
        "pipeline": {"steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}]},
        "plots": {
            "specs": [
                {
                    "id": "plot_a",
                    "uses": "plot/time_series",
                    "reads": {"df": "ingest/df"},
                    "with": {"y": ["OD600"]},
                }
            ]
        },
        "exports": {"specs": []},
    }
    cfg_path = _write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--preset", "notebook/plots", "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / "plots.py"
    assert nb_path.exists()
    content = nb_path.read_text(encoding="utf-8")
    assert "label=\"Dataset (artifact df.parquet)\"" in content
    assert "df.parquet" in content
    assert "df_active" in content
    assert "__PLOT_SPECS__" not in content
    assert "resolve_plot_specs" not in content
    assert "plot --mode save" not in content


def test_notebook_scaffold_defaults_to_outputs_dir(tmp_path: Path) -> None:
    cfg = {
        "schema": "reader/v2",
        "experiment": {"id": "exp_nb"},
        "paths": {"outputs": "./outputs", "plots": "plots", "exports": "exports"},
        "pipeline": {"steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}]},
        "plots": {"specs": []},
        "exports": {"specs": []},
    }
    cfg_path = _write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / "notebook.py"
    assert nb_path.exists()


def test_notebook_scaffold_includes_df_selector(tmp_path: Path) -> None:
    cfg = {
        "schema": "reader/v2",
        "experiment": {"id": "exp_nb"},
        "paths": {"outputs": "./outputs", "plots": "plots", "exports": "exports"},
        "pipeline": {"steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}]},
        "plots": {"specs": []},
        "exports": {"specs": []},
    }
    cfg_path = _write_config(tmp_path, cfg)
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / "notebook.py"
    content = nb_path.read_text(encoding="utf-8")
    assert "artifact dataset(s)" in content
    assert "label=\"Dataset (artifact df.parquet)\"" in content
    assert "df_active = None" in content
    assert "label=\"X\"" in content
    assert "label=\"Y\"" in content
    assert "label=\"Hue\"" in content
    assert "label=\"Facet row\"" in content
    assert "label=\"Facet col\"" in content
    assert "Filter:" in content
    assert "Design IDs" in content
    assert "Design + treatment summary" not in content
    assert "label=\"Group by\"" not in content
    assert "Interactive plot explorer" not in content
    assert "explore_x = mo.ui.dropdown" not in content
    assert "explore_y = mo.ui.dropdown" not in content
    assert "explore_hue = mo.ui.dropdown" not in content
    assert "mo.ui.table" in content
    assert "mo.ui.altair_chart" in content
    assert "Quick plot" not in content
    assert "Available plot modules" not in content
    assert "plot/time_series" not in content
    assert "plot/snapshot_barplot" not in content
    assert "line: value vs time" not in content
    assert "Metadata summary" not in content
    assert "Source artifact" not in content


def test_notebook_scaffold_uses_legacy_dir_when_present(tmp_path: Path) -> None:
    cfg = {
        "schema": "reader/v2",
        "experiment": {"id": "exp_nb"},
        "paths": {"outputs": "./outputs", "plots": "plots", "exports": "exports"},
        "pipeline": {"steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}]},
        "plots": {"specs": []},
        "exports": {"specs": []},
    }
    cfg_path = _write_config(tmp_path, cfg)
    legacy_dir = tmp_path / "notebooks"
    legacy_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    assert "Legacy notebooks/ detected" in result.output
    nb_path = legacy_dir / "notebook.py"
    assert nb_path.exists()


def test_notebook_scaffold_respects_notebooks_override(tmp_path: Path) -> None:
    cfg = {
        "schema": "reader/v2",
        "experiment": {"id": "exp_nb"},
        "paths": {
            "outputs": "./outputs",
            "plots": "plots",
            "exports": "exports",
            "notebooks": "custom_notes",
        },
        "pipeline": {"steps": [{"id": "ingest", "uses": "ingest/synergy_h1"}]},
        "plots": {"specs": []},
        "exports": {"specs": []},
    }
    cfg_path = _write_config(tmp_path, cfg)
    legacy_dir = tmp_path / "notebooks"
    legacy_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "custom_notes" / "notebook.py"
    assert nb_path.exists()
