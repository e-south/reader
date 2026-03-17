"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/notebooks/test_scaffold.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from reader.tests.support import base_reader_config, default_notebook_name, write_config
from reader.workbench.cli import app


def test_plot_notebook_scaffold_uses_specs(tmp_path: Path) -> None:
    cfg_path = write_config(
        tmp_path,
        base_reader_config(experiment_id="exp_nb"),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--template", "notebook/eda", "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    assert nb_path.exists()
    content = nb_path.read_text(encoding="utf-8")
    assert 'label="Dataset (dataframe record)"' in content
    assert "discover_dataframe_records" in content
    assert "df = None" in content
    assert "__PLOT_SPECS__" not in content
    assert "resolve_plot_specs" not in content
    assert "plot --mode save" not in content


def test_notebook_scaffold_defaults_to_outputs_dir(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    assert nb_path.exists()


def test_notebook_scaffold_includes_df_selector(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")
    assert "dataframe record(s)" in content
    assert 'label="Dataset (dataframe record)"' in content
    assert "df = None" in content
    assert "## Dataset table explorer" in content
    assert "Design IDs" in content
    assert "Design + treatment summary" not in content
    assert 'label="Group by"' not in content
    assert "Interactive plot explorer" not in content
    assert "explore_x = mo.ui.dropdown" not in content
    assert "explore_y = mo.ui.dropdown" not in content
    assert "explore_hue = mo.ui.dropdown" not in content
    assert "mo.ui.table" in content
    assert "mo.ui.altair_chart" not in content
    assert "Quick plot" not in content
    assert "Available plot modules" not in content
    assert "plot/time_series" not in content
    assert "plot/snapshot_barplot" not in content
    assert "line: value vs time" not in content
    assert "Metadata summary" not in content
    assert "Source artifact" not in content


def test_notebook_scaffold_ignores_legacy_dir_when_present(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    legacy_dir = tmp_path / "notebooks"
    legacy_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    assert "Legacy notebooks/ detected" not in result.output
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    assert nb_path.exists()


def test_notebook_scaffold_respects_notebooks_override(tmp_path: Path) -> None:
    cfg = base_reader_config(experiment_id="exp_nb")
    cfg["paths"]["notebooks"] = "custom_notes"
    cfg_path = write_config(tmp_path, cfg)
    legacy_dir = tmp_path / "notebooks"
    legacy_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "custom_notes" / default_notebook_name()
    assert nb_path.exists()


def test_notebook_scaffold_uses_configured_notebook_spec(tmp_path: Path) -> None:
    cfg_path = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="exp_nb",
            protocol_id="cytometry/flow_panel",
            protocol_outputs={"notebook": {"template": "notebook/cytometry"}},
            resources={"metadata": {"kind": "file", "path": "./inputs/metadata.csv"}},
        ),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")
    assert "Cytometry" in content


def test_notebook_scaffold_disables_record_scan_by_default(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")
    assert "allow_scan=False" in content


def test_notebook_scaffold_can_enable_record_scan(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none", "--scan-records"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")
    assert "allow_scan=True" in content
