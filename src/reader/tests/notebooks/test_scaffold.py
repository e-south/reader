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
    assert "records(experiment)" in content
    assert "read_dataframe(experiment, _record_id).dataframe" in content
    assert "load_notebook_context" in content
    assert "reader.workbench.notebooks.context" not in content
    assert "load_workbench_decl(cfg_path)" not in content
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


def test_notebook_scaffold_explores_dataframe_deliverables_in_primary_viewport(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--template", "notebook/eda", "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")
    assert "collect_notebook_deliverables" in content
    assert "build_notebook_deliverable_selector" in content
    assert "render_notebook_deliverable_viewport" in content
    assert "dataframe_loader=_load_dataframe" in content
    assert "read_dataframe(experiment, _record_id).dataframe" in content
    assert "render_notebook_overview_panel" in content
    assert "data_ready" not in content
    assert 'label="Group by"' not in content
    assert "Interactive plot explorer" not in content
    assert "explore_x = mo.ui.dropdown" not in content
    assert "explore_y = mo.ui.dropdown" not in content
    assert "explore_hue = mo.ui.dropdown" not in content
    assert "mo.ui.altair_chart" not in content
    assert "Quick plot" not in content
    assert "Available plot modules" not in content
    assert "plot/time_series" not in content
    assert "plot/snapshot_barplot" not in content
    assert "line: value vs time" not in content
    assert "Metadata summary" not in content
    assert "Source artifact" not in content


def test_notebook_scaffold_surfaces_deliverables_with_progressive_disclosure(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--template", "notebook/eda", "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")

    assert "collect_notebook_deliverables" in content
    assert "build_notebook_deliverable_selector" in content
    assert "render_notebook_deliverable_viewport" in content
    assert "mo.accordion" not in content


def test_generated_eda_uses_one_record_driven_selector_and_viewport(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--template", "notebook/eda", "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")

    assert content.count("build_notebook_deliverable_selector(mo, deliverables)") == 1
    assert content.count("render_notebook_deliverable_viewport(") == 1
    assert "dataframe_loader=_load_dataframe" in content
    assert "read_dataframe(experiment, _record_id).dataframe" in content
    assert "record_dropdown" not in content
    assert "build_dataframe_record_catalog" not in content
    assert "select_default_dataframe_record" not in content
    assert "## Dataset selection" not in content
    assert "## Dataset table explorer" not in content


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


def test_notebook_scaffold_has_no_uncataloged_record_scan_mode(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")
    assert "discover_dataframe_records" not in content
    assert "allow_scan" not in content
    assert "read_parquet" not in content


def test_notebook_cli_rejects_removed_record_scan_option(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none", "--scan-records"])
    assert result.exit_code == 2
    assert not (tmp_path / "outputs" / "notebooks" / default_notebook_name()).exists()
