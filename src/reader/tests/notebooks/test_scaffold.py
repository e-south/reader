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
    assert "load_notebook_workbench_context" in content
    assert "load_workbench_decl(cfg_path)" not in content
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
    result = runner.invoke(app, ["notebook", str(cfg_path), "--template", "notebook/eda", "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")
    assert "dataframe record(s)" in content
    assert 'label="Dataset (dataframe record)"' in content
    assert "df = None" in content
    assert "## Dataset table explorer" in content
    assert "build_design_treatment_summary_rows" in content
    assert "render_notebook_overview_panel" in content
    assert "Design + treatment summary" not in content
    assert "data_ready" not in content
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


def test_notebook_scaffold_surfaces_deliverables_with_progressive_disclosure(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path, base_reader_config(experiment_id="exp_nb"))
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--template", "notebook/eda", "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")

    assert "collect_notebook_deliverables" in content
    assert "render_notebook_deliverables_panel" in content
    assert "render_notebook_deliverables_panel(mo, deliverables)" in content


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


def test_retron_notebook_scaffold_surfaces_plot_portfolio_and_semantic_focus(tmp_path: Path) -> None:
    cfg_path = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="exp_retron_nb",
            protocol_id="plate_reader/retron_sponge_screen",
            protocol_analysis={
                "semantic_metrics": {
                    "relevant_stress_map": {"sulAp": "100 nM ciprofloxacin"},
                    "sensor_target_map": {"sulAp": ["LexA"]},
                }
            },
            protocol_outputs={"plots": {"include": ["baseline_shifted_kinetics"]}},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        ),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")
    assert "Retron sponge review" in content
    assert 'app = marimo.App(width="medium")' in content
    assert "What this notebook contains" not in content
    assert "Assay contract" in content
    assert "Plot map" in content
    assert "Transforms" in content
    assert "retron_figure_coverage_rows" in content
    assert "Available plots and exports" in content
    assert '"Analysis exports"' in content
    assert 'label="Assay view"' in content
    assert "retron_scope_control_items" in content
    assert "mo.hstack(" in content
    assert "baseline_shifted_kinetics" in content
    assert "retron_visible_plot_specs" in content
    assert "Math / transform" in content
    assert "mo.ui.data_explorer" not in content
    assert "## Semantic table focus" not in content


def test_retron_aggregate_notebook_scaffold_surfaces_cross_run_review_sections(tmp_path: Path) -> None:
    manifest_path = tmp_path / "inputs" / "review_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "relevant_stress_map: {spyP: '3% EtOH'}\nsensor_target_map: {spyP: [CpxR, BaeR]}\nsources: []\n",
        encoding="utf-8",
    )
    cfg_path = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="exp_retron_review",
            protocol_id="workbench/generic",
            protocol_outputs={"notebook": {"template": "notebook/retron_sponge_aggregate"}},
            resources={"review_manifest": {"kind": "file", "path": "./inputs/review_manifest.yaml"}},
        ),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"])
    assert result.exit_code == 0
    nb_path = tmp_path / "outputs" / "notebooks" / default_notebook_name()
    content = nb_path.read_text(encoding="utf-8")
    assert "Retron sponge review set" in content
    assert 'app = marimo.App(width="medium")' in content
    assert "What this notebook contains" not in content
    assert "Source experiments" in content
    assert "Workflow map" in content
    assert 'label="Review surface"' in content
    assert '"Aggregate view": "Aggregate view"' in content
    assert '"Source experiment view": "Source experiment view"' in content
    assert "Source experiment" in content
    assert "Source assay view" in content
    assert "retron_source_control_panel = mo.hstack" in content
    assert "retron_aggregate_control_panel = mo.hstack" in content
    assert "retron_source_selector_rows(retron_aggregate_bundle)" in content
    assert "wrap=False" in content
    assert 'label="Aggregate evidence family"' in content
    assert 'label="Fingerprint sponge"' not in content
    assert "retron_master_surface_selector" not in content
    assert content.count("full_width=False") >= 5
    assert "load_retron_source_surface" in content
    assert "retron_figure_coverage_rows" in content
    assert "retron_aggregate_figure_rows" in content
    assert "retron_aggregate_plot_rows" in content
    assert "mo.state({})" in content
    assert "expected_vs_observed" in content
    assert "review_manifest" in content
    assert "mo.ui.data_explorer" not in content
    assert "## Inspect one aggregate plot at a time" not in content
