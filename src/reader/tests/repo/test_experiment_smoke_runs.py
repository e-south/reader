from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from rich.console import Console

from reader.contracts import builtin_contract_catalog
from reader.tests.repo.experiment_matrix import END_TO_END_RUNNABLE_CONFIGS, repo_rel
from reader.tests.support import REPO_ROOT, load_decl
from reader.workbench import resolve_workbench
from reader.workbench.engine import run_spec
from reader.workbench.records import RecordStore


def _stage_experiment(tmp_path: Path, rel_dir: str) -> Path:
    source = REPO_ROOT / "experiments" / rel_dir
    target = tmp_path / source.name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "config.yaml", target / "config.yaml")
    shutil.copytree(source / "inputs", target / "inputs")
    return target / "config.yaml"


def _run(
    decl,
    *,
    include_pipeline: bool,
    include_plots: bool,
    include_exports: bool,
    plot_specs=None,
    export_specs=None,
) -> None:
    run_spec(
        decl,
        include_pipeline=include_pipeline,
        include_plots=include_plots,
        include_exports=include_exports,
        plot_specs=plot_specs,
        export_specs=export_specs,
        console=Console(force_terminal=False, color_system=None),
        log_level="ERROR",
        verbose=False,
    )


@pytest.mark.integration
@pytest.mark.fleet
@pytest.mark.parametrize("config_path", END_TO_END_RUNNABLE_CONFIGS, ids=repo_rel)
def test_repo_data_backed_experiments_run_end_to_end(tmp_path: Path, config_path: Path) -> None:
    rel_dir = str(config_path.parent.relative_to(REPO_ROOT / "experiments"))
    cfg_path = _stage_experiment(tmp_path, rel_dir)
    decl = load_decl(cfg_path)
    workbench = resolve_workbench(decl)

    _run(decl, include_pipeline=True, include_plots=True, include_exports=True)

    layout = decl.experiment_semantics.layout
    outputs = layout.outputs_dir
    store = RecordStore(
        outputs,
        contracts=builtin_contract_catalog(),
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        create=False,
    )

    latest_ids = {record.record_id for record in store.iter_latest_records()}
    expected_plot_ids = {f"plot:{plot.id}" for plot in workbench.plots}
    expected_export_ids = {f"export:{export.id}" for export in workbench.exports}
    plots_dir = outputs / layout.plots_subdir
    exports_dir = outputs / layout.exports_subdir

    assert (outputs / "manifests" / "records.json").exists()
    assert "ingest/df" in latest_ids
    assert expected_plot_ids.issubset(latest_ids)
    assert expected_export_ids.issubset(latest_ids)
    if expected_plot_ids:
        assert any(path.is_file() for path in plots_dir.rglob("*"))
    if expected_export_ids:
        assert any(path.is_file() for path in exports_dir.rglob("*"))


@pytest.mark.smoke
def test_plate_reader_panel_v3_generates_records_and_plots_from_clean_temp_copy(tmp_path: Path) -> None:
    cfg_path = _stage_experiment(tmp_path, "2025/20250614_sensor_panel_M9_glu")
    decl = load_decl(cfg_path)
    workbench = resolve_workbench(decl)
    plot_ts = next(plot for plot in workbench.plots if plot.id == "raw_kinetics")

    _run(decl, include_pipeline=True, include_plots=False, include_exports=False)
    _run(decl, include_pipeline=False, include_plots=True, include_exports=False, plot_specs=[plot_ts])

    layout = decl.experiment_semantics.layout
    outputs = layout.outputs_dir
    manifests = outputs / "manifests"
    plots_dir = outputs / layout.plots_subdir
    store = RecordStore(
        outputs,
        contracts=builtin_contract_catalog(),
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        create=False,
    )

    latest_ids = {record.record_id for record in store.iter_latest_records()}

    assert (manifests / "records.json").exists()
    assert not (manifests / "manifest.json").exists()
    assert not (manifests / "plots_manifest.json").exists()
    assert not (manifests / "exports_manifest.json").exists()
    assert "ingest/df" in latest_ids
    assert "plot:raw_kinetics" in latest_ids
    assert any(plots_dir.glob("*.pdf"))


@pytest.mark.smoke
def test_sfxi_v3_generates_records_and_export_from_clean_temp_copy(tmp_path: Path) -> None:
    cfg_path = _stage_experiment(tmp_path, "2025/20250915_sfxi_pSingle_ref")
    decl = load_decl(cfg_path)
    workbench = resolve_workbench(decl)
    export_vec8 = next(export for export in workbench.exports if export.id == "logic_summary_workbook")

    _run(decl, include_pipeline=True, include_plots=False, include_exports=False)
    _run(decl, include_pipeline=False, include_plots=False, include_exports=True, export_specs=[export_vec8])

    layout = decl.experiment_semantics.layout
    outputs = layout.outputs_dir
    manifests = outputs / "manifests"
    store = RecordStore(
        outputs,
        contracts=builtin_contract_catalog(),
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        create=False,
    )

    latest_ids = {record.record_id for record in store.iter_latest_records()}

    assert (manifests / "records.json").exists()
    assert not (manifests / "manifest.json").exists()
    assert not (manifests / "plots_manifest.json").exists()
    assert not (manifests / "exports_manifest.json").exists()
    assert "sfxi_vec8/vec8" in latest_ids
    assert "export:logic_summary_workbook" in latest_ids
    assert (outputs / layout.exports_subdir / "sfxi" / "vec8.xlsx").exists()


@pytest.mark.smoke
def test_sfxi_logic_geometry_experiment_runs_and_plots_from_clean_temp_copy(tmp_path: Path) -> None:
    cfg_path = _stage_experiment(tmp_path, "2025/20250825_sensors_1-7p_M9_logic_sym")
    decl = load_decl(cfg_path)
    workbench = resolve_workbench(decl)
    plot_logic = next(plot for plot in workbench.plots if plot.id == "logic_symmetry")

    _run(decl, include_pipeline=True, include_plots=False, include_exports=False)
    _run(decl, include_pipeline=False, include_plots=True, include_exports=False, plot_specs=[plot_logic])

    layout = decl.experiment_semantics.layout
    outputs = layout.outputs_dir
    manifests = outputs / "manifests"
    plots_dir = outputs / layout.plots_subdir
    store = RecordStore(
        outputs,
        contracts=builtin_contract_catalog(),
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        create=False,
    )

    latest_ids = {record.record_id for record in store.iter_latest_records()}

    assert (manifests / "records.json").exists()
    assert "promote_to_tidy_plus_map/df" in latest_ids
    assert "sfxi_vec8/vec8" not in latest_ids
    assert "plot:logic_symmetry" in latest_ids
    assert any(plots_dir.glob("*.pdf"))


@pytest.mark.smoke
def test_retron_sponge_experiment_generates_semantic_outputs_from_clean_temp_copy(tmp_path: Path) -> None:
    cfg_path = _stage_experiment(tmp_path, "2026/20260317_tetra_functional_sponges")
    decl = load_decl(cfg_path)
    workbench = resolve_workbench(decl)
    plot_summary = next(plot for plot in workbench.plots if plot.id == "interaction_summary")
    export_summary = next(export for export in workbench.exports if export.id == "semantic_summary_table")

    _run(decl, include_pipeline=True, include_plots=False, include_exports=False)
    _run(decl, include_pipeline=False, include_plots=True, include_exports=False, plot_specs=[plot_summary])
    _run(decl, include_pipeline=False, include_plots=False, include_exports=True, export_specs=[export_summary])

    layout = decl.experiment_semantics.layout
    outputs = layout.outputs_dir
    manifests = outputs / "manifests"
    plots_dir = outputs / layout.plots_subdir
    store = RecordStore(
        outputs,
        contracts=builtin_contract_catalog(),
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        create=False,
    )

    latest_ids = {record.record_id for record in store.iter_latest_records()}

    assert (manifests / "records.json").exists()
    assert "semantic_metrics/trace" in latest_ids
    assert "semantic_metrics/summary" in latest_ids
    assert "plot:interaction_summary" in latest_ids
    assert "export:semantic_summary_table" in latest_ids
    assert any(plots_dir.glob("*.pdf"))
    assert (outputs / layout.exports_subdir / "retron" / "semantic_summary.csv").exists()
