from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from rich.console import Console

from reader.contracts import builtin_contract_catalog
from reader.tests.support import REPO_ROOT, load_decl
from reader.workbench import resolve_workbench
from reader.workbench.engine import run_spec
from reader.workbench.records import RecordStore

pytestmark = pytest.mark.integration


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


def test_plate_reader_panel_v3_generates_records_and_plots_from_clean_temp_copy(tmp_path: Path) -> None:
    cfg_path = _stage_experiment(tmp_path, "2025/20250614_sensor_panel_M9_glu")
    decl = load_decl(cfg_path)
    workbench = resolve_workbench(decl)
    plot_ts = next(plot for plot in workbench.plots if plot.id == "plot_time_series")

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
    assert "plot:plot_time_series" in latest_ids
    assert any(plots_dir.glob("*.pdf"))


def test_sfxi_v3_generates_records_and_export_from_clean_temp_copy(tmp_path: Path) -> None:
    cfg_path = _stage_experiment(tmp_path, "2025/20250915_sfxi_pSingle_ref")
    decl = load_decl(cfg_path)
    workbench = resolve_workbench(decl)
    export_vec8 = next(export for export in workbench.exports if export.id == "export_vec8_xlsx")

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
    assert "export:export_vec8_xlsx" in latest_ids
    assert (outputs / layout.exports_subdir / "sfxi" / "vec8.xlsx").exists()
