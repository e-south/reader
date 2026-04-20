from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from reader.contracts import builtin_contract_catalog
from reader.tests.repo.experiment_matrix import END_TO_END_RUNNABLE_CONFIGS, repo_rel
from reader.tests.support import REPO_ROOT, default_notebook_name, load_decl
from reader.workbench import resolve_workbench
from reader.workbench.cli import app
from reader.workbench.engine import run_spec
from reader.workbench.records import RecordStore, record_paths

pytestmark = pytest.mark.integration


def _stage_experiment(tmp_path: Path, rel_dir: str) -> Path:
    source = REPO_ROOT / "experiments" / rel_dir
    if not source.exists():
        pytest.skip(f"Experiment fixture missing from checkout: {source}")
    if not (source / "inputs").exists():
        pytest.skip(f"Experiment inputs missing from checkout: {source / 'inputs'}")
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


def _assert_file_bundle_records_exist(store: RecordStore, record_ids: set[str]) -> None:
    for record_id in sorted(record_ids):
        record = store.read_record(record_id)
        paths = record_paths(record)
        assert paths, f"Record {record_id} did not include any materialized files."
        assert all(path.is_file() for path in paths), f"Record {record_id} referenced missing files: {paths!r}"


@pytest.mark.active_experiments
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

    assert (outputs / "manifests" / "records.json").exists()
    assert "ingest/df" in latest_ids
    assert expected_plot_ids.issubset(latest_ids)
    assert expected_export_ids.issubset(latest_ids)
    _assert_file_bundle_records_exist(store, expected_plot_ids | expected_export_ids)


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
    _assert_file_bundle_records_exist(store, {"plot:raw_kinetics"})


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
    _assert_file_bundle_records_exist(store, {"export:logic_summary_workbook"})
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
    _assert_file_bundle_records_exist(store, {"plot:logic_symmetry"})


@pytest.mark.smoke
def test_retron_sponge_experiment_generates_semantic_outputs_from_clean_temp_copy(tmp_path: Path) -> None:
    cfg_path = _stage_experiment(tmp_path, "2026/20260317_tetra_functional_sponges")
    decl = load_decl(cfg_path)
    workbench = resolve_workbench(decl)

    _run(decl, include_pipeline=True, include_plots=True, include_exports=True)

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
    expected_plot_ids = {f"plot:{plot.id}" for plot in workbench.plots}
    expected_export_ids = {f"export:{export.id}" for export in workbench.exports}

    assert (manifests / "records.json").exists()
    assert "semantic_metrics/trace" in latest_ids
    assert "semantic_metrics/summary" in latest_ids
    assert expected_plot_ids.issubset(latest_ids)
    assert expected_export_ids.issubset(latest_ids)
    assert "plot:baseline_shifted_kinetics" in latest_ids
    _assert_file_bundle_records_exist(store, expected_plot_ids | expected_export_ids)
    assert any(plots_dir.glob("raw_kinetics*.pdf"))
    assert (outputs / layout.exports_subdir / "retron" / "semantic_summary.csv").exists()
    assert (outputs / layout.exports_subdir / "retron" / "semantic_trace.csv").exists()


@pytest.mark.smoke
def test_cli_notebook_scaffold_on_staged_experiment_preserves_runnable_readiness(tmp_path: Path) -> None:
    cfg_path = _stage_experiment(tmp_path, "2025/20250614_sensor_panel_M9_glu")
    runner = CliRunner()

    notebook_result = runner.invoke(app, ["notebook", str(cfg_path), "--mode", "none"], env={"COLUMNS": "200"})
    assert notebook_result.exit_code == 0, notebook_result.output

    notebook_path = cfg_path.parent / "outputs" / "notebooks" / default_notebook_name()
    assert notebook_path.exists()

    inspect_result = runner.invoke(app, ["inspect", str(cfg_path), "--format", "json"])
    assert inspect_result.exit_code == 0
    inspect_payload = json.loads(inspect_result.output)
    assert inspect_payload["implementation"]["readiness"]["state"] == "runnable"


@pytest.mark.smoke
def test_cli_retron_sponge_experiment_runs_end_to_end_and_writes_artifact_journal(tmp_path: Path) -> None:
    cfg_path = _stage_experiment(tmp_path, "2026/20260317_tetra_functional_sponges")
    runner = CliRunner()

    validate_result = runner.invoke(app, ["validate", str(cfg_path), "--format", "json"])
    assert validate_result.exit_code == 0
    validate_payload = json.loads(validate_result.output)
    assert validate_payload["summary"]["status"] == "ok"

    for command in (
        ["run", str(cfg_path)],
        ["plot", str(cfg_path)],
        ["export", str(cfg_path)],
    ):
        result = runner.invoke(app, command, env={"COLUMNS": "200"})
        assert result.exit_code == 0, result.output

    inspect_result = runner.invoke(app, ["inspect", str(cfg_path), "--format", "json"])
    assert inspect_result.exit_code == 0
    inspect_payload = json.loads(inspect_result.output)
    assert inspect_payload["implementation"]["readiness"]["state"] == "records_ready"
    assert (
        inspect_payload["implementation"]["compiled"]["semantic_program"]["ranking"]["execution"]["status"]
        == "compiled"
    )

    records_result = runner.invoke(app, ["records", str(cfg_path), "--format", "json"])
    assert records_result.exit_code == 0
    records_payload = json.loads(records_result.output)
    record_ids = {item["record_id"] for item in records_payload["records"]}
    assert "semantic_metrics/trace" in record_ids
    assert "semantic_metrics/summary" in record_ids
    assert "plot:baseline_shifted_kinetics" in record_ids
    assert "export:semantic_summary_table" in record_ids

    decl = load_decl(cfg_path)
    workbench = resolve_workbench(decl)
    layout = decl.experiment_semantics.layout
    outputs = layout.outputs_dir
    store = RecordStore(
        outputs,
        contracts=builtin_contract_catalog(),
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        create=False,
    )
    expected_plot_ids = {f"plot:{plot.id}" for plot in workbench.plots}
    expected_export_ids = {f"export:{export.id}" for export in workbench.exports}
    _assert_file_bundle_records_exist(store, expected_plot_ids | expected_export_ids)

    journal = cfg_path.parent / "JOURNAL.md"
    assert journal.exists()
    journal_text = journal.read_text(encoding="utf-8")
    assert "uv run reader run" in journal_text
    assert "uv run reader plot" in journal_text
    assert "uv run reader export" in journal_text
