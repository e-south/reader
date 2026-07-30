from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reader_workbench.runtime import builtin_runtime
from reader_workbench.tests.support import base_reader_config, record_successful_invocation, write_config
from reader_workbench.workbench.config import ReaderSpec
from reader_workbench.workbench.decl import build_workbench_decl
from reader_workbench.workbench.inspection.readiness import experiment_readiness_payload


def _runtime_and_declaration(config_path: Path):
    runtime = builtin_runtime()
    spec = ReaderSpec.load(config_path)
    declaration = build_workbench_decl(spec, source_path=config_path, protocols=runtime.protocols)
    return runtime, declaration


def test_readiness_does_not_count_records_retired_from_the_current_workbench(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "config.yaml", base_reader_config(experiment_id="example"))
    runtime, decl = _runtime_and_declaration(config_path)
    layout = decl.experiment_semantics.layout
    store = runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=decl.experiment.root,
    )
    store.persist_dataframe(
        producer_id="retired",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="retired/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:retired-config",
        producer_config_digest="sha256:retired-producer",
    )

    readiness = experiment_readiness_payload(
        job_path=config_path,
        decl=decl,
        runtime=runtime,
        check_files=False,
    )

    assert readiness["state"] == "uncataloged_outputs_present"
    assert readiness["records"] == {
        "catalog": True,
        "available": False,
        "verification": None,
        "uncataloged_outputs_present": True,
    }
    assert readiness["capabilities"]["records"] is False
    assert readiness["capabilities"]["verify"] is True


def test_readiness_does_not_advertise_surfaces_with_missing_record_dependencies(tmp_path: Path) -> None:
    config = base_reader_config(
        experiment_id="partial_run",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [14.0]}},
        protocol_analysis={"crosstalk_pairs": {"enabled": True, "export": True}},
        protocol_outputs={
            "plots": {"profile": "none", "include": ["raw_kinetics"]},
            "exports": {"include": ["crosstalk_pairs_table"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    config_path = write_config(tmp_path / "config.yaml", config)
    runtime, decl = _runtime_and_declaration(config_path)
    store = runtime.record_store(
        decl.experiment_semantics.layout.outputs_dir,
        plots_subdir=decl.experiment_semantics.layout.plots_subdir,
        exports_subdir=decl.experiment_semantics.layout.exports_subdir,
        experiment_root=decl.experiment.root,
    )
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest=decl.config_digest,
        producer_config_digest="sha256:ingest-config",
    )
    record_successful_invocation(
        store,
        records=[record],
        config_digest=decl.config_digest,
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
    )

    readiness = experiment_readiness_payload(
        job_path=config_path,
        decl=decl,
        runtime=runtime,
        check_files=False,
    )

    assert readiness["records"]["verification"] == "ok"
    assert readiness["capabilities"]["plot"] is False
    assert readiness["capabilities"]["export"] is False
    commands = [step["command"] for step in readiness["next_steps"]]
    assert " reader run " in f" {commands[0]} "
    assert not any(" reader plot " in f" {command} " for command in commands)
    assert not any(" reader export " in f" {command} " for command in commands)


def test_readiness_routes_invalid_catalog_to_complete_epoch_reset(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "config.yaml", base_reader_config(experiment_id="invalid_catalog"))
    runtime, decl = _runtime_and_declaration(config_path)
    layout = decl.experiment_semantics.layout
    store = runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=decl.experiment.root,
    )
    store.records_path.write_text(
        json.dumps({"schema_version": 3, "latest": {}, "history": {}}),
        encoding="utf-8",
    )

    readiness = experiment_readiness_payload(
        job_path=config_path,
        decl=decl,
        runtime=runtime,
        check_files=False,
    )

    assert readiness["state"] == "blocked"
    assert readiness["summary"] == "blocked (invalid records catalog)"
    assert len(readiness["next_steps"]) == 1
    next_step = readiness["next_steps"][0]
    assert " reader run " in f" {next_step['command']} "
    assert str(config_path) in next_step["command"]
    assert next_step["command"].endswith(" --reset-records")
    assert "complete pipeline rerun" in next_step["description"]
