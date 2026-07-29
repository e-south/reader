from __future__ import annotations

from pathlib import Path

import pandas as pd

from reader.api import open_experiment
from reader.tests.support import base_reader_config, write_config
from reader.workbench.inspection.readiness import experiment_readiness_payload


def test_readiness_does_not_count_records_retired_from_the_current_workbench(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "config.yaml", base_reader_config(experiment_id="example"))
    experiment = open_experiment(config_path)
    decl = experiment.declaration
    layout = decl.experiment_semantics.layout
    store = experiment.runtime.record_store(
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
        runtime=experiment.runtime,
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
    experiment = open_experiment(config_path)
    decl = experiment.declaration
    store = experiment.runtime.record_store(
        decl.experiment_semantics.layout.outputs_dir,
        plots_subdir=decl.experiment_semantics.layout.plots_subdir,
        exports_subdir=decl.experiment_semantics.layout.exports_subdir,
        experiment_root=decl.experiment.root,
    )
    store.persist_dataframe(
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

    readiness = experiment_readiness_payload(
        job_path=config_path,
        decl=decl,
        runtime=experiment.runtime,
        check_files=False,
    )

    assert readiness["records"]["verification"] == "ok"
    assert readiness["capabilities"]["plot"] is False
    assert readiness["capabilities"]["export"] is False
    commands = [step["command"] for step in readiness["next_steps"]]
    assert " reader run " in f" {commands[0]} "
    assert not any(" reader plot " in f" {command} " for command in commands)
    assert not any(" reader export " in f" {command} " for command in commands)
