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
