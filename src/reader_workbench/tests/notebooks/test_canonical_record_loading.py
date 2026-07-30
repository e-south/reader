from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reader_workbench.api import open_experiment, read_dataframe, records
from reader_workbench.errors import RecordError
from reader_workbench.runtime import builtin_runtime
from reader_workbench.tests.support.configs import base_reader_config, write_config
from reader_workbench.workbench.config import ReaderSpec
from reader_workbench.workbench.decl import build_workbench_decl


def test_notebook_public_record_flow_rejects_valid_but_tampered_parquet(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "config.yaml", base_reader_config(experiment_id="notebook_integrity"))
    runtime = builtin_runtime()
    spec = ReaderSpec.load(config_path)
    declaration = build_workbench_decl(spec, source_path=config_path, protocols=runtime.protocols)
    experiment = open_experiment(config_path, runtime=runtime)
    layout = declaration.experiment_semantics.layout
    store = runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=declaration.experiment.root,
    )
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["signal"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest=declaration.config_digest,
    )
    catalog = records(experiment)
    selected_record_id = next(
        str(entry["record_id"]) for entry in catalog.entries if entry.get("kind") == "dataframe_artifact"
    )
    pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["signal"], "value": [999.0]}).to_parquet(
        record.path, index=False
    )

    with pytest.raises(RecordError, match="content digest mismatch"):
        read_dataframe(experiment, selected_record_id)
