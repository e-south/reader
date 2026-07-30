from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from reader_workbench.contracts import builtin_contract_catalog
from reader_workbench.tests.support import (
    base_reader_config,
    cli_error_data,
    cli_success_data,
    record_successful_invocation,
    write_config,
)
from reader_workbench.workbench.cli import app
from reader_workbench.workbench.config import ReaderSpec, reader_spec_digest
from reader_workbench.workbench.records import RecordStore


def _write_config(tmp_path) -> tuple[object, ReaderSpec]:
    config = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="exp",
            protocol_id="plate_reader/dual_reporter_screen",
            protocol_analysis={"include_fold_change": False},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
            annotations={
                "labels": {
                    "design_id": {
                        "source": "design_id",
                        "output": "design_id_alias",
                        "values": {},
                    }
                }
            },
        ),
    )
    return config, ReaderSpec.load(config)


def _write_record(tmp_path, spec: ReaderSpec):
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest=reader_spec_digest(spec),
        producer_config_digest="sha256:producer",
    )
    record_successful_invocation(
        store,
        records=[record],
        config_digest=reader_spec_digest(spec),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
    )
    return record


def test_verify_json_reports_verified_v5_records(tmp_path) -> None:
    config, spec = _write_config(tmp_path)
    _write_record(tmp_path, spec)

    result = CliRunner().invoke(app, ["verify", str(config), "--format", "json"])

    assert result.exit_code == 0
    payload = cli_success_data(result.output)
    assert payload["schema"] == "reader.verify/v1"
    assert payload["status"] == "ok"
    assert payload["summary"] == {
        "checked": 1,
        "failed": 0,
        "unverifiable": 0,
        "invocations_checked": 1,
        "invocation_failures": 0,
    }


def test_verify_ignores_records_owned_by_removed_workbench_surfaces(tmp_path) -> None:
    config, spec = _write_config(tmp_path)
    _write_record(tmp_path, spec)
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    retired_record = store.persist_dataframe(
        producer_id="retired",
        producer_plugin="transform/identity",
        out_name="df",
        record_id="retired/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:retired-config",
        producer_config_digest="sha256:retired-producer",
    )
    record_successful_invocation(
        store,
        records=[retired_record],
        config_digest="sha256:retired-config",
        operation="run",
        selected_step_ids={"pipeline": ["retired"], "plots": [], "exports": []},
    )

    result = CliRunner().invoke(app, ["verify", str(config), "--format", "json"])

    assert result.exit_code == 0
    payload = cli_success_data(result.output)
    assert payload["status"] == "ok"
    assert payload["summary"] == {
        "checked": 1,
        "failed": 0,
        "unverifiable": 0,
        "invocations_checked": 2,
        "invocation_failures": 0,
    }


def test_verify_json_is_machine_readable_and_nonzero_for_corruption(tmp_path) -> None:
    config, spec = _write_config(tmp_path)
    record = _write_record(tmp_path, spec)
    record.path.write_bytes(b"corrupt")

    result = CliRunner().invoke(app, ["verify", str(config), "--format", "json"])

    assert result.exit_code == 1
    error = cli_error_data(result.output)
    assert error["code"] == "artifact.size_mismatch"


def test_verify_json_reports_missing_catalog_without_rich_output(tmp_path) -> None:
    config, _spec = _write_config(tmp_path)

    result = CliRunner().invoke(app, ["verify", str(config), "--format", "json"])

    assert result.exit_code == 1
    error = cli_error_data(result.output)
    assert error["code"] == "catalog.missing"
