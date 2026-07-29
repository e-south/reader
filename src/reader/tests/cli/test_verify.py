from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from reader.contracts import builtin_contract_catalog
from reader.tests.support import cli_error_data, cli_success_data
from reader.workbench.cli import app
from reader.workbench.config import ReaderSpec, reader_spec_digest
from reader.workbench.records import RecordStore


def _write_config(tmp_path) -> tuple[object, ReaderSpec]:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema: reader/v8\nexperiment:\n  id: exp\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    return config, ReaderSpec.load(config)


def _write_record(tmp_path, spec: ReaderSpec):
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    return store.persist_dataframe(
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


def test_verify_json_reports_verified_v5_records(tmp_path) -> None:
    config, spec = _write_config(tmp_path)
    _write_record(tmp_path, spec)

    result = CliRunner().invoke(app, ["verify", str(config), "--format", "json"])

    assert result.exit_code == 0
    payload = cli_success_data(result.output)
    assert payload["schema"] == "reader.verify/v1"
    assert payload["status"] == "ok"
    assert payload["summary"] == {"checked": 1, "failed": 0, "unverifiable": 0}


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
