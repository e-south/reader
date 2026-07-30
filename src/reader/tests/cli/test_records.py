import json

import pandas as pd
from typer.testing import CliRunner

from reader.contracts import builtin_contract_catalog
from reader.tests.support import cli_success_data
from reader.workbench.cli import app
from reader.workbench.graph import ProvenanceInput, RecordRef
from reader.workbench.records import PathDescription, RecordStore


def test_records_requires_catalog(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema: reader/v8\nexperiment:\n  id: exp\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["records", str(config)])
    assert result.exit_code == 1
    assert "No outputs/manifests/records.json found" in result.output
    text = " ".join(result.output.split())
    assert "Run 'uv run reader run" in text
    assert config.parent.name in text


def test_records_lists_catalog_for_non_active_lifecycle(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema: reader/v8\nexperiment:\n  id: exp\n  lifecycle: draft\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["records", str(config)])
    assert result.exit_code == 0
    assert "ingest/df" in result.output


def test_records_lists_dataframe_and_file_bundle_entries(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema: reader/v8\nexperiment:\n  id: exp\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    plot_path = outputs / "plots" / "trace.png"
    summary_path = outputs / "plots" / "summary.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("png", encoding="utf-8")
    summary_path.write_text("png", encoding="utf-8")
    store.append_file_bundle(
        producer_kind="plot",
        producer_id="qc",
        producer_plugin="plot/time_series",
        record_id="plot:qc",
        inputs=store.capture_inputs([ProvenanceInput(label="df", ref=RecordRef(record_id="ingest/df"))]),
        config_digest="sha256:plot",
        files=[plot_path, summary_path],
        description="Render grouped time-series plots from tidy plate-reader traces.",
        path_descriptions=(
            PathDescription(path=plot_path, description="Grouped time-series traces for the configured channels."),
            PathDescription(path=summary_path, description="Endpoint summary by treatment."),
        ),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["records", str(config)])
    assert result.exit_code == 0
    assert "ingest/df" in result.output
    assert "plot:qc" in result.output
    assert "2 files" in result.output

    json_result = runner.invoke(app, ["records", str(config), "--format", "json"])
    assert json_result.exit_code == 0
    payload = cli_success_data(json_result.output)
    bundle = next(record for record in payload["records"] if record["record_id"] == "plot:qc")
    assert bundle["description"] == "Render grouped time-series plots from tidy plate-reader traces."
    assert bundle["path_descriptions"] == [
        {"path": "plots/summary.png", "description": "Endpoint summary by treatment."},
        {"path": "plots/trace.png", "description": "Grouped time-series traces for the configured channels."},
    ]
    assert "detail" not in bundle


def test_records_rejects_retired_record_schemas(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema: reader/v8\nexperiment:\n  id: exp\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    record = {
        "schema_version": 3,
        "record_id": "plot:qc",
        "kind": "file_bundle",
        "producer": {"kind": "plot", "id": "qc", "plugin": "plot/time_series"},
        "created_at": "2026-07-10T00:00:00+00:00",
        "inputs": [],
        "config_digest": "sha256:qc",
        "files": ["plots/qc.png"],
    }
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    catalog["latest"] = {"plot:qc": record}
    catalog["history"] = {"plot:qc": [record]}
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    runner = CliRunner()
    text_result = runner.invoke(app, ["records", str(config)])
    json_result = runner.invoke(app, ["records", str(config), "--format", "json"])

    assert text_result.exit_code == 1
    assert "schema_version must be 6" in text_result.output
    assert "--reset-records" in text_result.output
    assert json_result.exit_code == 1
    assert "schema_version must be 6" in json_result.output
    assert "--reset-records" in json_result.output


def test_records_all_shows_revision_counts(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema: reader/v8\nexperiment:\n  id: exp\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:one",
    )
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:two",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["records", str(config), "--all"])
    assert result.exit_code == 0
    assert "Records • history" in result.output
    assert "ingest/df" in result.output
    assert "2" in result.output
