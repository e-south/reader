import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from reader_workbench.contracts import builtin_contract_catalog
from reader_workbench.runtime import ReaderRuntime, builtin_runtime
from reader_workbench.tests.support import base_reader_config, cli_success_data, write_config
from reader_workbench.workbench.cli import app
from reader_workbench.workbench.config import ReaderSpec, reader_spec_digest
from reader_workbench.workbench.graph import ProvenanceInput, RecordRef
from reader_workbench.workbench.records import PathDescription, RecordStore
from reader_workbench.workbench.registry import Registry


def _collection_config(tmp_path: Path, *, lifecycle: str = "active") -> Path:
    return write_config(
        tmp_path / "config.yaml",
        base_reader_config(
            experiment_id="exp",
            lifecycle=lifecycle,
            protocol_id="logic/four_state_vector_collection",
        ),
    )


def _config_digest(config: Path) -> str:
    return reader_spec_digest(ReaderSpec.load(config))


def test_records_requires_catalog(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema: reader/v8\nexperiment:\n  id: exp\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["records", str(config)], env={"COLUMNS": "200"})
    assert result.exit_code == 1
    assert "No outputs/manifests/records.json found" in result.output
    text = " ".join(result.output.split())
    assert "Run 'uv run reader run" in text
    assert config.parent.name in text


def test_records_lists_catalog_for_non_active_lifecycle(tmp_path) -> None:
    config = _collection_config(tmp_path, lifecycle="draft")
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="four_state_vector_collection/vectors",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest=_config_digest(config),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["records", str(config)], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "four_state_vector_collection/vectors" in result.output


def test_records_lists_dataframe_and_file_bundle_entries(tmp_path) -> None:
    config = _collection_config(tmp_path)
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="four_state_vector_collection/vectors",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest=_config_digest(config),
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
        record_id="plot:four_state_vector_heatmap",
        inputs=store.capture_inputs(
            [ProvenanceInput(label="df", ref=RecordRef(record_id="four_state_vector_collection/vectors"))]
        ),
        config_digest=_config_digest(config),
        files=[plot_path, summary_path],
        description="Render grouped time-series plots from tidy plate-reader traces.",
        path_descriptions=(
            PathDescription(path=plot_path, description="Grouped time-series traces for the configured channels."),
            PathDescription(path=summary_path, description="Endpoint summary by treatment."),
        ),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["records", str(config)], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "four_state_vector_collection/vectors" in result.output
    assert "plot:four_state_vector_heatmap" in result.output
    assert "2 files" in result.output

    json_result = runner.invoke(app, ["records", str(config), "--format", "json"])
    assert json_result.exit_code == 0
    payload = cli_success_data(json_result.output)
    bundle = next(record for record in payload["records"] if record["record_id"] == "plot:four_state_vector_heatmap")
    assert bundle["description"] == "Render grouped time-series plots from tidy plate-reader traces."
    assert bundle["path_descriptions"] == [
        {"path": "plots/summary.png", "description": "Endpoint summary by treatment."},
        {"path": "plots/trace.png", "description": "Grouped time-series traces for the configured channels."},
    ]
    assert "detail" not in bundle


def test_records_default_isolates_retired_records_and_unavailable_lineage(tmp_path) -> None:
    config = _collection_config(tmp_path)
    current_digest = _config_digest(config)
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    frame = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    for producer_id, record_id, config_digest in (
        ("four_state_vector_collection", "four_state_vector_collection/vectors", current_digest),
        ("stale_heatmap", "plot:four_state_vector_heatmap", "sha256:stale"),
        ("retired_collection", "retired_collection/vectors", current_digest),
    ):
        store.persist_dataframe(
            producer_id=producer_id,
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id=record_id,
            df=frame,
            contract_id="tidy.v1",
            inputs=[],
            config_digest=config_digest,
        )

    runner = CliRunner()
    text_result = runner.invoke(app, ["records", str(config)], env={"COLUMNS": "200"})
    current_result = runner.invoke(app, ["records", str(config), "--format", "json"])
    history_result = runner.invoke(app, ["records", str(config), "--all", "--format", "json"])

    assert text_result.exit_code == 0
    assert "four_state_vector_collection/vectors" in text_result.output
    assert "retired_collection" not in text_result.output
    assert "plot:four_state_vector_heatmap" not in text_result.output
    assert current_result.exit_code == 0
    current = cli_success_data(current_result.output)
    assert current["summary"]["records"] == 1
    assert [record["record_id"] for record in current["records"]] == ["four_state_vector_collection/vectors"]
    assert history_result.exit_code == 0
    history = cli_success_data(history_result.output)
    assert history["selection"]["include_history"] is True
    assert {record["record_id"] for record in history["records"]} == {
        "four_state_vector_collection/vectors",
        "plot:four_state_vector_heatmap",
        "retired_collection/vectors",
    }

    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    retired = catalog["latest"]["retired_collection/vectors"]
    retired["inputs"] = [
        {
            "label": "source",
            "kind": "source_record",
            "resource": "source",
            "experiment": "removed-source",
            "record": "source/df",
            "discovery_policy": "source_record",
            "record_revision_digest": "sha256:" + "a" * 64,
        }
    ]
    catalog["history"]["retired_collection/vectors"][-1] = retired
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    isolated_current = runner.invoke(app, ["records", str(config), "--format", "json"])
    unresolvable_history = runner.invoke(app, ["records", str(config), "--all", "--format", "json"])

    assert isolated_current.exit_code == 0
    assert [record["record_id"] for record in cli_success_data(isolated_current.output)["records"]] == [
        "four_state_vector_collection/vectors"
    ]
    assert unresolvable_history.exit_code == 1
    assert "Could not resolve source experiment 'removed-source'" in unresolvable_history.output


def test_records_generic_workbench_uses_current_config_as_its_record_boundary(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema: reader/v8\nexperiment:\n  id: exp\nprotocol:\n  id: workbench/generic\n",
        encoding="utf-8",
    )
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    frame = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    for record_id, config_digest in (
        ("measurements/current", _config_digest(config)),
        ("measurements/prior", "sha256:prior"),
    ):
        store.persist_dataframe(
            producer_id=record_id.replace("/", "_"),
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id=record_id,
            df=frame,
            contract_id="tidy.v1",
            inputs=[],
            config_digest=config_digest,
        )

    runner = CliRunner()
    current_result = runner.invoke(app, ["records", str(config), "--format", "json"])
    history_result = runner.invoke(app, ["records", str(config), "--all", "--format", "json"])

    assert current_result.exit_code == 0
    current = cli_success_data(current_result.output)
    assert [record["record_id"] for record in current["records"]] == ["measurements/current"]
    assert history_result.exit_code == 0
    history = cli_success_data(history_result.output)
    assert {record["record_id"] for record in history["records"]} == {
        "measurements/current",
        "measurements/prior",
    }


def test_records_history_remains_readable_when_current_plugins_are_unavailable(tmp_path, monkeypatch) -> None:
    config = _collection_config(tmp_path)
    builtin = builtin_runtime()
    archived_runtime = ReaderRuntime(
        contracts=builtin.contracts,
        protocols=builtin.protocols,
        plugins=Registry(contracts=builtin.contracts),
    )
    monkeypatch.setattr("reader_workbench.runtime.builtin_runtime", lambda: archived_runtime)
    store = RecordStore(tmp_path / "outputs", contracts=builtin.contracts)
    store.persist_dataframe(
        producer_id="retired_collection",
        producer_plugin="transform/retired_collection",
        out_name="df",
        record_id="retired_collection/vectors",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:archived",
    )

    current_result = CliRunner().invoke(app, ["records", str(config), "--format", "json"])
    history_result = CliRunner().invoke(app, ["records", str(config), "--all", "--format", "json"])

    assert current_result.exit_code == 1
    assert "Unknown plugin" in current_result.output
    assert history_result.exit_code == 0
    history = cli_success_data(history_result.output)
    assert [record["record_id"] for record in history["records"]] == ["retired_collection/vectors"]
    assert history["records"][0]["description"] == (
        "Description unavailable because plugin 'transform/retired_collection' is not registered."
    )


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
