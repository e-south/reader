"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_record_store.py

Tests for the unified workbench record catalog.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from reader.contracts import builtin_contract_catalog
from reader.errors import ContractError, RecordError
from reader.workbench.graph import ProvenanceInput, RecipeSource, RecordRef
from reader.workbench.records import PathDescription, RecordStore, discover_dataframe_records


def test_records_catalog_invalid_json_raises(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manifests = outputs / "manifests"
    manifests.mkdir()
    (manifests / "records.json").write_text("{not json", encoding="utf-8")
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), create=False)
    with pytest.raises(RecordError):
        store.iter_latest_records()


def test_records_catalog_missing_keys_raises(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manifests = outputs / "manifests"
    manifests.mkdir()
    (manifests / "records.json").write_text(json.dumps({"latest": []}), encoding="utf-8")
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), create=False)
    with pytest.raises(RecordError):
        store.iter_latest_records()


def test_records_catalog_unknown_schema_version_raises(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manifests = outputs / "manifests"
    manifests.mkdir()
    (manifests / "records.json").write_text(
        json.dumps({"schema_version": 999, "latest": {}, "history": {}}),
        encoding="utf-8",
    )
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), create=False)
    with pytest.raises(RecordError, match="schema_version must be 3"):
        store.iter_latest_records()


def test_record_payload_unknown_schema_version_raises(tmp_path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    catalog["latest"]["ingest/df"]["schema_version"] = 999
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="record payload schema_version must be 3"):
        store.read_record("ingest/df")


def test_record_payload_rejects_ambiguous_provenance_input(tmp_path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    catalog["latest"]["ingest/df"]["inputs"] = [
        {"label": "raw", "record": "upstream/df", "file": "inputs/raw.xlsx", "extra": "ignored"}
    ]
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="invalid provenance input.*exactly one reference shape"):
        store.read_record("ingest/df")


def test_record_store_persists_dataframe_and_file_bundle(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
        source_recipe=RecipeSource(recipe="plate_reader/sample_map", with_={"channel": "OD600"}),
    )
    plot_path = outputs / "plots" / "trace.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("png", encoding="utf-8")
    bundle = store.append_file_bundle(
        producer_kind="plot",
        producer_id="qc_plot",
        producer_plugin="plot/time_series",
        record_id="plot:qc_plot",
        inputs=[ProvenanceInput(label="df", ref=RecordRef(record_id="ingest/df"))],
        config_digest="sha256:plot",
        files=[plot_path],
        description="Render grouped time-series plots from tidy plate-reader traces.",
        path_descriptions=(
            PathDescription(
                path=plot_path,
                description="Grouped time-series traces for the configured channels and treatment groups.",
            ),
        ),
    )

    latest = store.iter_latest_records()
    assert {item.record_id for item in latest} == {"ingest/df", "plot:qc_plot"}
    assert record.path.exists()
    assert bundle.files == (plot_path,)
    assert bundle.description == "Render grouped time-series plots from tidy plate-reader traces."
    assert bundle.description_for(plot_path) == (
        "Grouped time-series traces for the configured channels and treatment groups."
    )
    persisted = json.loads(store.records_path.read_text(encoding="utf-8"))
    assert persisted["latest"]["plot:qc_plot"]["description"] == bundle.description
    assert persisted["latest"]["plot:qc_plot"]["path_descriptions"] == [
        {
            "description": "Grouped time-series traces for the configured channels and treatment groups.",
            "path": "plots/trace.png",
        }
    ]
    restored_bundle = store.read_record("plot:qc_plot")
    assert restored_bundle.description == bundle.description
    assert restored_bundle.description_for(plot_path) == bundle.description_for(plot_path)
    assert len(store.record_history("ingest/df")) == 1
    assert len(store.record_history("plot:qc_plot")) == 1
    assert store.revision_counts(["ingest/df", "plot:qc_plot"]) == {"ingest/df": 1, "plot:qc_plot": 1}
    assert record.producer.source_recipe is not None
    assert record.producer.source_recipe.recipe == "plate_reader/sample_map"


@pytest.mark.parametrize("description", ["   ", None])
def test_file_bundle_description_rejects_missing_or_blank_text(tmp_path, description) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())

    with pytest.raises(RecordError, match="description must be a non-empty string"):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[outputs / "plots" / "trace.png"],
            description=description,
        )


def test_plot_file_bundle_rejects_missing_path_descriptions(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())

    with pytest.raises(RecordError, match="plot file bundles must describe every file"):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[outputs / "plots" / "trace.png"],
            description="Render grouped time-series plots from tidy plate-reader traces.",
        )


def test_file_bundle_rejects_empty_file_list(tmp_path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())

    with pytest.raises(RecordError, match="must contain at least one file"):
        store.append_file_bundle(
            producer_kind="export",
            producer_id="summary",
            producer_plugin="export/xlsx",
            record_id="export:summary",
            inputs=[],
            config_digest="sha256:export",
            files=[],
            description="Workbook export.",
        )


def test_path_description_rejects_non_pathlike_values() -> None:
    with pytest.raises(RecordError, match="path must be path-like"):
        PathDescription(path=None, description="Plot description.")  # type: ignore[arg-type]


def test_file_bundle_rejects_untyped_path_description_entries(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())

    with pytest.raises(RecordError, match="entries must be PathDescription"):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[outputs / "plots" / "trace.png"],
            description="Render grouped time-series plots from tidy plate-reader traces.",
            path_descriptions=(object(),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("path_state", "message"),
    [
        ("missing", "missing files"),
        ("directory", "non-file paths"),
    ],
)
def test_file_bundle_rejects_paths_that_are_not_existing_files(tmp_path, path_state, message) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    plot_path = outputs / "plots" / "trace.png"
    if path_state == "directory":
        plot_path.mkdir(parents=True)

    with pytest.raises(RecordError, match=message):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[plot_path],
            description="Render grouped time-series plots from tidy plate-reader traces.",
            path_descriptions=(
                PathDescription(path=plot_path, description="Grouped time-series traces for the configured channels."),
            ),
        )

    assert store.iter_latest_records() == ()


@pytest.mark.parametrize(
    ("descriptions", "message"),
    [
        (
            (
                ("plots/trace.png", "First description."),
                ("plots/trace.png", "Second description."),
            ),
            "duplicate path descriptions",
        ),
        (
            (("plots/other.png", "Description for a different file."),),
            "unmatched path descriptions",
        ),
        (
            (("plots/trace.png", "Description for only one file."),),
            "missing path descriptions",
        ),
    ],
)
def test_plot_file_bundle_rejects_invalid_path_description_coverage(tmp_path, descriptions, message) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    files = [outputs / "plots" / "trace.png"]
    if "missing" in message:
        files.append(outputs / "plots" / "summary.png")

    with pytest.raises(RecordError, match=message):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=files,
            description="Render grouped time-series plots from tidy plate-reader traces.",
            path_descriptions=tuple(
                PathDescription(path=outputs / path, description=description) for path, description in descriptions
            ),
        )


def test_file_bundle_without_description_is_rejected(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    payload = {
        "schema_version": 3,
        "record_id": "plot:legacy",
        "kind": "file_bundle",
        "producer": {"kind": "plot", "id": "legacy", "plugin": "plot/time_series"},
        "created_at": "2026-07-10T00:00:00+00:00",
        "inputs": [],
        "config_digest": "sha256:legacy",
        "files": ["plots/legacy.png"],
    }
    catalog = {"schema_version": 3, "latest": {"plot:legacy": payload}, "history": {"plot:legacy": [payload]}}
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="must include a non-empty description"):
        store.read_record("plot:legacy")


def test_dataframe_record_load_rejects_content_digest_mismatch(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    df.assign(value=[2.0]).to_parquet(record.path, index=False)

    with pytest.raises(RecordError, match="content digest mismatch"):
        record.load_dataframe()


def test_same_config_changed_dataframe_uses_immutable_revision_paths(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    first_df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})

    first = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=first_df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:same-config",
    )
    second = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=first_df.assign(value=[2.0]),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:same-config",
    )

    history = store.record_history("ingest/df")
    assert len(history) == 2
    assert first.path != second.path
    assert [record.path for record in history] == [first.path, second.path]
    pd.testing.assert_frame_equal(history[0].load_dataframe(), first_df)
    pd.testing.assert_frame_equal(history[1].load_dataframe(), first_df.assign(value=[2.0]))


def test_identical_dataframe_revision_is_idempotent(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    persist = {
        "producer_id": "ingest",
        "producer_plugin": "ingest/synergy_h1",
        "out_name": "df",
        "record_id": "ingest/df",
        "df": df,
        "contract_id": "tidy.v1",
        "inputs": [],
        "config_digest": "sha256:same-config",
    }

    first = store.persist_dataframe(**persist)
    second = store.persist_dataframe(**persist)

    assert second == first
    assert store.record_history("ingest/df") == (first,)
    assert sorted(path.name for path in store.artifacts_dir.iterdir()) == ["ingest.ingest_synergy_h1"]


def test_dataframe_digesting_does_not_read_entire_artifact(tmp_path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})

    def reject_read_bytes(_path: Path) -> bytes:
        raise AssertionError("artifact hashing must stream file content")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )

    record.verify_content_digest()


def test_catalog_replace_failure_preserves_catalog_and_cleans_staging(tmp_path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    original_catalog = store.records_path.read_text(encoding="utf-8")
    plot_path = outputs / "plots" / "trace.png"
    plot_path.write_text("png", encoding="utf-8")
    original_replace = Path.replace

    def fail_catalog_replace(source: Path, target: Path) -> Path:
        if target == store.records_path:
            raise OSError("injected catalog replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_catalog_replace)

    with pytest.raises(RecordError, match="atomically replace records.json"):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[plot_path],
            description="Render grouped time-series plots from tidy plate-reader traces.",
            path_descriptions=(
                PathDescription(path=plot_path, description="Grouped time-series traces for configured channels."),
            ),
        )

    assert store.records_path.read_text(encoding="utf-8") == original_catalog
    assert list(store.manifests_dir.glob(".records.json.*.tmp")) == []


def test_dataframe_catalog_failure_removes_unpublished_revision(tmp_path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    original_replace = Path.replace

    def fail_catalog_replace(source: Path, target: Path) -> Path:
        if target == store.records_path:
            raise OSError("injected catalog replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_catalog_replace)

    with pytest.raises(RecordError, match="atomically replace records.json"):
        store.persist_dataframe(
            producer_id="ingest",
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id="ingest/df",
            df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
            contract_id="tidy.v1",
            inputs=[],
            config_digest="sha256:test",
        )

    assert store.record_history("ingest/df") == ()
    assert list(store.artifacts_dir.iterdir()) == []
    assert list(store.manifests_dir.glob(".records.json.*.tmp")) == []


def test_dataframe_contract_validation_cannot_be_disabled(tmp_path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())

    with pytest.raises(ContractError):
        store.persist_dataframe(
            producer_id="ingest",
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id="ingest/df",
            df=pd.DataFrame({"value": [1.0]}),
            contract_id="tidy.v1",
            inputs=[],
            config_digest="sha256:test",
        )

    assert store.record_history("ingest/df") == ()
    assert list(store.artifacts_dir.iterdir()) == []


@pytest.mark.parametrize("contract_id", ["none", "files"])
def test_dataframe_record_rejects_reserved_contract_ids(tmp_path, contract_id: str) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())

    with pytest.raises(ContractError, match="unknown contract id"):
        store.persist_dataframe(
            producer_id="ingest",
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id="ingest/df",
            df=pd.DataFrame({"value": [1.0]}),
            contract_id=contract_id,
            inputs=[],
            config_digest="sha256:test",
        )

    assert store.record_history("ingest/df") == ()
    assert list(store.artifacts_dir.iterdir()) == []


@pytest.mark.parametrize("raw_path", ["../outside.parquet", "/tmp/outside.parquet"])
def test_catalog_rejects_dataframe_paths_outside_outputs(tmp_path, raw_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    payload = {
        "schema_version": 3,
        "record_id": "ingest/df",
        "kind": "dataframe_artifact",
        "producer": {"kind": "pipeline", "id": "ingest", "plugin": "ingest/synergy_h1"},
        "created_at": "2026-07-11T00:00:00+00:00",
        "inputs": [],
        "config_digest": "sha256:test",
        "contract_id": "tidy.v1",
        "path": raw_path,
        "content_digest": "sha256:test",
        "code_digest": "",
    }
    catalog = {"schema_version": 3, "latest": {"ingest/df": payload}, "history": {"ingest/df": [payload]}}
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="must resolve within the outputs directory|must be relative"):
        store.read_dataframe("ingest/df")


@pytest.mark.parametrize("raw_path", ["../outside.png", "/tmp/outside.png"])
def test_catalog_rejects_file_bundle_paths_outside_outputs(tmp_path, raw_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    payload = {
        "schema_version": 3,
        "record_id": "plot:qc_plot",
        "kind": "file_bundle",
        "producer": {"kind": "plot", "id": "qc_plot", "plugin": "plot/time_series"},
        "created_at": "2026-07-11T00:00:00+00:00",
        "inputs": [],
        "config_digest": "sha256:test",
        "files": [raw_path],
        "description": "Time-series plot.",
        "path_descriptions": [{"path": raw_path, "description": "Time-series plot."}],
    }
    catalog = {
        "schema_version": 3,
        "latest": {"plot:qc_plot": payload},
        "history": {"plot:qc_plot": [payload]},
    }
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="must resolve within the outputs directory|must be relative"):
        store.read_record("plot:qc_plot")


def test_file_bundle_rejects_path_that_resolves_outside_outputs(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_plot = outside / "trace.png"
    outside_plot.write_text("png", encoding="utf-8")
    plots_link = outputs / "plots" / "external"
    plots_link.symlink_to(outside, target_is_directory=True)
    linked_plot = plots_link / "trace.png"

    with pytest.raises(RecordError, match="must resolve within the outputs directory"):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[linked_plot],
            description="Render grouped time-series plots from tidy plate-reader traces.",
            path_descriptions=(
                PathDescription(path=linked_plot, description="Grouped time-series traces for configured channels."),
            ),
        )

    assert store.iter_latest_records() == ()


def test_record_store_revision_counts_bulk_read(tmp_path) -> None:
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
        config_digest="sha256:first",
    )
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df.assign(value=[2.0]),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:second",
    )

    counts = store.revision_counts(["ingest/df", "missing/df"])

    assert counts == {"ingest/df": 2, "missing/df": 0}


def test_discover_dataframe_records_is_catalog_only_by_default(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    data_dir = outputs / "artifacts" / "ratio.transform_ratio"
    data_dir.mkdir(parents=True)
    df = pd.DataFrame({"value": [1.0]})
    df.to_parquet(data_dir / "df.parquet", index=False)

    info, labels, note, warning = discover_dataframe_records(outputs)
    assert info == {}
    assert labels == []
    assert "records.json" in note
    assert warning == ""


def test_discover_dataframe_records_can_fall_back_to_scan(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    data_dir = outputs / "artifacts" / "ratio.transform_ratio"
    data_dir.mkdir(parents=True)
    df = pd.DataFrame({"value": [1.0]})
    df.to_parquet(data_dir / "df.parquet", index=False)

    info, labels, note, warning = discover_dataframe_records(outputs, allow_scan=True)
    assert labels
    assert note == ""
    assert "scanning outputs/artifacts" in warning
    first = info[labels[0]]
    assert first["path"].name == "df.parquet"
