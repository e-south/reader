from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import pandas as pd
import pyarrow.parquet as pq
import pytest

from reader.api import DataFrameRecordResult, Experiment, open_experiment, read_dataframe, records
from reader.errors import RecordError
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.tests.support.configs import base_reader_config, write_config
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl, build_workbench_decl
from reader.workbench.records import DataFrameArtifactRecord, record_revision_digest
from reader.workbench.records import model as record_model


@dataclass(frozen=True)
class _ExperimentFixture:
    experiment: Experiment
    runtime: ReaderRuntime
    declaration: WorkbenchDecl


def _experiment(tmp_path: Path) -> _ExperimentFixture:
    config_path = write_config(tmp_path / "config.yaml", base_reader_config(experiment_id="example"))
    runtime = builtin_runtime()
    spec = ReaderSpec.load(config_path)
    declaration = build_workbench_decl(spec, source_path=config_path, protocols=runtime.protocols)
    return _ExperimentFixture(
        experiment=open_experiment(config_path, runtime=runtime),
        runtime=runtime,
        declaration=declaration,
    )


def _record_store(fixture: _ExperimentFixture):
    declaration = fixture.declaration
    layout = declaration.experiment_semantics.layout
    return fixture.runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=declaration.experiment.root,
    )


def _persist_dataframe(store, fixture: _ExperimentFixture, frame: pd.DataFrame) -> DataFrameArtifactRecord:
    return store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=frame,
        contract_id="tidy.v1",
        inputs=[],
        config_digest=fixture.declaration.config_digest,
    )


def test_read_dataframe_returns_verified_revision_and_defensive_copy(tmp_path: Path, monkeypatch) -> None:
    fixture = _experiment(tmp_path)
    store = _record_store(fixture)
    first = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["signal"], "value": [1.0]})
    second = first.assign(value=[2.0])
    _persist_dataframe(store, fixture, first)
    latest = _persist_dataframe(store, fixture, second)
    loaded: dict[str, pd.DataFrame] = {}
    original_load = DataFrameArtifactRecord.load_dataframe

    def _load(record: DataFrameArtifactRecord) -> pd.DataFrame:
        frame = original_load(record)
        loaded["frame"] = frame
        return frame

    monkeypatch.setattr(DataFrameArtifactRecord, "load_dataframe", _load)

    result = read_dataframe(fixture.experiment, "ingest/df")

    assert isinstance(result, DataFrameRecordResult)
    assert result.experiment == fixture.experiment.identity
    assert result.record.record_id == "ingest/df"
    assert result.record.revision == 2
    assert result.record.revision_digest.startswith("sha256:")
    assert result.contract_id == "tidy.v1"
    assert result.content_digest == latest.content_digest
    pd.testing.assert_frame_equal(result.dataframe, second)
    assert result.dataframe is not loaded["frame"]
    result.dataframe.loc[0, "value"] = 99.0
    assert loaded["frame"].loc[0, "value"] == 2.0
    assert json.loads(json.dumps(result.to_dict()))["dataframe"] == {
        "rows": 1,
        "columns": ["position", "time", "channel", "value"],
    }


def test_records_and_read_dataframe_bind_one_exact_revision(tmp_path: Path) -> None:
    fixture = _experiment(tmp_path)
    store = _record_store(fixture)
    first_frame = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["signal"], "value": [1.0]})
    second_frame = first_frame.assign(value=[2.0])
    first = _persist_dataframe(store, fixture, first_frame)
    first_digest = record_revision_digest(first, outputs_dir=store.root)
    second = _persist_dataframe(store, fixture, second_frame)

    entry = records(fixture.experiment).entries[0]
    loaded = read_dataframe(
        fixture.experiment,
        "ingest/df",
        revision=1,
        revision_digest=first_digest,
    )

    assert entry["revision"] == 2
    assert entry["revision_digest"] == record_revision_digest(second, outputs_dir=store.root)
    assert loaded.record.revision == 1
    assert loaded.record.revision_digest == first_digest
    pd.testing.assert_frame_equal(loaded.dataframe, first_frame)


def test_read_dataframe_rejects_mismatched_exact_revision_identity(tmp_path: Path) -> None:
    fixture = _experiment(tmp_path)
    store = _record_store(fixture)
    frame = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["signal"], "value": [1.0]})
    first = _persist_dataframe(store, fixture, frame)
    first_digest = record_revision_digest(first, outputs_dir=store.root)
    second = _persist_dataframe(store, fixture, frame.assign(value=[2.0]))
    second_digest = record_revision_digest(second, outputs_dir=store.root)

    with pytest.raises(RecordError, match="revision digest mismatch.*refresh"):
        read_dataframe(
            fixture.experiment,
            "ingest/df",
            revision=1,
            revision_digest=second_digest,
        )

    assert first_digest != second_digest


def test_read_dataframe_rejects_missing_catalog_without_creating_outputs(tmp_path: Path) -> None:
    fixture = _experiment(tmp_path)

    with pytest.raises(RecordError, match="Record catalog is missing"):
        read_dataframe(fixture.experiment, "ingest/df")

    assert not (tmp_path / "outputs").exists()


def test_read_dataframe_rejects_missing_record(tmp_path: Path) -> None:
    fixture = _experiment(tmp_path)
    _record_store(fixture)

    with pytest.raises(RecordError, match="Dataframe record 'missing/df' is missing"):
        read_dataframe(fixture.experiment, "missing/df")


def test_read_dataframe_rejects_non_dataframe_record(tmp_path: Path) -> None:
    fixture = _experiment(tmp_path)
    store = _record_store(fixture)
    artifact = store.exports_dir / "summary.txt"
    artifact.write_text("summary", encoding="utf-8")
    store.append_file_bundle(
        producer_kind="export",
        producer_id="summary",
        producer_plugin="export/table",
        record_id="summary/files",
        inputs=[],
        config_digest=fixture.declaration.config_digest,
        files=[artifact],
        description="Example exported summary.",
    )

    with pytest.raises(RecordError, match="exists but is not a dataframe artifact"):
        read_dataframe(fixture.experiment, "summary/files")


def test_read_dataframe_rejects_content_digest_mismatch(tmp_path: Path) -> None:
    fixture = _experiment(tmp_path)
    store = _record_store(fixture)
    frame = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["signal"], "value": [1.0]})
    record = _persist_dataframe(store, fixture, frame)
    record.path.write_bytes(record.path.read_bytes() + b"tampered")

    with pytest.raises(RecordError, match="content digest mismatch"):
        read_dataframe(fixture.experiment, "ingest/df")


def test_read_dataframe_row_limit_returns_verified_bounded_preview(tmp_path: Path, monkeypatch) -> None:
    fixture = _experiment(tmp_path)
    store = _record_store(fixture)
    frame = pd.DataFrame(
        {
            "position": [f"A{index + 1}" for index in range(500)],
            "time": [float(index) for index in range(500)],
            "channel": ["signal"] * 500,
            "value": [float(index) for index in range(500)],
        }
    )
    record = _persist_dataframe(store, fixture, frame)

    def _reject_full_materialization(*_args, **_kwargs):
        raise AssertionError("bounded preview must not use pandas.read_parquet")

    monkeypatch.setattr(pd, "read_parquet", _reject_full_materialization)

    result = read_dataframe(fixture.experiment, "ingest/df", row_limit=200)

    assert result.record.revision == 1
    assert result.record.revision_digest == record_revision_digest(record, outputs_dir=store.root)
    assert result.content_digest == record.content_digest
    pd.testing.assert_frame_equal(result.dataframe, frame.head(200))


def test_read_dataframe_row_limit_rejects_tampered_content(tmp_path: Path) -> None:
    fixture = _experiment(tmp_path)
    store = _record_store(fixture)
    frame = pd.DataFrame({"position": ["A1", "A2"], "time": [0.0, 1.0], "channel": ["signal"] * 2, "value": [1.0, 2.0]})
    record = _persist_dataframe(store, fixture, frame)
    record.path.write_bytes(record.path.read_bytes() + b"tampered")

    with pytest.raises(RecordError, match="content digest mismatch"):
        read_dataframe(fixture.experiment, "ingest/df", row_limit=1)


def test_read_dataframe_row_limit_reads_verified_open_descriptor_when_path_is_replaced(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _experiment(tmp_path)
    store = _record_store(fixture)
    original_frame = pd.DataFrame(
        {"position": ["A1", "A2"], "time": [0.0, 1.0], "channel": ["signal"] * 2, "value": [1.0, 2.0]}
    )
    replacement_frame = original_frame.assign(value=[99.0, 100.0])
    record = _persist_dataframe(store, fixture, original_frame)
    replacement_path = tmp_path / "replacement.parquet"
    replacement_frame.to_parquet(replacement_path, index=False)
    original_parquet_file = pq.ParquetFile

    def _replace_path_then_open(source, *args, **kwargs):
        replacement_path.replace(record.path)
        return original_parquet_file(source, *args, **kwargs)

    monkeypatch.setattr(record_model.pq, "ParquetFile", _replace_path_then_open)

    result = read_dataframe(fixture.experiment, "ingest/df", row_limit=1)

    pd.testing.assert_frame_equal(result.dataframe, original_frame.head(1))


@pytest.mark.parametrize("row_limit", [0, -1, True, 1.5])
def test_read_dataframe_rejects_invalid_row_limit(tmp_path: Path, row_limit: object) -> None:
    fixture = _experiment(tmp_path)
    store = _record_store(fixture)
    frame = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["signal"], "value": [1.0]})
    _persist_dataframe(store, fixture, frame)

    with pytest.raises(RecordError, match="row_limit must be a positive integer"):
        read_dataframe(fixture.experiment, "ingest/df", row_limit=row_limit)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("catalog_change", "message"),
    [
        ("invalid_frame", "violates recorded contract 'tidy.v1'.*missing required column 'channel'"),
        ("unknown_contract", "violates recorded contract 'unknown.v1'.*unknown contract id 'unknown.v1'"),
    ],
)
def test_read_dataframe_revalidates_the_recorded_contract(
    tmp_path: Path,
    catalog_change: str,
    message: str,
) -> None:
    fixture = _experiment(tmp_path)
    store = _record_store(fixture)
    frame = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["signal"], "value": [1.0]})
    record = _persist_dataframe(store, fixture, frame)
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    latest = catalog["latest"]["ingest/df"]
    history = catalog["history"]["ingest/df"][-1]
    if catalog_change == "invalid_frame":
        pd.DataFrame({"position": ["A1"], "time": [0.0], "value": [1.0]}).to_parquet(record.path, index=False)
        digest = "sha256:" + hashlib.sha256(record.path.read_bytes()).hexdigest()
        for payload in (latest, history):
            payload["content_digest"] = digest
            payload["size_bytes"] = record.path.stat().st_size
    else:
        for payload in (latest, history):
            payload["contract_id"] = "unknown.v1"
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match=message):
        read_dataframe(fixture.experiment, "ingest/df")


def test_public_api_result_annotations_are_runtime_resolvable() -> None:
    experiment_hints = get_type_hints(Experiment)
    dataframe_hints = get_type_hints(DataFrameRecordResult)

    assert experiment_hints["_runtime"] is ReaderRuntime
    assert dataframe_hints["dataframe"] is pd.DataFrame
