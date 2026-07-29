from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reader.contracts import builtin_contract_catalog
from reader.errors import RecordError
from reader.workbench.graph import ProvenanceInput, RecordRef
from reader.workbench.records import RecordStore, RecordVerificationScope, verify_record_store


def _frame(value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["signal"],
            "value": [value],
        }
    )


def _source_record(store: RecordStore, kind: str):
    if kind == "dataframe":
        record = store.persist_dataframe(
            producer_id="source",
            producer_plugin="transform/source",
            out_name="df",
            record_id="source/df",
            df=_frame(),
            contract_id="tidy.v1",
            inputs=(),
            config_digest="sha256:test",
        )
        return record, record.path
    artifact = store.exports_dir / "source.txt"
    artifact.write_text("source", encoding="utf-8")
    record = store.append_file_bundle(
        producer_kind="export",
        producer_id="source",
        producer_plugin="export/source",
        record_id="source/files",
        inputs=(),
        config_digest="sha256:test",
        files=[artifact],
        description="Source files.",
    )
    return record, artifact


def _corrupt_same_size(path: Path) -> None:
    content = path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])


@pytest.mark.parametrize("kind", ["dataframe", "file_bundle"])
def test_capture_rejects_corrupt_local_record_artifacts(kind: str, tmp_path: Path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    record, artifact = _source_record(store, kind)
    _corrupt_same_size(artifact)
    counts_before = store.revision_counts()

    with pytest.raises(RecordError, match="content digest mismatch"):
        store.capture_inputs(
            [ProvenanceInput(label="source", ref=RecordRef(record.record_id))],
            resolved_inputs={"source": record},
        )

    assert store.revision_counts() == counts_before


@pytest.mark.parametrize("kind", ["dataframe", "file_bundle"])
def test_persistence_rechecks_captured_local_record_artifacts(kind: str, tmp_path: Path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    record, artifact = _source_record(store, kind)
    captured = store.capture_inputs(
        [ProvenanceInput(label="source", ref=RecordRef(record.record_id))],
        resolved_inputs={"source": record},
    )
    _corrupt_same_size(artifact)

    with pytest.raises(RecordError, match="changed after input evidence was captured.*content digest mismatch"):
        store.persist_dataframe(
            producer_id="downstream",
            producer_plugin="transform/downstream",
            out_name="df",
            record_id="downstream/df",
            df=_frame(2.0),
            contract_id="tidy.v1",
            inputs=captured,
            config_digest="sha256:test",
        )

    assert store.latest_record("downstream/df") is None


@pytest.mark.parametrize("kind", ["dataframe", "file_bundle"])
def test_scoped_verification_rejects_corrupt_local_input_artifacts(kind: str, tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    store = RecordStore(
        experiment_root / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=experiment_root,
    )
    record, artifact = _source_record(store, kind)
    captured = store.capture_inputs(
        [ProvenanceInput(label="source", ref=RecordRef(record.record_id))],
        resolved_inputs={"source": record},
    )
    downstream = store.persist_dataframe(
        producer_id="downstream",
        producer_plugin="transform/downstream",
        out_name="df",
        record_id="downstream/df",
        df=_frame(2.0),
        contract_id="tidy.v1",
        inputs=captured,
        config_digest="sha256:test",
    )
    _corrupt_same_size(artifact)

    report = verify_record_store(
        store,
        experiment_root=experiment_root,
        expected_config_digest="sha256:test",
        expected_build_identity=downstream.build_identity,
        scope=RecordVerificationScope(record_ids=frozenset({"downstream/df"})),
    )

    assert report["status"] == "failed"
    assert report["records"][0]["issues"][0]["code"] == "input.record_artifact_invalid"
