from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from reader.contracts import builtin_contract_catalog
from reader.errors import RecordError
from reader.workbench.graph import FileRef, ProvenanceInput, RecordRef
from reader.workbench.records import ArtifactEvidence, RecordStore, RecordVerificationScope, verify_record_store
from reader.workbench.records.identity import current_build_identity


def _tidy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["OD600"],
            "value": [1.0],
        }
    )


def test_artifact_evidence_requires_a_complete_sha256_digest() -> None:
    with pytest.raises(RecordError, match="content_digest must be a sha256 digest"):
        ArtifactEvidence(relative_path=Path("inputs/raw.xlsx"), size_bytes=1, content_digest="sha256:short")


def _write_record(tmp_path: Path):
    outputs = tmp_path / "outputs"
    store = RecordStore(
        outputs,
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:experiment-config",
        producer_config_digest="sha256:producer-config",
    )
    return store, record


def test_new_records_are_schema_v5_and_verify_without_loading_dataframe(tmp_path: Path) -> None:
    store, record = _write_record(tmp_path)

    payload = json.loads(store.records_path.read_text(encoding="utf-8"))["latest"]["ingest/df"]
    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert payload["schema_version"] == 5
    assert payload["producer_config_digest"] == "sha256:producer-config"
    assert payload["build_identity"]["reader_version"]
    assert payload["build_identity"]["source_digest"].startswith("sha256:")
    assert payload["size_bytes"] == record.path.stat().st_size
    assert report["schema"] == "reader.verify/v1"
    assert report["status"] == "ok"
    assert report["summary"] == {"checked": 1, "failed": 0, "unverifiable": 0}


@pytest.mark.parametrize("corruption", ["empty_history", "divergent_final"])
def test_verifier_rejects_latest_history_lineage_corruption(tmp_path: Path, corruption: str) -> None:
    store, _record = _write_record(tmp_path)
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    if corruption == "empty_history":
        catalog["history"]["ingest/df"] = []
    else:
        catalog["history"]["ingest/df"][-1]["config_digest"] = "sha256:divergent-history"
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"] == {"checked": 0, "failed": 1, "unverifiable": 0}
    assert report["issues"][0]["code"] == "catalog.invalid"
    assert "history" in report["issues"][0]["reason"]


@pytest.mark.parametrize("corruption", ["invalid_utf8", "directory"])
def test_verifier_reports_unreadable_catalogs_as_invalid(tmp_path: Path, corruption: str) -> None:
    store, _record = _write_record(tmp_path)
    if corruption == "invalid_utf8":
        store.records_path.write_bytes(b"\xff")
    else:
        store.records_path.unlink()
        store.records_path.mkdir()

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"] == {"checked": 0, "failed": 1, "unverifiable": 0}
    assert report["issues"][0]["code"] == "catalog.invalid"
    assert "records.json" in report["issues"][0]["reason"]


def test_verifier_marks_records_from_another_reader_build_unverifiable(tmp_path: Path) -> None:
    store, _record = _write_record(tmp_path)
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    payload = catalog["latest"]["ingest/df"]
    different_source_digest = "sha256:" + ("0" * 64)
    assert different_source_digest != current_build_identity().source_digest
    payload["build_identity"]["source_digest"] = different_source_digest
    payload["code_digest"] = different_source_digest
    catalog["history"]["ingest/df"][-1] = dict(payload)
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "unverifiable"
    assert report["summary"] == {"checked": 1, "failed": 0, "unverifiable": 1}
    assert report["records"][0]["issues"][0]["code"] == "build.identity_mismatch"


@pytest.mark.parametrize("field", ["build_identity", "code_digest"])
def test_verifier_rejects_malformed_build_digests_as_invalid_catalogs(tmp_path: Path, field: str) -> None:
    store, _record = _write_record(tmp_path)
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    payload = catalog["latest"]["ingest/df"]
    if field == "build_identity":
        payload["build_identity"]["source_digest"] = "sha256:"
    else:
        payload["code_digest"] = "sha256:"
    catalog["history"]["ingest/df"][-1] = dict(payload)
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"] == {"checked": 0, "failed": 1, "unverifiable": 0}
    assert report["issues"][0]["code"] == "catalog.invalid"
    assert "sha256" in report["issues"][0]["reason"]


def test_verifier_scopes_records_to_the_current_workbench_declaration(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    for record_id, config_digest in (
        ("current/df", "sha256:current-config"),
        ("retired/df", "sha256:retired-config"),
    ):
        store.persist_dataframe(
            producer_id=record_id.split("/", 1)[0],
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id=record_id,
            df=_tidy_df(),
            contract_id="tidy.v1",
            inputs=[],
            config_digest=config_digest,
            producer_config_digest="sha256:producer-config",
        )

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:current-config",
        scope=RecordVerificationScope(record_ids=frozenset({"current/df"})),
    )

    assert report["status"] == "ok"
    assert report["summary"] == {"checked": 1, "failed": 0, "unverifiable": 0}
    assert [record["record_id"] for record in report["records"]] == ["current/df"]


def test_verifier_rejects_retired_record_schemas_as_invalid_catalogs(tmp_path: Path) -> None:
    store, _record = _write_record(tmp_path)
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    current = catalog["latest"]["ingest/df"]
    retired = {
        "schema_version": 3,
        "record_id": current["record_id"],
        "kind": current["kind"],
        "producer": current["producer"],
        "created_at": current["created_at"],
        "inputs": [],
        "config_digest": current["config_digest"],
        "contract_id": current["contract_id"],
        "path": current["path"],
        "content_digest": current["content_digest"],
        "code_digest": "",
    }
    catalog["latest"]["ingest/df"] = retired
    catalog["history"]["ingest/df"] = [retired]
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"] == {"checked": 0, "failed": 1, "unverifiable": 0}
    assert report["issues"][0]["code"] == "catalog.invalid"
    assert "schema_version must be 5" in report["issues"][0]["reason"]


def test_verifier_reports_corrupt_dataframe_bytes(tmp_path: Path) -> None:
    store, record = _write_record(tmp_path)
    record.path.write_bytes(record.path.read_bytes() + b"corrupt")

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"] == {"checked": 1, "failed": 1, "unverifiable": 0}
    assert report["records"][0]["issues"][0]["code"] == "artifact.size_mismatch"


def test_verifier_reports_artifact_io_failures_as_structured_issues(tmp_path: Path, monkeypatch) -> None:
    store, _record = _write_record(tmp_path)

    def _unreadable(_path: Path) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr("reader.workbench.records.verification.sha256_file", _unreadable)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"] == {"checked": 1, "failed": 1, "unverifiable": 0}
    assert report["records"][0]["issues"][0]["code"] == "artifact.io_error"
    assert "permission denied" in report["records"][0]["issues"][0]["reason"]


def test_verifier_detects_source_file_drift_and_records_discovery_policy(tmp_path: Path) -> None:
    raw = tmp_path / "inputs" / "raw.xlsx"
    raw.parent.mkdir()
    raw.write_bytes(b"original")
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=store.capture_inputs(
            [
                ProvenanceInput(
                    label="raw",
                    ref=FileRef(path=raw),
                    discovery_policy="plugin_discovery",
                )
            ]
        ),
        config_digest="sha256:experiment-config",
        producer_config_digest="sha256:producer-config",
    )
    payload = json.loads(store.records_path.read_text(encoding="utf-8"))["latest"]["ingest/df"]

    assert payload["inputs"][0]["discovery_policy"] == "plugin_discovery"
    assert payload["inputs"][0]["artifact"]["path"] == "inputs/raw.xlsx"

    raw.write_bytes(b"changed-source")
    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["records"][0]["issues"][0]["code"] == "input.size_mismatch"


def test_verifier_rejects_source_symlink_escape(tmp_path: Path) -> None:
    raw = tmp_path / "inputs" / "raw.xlsx"
    raw.parent.mkdir()
    raw.write_bytes(b"original")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.xlsx"
    outside.write_bytes(b"original")
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=store.capture_inputs([ProvenanceInput(label="raw", ref=FileRef(path=raw))]),
        config_digest="sha256:experiment-config",
        producer_config_digest="sha256:producer-config",
    )
    raw.unlink()
    raw.symlink_to(outside)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["records"][0]["issues"][0]["code"] == "input.outside_root"


def test_verifier_binds_downstream_to_exact_upstream_revision(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:experiment-config",
        producer_config_digest="sha256:ingest-config",
    )
    store.persist_dataframe(
        producer_id="transform",
        producer_plugin="transform/identity",
        out_name="df",
        record_id="transform/df",
        df=_tidy_df(),
        contract_id="tidy.v1",
        inputs=store.capture_inputs([ProvenanceInput(label="df", ref=RecordRef(record_id="ingest/df"))]),
        config_digest="sha256:experiment-config",
        producer_config_digest="sha256:transform-config",
    )

    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=_tidy_df().assign(value=[2.0]),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:experiment-config",
        producer_config_digest="sha256:ingest-config",
    )
    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    by_id = {item["record_id"]: item for item in report["records"]}
    assert by_id["ingest/df"]["status"] == "ok"
    assert by_id["transform/df"]["status"] == "failed"
    assert by_id["transform/df"]["issues"][0]["code"] == "input.record_revision_mismatch"
