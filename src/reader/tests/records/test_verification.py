from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from filelock import Timeout

import reader.workbench.records.verification as verification_module
from reader.contracts import builtin_contract_catalog
from reader.errors import RecordError
from reader.workbench.engine.invocations import InvocationLedger
from reader.workbench.graph import FileRef, ProvenanceInput, RecordRef
from reader.workbench.records import (
    ArtifactEvidence,
    RecordStore,
    RecordVerificationScope,
    record_revision_digest,
    verify_record_store,
)
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


def _write_valid_invocation(tmp_path: Path):
    store, record = _write_record(tmp_path)
    ledger = InvocationLedger.for_store(store=store)
    attempt = ledger.append_attempt(
        config_digest="sha256:" + ("a" * 64),
        build_identity=current_build_identity(),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    ledger.append_result(
        attempt,
        exit_status=0,
        produced_record_revisions=[
            {
                "record_id": record.record_id,
                "revision": 1,
                "revision_digest": record_revision_digest(record, outputs_dir=store.root),
            }
        ],
    )
    events = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    return store, ledger.path, events


def _rewrite_invocations(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


class _SnapshotLock:
    def __init__(self, *, failure: BaseException, failure_entry: int) -> None:
        self._failure = failure
        self._failure_entry = failure_entry
        self.entries = 0
        self.depth = 0

    def acquire(self):
        self.entries += 1
        if self.entries == self._failure_entry:
            raise self._failure
        self.depth += 1
        return self

    def release(self) -> None:
        self.depth -= 1

    def __enter__(self):
        return self.acquire()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.release()


@pytest.mark.parametrize(
    "lock_failure",
    [Timeout("provenance lease"), OSError("lock unavailable"), NotImplementedError("lock unsupported")],
)
def test_verifier_reports_start_snapshot_lock_failure_as_retryable_concurrent_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_failure: BaseException,
) -> None:
    store, _ledger_path, _events = _write_valid_invocation(tmp_path)
    lock = _SnapshotLock(failure=lock_failure, failure_entry=1)
    monkeypatch.setattr(RecordStore, "provenance_lock", property(lambda _store: lock))

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"] == {
        "checked": 0,
        "failed": 0,
        "unverifiable": 0,
        "invocations_checked": 0,
        "invocation_failures": 1,
    }
    assert report["records"] == []
    assert report["issues"] == [
        {
            "code": "verification.concurrent_change",
            "field": "outputs/manifests",
            "reason": "Reader could not establish a stable provenance snapshot at the start of verification.",
            "remediation": "Retry verification after the active writer finishes.",
            "retryable": True,
        }
    ]


@pytest.mark.parametrize(
    "lock_failure",
    [Timeout("provenance lease"), OSError("lock unavailable"), NotImplementedError("lock unsupported")],
)
def test_verifier_reports_end_snapshot_lock_failure_as_retryable_concurrent_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_failure: BaseException,
) -> None:
    store, _ledger_path, _events = _write_valid_invocation(tmp_path)
    lock = _SnapshotLock(failure=lock_failure, failure_entry=2)
    monkeypatch.setattr(RecordStore, "provenance_lock", property(lambda _store: lock))

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"] == {
        "checked": 1,
        "failed": 0,
        "unverifiable": 0,
        "invocations_checked": 1,
        "invocation_failures": 1,
    }
    assert report["records"][0]["status"] == "ok"
    assert report["issues"] == [
        {
            "code": "verification.concurrent_change",
            "field": "outputs/manifests",
            "reason": "Reader could not confirm a stable provenance snapshot at the end of verification.",
            "remediation": "Retry verification after the active writer finishes.",
            "retryable": True,
        }
    ]


def test_verifier_binds_and_reads_the_epoch_inside_the_snapshot_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ledger_path, _events = _write_valid_invocation(tmp_path)
    lock = _SnapshotLock(failure=AssertionError("unused"), failure_entry=99)
    original_provenance_epoch_id = store.provenance_epoch_id
    observed_depths: list[int] = []

    def _guarded_provenance_epoch_id() -> str:
        observed_depths.append(lock.depth)
        if lock.depth < 1:
            raise AssertionError("verification read the provenance epoch before acquiring its snapshot lease")
        return original_provenance_epoch_id()

    monkeypatch.setattr(RecordStore, "provenance_lock", property(lambda _store: lock))
    monkeypatch.setattr(store, "provenance_epoch_id", _guarded_provenance_epoch_id)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "ok"
    assert observed_depths
    assert all(depth >= 1 for depth in observed_depths)


def test_verifier_reports_an_epoch_reset_after_the_start_snapshot_as_concurrent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ledger_path, _events = _write_valid_invocation(tmp_path)
    concurrent_store = RecordStore(
        store.root,
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    original_verify_ledger = verification_module._verify_invocation_ledger

    def _reset_then_verify_ledger(*args, **kwargs):
        concurrent_store.reset_catalog()
        return original_verify_ledger(*args, **kwargs)

    monkeypatch.setattr(verification_module, "_verify_invocation_ledger", _reset_then_verify_ledger)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"]["invocation_failures"] == 1
    assert report["issues"] == [
        {
            "code": "verification.concurrent_change",
            "field": "outputs/manifests",
            "reason": "The active provenance epoch changed during verification.",
            "remediation": "Retry verification after the active writer finishes.",
            "retryable": True,
        }
    ]


def test_verifier_never_opens_a_ledger_through_a_symlinked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, ledger_path, _events = _write_valid_invocation(tmp_path)
    external_dir = tmp_path / "external-ledgers"
    external_dir.mkdir()
    external_target = external_dir / ledger_path.name
    external_target.write_bytes(b"external ledger contents must not be read")
    ledger_path.unlink()
    ledger_path.parent.rmdir()
    ledger_path.parent.symlink_to(external_dir, target_is_directory=True)

    opened_external_target: list[str] = []
    original_path_open = Path.open
    original_os_open = os.open

    def _guard_path_open(path: Path, *args, **kwargs):
        if path == ledger_path:
            opened_external_target.append("Path.open")
            raise AssertionError("external ledger target was opened")
        return original_path_open(path, *args, **kwargs)

    def _guard_os_open(path, *args, **kwargs):
        if Path(path) == ledger_path:
            opened_external_target.append("os.open")
            raise AssertionError("external ledger target was opened")
        return original_os_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _guard_path_open)
    monkeypatch.setattr(os, "open", _guard_os_open)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert "invocation.ledger_unconfined" in {issue["code"] for issue in report["issues"]}
    assert opened_external_target == []


def test_verifier_rejects_a_hard_linked_ledger_before_reading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, ledger_path, _events = _write_valid_invocation(tmp_path)
    external_target = tmp_path / "external-ledger.jsonl"
    external_target.write_bytes(b"external ledger contents must not be read")
    ledger_path.unlink()
    os.link(external_target, ledger_path)
    external_identity = (external_target.stat().st_dev, external_target.stat().st_ino)

    read_external_target: list[str] = []
    original_fdopen = os.fdopen

    def _guard_fdopen(descriptor: int, *args, **kwargs):
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == external_identity:
            read_external_target.append("os.fdopen")
            raise AssertionError("hard-linked ledger contents were read")
        return original_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", _guard_fdopen)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert "invocation.ledger_unconfined" in {issue["code"] for issue in report["issues"]}
    assert read_external_target == []


def test_new_records_are_schema_v5_and_verify_without_loading_dataframe(tmp_path: Path) -> None:
    store, _ledger_path, _events = _write_valid_invocation(tmp_path)
    record = store.read_record("ingest/df")

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
    assert report["summary"] == {
        "checked": 1,
        "failed": 0,
        "unverifiable": 0,
        "invocations_checked": 1,
        "invocation_failures": 0,
    }


def test_verifier_accepts_writer_invocation_contract_and_catalog_revision(tmp_path: Path) -> None:
    store, _ledger_path, _events = _write_valid_invocation(tmp_path)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "ok"
    assert report["summary"]["invocations_checked"] == 1
    assert report["summary"]["invocation_failures"] == 0


def test_verifier_accepts_consistent_failed_invocation_result(tmp_path: Path) -> None:
    store, record = _write_record(tmp_path)
    ledger = InvocationLedger.for_store(store=store)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=current_build_identity(),
        operation="run",
        selected_step_ids={"pipeline": ["ingest"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    ledger.append_result(
        attempt,
        exit_status=1,
        produced_record_revisions=[
            {
                "record_id": record.record_id,
                "revision": 1,
                "revision_digest": record_revision_digest(record, outputs_dir=store.root),
            }
        ],
        failure=RuntimeError("synthetic failure"),
    )

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "ok"
    assert report["summary"]["invocation_failures"] == 0


def test_verifier_rejects_result_before_attempt(tmp_path: Path) -> None:
    store, ledger_path, events = _write_valid_invocation(tmp_path)
    _rewrite_invocations(ledger_path, list(reversed(events)))

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert "invocation.order_invalid" in {issue["code"] for issue in report["issues"]}


def test_verifier_rejects_malformed_invocation_fields(tmp_path: Path) -> None:
    store, ledger_path, events = _write_valid_invocation(tmp_path)
    events[0]["unexpected"] = True
    _rewrite_invocations(ledger_path, events)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert "invocation.event_invalid" in {issue["code"] for issue in report["issues"]}


def test_verifier_rejects_invocation_from_a_different_provenance_epoch(tmp_path: Path) -> None:
    store, ledger_path, events = _write_valid_invocation(tmp_path)
    events[0]["provenance_epoch_id"] = str(uuid4())
    events[1]["provenance_epoch_id"] = events[0]["provenance_epoch_id"]
    _rewrite_invocations(ledger_path, events)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert "invocation.event_invalid" in {issue["code"] for issue in report["issues"]}
    assert any("different provenance epoch" in issue["reason"] for issue in report["issues"])


def test_verifier_rejects_unclaimed_catalog_revisions(tmp_path: Path) -> None:
    store, _record = _write_record(tmp_path)
    ledger = InvocationLedger.for_store(store=store)
    attempt = ledger.append_attempt(
        config_digest="sha256:config",
        build_identity=current_build_identity(),
        operation="run",
        selected_step_ids={"pipeline": [], "plots": [], "exports": []},
        declared_inputs=[],
    )
    ledger.append_result(
        attempt,
        exit_status=1,
        produced_record_revisions=[],
        failure=RuntimeError("unrelated failure"),
    )

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert "invocation.revision_unclaimed" in {issue["code"] for issue in report["issues"]}


def test_verifier_rejects_an_empty_active_ledger_for_a_nonempty_catalog(tmp_path: Path) -> None:
    store, _record = _write_record(tmp_path)
    ledger_path = store.invocation_ledger_path()
    ledger_path.parent.mkdir(parents=True)
    ledger_path.touch()

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert "invocation.ledger_empty" in {issue["code"] for issue in report["issues"]}


def test_verifier_ignores_corrupt_inactive_epoch_ledgers(tmp_path: Path) -> None:
    store, _ledger_path, _events = _write_valid_invocation(tmp_path)
    (store.manifests_dir / "invocations" / f"{uuid4()}.jsonl").write_text("not json\n", encoding="utf-8")

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "ok"


def test_verifier_fails_if_provenance_changes_during_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ledger_path, _events = _write_valid_invocation(tmp_path)
    concurrent_store = RecordStore(
        store.root,
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    original_iter = store.iter_latest_records
    injected = False

    def _inject_concurrent_record(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            record = concurrent_store.persist_dataframe(
                producer_id="concurrent",
                producer_plugin="ingest/synergy_h1",
                out_name="df",
                record_id="concurrent/df",
                df=_tidy_df(),
                contract_id="tidy.v1",
                inputs=[],
                config_digest="sha256:experiment-config",
                producer_config_digest="sha256:producer-config",
            )
            ledger = InvocationLedger.for_store(store=concurrent_store)
            attempt = ledger.append_attempt(
                config_digest="sha256:experiment-config",
                build_identity=current_build_identity(),
                operation="run",
                selected_step_ids={"pipeline": ["concurrent"], "plots": [], "exports": []},
                declared_inputs=[],
            )
            ledger.append_result(
                attempt,
                exit_status=0,
                produced_record_revisions=[
                    {
                        "record_id": record.record_id,
                        "revision": 1,
                        "revision_digest": record_revision_digest(record, outputs_dir=concurrent_store.root),
                    }
                ],
            )
        return original_iter(*args, **kwargs)

    monkeypatch.setattr(store, "iter_latest_records", _inject_concurrent_record)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert "verification.concurrent_change" in {issue["code"] for issue in report["issues"]}


def test_verifier_rejects_changed_invocation_identity(tmp_path: Path) -> None:
    store, ledger_path, events = _write_valid_invocation(tmp_path)
    events[1]["config_digest"] = "sha256:" + ("b" * 64)
    _rewrite_invocations(ledger_path, events)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert "invocation.identity_mismatch" in {issue["code"] for issue in report["issues"]}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "failed"),
        ("exit_status", 1),
        ("failure", {"type": "RuntimeError", "reason": "contradicts success"}),
    ],
)
def test_verifier_rejects_inconsistent_invocation_result_state(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    store, ledger_path, events = _write_valid_invocation(tmp_path)
    events[1][field] = value
    _rewrite_invocations(ledger_path, events)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert "invocation.event_invalid" in {issue["code"] for issue in report["issues"]}


def test_verifier_rejects_malformed_produced_revision(tmp_path: Path) -> None:
    store, ledger_path, events = _write_valid_invocation(tmp_path)
    events[1]["produced_record_revisions"][0]["revision"] = 0
    _rewrite_invocations(ledger_path, events)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert "invocation.event_invalid" in {issue["code"] for issue in report["issues"]}


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("revision", 2, "invocation.revision_missing"),
        ("revision_digest", "sha256:" + ("b" * 64), "invocation.revision_mismatch"),
    ],
)
def test_verifier_reconciles_produced_revisions_with_catalog_history(
    tmp_path: Path,
    field: str,
    value: object,
    expected_code: str,
) -> None:
    store, ledger_path, events = _write_valid_invocation(tmp_path)
    events[1]["produced_record_revisions"][0][field] = value
    _rewrite_invocations(ledger_path, events)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert expected_code in {issue["code"] for issue in report["issues"]}


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
    assert report["summary"] == {
        "checked": 0,
        "failed": 1,
        "unverifiable": 0,
        "invocations_checked": 0,
        "invocation_failures": 0,
    }
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
    assert report["summary"] == {
        "checked": 0,
        "failed": 1,
        "unverifiable": 0,
        "invocations_checked": 0,
        "invocation_failures": 0,
    }
    assert report["issues"][0]["code"] == "catalog.invalid"
    assert "records.json" in report["issues"][0]["reason"]


def test_verifier_marks_records_from_another_reader_build_unverifiable(tmp_path: Path) -> None:
    store, ledger_path, events = _write_valid_invocation(tmp_path)
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    payload = catalog["latest"]["ingest/df"]
    different_source_digest = "sha256:" + ("0" * 64)
    assert different_source_digest != current_build_identity().source_digest
    payload["build_identity"]["source_digest"] = different_source_digest
    payload["code_digest"] = different_source_digest
    catalog["history"]["ingest/df"][-1] = dict(payload)
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")
    current = store.read_record("ingest/df")
    events[1]["produced_record_revisions"][0]["revision_digest"] = record_revision_digest(
        current,
        outputs_dir=store.root,
    )
    _rewrite_invocations(ledger_path, events)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "unverifiable"
    assert report["summary"] == {
        "checked": 1,
        "failed": 0,
        "unverifiable": 1,
        "invocations_checked": 1,
        "invocation_failures": 0,
    }
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
    assert report["summary"] == {
        "checked": 0,
        "failed": 1,
        "unverifiable": 0,
        "invocations_checked": 0,
        "invocation_failures": 0,
    }
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
    ledger = InvocationLedger.for_store(store=store)
    attempt = ledger.append_attempt(
        config_digest="sha256:current-config",
        build_identity=current_build_identity(),
        operation="run",
        selected_step_ids={"pipeline": ["current", "retired"], "plots": [], "exports": []},
        declared_inputs=[],
    )
    revisions = []
    for record in store.iter_latest_records():
        revisions.append(
            {
                "record_id": record.record_id,
                "revision": 1,
                "revision_digest": record_revision_digest(record, outputs_dir=store.root),
            }
        )
    ledger.append_result(attempt, exit_status=0, produced_record_revisions=revisions)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:current-config",
        scope=RecordVerificationScope(record_ids=frozenset({"current/df"})),
    )

    assert report["status"] == "ok"
    assert report["summary"] == {
        "checked": 1,
        "failed": 0,
        "unverifiable": 0,
        "invocations_checked": 1,
        "invocation_failures": 0,
    }
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
    assert report["summary"] == {
        "checked": 0,
        "failed": 1,
        "unverifiable": 0,
        "invocations_checked": 0,
        "invocation_failures": 0,
    }
    assert report["issues"][0]["code"] == "catalog.invalid"
    assert "schema_version must be 5" in report["issues"][0]["reason"]


def test_verifier_reports_corrupt_dataframe_bytes(tmp_path: Path) -> None:
    store, _ledger_path, _events = _write_valid_invocation(tmp_path)
    record = store.read_record("ingest/df")
    record.path.write_bytes(record.path.read_bytes() + b"corrupt")

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"] == {
        "checked": 1,
        "failed": 1,
        "unverifiable": 0,
        "invocations_checked": 1,
        "invocation_failures": 0,
    }
    assert report["records"][0]["issues"][0]["code"] == "artifact.size_mismatch"


def test_verifier_reports_artifact_io_failures_as_structured_issues(tmp_path: Path, monkeypatch) -> None:
    store, _ledger_path, _events = _write_valid_invocation(tmp_path)

    def _unreadable(_path: Path) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr("reader.workbench.records.verification.sha256_file", _unreadable)

    report = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest="sha256:experiment-config",
    )

    assert report["status"] == "failed"
    assert report["summary"] == {
        "checked": 1,
        "failed": 1,
        "unverifiable": 0,
        "invocations_checked": 1,
        "invocation_failures": 0,
    }
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
