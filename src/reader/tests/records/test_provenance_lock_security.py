"""Adversarial coverage for record-catalog locking and reads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from filelock import Timeout

from reader.contracts import builtin_contract_catalog
from reader.errors import RecordError
from reader.workbench.records import RecordStore


def _catalog_payload() -> dict[str, object]:
    return {
        "schema_version": 4,
        "provenance_epoch_id": "2c1e4014-6217-4f10-9241-f4efb748bd75",
        "latest": {},
        "history": {},
    }


def test_provenance_lock_rejects_a_hard_link_without_mutating_its_target(tmp_path: Path) -> None:
    manifests = tmp_path / "outputs" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "records.json").write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    outside = tmp_path / "outside.txt"
    sentinel = b"outside evidence must remain byte-identical\n"
    outside.write_bytes(sentinel)
    try:
        os.link(outside, manifests / ".records.lock")
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
        create=False,
    )

    with pytest.raises(OSError, match="single link"):
        store.provenance_lock.acquire(timeout=0)

    assert outside.read_bytes() == sentinel


def test_catalog_snapshot_normalizes_an_unsafe_lock_without_mutating_its_target(tmp_path: Path) -> None:
    manifests = tmp_path / "outputs" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "records.json").write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    outside = tmp_path / "outside.txt"
    sentinel = b"outside evidence must remain byte-identical\n"
    outside.write_bytes(sentinel)
    try:
        os.link(outside, manifests / ".records.lock")
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
        create=False,
    )

    with pytest.raises(RecordError, match="inspect the record catalog"):
        store.catalog_snapshot()

    assert outside.read_bytes() == sentinel


def test_catalog_snapshot_does_not_mask_a_protected_body_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )

    def _fail_inside_scope() -> dict[str, object]:
        raise NotImplementedError("body failure remains visible")

    monkeypatch.setattr(store, "_read_catalog", _fail_inside_scope)

    with pytest.raises(NotImplementedError, match="body failure remains visible"):
        store.catalog_snapshot()

    assert not store.provenance_lock.is_locked


def test_provenance_lock_is_reentrant_and_contends_across_instances(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    first = RecordStore(outputs, contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    second = RecordStore(outputs, contracts=builtin_contract_catalog(), experiment_root=tmp_path)

    with first.provenance_lock:
        with first.provenance_lock:
            assert first.provenance_lock.is_locked
        assert first.provenance_lock.is_locked
        with pytest.raises(Timeout):
            second.provenance_lock.acquire(timeout=0)

    assert not first.provenance_lock.is_locked


def test_provenance_lock_contends_across_processes(tmp_path: Path) -> None:
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    probe = """
import sys
from filelock import Timeout
from reader.workbench.records.locking import ProvenanceFileLock

lock = ProvenanceFileLock(sys.argv[1], timeout=0)
try:
    lock.acquire()
except Timeout:
    raise SystemExit(0)
else:
    lock.release()
    raise SystemExit(1)
"""

    with store.provenance_lock:
        result = subprocess.run(
            [sys.executable, "-c", probe, str(store.manifests_dir / ".records.lock")],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

    assert result.returncode == 0, result.stderr


def test_provenance_lock_rejects_a_symlink_without_mutating_its_target(tmp_path: Path) -> None:
    manifests = tmp_path / "outputs" / "manifests"
    manifests.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    sentinel = b"outside evidence must remain byte-identical\n"
    outside.write_bytes(sentinel)
    try:
        (manifests / ".records.lock").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
        create=False,
    )

    with pytest.raises(OSError):
        store.provenance_lock.acquire(timeout=0)

    assert outside.read_bytes() == sentinel


def test_provenance_lock_rejects_a_fifo_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs unavailable")
    manifests = tmp_path / "outputs" / "manifests"
    manifests.mkdir(parents=True)
    os.mkfifo(manifests / ".records.lock")
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
        create=False,
    )

    with pytest.raises(OSError, match="regular file"):
        store.provenance_lock.acquire(timeout=0)


def test_records_catalog_rejects_a_hard_link(tmp_path: Path) -> None:
    manifests = tmp_path / "outputs" / "manifests"
    manifests.mkdir(parents=True)
    outside = tmp_path / "outside-records.json"
    sentinel = json.dumps(_catalog_payload()).encode()
    outside.write_bytes(sentinel)
    try:
        os.link(outside, manifests / "records.json")
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
        create=False,
    )

    with pytest.raises(RecordError, match="single link"):
        store.provenance_epoch_id()

    assert outside.read_bytes() == sentinel


def test_records_catalog_rejects_a_fifo_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs unavailable")
    manifests = tmp_path / "outputs" / "manifests"
    manifests.mkdir(parents=True)
    os.mkfifo(manifests / "records.json")
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
        create=False,
    )

    with pytest.raises(RecordError, match="regular file"):
        store.provenance_epoch_id()
