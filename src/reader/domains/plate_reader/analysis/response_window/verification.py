"""Fail-fast verification for published response-window bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reader.contracts import ContractCatalog

from .display import validate_display_manifest
from .provenance import sha256_file
from .verification_invariants import verify_record_invariants

BUNDLE_SCHEMA_VERSION = "reader.response_window.bundle.v3"
RECORD_CONTRACTS = {
    "wells": "plate_reader.response_window.wells.v2",
    "designs": "plate_reader.response_window.designs.v2",
    "bootstrap_draws": "plate_reader.response_window.bootstrap_draws.v2",
    "traces": "plate_reader.response_window.traces.v2",
    "events": "plate_reader.response_window.events.v2",
}
RECORD_ARTIFACTS = {record_id: f"tables/{record_id}.parquet" for record_id in RECORD_CONTRACTS}
_COUNT_KEYS = {
    "wells": "well_rows",
    "designs": "design_rows",
    "bootstrap_draws": "bootstrap_draw_rows",
    "traces": "trace_rows",
    "events": "experiments",
}
_REQUIRED_COUNT_KEYS = {
    *_COUNT_KEYS.values(),
    "unique_design_ids",
    "repeated_design_ids",
    "reductions",
    "plots",
}


def verify_bundle_payload(
    root: Path,
    *,
    contracts: ContractCatalog,
) -> tuple[Path, dict[str, object], dict[str, int]]:
    """Verify one bundle's manifest, artifacts, schemas, and cross-table semantics."""

    bundle_root = Path(root).expanduser().resolve()
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"response-window manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"response-window bundle must use {BUNDLE_SCHEMA_VERSION!r}.")
    if manifest.get("contracts") != RECORD_CONTRACTS:
        raise ValueError("response-window record contracts disagree with the public bundle contract.")
    if manifest.get("state_order") != ["00", "10", "01", "11"]:
        raise ValueError("response-window bundle must declare state order [00, 10, 01, 11].")
    display = validate_display_manifest(manifest.get("display"))
    expected_records = {
        record_id: {"contract_id": RECORD_CONTRACTS[record_id], "artifact_id": artifact_id}
        for record_id, artifact_id in RECORD_ARTIFACTS.items()
    }
    if manifest.get("records") != expected_records:
        raise ValueError("response-window record artifact map disagrees with the public bundle contract.")
    artifacts = _verify_artifacts(bundle_root, manifest)
    _verify_request_record(manifest, artifacts)
    required = (*RECORD_ARTIFACTS.values(), "request.yaml", "review.py", "tables/plot_manifest.csv")
    for artifact_id in required:
        if artifact_id not in artifacts:
            raise RuntimeError(f"response-window bundle lacks required artifact {artifact_id!r}.")
    counts = _integer_counts(manifest)
    frames = _verify_record_frames(bundle_root, counts, contracts=contracts)
    _verify_source_records(manifest, counts, artifacts)
    verify_record_invariants(
        root=bundle_root,
        manifest=manifest,
        artifacts=artifacts,
        counts=counts,
        display=display,
        frames=frames,
    )
    return manifest_path, manifest, counts


def _verify_request_record(manifest: dict[str, object], artifacts: dict[str, dict[str, object]]) -> None:
    request = manifest.get("request")
    if not isinstance(request, dict):
        raise ValueError("response-window bundle lacks request provenance.")
    artifact_id = request.get("artifact_id")
    if artifact_id != "request.yaml":
        raise ValueError("response-window request provenance must reference bundled request.yaml.")
    digest = request.get("sha256")
    if not _is_sha256(digest):
        raise ValueError("response-window request provenance lacks a valid sha256 digest.")
    if artifacts.get("request.yaml", {}).get("sha256") != digest:
        raise ValueError("response-window request digest disagrees with bundled request.yaml.")


def _verify_artifacts(root: Path, manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise RuntimeError("response-window bundle has no artifact manifest.")
    artifacts: dict[str, dict[str, object]] = {}
    for artifact_id, raw in raw_artifacts.items():
        if not isinstance(artifact_id, str) or not isinstance(raw, dict):
            raise RuntimeError("response-window artifact metadata must use string keys and mappings.")
        relative = raw.get("path")
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(f"artifact {artifact_id!r} lacks a path.")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeError(f"artifact {artifact_id!r} lacks a valid byte count.")
        if not _is_sha256(digest):
            raise RuntimeError(f"artifact {artifact_id!r} lacks a valid digest.")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"artifact {artifact_id!r} escapes the bundle root.") from exc
        if relative != artifact_id:
            raise RuntimeError(f"artifact {artifact_id!r} path disagrees with its manifest identity.")
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"artifact {artifact_id!r} is missing or has the wrong size.")
        if sha256_file(path) != digest:
            raise RuntimeError(f"artifact {artifact_id!r} digest mismatch.")
        artifacts[artifact_id] = raw
    return artifacts


def _integer_counts(manifest: dict[str, object]) -> dict[str, int]:
    raw = manifest.get("counts")
    if not isinstance(raw, dict) or set(raw) != _REQUIRED_COUNT_KEYS:
        raise ValueError(f"response-window bundle counts must use exactly {sorted(_REQUIRED_COUNT_KEYS)}.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw.values()):
        raise ValueError("response-window bundle counts must be non-negative integers.")
    return {str(key): int(value) for key, value in raw.items()}


def _verify_record_frames(
    root: Path,
    counts: dict[str, int],
    *,
    contracts: ContractCatalog,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for record_id, artifact_id in RECORD_ARTIFACTS.items():
        frame = pd.read_parquet(root / artifact_id)
        contracts.validate(frame, contract_id=RECORD_CONTRACTS[record_id], where=f"response-window:{record_id}")
        count_key = _COUNT_KEYS[record_id]
        if counts[count_key] != len(frame):
            raise ValueError(
                f"response-window count mismatch for {record_id!r}: "
                f"manifest={counts[count_key]!r}, observed={len(frame)}."
            )
        frames[record_id] = frame
    return frames


def _verify_source_records(
    manifest: dict[str, object],
    counts: dict[str, int],
    artifacts: dict[str, dict[str, object]],
) -> None:
    records = manifest.get("source_records")
    if not isinstance(records, list) or len(records) != counts["experiments"]:
        raise ValueError("response-window source-record provenance must match the experiment count.")
    experiment_ids: list[str] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("experiment_id"), str):
            raise ValueError("response-window source-record provenance is malformed.")
        for field in ("config_artifact", "records_artifact"):
            artifact_id = record.get(field)
            if not isinstance(artifact_id, str) or artifact_id not in artifacts:
                raise ValueError(f"response-window source record lacks bundled {field}.")
        digests = record.get("records")
        if not isinstance(digests, dict) or not digests:
            raise ValueError("response-window source-record provenance lacks record digests.")
        if any(not _is_sha256(value) for value in digests.values()):
            raise ValueError("response-window source-record provenance contains an invalid digest.")
        experiment_ids.append(str(record["experiment_id"]))
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("response-window source-record experiment identities are not unique.")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "RECORD_ARTIFACTS",
    "RECORD_CONTRACTS",
    "verify_bundle_payload",
]
