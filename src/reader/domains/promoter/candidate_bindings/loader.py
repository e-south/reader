"""Load verified study bindings without importing study implementation code."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .annotations import safe_relative_posix_reference
from .contract import (
    BASERENDER_CONTRACT_ID,
    BASERENDER_CONTRACT_VERSION,
    BINDING_ARTIFACT_ID,
    BINDING_COLUMNS,
    BINDING_RECORD_ID,
    BINDING_SCHEMA_ID,
    BINDING_SCHEMA_VERSION,
    READER_ALIAS_NAMESPACE,
    PromoterCandidateBindings,
)
from .rows import bindings_from_frame

_SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load_promoter_candidate_bindings(source: Path) -> PromoterCandidateBindings:
    """Load the exact ``reader.design_id`` projection from one study bundle."""

    supplied = Path(source).expanduser().resolve()
    manifest_path = supplied if supplied.is_file() else supplied / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Promoter candidate-binding manifest not found: {manifest_path}")
    if manifest_path.name != "manifest.json":
        raise ValueError("Promoter candidate-binding resource must reference manifest.json.")
    root = manifest_path.parent
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), context="binding manifest")
    fields = {
        "schema_id",
        "schema_version",
        "study_id",
        "created_at",
        "record",
        "candidate_table",
        "source_artifacts",
        "baserender_contract",
    }
    if set(manifest) != fields:
        raise ValueError(f"Promoter candidate-binding manifest fields must be exactly {sorted(fields)}.")
    if manifest["schema_id"] != BINDING_SCHEMA_ID or str(manifest["schema_version"]) != BINDING_SCHEMA_VERSION:
        raise ValueError(
            f"Promoter candidate bindings must use {BINDING_SCHEMA_ID!r} at version {BINDING_SCHEMA_VERSION!r}."
        )
    study_id = _nonempty(manifest["study_id"], context="binding manifest.study_id")
    _created_at(manifest["created_at"])
    _verify_baserender_contract(manifest["baserender_contract"])
    record_id, records_path, records_sha256, expected_rows = _record(manifest["record"], root=root)
    candidate_table_id, candidate_selection_sha256 = _candidate_table(manifest["candidate_table"])
    source_artifacts = _source_artifacts(manifest["source_artifacts"])
    _verify_parquet_metadata(records_path, study_id=study_id)
    frame = pd.read_parquet(records_path)
    if tuple(frame.columns) != BINDING_COLUMNS:
        raise ValueError(f"Promoter candidate-binding columns must be exactly {list(BINDING_COLUMNS)}.")
    if len(frame) != expected_rows:
        raise ValueError(
            f"Promoter candidate-binding row count mismatch: manifest={expected_rows}, observed={len(frame)}."
        )
    duplicates = frame.duplicated(subset=["alias_namespace", "alias"], keep=False)
    if duplicates.any():
        identities = sorted(
            f"{row.alias_namespace}:{row.alias}"
            for row in frame.loc[duplicates, ["alias_namespace", "alias"]].itertuples(index=False)
        )
        raise ValueError(f"Promoter candidate-binding typed aliases must be unique: {identities}.")
    reader_frame = frame.loc[frame["alias_namespace"].astype(str).eq(READER_ALIAS_NAMESPACE)].reset_index(drop=True)
    if reader_frame.empty:
        raise ValueError(f"Promoter candidate bindings contain no {READER_ALIAS_NAMESPACE!r} aliases.")
    rows = bindings_from_frame(reader_frame)
    if any(
        row.candidate_table_id != candidate_table_id or row.candidate_selection_sha256 != candidate_selection_sha256
        for row in rows
    ):
        raise ValueError("Reader binding rows disagree with manifest candidate-table provenance.")
    return PromoterCandidateBindings(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        records_sha256="sha256:" + records_sha256,
        schema_id=BINDING_SCHEMA_ID,
        schema_version=BINDING_SCHEMA_VERSION,
        study_id=study_id,
        record_id=record_id,
        candidate_table_id=candidate_table_id,
        candidate_selection_sha256=candidate_selection_sha256,
        source_artifacts=source_artifacts,
        rows=rows,
    )


def _record(value: object, *, root: Path) -> tuple[str, Path, str, int]:
    record = _exact_mapping(
        value,
        context="binding manifest.record",
        fields={"record_id", "path", "sha256", "row_count"},
    )
    record_id = _nonempty(record["record_id"], context="binding manifest.record.record_id")
    if record_id != BINDING_RECORD_ID or record["path"] != BINDING_ARTIFACT_ID:
        raise ValueError("Promoter candidate-binding record identity or path is unsupported.")
    path = _confined_path(root, str(record["path"]), context="binding record")
    digest = _digest(record["sha256"], context="binding manifest.record.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"Promoter candidate-binding table not found: {path}")
    if sha256_file(path).removeprefix("sha256:") != digest:
        raise ValueError("Promoter candidate-binding table digest mismatch.")
    count = record["row_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Promoter candidate-binding row count must be a positive integer.")
    return record_id, path, digest, count


def _candidate_table(value: object) -> tuple[str, str]:
    table = _exact_mapping(
        value,
        context="binding manifest.candidate_table",
        fields={"dataset_id", "selection_sha256"},
    )
    return (
        _nonempty(table["dataset_id"], context="binding manifest.candidate_table.dataset_id"),
        _digest(table["selection_sha256"], context="binding manifest.candidate_table.selection_sha256"),
    )


def _verify_baserender_contract(value: object) -> None:
    contract = _exact_mapping(
        value,
        context="binding manifest.baserender_contract",
        fields={"contract_id", "contract_version"},
    )
    if contract != {
        "contract_id": BASERENDER_CONTRACT_ID,
        "contract_version": BASERENDER_CONTRACT_VERSION,
    }:
        raise ValueError("Promoter candidate bindings require the supported BaseRender sequence-panel contract.")


def _source_artifacts(value: object) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("binding manifest.source_artifacts must be a non-empty list.")
    result: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        item = _exact_mapping(
            raw,
            context=f"binding manifest.source_artifacts[{index}]",
            fields={"artifact_id", "path", "sha256"},
        )
        result.append(
            {
                "artifact_id": _nonempty(item["artifact_id"], context=f"source_artifacts[{index}].artifact_id"),
                "path": safe_relative_posix_reference(item["path"], context=f"source_artifacts[{index}].path"),
                "sha256": _digest(item["sha256"], context=f"source_artifacts[{index}].sha256"),
            }
        )
    identities = [item["artifact_id"] for item in result]
    if len(identities) != len(set(identities)):
        raise ValueError("binding manifest source artifact identities must be unique.")
    return tuple(result)


def _verify_parquet_metadata(path: Path, *, study_id: str) -> None:
    metadata = pq.read_schema(path).metadata or {}
    expected = {
        b"schema_id": BINDING_SCHEMA_ID.encode(),
        b"schema_version": BINDING_SCHEMA_VERSION.encode(),
        b"study_id": study_id.encode(),
        b"record_id": BINDING_RECORD_ID.encode(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Promoter candidate-binding Parquet metadata disagrees with its contract.")


def _confined_path(root: Path, relative: str, *, context: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{context} escapes its bundle root.") from exc
    return path


def _exact_mapping(value: object, *, context: str, fields: set[str]) -> dict[str, Any]:
    result = _mapping(value, context=context)
    if set(result) != fields:
        raise ValueError(f"{context} fields must be exactly {sorted(fields)}.")
    return result


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _nonempty(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, context: str) -> str:
    text = _nonempty(value, context=context).lower().removeprefix("sha256:")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{context} must be a 64-character hexadecimal SHA-256 digest.")
    return text


def _created_at(value: object) -> None:
    text = _nonempty(value, context="binding manifest.created_at")
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("binding manifest.created_at must be an ISO-8601 timestamp.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("binding manifest.created_at must include a timezone.")


__all__ = ["load_promoter_candidate_bindings", "sha256_file"]
