from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from reader.core.errors import RecordError

from .ontology import WorkbenchProducerKind, WorkbenchRecordKind


@dataclass(frozen=True)
class RecordProducer:
    kind: WorkbenchProducerKind
    id: str
    uses: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id, "uses": self.uses}


@dataclass(frozen=True)
class WorkbenchRecord:
    record_id: str
    kind: WorkbenchRecordKind
    producer: RecordProducer
    created_at: str
    inputs: tuple[str, ...]
    config_digest: str


@dataclass(frozen=True)
class DataFrameArtifactRecord(WorkbenchRecord):
    contract_id: str
    path: Path
    content_digest: str
    code_digest: str = ""

    def __post_init__(self) -> None:
        if self.kind != "dataframe_artifact":
            raise RecordError(f"DataFrameArtifactRecord must use kind 'dataframe_artifact', got {self.kind!r}")

    def load_dataframe(self) -> pd.DataFrame:
        if self.path.suffix.lower() == ".parquet":
            return pd.read_parquet(self.path)
        raise RecordError(f"Record {self.record_id} is not a parquet dataframe: {self.path}")


@dataclass(frozen=True)
class FileBundleRecord(WorkbenchRecord):
    files: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.kind != "file_bundle":
            raise RecordError(f"FileBundleRecord must use kind 'file_bundle', got {self.kind!r}")


def record_paths(record: DataFrameArtifactRecord | FileBundleRecord) -> tuple[Path, ...]:
    if isinstance(record, DataFrameArtifactRecord):
        return (record.path,)
    return record.files


def _decode_path(raw: Any, *, outputs_dir: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise RecordError("record path entries must be non-empty strings")
    path = Path(raw)
    return path if path.is_absolute() else (outputs_dir / path)


def _encode_path(path: Path, *, outputs_dir: Path) -> str:
    try:
        return str(path.relative_to(outputs_dir))
    except ValueError:
        return str(path)


def record_to_dict(record: DataFrameArtifactRecord | FileBundleRecord, *, outputs_dir: Path) -> dict[str, Any]:
    base = {
        "schema_version": 1,
        "record_id": record.record_id,
        "kind": record.kind,
        "producer": record.producer.to_dict(),
        "created_at": record.created_at,
        "inputs": list(record.inputs),
        "config_digest": record.config_digest,
    }
    if isinstance(record, DataFrameArtifactRecord):
        base.update(
            {
                "contract_id": record.contract_id,
                "path": _encode_path(record.path, outputs_dir=outputs_dir),
                "content_digest": record.content_digest,
                "code_digest": record.code_digest,
            }
        )
        return base
    base["files"] = [_encode_path(path, outputs_dir=outputs_dir) for path in record.files]
    return base


def record_from_dict(payload: dict[str, Any], *, outputs_dir: Path) -> DataFrameArtifactRecord | FileBundleRecord:
    if not isinstance(payload, dict):
        raise RecordError("record payload must be a JSON object")
    record_id = payload.get("record_id")
    kind = payload.get("kind")
    producer_payload = payload.get("producer")
    created_at = payload.get("created_at")
    inputs = payload.get("inputs")
    config_digest = payload.get("config_digest")
    if not isinstance(record_id, str) or not record_id:
        raise RecordError("record payload must include non-empty string 'record_id'")
    if kind not in {"dataframe_artifact", "file_bundle"}:
        raise RecordError(f"record {record_id!r} has unknown kind {kind!r}")
    if not isinstance(producer_payload, dict):
        raise RecordError(f"record {record_id!r} must include producer metadata")
    producer_kind = producer_payload.get("kind")
    producer_id = producer_payload.get("id")
    producer_uses = producer_payload.get("uses")
    if producer_kind not in {"pipeline", "plot", "export", "notebook"}:
        raise RecordError(f"record {record_id!r} has invalid producer kind {producer_kind!r}")
    if not isinstance(producer_id, str) or not producer_id:
        raise RecordError(f"record {record_id!r} must include producer.id")
    if not isinstance(producer_uses, str) or not producer_uses:
        raise RecordError(f"record {record_id!r} must include producer.uses")
    if not isinstance(created_at, str) or not created_at:
        raise RecordError(f"record {record_id!r} must include created_at")
    if not isinstance(inputs, list) or any(not isinstance(item, str) for item in inputs):
        raise RecordError(f"record {record_id!r} must include string inputs")
    if not isinstance(config_digest, str) or not config_digest:
        raise RecordError(f"record {record_id!r} must include config_digest")
    producer = RecordProducer(kind=producer_kind, id=producer_id, uses=producer_uses)
    if kind == "dataframe_artifact":
        contract_id = payload.get("contract_id")
        path = payload.get("path")
        content_digest = payload.get("content_digest")
        code_digest = payload.get("code_digest", "")
        if not isinstance(contract_id, str) or not contract_id:
            raise RecordError(f"record {record_id!r} must include contract_id")
        if not isinstance(content_digest, str) or not content_digest:
            raise RecordError(f"record {record_id!r} must include content_digest")
        if not isinstance(code_digest, str):
            raise RecordError(f"record {record_id!r} must include string code_digest")
        return DataFrameArtifactRecord(
            record_id=record_id,
            kind="dataframe_artifact",
            producer=producer,
            created_at=created_at,
            inputs=tuple(inputs),
            config_digest=config_digest,
            contract_id=contract_id,
            path=_decode_path(path, outputs_dir=outputs_dir),
            content_digest=content_digest,
            code_digest=code_digest,
        )
    files = payload.get("files")
    if not isinstance(files, list) or any(not isinstance(item, str) or not item for item in files):
        raise RecordError(f"record {record_id!r} must include non-empty string file paths")
    return FileBundleRecord(
        record_id=record_id,
        kind="file_bundle",
        producer=producer,
        created_at=created_at,
        inputs=tuple(inputs),
        config_digest=config_digest,
        files=tuple(_decode_path(path, outputs_dir=outputs_dir) for path in files),
    )
