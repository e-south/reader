from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from reader.errors import RecordError
from reader.workbench.ontology import WorkbenchProducerKind, WorkbenchRecordKind

from .evidence import ArtifactEvidence, RecordInputEvidence
from .identity import BuildIdentity, digest_json

RECORD_SCHEMA_VERSION = 5
_V5_BASE_FIELDS = {
    "schema_version",
    "record_id",
    "kind",
    "producer",
    "created_at",
    "inputs",
    "config_digest",
}
_V5_DATAFRAME_FIELDS = _V5_BASE_FIELDS | {
    "contract_id",
    "path",
    "content_digest",
    "code_digest",
    "producer_config_digest",
    "build_identity",
    "size_bytes",
}
_V5_FILE_BUNDLE_REQUIRED_FIELDS = _V5_BASE_FIELDS | {
    "files",
    "description",
    "file_evidence",
    "producer_config_digest",
    "build_identity",
}
_V5_FILE_BUNDLE_FIELDS = _V5_FILE_BUNDLE_REQUIRED_FIELDS | {"path_descriptions"}


@dataclass(frozen=True)
class RecordRecipeSource:
    recipe: str
    with_: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"recipe": self.recipe, "with": dict(self.with_ or {})}


@dataclass(frozen=True)
class RecordProducer:
    kind: WorkbenchProducerKind
    id: str
    plugin: str | None = None
    template: str | None = None
    source_recipe: RecordRecipeSource | None = None

    def __post_init__(self) -> None:
        if self.kind == "notebook":
            if not isinstance(self.template, str) or not self.template:
                raise RecordError("notebook producers must include a non-empty template id")
            if self.plugin is not None:
                raise RecordError("notebook producers must not include a plugin id")
            if self.source_recipe is not None:
                raise RecordError("notebook producers must not include recipe provenance")
            return
        if not isinstance(self.plugin, str) or not self.plugin:
            raise RecordError("pipeline/plot/export producers must include a non-empty plugin id")
        if self.template is not None:
            raise RecordError("pipeline/plot/export producers must not include a template id")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "id": self.id}
        if self.kind == "notebook":
            payload["template"] = self.template or ""
        else:
            payload["plugin"] = self.plugin or ""
        if self.source_recipe is not None:
            payload["source_recipe"] = self.source_recipe.to_dict()
        return payload


@dataclass(frozen=True)
class WorkbenchRecord:
    record_id: str
    kind: WorkbenchRecordKind
    producer: RecordProducer
    created_at: str
    inputs: tuple[RecordInputEvidence, ...]
    config_digest: str

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        if any(not isinstance(item, RecordInputEvidence) for item in inputs):
            raise RecordError("record inputs must contain immutable RecordInputEvidence values")
        object.__setattr__(self, "inputs", inputs)


@dataclass(frozen=True)
class DataFrameArtifactRecord(WorkbenchRecord):
    contract_id: str
    path: Path
    content_digest: str
    code_digest: str = ""
    producer_config_digest: str = ""
    build_identity: BuildIdentity | None = None
    size_bytes: int | None = None
    schema_version: int = RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "dataframe_artifact":
            raise RecordError(f"DataFrameArtifactRecord must use kind 'dataframe_artifact', got {self.kind!r}")
        if self.schema_version != RECORD_SCHEMA_VERSION:
            raise RecordError(
                f"DataFrameArtifactRecord schema_version must be {RECORD_SCHEMA_VERSION} (got {self.schema_version!r})"
            )
        if not self.producer_config_digest:
            raise RecordError("schema-v5 dataframe records require producer_config_digest")
        if self.build_identity is None:
            raise RecordError("schema-v5 dataframe records require build_identity")
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise RecordError("schema-v5 dataframe records require non-negative size_bytes")

    def load_dataframe(self) -> pd.DataFrame:
        self.verify_content_digest()
        if self.path.suffix.lower() == ".parquet":
            return pd.read_parquet(self.path)
        raise RecordError(f"Record {self.record_id} is not a parquet dataframe: {self.path}")

    def verify_content_digest(self) -> None:
        if not self.path.exists():
            raise RecordError(f"Record {self.record_id} dataframe artifact is missing: {self.path}")
        actual = sha256_file(self.path)
        if actual != self.content_digest:
            raise RecordError(
                f"Record {self.record_id} content digest mismatch for {self.path}: "
                f"expected {self.content_digest}, got {actual}"
            )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(chunk_size), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _normalized_description(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{where} must be a non-empty string")
    normalized = value.strip()
    if "\n" in normalized or "\r" in normalized:
        raise RecordError(f"{where} must be a single line")
    return normalized


@dataclass(frozen=True)
class PathDescription:
    path: Path
    description: str

    def __post_init__(self) -> None:
        try:
            path = Path(self.path)
        except (TypeError, ValueError) as exc:
            raise RecordError("PathDescription path must be path-like") from exc
        if not str(path).strip() or str(path) == ".":
            raise RecordError("PathDescription path must be a non-empty path")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "description",
            _normalized_description(self.description, where="PathDescription description"),
        )


def normalize_file_bundle_metadata(
    *,
    producer_kind: WorkbenchProducerKind,
    files: tuple[Path, ...],
    description: str | None,
    path_descriptions: tuple[PathDescription, ...],
) -> tuple[str, tuple[PathDescription, ...]]:
    """Validate and normalize the non-filesystem portion of a file-bundle contract."""

    if not files:
        raise RecordError("FileBundleRecord must contain at least one file")
    normalized_description = _normalized_description(description, where="FileBundleRecord description")
    try:
        descriptions = tuple(path_descriptions)
    except TypeError as exc:
        raise RecordError("FileBundleRecord path description entries must be PathDescription values") from exc
    if any(not isinstance(item, PathDescription) for item in descriptions):
        raise RecordError("FileBundleRecord path description entries must be PathDescription values")
    if producer_kind == "plot" and not descriptions:
        raise RecordError("plot file bundles must describe every file")

    described_paths = [item.path for item in descriptions]
    duplicate_paths = sorted(
        (path for path, count in Counter(described_paths).items() if count > 1),
        key=str,
    )
    if duplicate_paths:
        raise RecordError(
            "FileBundleRecord has duplicate path descriptions: " + ", ".join(str(path) for path in duplicate_paths)
        )
    file_paths = set(files)
    unmatched = sorted(set(described_paths) - file_paths, key=str)
    if unmatched:
        raise RecordError(
            "FileBundleRecord has unmatched path descriptions: " + ", ".join(str(path) for path in unmatched)
        )
    if described_paths:
        missing = sorted(file_paths - set(described_paths), key=str)
        if missing:
            raise RecordError(
                "FileBundleRecord has missing path descriptions: " + ", ".join(str(path) for path in missing)
            )
        descriptions_by_path = {item.path: item for item in descriptions}
        descriptions = tuple(descriptions_by_path[path] for path in files)
    return normalized_description, descriptions


@dataclass(frozen=True)
class FileBundleRecord(WorkbenchRecord):
    files: tuple[Path, ...]
    description: str | None
    path_descriptions: tuple[PathDescription, ...] = ()
    file_evidence: tuple[ArtifactEvidence, ...] = ()
    producer_config_digest: str = ""
    build_identity: BuildIdentity | None = None
    schema_version: int = RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "file_bundle":
            raise RecordError(f"FileBundleRecord must use kind 'file_bundle', got {self.kind!r}")
        if self.schema_version != RECORD_SCHEMA_VERSION:
            raise RecordError(
                f"FileBundleRecord schema_version must be {RECORD_SCHEMA_VERSION} (got {self.schema_version!r})"
            )
        description, path_descriptions = normalize_file_bundle_metadata(
            producer_kind=self.producer.kind,
            files=self.files,
            description=self.description,
            path_descriptions=self.path_descriptions,
        )
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "path_descriptions", path_descriptions)
        if not self.producer_config_digest:
            raise RecordError("schema-v5 file-bundle records require producer_config_digest")
        if self.build_identity is None:
            raise RecordError("schema-v5 file-bundle records require build_identity")
        evidence = tuple(self.file_evidence)
        if len(evidence) != len(self.files):
            raise RecordError("schema-v5 file-bundle records must include evidence for every file")
        evidence_paths = [item.relative_path for item in evidence]
        if len(set(evidence_paths)) != len(evidence_paths):
            raise RecordError("schema-v5 file-bundle records contain duplicate evidence paths")
        object.__setattr__(self, "file_evidence", evidence)

    def description_for(self, path: Path) -> str | None:
        requested = Path(path)
        for item in self.path_descriptions:
            if item.path == requested:
                return item.description
        return None


def record_paths(record: DataFrameArtifactRecord | FileBundleRecord) -> tuple[Path, ...]:
    if isinstance(record, DataFrameArtifactRecord):
        return (record.path,)
    return record.files


def _path_within_outputs(path: Path, *, outputs_dir: Path) -> tuple[Path, Path]:
    try:
        outputs_root = outputs_dir.resolve(strict=False)
        candidate = path if path.is_absolute() else outputs_root / path
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise RecordError(f"record path {path!s} must resolve within the outputs directory") from exc
    except (OSError, RuntimeError) as exc:
        raise RecordError(f"record path {path!s} could not be resolved safely: {exc}") from exc
    if relative == Path("."):
        raise RecordError("record paths must identify files below the outputs directory")
    return resolved, relative


def _decode_path(raw: Any, *, outputs_dir: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise RecordError("record path entries must be non-empty strings")
    path = Path(raw)
    if path.is_absolute():
        raise RecordError("record path entries must be relative to the outputs directory")
    resolved, _relative = _path_within_outputs(path, outputs_dir=outputs_dir)
    return resolved


def _encode_path(path: Path, *, outputs_dir: Path) -> str:
    _resolved, relative = _path_within_outputs(Path(path), outputs_dir=outputs_dir)
    return str(relative)


def record_to_dict(record: DataFrameArtifactRecord | FileBundleRecord, *, outputs_dir: Path) -> dict[str, Any]:
    base = {
        "schema_version": record.schema_version,
        "record_id": record.record_id,
        "kind": record.kind,
        "producer": record.producer.to_dict(),
        "created_at": record.created_at,
        "inputs": [item.to_dict() for item in record.inputs],
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
        base.update(
            {
                "producer_config_digest": record.producer_config_digest,
                "build_identity": record.build_identity.to_dict() if record.build_identity else None,
                "size_bytes": record.size_bytes,
            }
        )
        return base
    base["files"] = [_encode_path(path, outputs_dir=outputs_dir) for path in record.files]
    base["description"] = record.description
    if record.path_descriptions:
        base["path_descriptions"] = [
            {
                "path": _encode_path(item.path, outputs_dir=outputs_dir),
                "description": item.description,
            }
            for item in record.path_descriptions
        ]
    base.update(
        {
            "producer_config_digest": record.producer_config_digest,
            "build_identity": record.build_identity.to_dict() if record.build_identity else None,
            "file_evidence": [item.to_dict() for item in record.file_evidence],
        }
    )
    return base


def record_revision_digest(record: DataFrameArtifactRecord | FileBundleRecord, *, outputs_dir: Path) -> str:
    """Return the stable identity of one exact persisted record revision."""
    return digest_json(record_to_dict(record, outputs_dir=outputs_dir))


def record_from_dict(
    payload: dict[str, Any],
    *,
    outputs_dir: Path,
    experiment_root: Path | None = None,
) -> DataFrameArtifactRecord | FileBundleRecord:
    if not isinstance(payload, dict):
        raise RecordError("record payload must be a JSON object")
    schema_version = payload.get("schema_version")
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
    if schema_version != RECORD_SCHEMA_VERSION:
        raise RecordError(f"record payload schema_version must be {RECORD_SCHEMA_VERSION} (got {schema_version!r})")
    allowed = _V5_DATAFRAME_FIELDS if kind == "dataframe_artifact" else _V5_FILE_BUNDLE_FIELDS
    required = _V5_DATAFRAME_FIELDS if kind == "dataframe_artifact" else _V5_FILE_BUNDLE_REQUIRED_FIELDS
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown=" + ", ".join(unknown))
        if missing:
            details.append("missing=" + ", ".join(missing))
        raise RecordError(f"schema-v5 record payload has unknown or missing fields: {'; '.join(details)}")
    if not isinstance(producer_payload, dict):
        raise RecordError(f"record {record_id!r} must include producer metadata")
    producer_kind = producer_payload.get("kind")
    producer_id = producer_payload.get("id")
    producer_plugin = producer_payload.get("plugin")
    producer_template = producer_payload.get("template")
    source_recipe_payload = producer_payload.get("source_recipe")
    if producer_kind not in {"pipeline", "plot", "export", "notebook"}:
        raise RecordError(f"record {record_id!r} has invalid producer kind {producer_kind!r}")
    if not isinstance(producer_id, str) or not producer_id:
        raise RecordError(f"record {record_id!r} must include producer.id")
    if producer_kind == "notebook":
        if not isinstance(producer_template, str) or not producer_template:
            raise RecordError(f"record {record_id!r} must include producer.template")
    elif not isinstance(producer_plugin, str) or not producer_plugin:
        raise RecordError(f"record {record_id!r} must include producer.plugin")
    source_recipe = None
    if source_recipe_payload is not None:
        if not isinstance(source_recipe_payload, dict):
            raise RecordError(f"record {record_id!r} must include producer.source_recipe as an object")
        recipe_name = source_recipe_payload.get("recipe")
        with_block = source_recipe_payload.get("with", {}) or {}
        if not isinstance(recipe_name, str) or not recipe_name:
            raise RecordError(f"record {record_id!r} must include producer.source_recipe.recipe")
        if not isinstance(with_block, dict):
            raise RecordError(f"record {record_id!r} producer.source_recipe.with must be a mapping")
        source_recipe = RecordRecipeSource(recipe=recipe_name, with_=dict(with_block))
    if not isinstance(created_at, str) or not created_at:
        raise RecordError(f"record {record_id!r} must include created_at")
    if not isinstance(inputs, list) or any(not isinstance(item, dict) for item in inputs):
        raise RecordError(f"record {record_id!r} must include structured inputs")
    if not isinstance(config_digest, str) or not config_digest:
        raise RecordError(f"record {record_id!r} must include config_digest")
    producer = RecordProducer(
        kind=producer_kind,
        id=producer_id,
        plugin=producer_plugin if isinstance(producer_plugin, str) else None,
        template=producer_template if isinstance(producer_template, str) else None,
        source_recipe=source_recipe,
    )
    try:
        evidence_root = (experiment_root or outputs_dir.parent).resolve(strict=False)
        parsed_inputs = tuple(RecordInputEvidence.from_dict(item, experiment_root=evidence_root) for item in inputs)
    except (TypeError, ValueError, RecordError) as exc:
        raise RecordError(f"record {record_id!r} has invalid provenance input: {exc}") from exc
    if kind == "dataframe_artifact":
        contract_id = payload.get("contract_id")
        path = payload.get("path")
        content_digest = payload.get("content_digest")
        code_digest = payload.get("code_digest", "")
        producer_config_digest = payload.get("producer_config_digest", "")
        build_identity = BuildIdentity.from_dict(payload.get("build_identity"))
        size_bytes = payload.get("size_bytes")
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
            inputs=parsed_inputs,
            config_digest=config_digest,
            contract_id=contract_id,
            path=_decode_path(path, outputs_dir=outputs_dir),
            content_digest=content_digest,
            code_digest=code_digest,
            producer_config_digest=producer_config_digest,
            build_identity=build_identity,
            size_bytes=size_bytes,
            schema_version=schema_version,
        )
    files = payload.get("files")
    if not isinstance(files, list) or any(not isinstance(item, str) or not item for item in files):
        raise RecordError(f"record {record_id!r} must include non-empty string file paths")
    description = payload.get("description")
    if not isinstance(description, str) or not description.strip():
        raise RecordError(f"file-bundle record {record_id!r} must include a non-empty description")
    path_descriptions_payload = payload.get("path_descriptions", [])
    if not isinstance(path_descriptions_payload, list):
        raise RecordError(f"record {record_id!r} path_descriptions must be a list when provided")
    path_descriptions: list[PathDescription] = []
    for item in path_descriptions_payload:
        if not isinstance(item, dict) or set(item) != {"path", "description"}:
            raise RecordError(f"record {record_id!r} path_descriptions entries must contain only path and description")
        item_path = item.get("path")
        item_description = item.get("description")
        if not isinstance(item_description, str):
            raise RecordError(f"record {record_id!r} path_descriptions entries must include string descriptions")
        path_descriptions.append(
            PathDescription(
                path=_decode_path(item_path, outputs_dir=outputs_dir),
                description=item_description,
            )
        )
    file_evidence_payload = payload.get("file_evidence")
    if not isinstance(file_evidence_payload, list):
        raise RecordError(f"record {record_id!r} file_evidence must be a list")
    file_evidence = tuple(ArtifactEvidence.from_dict(item) for item in file_evidence_payload)
    producer_config_digest = payload.get("producer_config_digest", "")
    build_identity = BuildIdentity.from_dict(payload.get("build_identity"))
    return FileBundleRecord(
        record_id=record_id,
        kind="file_bundle",
        producer=producer,
        created_at=created_at,
        inputs=parsed_inputs,
        config_digest=config_digest,
        files=tuple(_decode_path(path, outputs_dir=outputs_dir) for path in files),
        description=description,
        schema_version=schema_version,
        path_descriptions=tuple(path_descriptions),
        file_evidence=file_evidence,
        producer_config_digest=producer_config_digest,
        build_identity=build_identity,
    )
