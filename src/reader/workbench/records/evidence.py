from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reader.errors import RecordError
from reader.workbench.experiments import ExperimentCatalog
from reader.workbench.graph import FileRef, InputRef, RecordRef, ResourceRef, SourceRecordRef

from .identity import is_sha256_digest

_DISCOVERY_POLICIES = frozenset(
    {
        "declared_file",
        "declared_resource",
        "plugin_discovery",
        "record",
        "source_record",
    }
)


@dataclass(frozen=True)
class ArtifactEvidence:
    relative_path: Path
    size_bytes: int
    content_digest: str

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or not path.parts or any(part == ".." for part in path.parts):
            raise RecordError("artifact evidence path must be relative and confined")
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise RecordError("artifact evidence size_bytes must be a non-negative integer")
        if not is_sha256_digest(self.content_digest):
            raise RecordError("artifact evidence content_digest must be a sha256 digest")
        object.__setattr__(self, "relative_path", path)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.relative_path.as_posix(),
            "size_bytes": self.size_bytes,
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ArtifactEvidence:
        if not isinstance(payload, dict) or set(payload) != {"path", "size_bytes", "content_digest"}:
            raise RecordError("artifact evidence must contain only path, size_bytes, and content_digest")
        raw_path = payload.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RecordError("artifact evidence path must be a non-empty string")
        return cls(
            relative_path=Path(raw_path),
            size_bytes=payload.get("size_bytes"),
            content_digest=payload.get("content_digest"),
        )


@dataclass(frozen=True)
class RecordInputEvidence:
    """Immutable evidence for one input consumed by a persisted record."""

    label: str
    ref: InputRef
    discovery_policy: str
    artifact: ArtifactEvidence | None = None
    record_revision_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise RecordError("record input evidence label must be a non-empty string")
        if self.discovery_policy not in _DISCOVERY_POLICIES:
            raise RecordError(
                "record input evidence discovery_policy must be one of " + ", ".join(sorted(_DISCOVERY_POLICIES))
            )
        if isinstance(self.ref, RecordRef):
            if self.discovery_policy != "record":
                raise RecordError("record references must use discovery_policy 'record'")
            if self.artifact is not None:
                raise RecordError("record references must not include artifact evidence")
            if not is_sha256_digest(self.record_revision_digest):
                raise RecordError("record references must include a sha256 record_revision_digest")
            return
        if isinstance(self.ref, SourceRecordRef):
            if self.discovery_policy != "source_record":
                raise RecordError("source record references must use discovery_policy 'source_record'")
            if self.artifact is not None:
                raise RecordError("source record references must not include artifact evidence")
            if not is_sha256_digest(self.record_revision_digest):
                raise RecordError("source record references must include a sha256 record_revision_digest")
            return
        if not isinstance(self.ref, (FileRef, ResourceRef)):
            raise RecordError("record input evidence contains an unsupported reference")
        if self.discovery_policy in {"record", "source_record"}:
            raise RecordError("file and resource inputs must not use record discovery policies")
        if self.artifact is None:
            raise RecordError("file and resource inputs must include artifact evidence")
        if self.record_revision_digest is not None:
            raise RecordError("file and resource inputs must not include record_revision_digest")

    def to_dict(self) -> dict[str, object]:
        base: dict[str, object] = {
            "label": self.label,
            "discovery_policy": self.discovery_policy,
        }
        if isinstance(self.ref, RecordRef):
            return {
                **base,
                "kind": "record",
                "record": self.ref.record_id,
                "record_revision_digest": self.record_revision_digest,
            }
        if isinstance(self.ref, SourceRecordRef):
            return {
                **base,
                "kind": "source_record",
                "resource": self.ref.resource_id,
                "experiment": self.ref.experiment_id,
                "record": self.ref.record_id,
                "record_revision_digest": self.record_revision_digest,
            }
        if isinstance(self.ref, ResourceRef):
            base.update({"kind": "resource", "resource": self.ref.resource_id})
        else:
            base["kind"] = "file"
        base["artifact"] = self.artifact.to_dict() if self.artifact else None
        return base

    @classmethod
    def from_dict(cls, payload: Any, *, experiment_root: Path) -> RecordInputEvidence:
        if not isinstance(payload, dict):
            raise RecordError("record input evidence must be a JSON object")
        kind = payload.get("kind")
        label = payload.get("label")
        discovery_policy = payload.get("discovery_policy")
        if kind == "record":
            expected = {
                "label",
                "kind",
                "record",
                "discovery_policy",
                "record_revision_digest",
            }
            if set(payload) != expected:
                raise RecordError("record input evidence has unknown or missing fields")
            record_id = payload.get("record")
            if not isinstance(record_id, str) or not record_id:
                raise RecordError("record input evidence must include a non-empty record id")
            return cls(
                label=label,
                ref=RecordRef(record_id=record_id),
                discovery_policy=discovery_policy,
                record_revision_digest=payload.get("record_revision_digest"),
            )
        if kind == "source_record":
            expected = {
                "label",
                "kind",
                "resource",
                "experiment",
                "record",
                "discovery_policy",
                "record_revision_digest",
            }
            if set(payload) != expected:
                raise RecordError("source record input evidence has unknown or missing fields")
            resource_id = payload.get("resource")
            experiment_id = payload.get("experiment")
            record_id = payload.get("record")
            if any(not isinstance(value, str) or not value for value in (resource_id, experiment_id, record_id)):
                raise RecordError("source record input evidence requires resource, experiment, and record identities")
            try:
                location = ExperimentCatalog.from_experiment_root(experiment_root).resolve(experiment_id)
            except Exception as exc:
                raise RecordError(f"Could not resolve source experiment {experiment_id!r}: {exc}") from exc
            return cls(
                label=label,
                ref=SourceRecordRef(
                    resource_id=resource_id,
                    experiment_id=location.id,
                    record_id=record_id,
                    experiment_root=location.root,
                    outputs_dir=location.outputs_dir,
                ),
                discovery_policy=discovery_policy,
                record_revision_digest=payload.get("record_revision_digest"),
            )
        if kind not in {"file", "resource"}:
            raise RecordError(f"record input evidence has unknown kind {kind!r}")
        expected = {"label", "kind", "discovery_policy", "artifact"}
        if kind == "resource":
            expected.add("resource")
        if set(payload) != expected:
            raise RecordError("record input evidence has unknown or missing fields")
        artifact = ArtifactEvidence.from_dict(payload.get("artifact"))
        resolved = experiment_root.resolve(strict=False) / artifact.relative_path
        if kind == "resource":
            resource_id = payload.get("resource")
            if not isinstance(resource_id, str) or not resource_id:
                raise RecordError("resource input evidence must include a non-empty resource id")
            ref: InputRef = ResourceRef(resource_id=resource_id, path=resolved)
        else:
            ref = FileRef(path=resolved)
        return cls(
            label=label,
            ref=ref,
            discovery_policy=discovery_policy,
            artifact=artifact,
        )


def capture_artifact_evidence(path: Path, *, root: Path) -> ArtifactEvidence:
    root_resolved = root.resolve(strict=False)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RecordError(f"artifact {path} must resolve within the outputs directory {root}") from exc
    if not resolved.is_file():
        raise RecordError(f"artifact {path} must be a regular file")
    return ArtifactEvidence(
        relative_path=relative,
        size_bytes=resolved.stat().st_size,
        content_digest=_sha256_file(resolved),
    )


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(chunk_size), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
