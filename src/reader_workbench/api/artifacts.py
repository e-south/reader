from __future__ import annotations

import hashlib
from mimetypes import guess_type
from pathlib import Path

from reader_workbench.errors import RecordError
from reader_workbench.workbench.records import (
    FileBundleRecord,
    record_to_dict,
    verify_record_artifact_integrity,
)

from ._record_reads import resolve_record_revision
from .models import ArtifactFileResult, Experiment


def read_artifact(
    experiment: Experiment,
    record_id: str,
    *,
    revision: int,
    revision_digest: str,
    path: str,
) -> ArtifactFileResult:
    """Read verified bytes for one file in an exact file-bundle record revision."""

    if revision is None or revision_digest is None:
        raise RecordError("read_artifact requires both revision and revision_digest")
    if not isinstance(path, str) or not path.strip() or path != path.strip():
        raise RecordError("path must be one non-empty outputs-relative artifact path")
    requested_path = Path(path)
    if requested_path.is_absolute() or ".." in requested_path.parts or requested_path == Path("."):
        raise RecordError("path must be one confined outputs-relative artifact path")
    normalized_path = requested_path.as_posix()
    if normalized_path != path:
        raise RecordError("path must use the canonical outputs-relative artifact path from records()")

    store, record, bound_revision = resolve_record_revision(
        experiment,
        record_id,
        revision=revision,
        revision_digest=revision_digest,
        missing_label="File-bundle record",
    )
    if not isinstance(record, FileBundleRecord):
        raise RecordError(f"Record {record_id!r} exists but is not a file bundle")
    payload = record_to_dict(record, outputs_dir=store.root)
    raw_files = payload.get("files")
    encoded_files = raw_files if isinstance(raw_files, list) else []
    try:
        file_index = encoded_files.index(normalized_path)
    except ValueError as exc:
        raise RecordError(
            f"Artifact {normalized_path!r} is not part of record {record_id!r} revision {revision}."
        ) from exc

    verify_record_artifact_integrity(record, outputs_dir=store.root)
    evidence_by_path = {item.relative_path.as_posix(): item for item in record.file_evidence}
    evidence = evidence_by_path[normalized_path]
    try:
        content = record.files[file_index].read_bytes()
    except OSError as exc:
        raise RecordError(f"Artifact {normalized_path!r} could not be read after verification.") from exc
    actual_digest = "sha256:" + hashlib.sha256(content).hexdigest()
    if len(content) != evidence.size_bytes or actual_digest != evidence.content_digest:
        raise RecordError(f"Artifact {normalized_path!r} changed while it was being read; refresh and retry.")
    return ArtifactFileResult(
        experiment=experiment.identity,
        record=bound_revision,
        relative_path=normalized_path,
        media_type=guess_type(normalized_path)[0] or "application/octet-stream",
        content_digest=evidence.content_digest,
        size_bytes=evidence.size_bytes,
        content=content,
    )


__all__ = ["ArtifactFileResult", "read_artifact"]
