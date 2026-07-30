from __future__ import annotations

from reader.errors import RecordError
from reader.workbench.records import (
    DataFrameArtifactRecord,
    FileBundleRecord,
    RecordStore,
    record_revision_digest,
)
from reader.workbench.records.identity import is_sha256_digest

from .models import Experiment, RecordRevision


def resolve_record_revision(
    experiment: Experiment,
    record_id: str,
    *,
    revision: int | None,
    revision_digest: str | None,
    missing_label: str = "Record",
) -> tuple[RecordStore, DataFrameArtifactRecord | FileBundleRecord, RecordRevision]:
    """Resolve one catalog revision and prove its caller-supplied identity."""

    if not isinstance(record_id, str) or not record_id.strip():
        raise RecordError("record_id must be a non-empty string")
    if record_id != record_id.strip():
        raise RecordError("record_id must not contain leading or trailing whitespace")
    if (revision is None) != (revision_digest is None):
        raise RecordError("revision and revision_digest must be provided together")
    if revision is not None and (isinstance(revision, bool) or not isinstance(revision, int) or revision < 1):
        raise RecordError("revision must be a positive integer")
    if revision_digest is not None and not is_sha256_digest(revision_digest):
        raise RecordError("revision_digest must be a sha256 digest")

    declaration = experiment._declaration
    layout = declaration.experiment_semantics.layout
    store = experiment._runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=declaration.experiment.root,
        create=False,
    )
    if not store.catalog_exists():
        raise RecordError(
            f"Record catalog is missing for experiment {declaration.experiment.id!r}: {store.records_path}"
        )

    history = store.record_history(record_id)
    if not history:
        raise RecordError(f"{missing_label} {record_id!r} is missing; produce it in an earlier step.")
    resolved_revision = len(history) if revision is None else revision
    if resolved_revision > len(history):
        raise RecordError(
            f"Record {record_id!r} revision {resolved_revision} is unavailable; refresh the record catalog."
        )
    record = history[resolved_revision - 1]
    actual_revision_digest = record_revision_digest(record, outputs_dir=store.root)
    if revision_digest is not None and actual_revision_digest != revision_digest:
        raise RecordError(
            f"Record {record_id!r} revision digest mismatch; refresh the record catalog before reading it."
        )
    return (
        store,
        record,
        RecordRevision(
            record_id=record.record_id,
            revision=resolved_revision,
            revision_digest=actual_revision_digest,
        ),
    )


__all__ = ["resolve_record_revision"]
