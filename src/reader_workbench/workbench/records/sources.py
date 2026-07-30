from __future__ import annotations

from dataclasses import dataclass

from reader_workbench.contracts import ContractCatalog
from reader_workbench.errors import RecordError
from reader_workbench.workbench.graph import SourceRecordRef

from .model import (
    DataFrameArtifactRecord,
    FileBundleRecord,
    record_revision_digest,
    verify_record_artifact_integrity,
)
from .store import RecordStore


@dataclass(frozen=True)
class ResolvedSourceRecord:
    """One exact record revision owned by another Reader experiment."""

    ref: SourceRecordRef
    record: DataFrameArtifactRecord | FileBundleRecord
    revision_digest: str

    def verify_artifact_integrity(self) -> None:
        """Verify the live artifacts bound to this exact catalog revision."""

        verify_record_artifact_integrity(self.record, outputs_dir=self.ref.outputs_dir)

    def load_dataframe(self):
        if not isinstance(self.record, DataFrameArtifactRecord):
            raise RecordError(
                f"Source record {self.ref.experiment_id}:{self.ref.record_id} is not a dataframe artifact"
            )
        return self.record.load_dataframe()


@dataclass(frozen=True)
class SourceRecordCollection:
    """Ordered, provenance-bearing source records supplied to an aggregate plugin."""

    records: tuple[ResolvedSourceRecord, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise RecordError("A source record collection must contain at least one record")

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


def resolve_source_record(ref: SourceRecordRef, *, contracts: ContractCatalog) -> ResolvedSourceRecord:
    store = RecordStore(
        ref.outputs_dir,
        contracts=contracts,
        experiment_root=ref.experiment_root,
        create=False,
    )
    record = store.latest_record(ref.record_id)
    if record is None:
        raise RecordError(f"Source record {ref.record_id!r} is missing from experiment {ref.experiment_id!r}")
    return ResolvedSourceRecord(
        ref=ref,
        record=record,
        revision_digest=record_revision_digest(record, outputs_dir=ref.outputs_dir),
    )


__all__ = ["ResolvedSourceRecord", "SourceRecordCollection", "resolve_source_record"]
