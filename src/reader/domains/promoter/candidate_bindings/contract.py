"""Reader consumer contract for study-issued promoter candidate bindings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BINDING_SCHEMA_ID = "dnadesign.study.promoter_candidate_bindings.v1"
BINDING_SCHEMA_VERSION = "1"
BINDING_ARTIFACT_ID = "bindings.parquet"
BINDING_RECORD_ID = "promoter_candidate_bindings/bindings"
READER_ALIAS_NAMESPACE = "reader.design_id"
BASERENDER_CONTRACT_ID = "dnadesign.baserender.sequence_panel.v1"
BASERENDER_CONTRACT_VERSION = "1"
SUPPORTED_SEQUENCE_ADAPTERS = frozenset({"densegen_tfbs", "usr_genbank_annotations_v1"})

BINDING_COLUMNS = (
    "alias_namespace",
    "alias",
    "display_label",
    "candidate_id",
    "canonical_sequence",
    "sequence_sha256",
    "candidate_table_id",
    "candidate_selection_sha256",
    "sequence_authority_dataset_id",
    "sequence_authority_id",
    "sequence_authority_sha256",
    "source_class",
    "design_family",
    "baserender_adapter_kind",
    "baserender_annotation_column",
    "densegen__plan",
    "densegen__run_id",
    "densegen__sampling_library_hash",
    "densegen__used_tfbs_detail",
    "densegen__required_regulators",
    "seq_annot__features",
    "seq_annot__source_file",
    "usr_label__primary",
    "derived__product_kind",
    "binding_status",
    "binding_method",
)


@dataclass(frozen=True)
class PromoterCandidateBinding:
    reader_design_id: str
    display_label: str
    candidate_id: str
    canonical_sequence: str
    sequence_sha256: str
    candidate_table_id: str
    candidate_selection_sha256: str
    sequence_authority_dataset_id: str
    sequence_authority_id: str
    sequence_authority_sha256: str
    source_class: str
    design_family: str
    densegen_plan: str | None
    densegen_run_id: str | None
    densegen_sampling_library_hash: str | None
    baserender_adapter_kind: str
    baserender_record: dict[str, object]
    binding_status: str
    binding_method: str


@dataclass(frozen=True)
class PromoterCandidateBindings:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    records_sha256: str
    schema_id: str
    schema_version: str
    study_id: str
    record_id: str
    candidate_table_id: str
    candidate_selection_sha256: str
    source_artifacts: tuple[dict[str, str], ...]
    rows: tuple[PromoterCandidateBinding, ...]

    def resolve(self, reader_design_id: str) -> PromoterCandidateBinding:
        matches = [row for row in self.rows if row.reader_design_id == reader_design_id]
        if len(matches) != 1:
            raise ValueError(
                "Reader design binding must resolve exactly once: "
                f"reader_design_id={reader_design_id!r}, matches={len(matches)}."
            )
        return matches[0]


__all__ = [
    "BASERENDER_CONTRACT_ID",
    "BASERENDER_CONTRACT_VERSION",
    "BINDING_ARTIFACT_ID",
    "BINDING_COLUMNS",
    "BINDING_RECORD_ID",
    "BINDING_SCHEMA_ID",
    "BINDING_SCHEMA_VERSION",
    "READER_ALIAS_NAMESPACE",
    "PromoterCandidateBinding",
    "PromoterCandidateBindings",
    "SUPPORTED_SEQUENCE_ADAPTERS",
]
