"""Row-level validation for Reader design aliases in study bindings."""

from __future__ import annotations

import hashlib
import re

import pandas as pd

from .annotations import densegen_annotations, genbank_annotations, safe_relative_posix_reference
from .contract import READER_ALIAS_NAMESPACE, SUPPORTED_SEQUENCE_ADAPTERS, PromoterCandidateBinding

_IUPAC_DNA = re.compile(r"[ACGTRYSWKMBDHVN]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def bindings_from_frame(frame: pd.DataFrame) -> tuple[PromoterCandidateBinding, ...]:
    return tuple(_binding_from_row(row) for _, row in frame.iterrows())


def _binding_from_row(row: pd.Series) -> PromoterCandidateBinding:
    namespace = _nonempty(row["alias_namespace"], context="alias_namespace")
    if namespace != READER_ALIAS_NAMESPACE:
        raise ValueError(f"Reader candidate binding must use namespace {READER_ALIAS_NAMESPACE!r}.")
    alias = _nonempty(row["alias"], context="alias")
    candidate_id = _nonempty(row["candidate_id"], context=f"{alias}.candidate_id")
    sequence = _nonempty(row["canonical_sequence"], context=f"{alias}.canonical_sequence")
    if sequence != sequence.upper() or _IUPAC_DNA.fullmatch(sequence) is None:
        raise ValueError(f"{alias!r} canonical sequence must be uppercase IUPAC DNA without whitespace.")
    sequence_digest = _digest(row["sequence_sha256"], context=f"{alias}.sequence_sha256")
    if sequence_digest != hashlib.sha256(sequence.encode()).hexdigest():
        raise ValueError(f"{alias!r} canonical sequence digest mismatch.")
    adapter = _nonempty(row["baserender_adapter_kind"], context=f"{alias}.baserender_adapter_kind")
    if adapter not in SUPPORTED_SEQUENCE_ADAPTERS:
        raise ValueError(f"{alias!r} uses unsupported BaseRender adapter {adapter!r}.")
    status = _nonempty(row["binding_status"], context=f"{alias}.binding_status")
    method = _nonempty(row["binding_method"], context=f"{alias}.binding_method")
    if status != "resolved" or method != "exact_alias":
        raise ValueError(f"{alias!r} must be an exact resolved candidate binding.")
    densegen_plan, densegen_run, densegen_library = _densegen_provenance(row, adapter=adapter, alias=alias)
    return PromoterCandidateBinding(
        reader_design_id=alias,
        display_label=_nonempty(row["display_label"], context=f"{alias}.display_label"),
        candidate_id=candidate_id,
        canonical_sequence=sequence,
        sequence_sha256=sequence_digest,
        candidate_table_id=_nonempty(row["candidate_table_id"], context=f"{alias}.candidate_table_id"),
        candidate_selection_sha256=_digest(
            row["candidate_selection_sha256"], context=f"{alias}.candidate_selection_sha256"
        ),
        sequence_authority_dataset_id=_nonempty(
            row["sequence_authority_dataset_id"], context=f"{alias}.sequence_authority_dataset_id"
        ),
        sequence_authority_id=_nonempty(row["sequence_authority_id"], context=f"{alias}.sequence_authority_id"),
        sequence_authority_sha256=_digest(
            row["sequence_authority_sha256"], context=f"{alias}.sequence_authority_sha256"
        ),
        source_class=_nonempty(row["source_class"], context=f"{alias}.source_class"),
        design_family=_nonempty(row["design_family"], context=f"{alias}.design_family"),
        densegen_plan=densegen_plan,
        densegen_run_id=densegen_run,
        densegen_sampling_library_hash=densegen_library,
        baserender_adapter_kind=adapter,
        baserender_record=_baserender_record(
            row,
            adapter=adapter,
            alias=alias,
            candidate_id=candidate_id,
            sequence=sequence,
        ),
        binding_status=status,
        binding_method=method,
    )


def _baserender_record(
    row: pd.Series,
    *,
    adapter: str,
    alias: str,
    candidate_id: str,
    sequence: str,
) -> dict[str, object]:
    column = _nonempty(row["baserender_annotation_column"], context=f"{alias}.baserender_annotation_column")
    if adapter == "densegen_tfbs":
        if column != "densegen__used_tfbs_detail":
            raise ValueError(f"{alias!r} DenseGen binding declares the wrong annotation column.")
        annotations = densegen_annotations(
            row["densegen__used_tfbs_detail"],
            required_regulators=row["densegen__required_regulators"],
            sequence=sequence,
            alias=alias,
        )
        return {"id": candidate_id, "sequence": sequence, "densegen__used_tfbs_detail": annotations}
    if column != "seq_annot__features":
        raise ValueError(f"{alias!r} GenBank binding declares the wrong annotation column.")
    features = genbank_annotations(row["seq_annot__features"], sequence=sequence, alias=alias)
    source_file = _nonempty(row["seq_annot__source_file"], context=f"{alias}.seq_annot__source_file")
    safe_relative_posix_reference(source_file, context=f"{alias}.seq_annot__source_file")
    return {
        "id": candidate_id,
        "sequence": sequence,
        "seq_annot__features": features,
        "seq_annot__source_file": source_file,
        "usr_label__primary": _nonempty(row["usr_label__primary"], context=f"{alias}.usr_label__primary"),
        "derived__product_kind": _nonempty(row["derived__product_kind"], context=f"{alias}.derived__product_kind"),
    }


def _densegen_provenance(
    row: pd.Series,
    *,
    adapter: str,
    alias: str,
) -> tuple[str | None, str | None, str | None]:
    fields = ("densegen__plan", "densegen__run_id", "densegen__sampling_library_hash")
    if adapter == "densegen_tfbs":
        return tuple(_nonempty(row[field], context=f"{alias}.{field}") for field in fields)  # type: ignore[return-value]
    if any(not _missing(row[field]) for field in fields):
        raise ValueError(f"{alias!r} GenBank binding must carry null DenseGen provenance.")
    return None, None, None


def _nonempty(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _digest(value: object, *, context: str) -> str:
    text = _nonempty(value, context=context).lower().removeprefix("sha256:")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{context} must be a 64-character hexadecimal SHA-256 digest.")
    return text


__all__ = ["bindings_from_frame"]
