"""Manifest projection and verification for one selected study binding."""

from __future__ import annotations

import re

from reader.domains.promoter.candidate_bindings import PromoterCandidateBinding

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_FIELDS = {
    "sequence_sha256",
    "sequence_authority_dataset_id",
    "sequence_authority_id",
    "sequence_authority_sha256",
    "source_class",
    "design_family",
    "binding_status",
    "binding_method",
    "densegen_plan",
    "densegen_run_id",
    "densegen_sampling_library_hash",
}
_DENSEGEN_FIELDS = ("densegen_plan", "densegen_run_id", "densegen_sampling_library_hash")


def selected_binding_record(binding: PromoterCandidateBinding) -> dict[str, object]:
    """Project binding provenance without exporting the canonical sequence."""

    return {
        "sequence_sha256": "sha256:" + binding.sequence_sha256,
        "sequence_authority_dataset_id": binding.sequence_authority_dataset_id,
        "sequence_authority_id": binding.sequence_authority_id,
        "sequence_authority_sha256": "sha256:" + binding.sequence_authority_sha256,
        "source_class": binding.source_class,
        "design_family": binding.design_family,
        "binding_status": binding.binding_status,
        "binding_method": binding.binding_method,
        "densegen_plan": binding.densegen_plan,
        "densegen_run_id": binding.densegen_run_id,
        "densegen_sampling_library_hash": binding.densegen_sampling_library_hash,
    }


def verify_selected_binding(
    value: object,
    *,
    baserender_adapter_kind: object,
) -> None:
    """Verify exact selected-binding provenance and adapter-specific null policy."""

    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError(f"promoter-evidence selected-binding fields must be exactly {sorted(_FIELDS)}.")
    digests = ("sequence_sha256", "sequence_authority_sha256")
    required_text = _FIELDS - set(digests) - set(_DENSEGEN_FIELDS)
    if (
        any(not _is_sha256(value[field]) for field in digests)
        or any(not _is_nonempty(value[field]) for field in required_text)
        or value["binding_status"] != "resolved"
        or value["binding_method"] != "exact_alias"
    ):
        raise ValueError("promoter-evidence selected-binding provenance is malformed.")
    if baserender_adapter_kind == "densegen_tfbs":
        if any(not _is_nonempty(value[field]) for field in _DENSEGEN_FIELDS):
            raise ValueError("DenseGen promoter evidence requires selected-binding plan, run, and library provenance.")
    elif baserender_adapter_kind == "usr_genbank_annotations_v1":
        if any(value[field] is not None for field in _DENSEGEN_FIELDS):
            raise ValueError("GenBank promoter evidence requires null DenseGen selected-binding provenance.")
    else:
        raise ValueError("promoter-evidence selected-binding uses an unsupported BaseRender adapter.")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = ["selected_binding_record", "verify_selected_binding"]
