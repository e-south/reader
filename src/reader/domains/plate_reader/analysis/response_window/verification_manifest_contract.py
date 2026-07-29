"""Exact field contracts for response-window bundle provenance."""

from __future__ import annotations

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "study_id",
        "request_id",
        "state_order",
        "display",
        "created_at",
        "primary_reduction_id",
        "request",
        "contracts",
        "records",
        "counts",
        "source_records",
        "artifacts",
    }
)
_REQUEST_PROVENANCE_FIELDS = frozenset({"artifact_id", "sha256"})
_ARTIFACT_FIELDS = frozenset({"path", "bytes", "sha256"})
_SOURCE_RECORD_FIELDS = frozenset({"experiment_id", "config_artifact", "records_artifact", "records"})


def require_manifest_fields(value: dict[object, object]) -> None:
    _require_exact_fields(value, expected=_MANIFEST_FIELDS, context="response-window manifest fields")


def require_request_provenance_fields(value: dict[object, object]) -> None:
    _require_exact_fields(
        value,
        expected=_REQUEST_PROVENANCE_FIELDS,
        context="response-window request provenance fields",
    )


def require_artifact_fields(value: dict[object, object]) -> None:
    _require_exact_fields(value, expected=_ARTIFACT_FIELDS, context="response-window artifact metadata fields")


def require_source_record_fields(value: dict[object, object]) -> None:
    _require_exact_fields(value, expected=_SOURCE_RECORD_FIELDS, context="response-window source-record fields")


def require_record_digest_keys(value: dict[object, object]) -> None:
    if any(not isinstance(record_id, str) or not record_id.strip() for record_id in value):
        raise ValueError("response-window source record-digest identities must be non-empty strings.")


def _require_exact_fields(
    value: dict[object, object],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    observed = set(value)
    if observed == expected:
        return
    missing = sorted(expected - observed)
    unexpected = sorted(str(field) for field in observed - expected)
    raise ValueError(f"{context} must be exact; missing={missing!r}, unexpected={unexpected!r}.")


__all__ = [
    "require_artifact_fields",
    "require_manifest_fields",
    "require_record_digest_keys",
    "require_request_provenance_fields",
    "require_source_record_fields",
]
