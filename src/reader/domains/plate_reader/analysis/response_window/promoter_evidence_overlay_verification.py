"""Strict verification for persisted objective display overlays."""

from __future__ import annotations

import math
import re

from .promoter_evidence_overlay import OBJECTIVE_OVERLAY_SCHEMA_VERSION

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def verify_overlay_record(value: object, *, claim_status: str, selection: dict[str, str]) -> None:
    if value is None:
        if claim_status != "objective_neutral":
            raise ValueError("non-neutral promoter evidence requires an objective overlay record.")
        return
    fields = {
        "schema_version",
        "objective_id",
        "claim_status",
        "experiment_id",
        "reader_design_id",
        "reduction_id",
        "manifest_sha256",
        "components",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"promoter-evidence objective overlay fields must be exactly {sorted(fields)}.")
    if (
        value["schema_version"] != OBJECTIVE_OVERLAY_SCHEMA_VERSION
        or not _is_nonempty(value["objective_id"])
        or value["claim_status"] != claim_status
        or claim_status == "objective_neutral"
        or not _is_sha256(value["manifest_sha256"])
    ):
        raise ValueError("promoter-evidence objective overlay identity or claim status is invalid.")
    if (
        value["experiment_id"] != selection["experiment_id"]
        or value["reader_design_id"] != selection["design_id"]
        or value["reduction_id"] != selection["reduction_id"]
    ):
        raise ValueError("promoter-evidence objective overlay selection disagrees with evidence selection.")
    _verify_components(value["components"])


def _verify_components(value: object) -> None:
    if not isinstance(value, list) or not 1 <= len(value) <= 6:
        raise ValueError(
            "promoter-evidence objective overlay components must contain between one and six raw components."
        )
    fields = {"component_id", "label", "value", "unit"}
    ids: list[str] = []
    for component in value:
        if (
            not isinstance(component, dict)
            or set(component) != fields
            or not _is_nonempty(component["component_id"])
            or not _is_nonempty(component["label"])
            or not _is_nonempty(component["unit"])
            or not _is_finite(component["value"])
        ):
            raise ValueError("promoter-evidence objective component is malformed.")
        ids.append(component["component_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("promoter-evidence objective component identities must be unique.")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


__all__ = ["verify_overlay_record"]
