"""Display-only objective overlay contract; no objective math lives here."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .provenance import sha256_file

OBJECTIVE_OVERLAY_SCHEMA_VERSION = "reader.response_window.objective_display_overlay.v2"
OBJECTIVE_DISPLAY_LABEL_MAX_LENGTH = 40


@dataclass(frozen=True)
class ObjectiveComponent:
    component_id: str
    label: str
    value: float
    unit: str


@dataclass(frozen=True)
class ObjectiveDisplayOverlay:
    path: Path
    manifest_sha256: str
    objective_id: str
    objective_display_label: str
    claim_status: str
    experiment_id: str
    reader_design_id: str
    reduction_id: str
    components: tuple[ObjectiveComponent, ...]


def load_objective_display_overlay(path: Path) -> ObjectiveDisplayOverlay:
    """Verify study-supplied display values without deriving or recalibrating them."""

    overlay_path = Path(path).expanduser().resolve()
    if not overlay_path.is_file():
        raise FileNotFoundError(f"objective display overlay not found: {overlay_path}")
    payload = _exact_mapping(
        json.loads(overlay_path.read_text(encoding="utf-8")),
        context="objective overlay",
        fields={
            "schema_version",
            "created_at",
            "objective_id",
            "objective_display_label",
            "claim_status",
            "selection",
            "components",
        },
    )
    if payload["schema_version"] != OBJECTIVE_OVERLAY_SCHEMA_VERSION:
        raise ValueError(f"objective overlay must use {OBJECTIVE_OVERLAY_SCHEMA_VERSION!r}.")
    _created_at(payload["created_at"])
    objective_id = _nonempty(payload["objective_id"], context="objective overlay.objective_id")
    objective_display_label = _objective_display_label(payload["objective_display_label"])
    selection = _exact_mapping(
        payload["selection"],
        context="objective overlay.selection",
        fields={"experiment_id", "reader_design_id", "reduction_id"},
    )
    components = _components(payload["components"])
    claim_status = _nonempty(payload["claim_status"], context="objective overlay.claim_status")
    if claim_status != "screen_only":
        raise ValueError("objective overlay v2 supports screen_only evidence only.")
    return ObjectiveDisplayOverlay(
        path=overlay_path,
        manifest_sha256=sha256_file(overlay_path),
        objective_id=objective_id,
        objective_display_label=objective_display_label,
        claim_status=claim_status,
        experiment_id=_nonempty(selection["experiment_id"], context="objective overlay.selection.experiment_id"),
        reader_design_id=_nonempty(
            selection["reader_design_id"], context="objective overlay.selection.reader_design_id"
        ),
        reduction_id=_nonempty(selection["reduction_id"], context="objective overlay.selection.reduction_id"),
        components=components,
    )


def _components(value: object) -> tuple[ObjectiveComponent, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 6:
        raise ValueError("objective overlay.components must contain between one and six raw components.")
    result: list[ObjectiveComponent] = []
    for index, raw in enumerate(value):
        item = _exact_mapping(
            raw,
            context=f"objective overlay.components[{index}]",
            fields={"component_id", "label", "value", "unit"},
        )
        result.append(
            ObjectiveComponent(
                component_id=_nonempty(
                    item["component_id"], context=f"objective overlay.components[{index}].component_id"
                ),
                label=_nonempty(item["label"], context=f"objective overlay.components[{index}].label"),
                value=_finite(item["value"], context=f"objective overlay.components[{index}].value"),
                unit=_nonempty(item["unit"], context=f"objective overlay.components[{index}].unit"),
            )
        )
    identities = [component.component_id for component in result]
    if len(identities) != len(set(identities)):
        raise ValueError("objective overlay component identities must be unique.")
    return tuple(result)


def _exact_mapping(value: object, *, context: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    payload = {str(key): item for key, item in value.items()}
    if set(payload) != fields:
        raise ValueError(f"{context} fields must be exactly {sorted(fields)}.")
    return payload


def _nonempty(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context} must be a finite number.")
    return float(value)


def _objective_display_label(value: object) -> str:
    context = "objective overlay.objective_display_label"
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    text = value.strip()
    if value != text or not value.isprintable() or len(text) > OBJECTIVE_DISPLAY_LABEL_MAX_LENGTH:
        raise ValueError(
            f"{context} must be a trimmed, printable, single-line string of at most "
            f"{OBJECTIVE_DISPLAY_LABEL_MAX_LENGTH} characters."
        )
    return text


def _created_at(value: object) -> None:
    text = _nonempty(value, context="objective overlay.created_at")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("objective overlay.created_at must be ISO-8601.") from exc
    if parsed.tzinfo is None:
        raise ValueError("objective overlay.created_at must include a timezone.")


__all__ = [
    "OBJECTIVE_DISPLAY_LABEL_MAX_LENGTH",
    "OBJECTIVE_OVERLAY_SCHEMA_VERSION",
    "ObjectiveComponent",
    "ObjectiveDisplayOverlay",
    "load_objective_display_overlay",
]
