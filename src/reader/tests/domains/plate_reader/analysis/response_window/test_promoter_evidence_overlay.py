from __future__ import annotations

import json
from pathlib import Path

import pytest

from reader.domains.plate_reader.analysis.response_window.promoter_evidence_overlay import (
    load_objective_display_overlay,
)
from reader.response_window_review import (
    OBJECTIVE_DISPLAY_LABEL_MAX_LENGTH,
    OBJECTIVE_OVERLAY_SCHEMA_VERSION,
)


def test_screen_only_overlay_carries_raw_components_without_a_calibrated_claim(tmp_path: Path) -> None:
    path = _write_overlay(tmp_path)

    overlay = load_objective_display_overlay(path)

    assert overlay.claim_status == "screen_only"
    assert overlay.objective_id == "response_magnitude_feasibility_v1"
    assert overlay.objective_display_label == "RMF"
    assert [component.component_id for component in overlay.components] == [
        "response_separation",
        "on_fluorescence_floor",
        "off_fluorescence_ceiling",
    ]


def test_screen_only_overlay_rejects_calibrated_or_production_fields(tmp_path: Path) -> None:
    path = _write_overlay(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibrated_score"] = 0.4
    payload["limiting_component_id"] = "on_fluorescence_floor"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields must be exactly"):
        load_objective_display_overlay(path)


def test_v2_overlay_rejects_production_claim_status(tmp_path: Path) -> None:
    path = _write_overlay(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["claim_status"] = "production"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="screen_only"):
        load_objective_display_overlay(path)


def test_v2_overlay_rejects_v1_schema_without_a_compatibility_path(tmp_path: Path) -> None:
    path = _write_overlay(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "reader.response_window.objective_display_overlay.v1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="objective_display_overlay.v2"):
        load_objective_display_overlay(path)


@pytest.mark.parametrize(
    "label",
    [
        "",
        " RMF",
        "Response margin\nfeasibility",
        "RMF\n",
        "RMF\tview",
        "x" * (OBJECTIVE_DISPLAY_LABEL_MAX_LENGTH + 1),
    ],
    ids=("empty", "leading-space", "multiline", "trailing-newline", "control-character", "overlong"),
)
def test_v2_overlay_rejects_display_labels_that_cannot_fit_the_evidence_card(
    tmp_path: Path,
    label: str,
) -> None:
    path = _write_overlay(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["objective_display_label"] = label
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="objective_display_label"):
        load_objective_display_overlay(path)


def _write_overlay(tmp_path: Path) -> Path:
    path = tmp_path / "objective-overlay.json"
    payload = {
        "schema_version": OBJECTIVE_OVERLAY_SCHEMA_VERSION,
        "created_at": "2026-07-13T00:00:00+00:00",
        "objective_id": "response_magnitude_feasibility_v1",
        "objective_display_label": "RMF",
        "claim_status": "screen_only",
        "selection": {
            "experiment_id": "experiment",
            "reader_design_id": "design",
            "reduction_id": "primary",
        },
        "components": [
            {
                "component_id": "response_separation",
                "label": "Response separation",
                "value": 0.8,
                "unit": "raw log2 units",
            },
            {
                "component_id": "on_fluorescence_floor",
                "label": "ON fluorescence floor",
                "value": 0.3,
                "unit": "pDual-10-relative log2 fluorescence",
            },
            {
                "component_id": "off_fluorescence_ceiling",
                "label": "OFF fluorescence ceiling",
                "value": -0.2,
                "unit": "pDual-10-relative log2 fluorescence",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
