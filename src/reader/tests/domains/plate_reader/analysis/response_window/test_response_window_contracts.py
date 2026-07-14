from __future__ import annotations

import pytest

from reader.domains.plate_reader.analysis.response_window.contracts import (
    ResponseWindowRequest,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "reader.response_window.request.v3",
        "study_id": "stress_ethanol_cipro_growth",
        "request_id": "stress-response-window-v1",
        "experiment_ids": ["20260101_example", "20260102_example"],
        "state_order": ["00", "10", "01", "11"],
        "display": {
            "study_label": "Ethanol and ciprofloxacin response",
            "event_label": "Stress addition",
            "state_labels": {
                "00": "No stress",
                "10": "Ethanol",
                "01": "Ciprofloxacin",
                "11": "Ethanol + ciprofloxacin",
            },
            "examples": [
                {
                    "design_id": "reference",
                    "label": "Reference fluorescence anchor",
                    "role": "reference_anchor",
                },
                {
                    "design_id": "ethanol-example",
                    "label": "Ethanol-response example",
                    "role": "response_example",
                },
            ],
        },
        "source": {
            "response_record_id": "ratio_response/df",
            "magnitude_record_id": "ratio_magnitude/df",
            "trajectory_record_id": "annotated/df",
            "response_channel": "response_ratio",
            "magnitude_channel": "magnitude_ratio",
            "growth_channel": "growth",
            "reference_design_id": "reference",
            "state_map_ref": "stress_states",
        },
        "event": {
            "event_id": "induction",
            "event_kind": "perturbation_addition",
            "segment_column": "sheet_index",
            "pre_segment_index": 0,
            "post_segment_index": 1,
            "estimate_method": "segment_gap_midpoint",
            "declaration": "The intervention occurred between acquisition segments 0 and 1.",
        },
        "reductions": [
            {
                "id": "event_logmean_6_12h_post",
                "window_start_event_h": 6.0,
                "window_end_event_h": 12.0,
                "method": "geometric_time_mean",
                "response_basis": "post_window",
                "role": "primary",
            },
            {
                "id": "event_linear_auc_6_12h_post",
                "window_start_event_h": 6.0,
                "window_end_event_h": 12.0,
                "method": "integrated_linear_mean",
                "response_basis": "post_window",
                "role": "sensitivity",
            },
        ],
        "aggregation": {
            "replicate_stat": "median",
            "bootstrap_samples": 200,
            "confidence_level": 0.9,
            "random_seed": 17,
        },
        "quality": {
            "positive_floor": 1.0e-12,
            "max_interior_gap_h": 0.75,
            "min_replicates_per_state": 2,
        },
    }


def test_request_requires_one_primary_reduction() -> None:
    payload = _payload()
    payload["reductions"][0]["role"] = "sensitivity"  # type: ignore[index]

    with pytest.raises(ValueError, match="exactly one primary reduction"):
        ResponseWindowRequest.from_mapping(payload)


def test_request_rejects_unknown_fields() -> None:
    payload = _payload()
    payload["event"]["infer_from_sheet_order"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="unknown fields"):
        ResponseWindowRequest.from_mapping(payload)


def test_request_parses_explicit_contract() -> None:
    request = ResponseWindowRequest.from_mapping(_payload())

    assert request.primary_reduction.id == "event_logmean_6_12h_post"
    assert request.study_id == "stress_ethanol_cipro_growth"
    assert request.state_order == ("00", "10", "01", "11")
    assert request.source.reference_design_id == "reference"
    assert request.source.state_map_ref == "stress_states"
    assert request.event.estimate_method == "segment_gap_midpoint"
    assert request.quality.max_interior_gap_h == 0.75
    assert request.display.state_labels["01"] == "Ciprofloxacin"
    assert request.display.reference_anchor.design_id == "reference"


def test_request_rejects_blank_study_identity() -> None:
    payload = _payload()
    payload["study_id"] = " "

    with pytest.raises(ValueError, match="study_id"):
        ResponseWindowRequest.from_mapping(payload)


def test_request_rejects_metric_specific_reference_authority_fields() -> None:
    payload = _payload()
    payload["source"]["reference_authority_record_id"] = "sfxi_vec8/vec8"  # type: ignore[index]
    payload["source"]["reference_authority_contract_id"] = "sfxi.vec8.v3"  # type: ignore[index]

    with pytest.raises(ValueError, match="unknown fields"):
        ResponseWindowRequest.from_mapping(payload)


def test_request_rejects_implicit_or_reordered_state_ontology() -> None:
    payload = _payload()
    payload["state_order"] = ["00", "01", "10", "11"]

    with pytest.raises(ValueError, match="must be exactly"):
        ResponseWindowRequest.from_mapping(payload)


def test_request_requires_display_labels_for_every_state() -> None:
    payload = _payload()
    del payload["display"]["state_labels"]["01"]  # type: ignore[index]

    with pytest.raises(ValueError, match="state_labels"):
        ResponseWindowRequest.from_mapping(payload)


def test_request_rejects_display_anchor_that_disagrees_with_source() -> None:
    payload = _payload()
    payload["display"]["examples"][0]["design_id"] = "other-reference"  # type: ignore[index]

    with pytest.raises(ValueError, match="reference anchor"):
        ResponseWindowRequest.from_mapping(payload)
