from __future__ import annotations

import pytest

from reader.domains.plate_reader.analysis.response_window.contracts import ResponseWindowAnalysisSpec


def _payload() -> dict[str, object]:
    return {
        "source": {
            "response_channel": "response_ratio",
            "magnitude_channel": "magnitude_ratio",
            "growth_channel": "growth",
            "reference_design_id": "reference",
            "state_column": "condition",
            "state_values": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
            "state_values_case_sensitive": True,
        },
        "event": {
            "event_id": "intervention",
            "event_kind": "addition",
            "segment_column": "segment",
            "pre_segment_index": 0,
            "post_segment_index": 1,
            "estimate_method": "segment_gap_midpoint",
            "declaration": "The event occurred between acquisition segments 0 and 1.",
        },
        "reductions": [
            {
                "id": "primary",
                "window_start_event_h": 1.0,
                "window_end_event_h": 2.0,
                "method": "geometric_time_mean",
                "response_basis": "post_window",
                "role": "primary",
            },
            {
                "id": "sensitivity",
                "window_start_event_h": 1.0,
                "window_end_event_h": 2.0,
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


def test_analysis_spec_is_domain_config_not_a_parallel_request_schema() -> None:
    spec = ResponseWindowAnalysisSpec.from_mapping(_payload())

    assert spec.primary_reduction.id == "primary"
    assert spec.source.state_values == {"00": "none", "01": "b", "10": "a", "11": "a+b"}
    assert not hasattr(spec, "experiment_ids")
    assert not hasattr(spec, "study_id")
    assert not hasattr(spec, "request_id")


def test_analysis_requires_one_primary_reduction() -> None:
    payload = _payload()
    payload["reductions"][0]["role"] = "sensitivity"  # type: ignore[index]

    with pytest.raises(ValueError, match="exactly one primary reduction"):
        ResponseWindowAnalysisSpec.from_mapping(payload)


def test_analysis_rejects_unknown_fields() -> None:
    payload = _payload()
    payload["study_id"] = "belongs-elsewhere"

    with pytest.raises(ValueError, match="unknown fields"):
        ResponseWindowAnalysisSpec.from_mapping(payload)


def test_analysis_requires_exact_four_state_mapping() -> None:
    payload = _payload()
    del payload["source"]["state_values"]["01"]  # type: ignore[index]

    with pytest.raises(ValueError, match="exactly 00, 10, 01, and 11"):
        ResponseWindowAnalysisSpec.from_mapping(payload)
