from __future__ import annotations

import pytest

from reader_workbench.domains.plate_reader.analysis.four_state_event_window.contracts import (
    FourStateEventWindowAnalysisSpec,
)
from reader_workbench.plugins.transform.four_state_event_window import FourStateEventWindowTransform
from reader_workbench.protocols import builtin_protocol_catalog


def _payload() -> dict[str, object]:
    return {
        "source": {
            "response_channel": "response_ratio",
            "magnitude_channel": "magnitude_ratio",
            "growth_channel": "growth",
            "reference_design_id": "reference",
            "state_column": "condition",
            "state_values": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
            "state_labels": {"00": "No treatment", "10": "Treatment A", "01": "Treatment B", "11": "A + B"},
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
            "observation_stat": "median",
            "descriptive_resampling_draws": 200,
            "descriptive_interval_mass": 0.9,
            "random_seed": 17,
        },
        "quality": {
            "positive_floor": 1.0e-12,
            "max_interior_gap_h": 0.75,
            "min_observations_per_state": 2,
        },
    }


def test_analysis_spec_is_domain_config_not_a_parallel_request_schema() -> None:
    spec = FourStateEventWindowAnalysisSpec.from_mapping(_payload())

    assert spec.primary_reduction.id == "primary"
    assert spec.source.state_values == {"00": "none", "01": "b", "10": "a", "11": "a+b"}
    assert spec.source.state_labels == {
        "00": "No treatment",
        "01": "Treatment B",
        "10": "Treatment A",
        "11": "A + B",
    }
    assert not hasattr(spec, "experiment_ids")
    assert not hasattr(spec, "study_id")
    assert not hasattr(spec, "request_id")


def test_source_labels_default_to_source_values() -> None:
    payload = _payload()
    del payload["source"]["state_labels"]  # type: ignore[index]

    spec = FourStateEventWindowAnalysisSpec.from_mapping(payload)

    assert spec.source.state_labels == spec.source.state_values


def test_source_rejects_design_prefilters_that_would_remove_measured_traces() -> None:
    payload = _payload()
    payload["source"]["included_design_ids_by_experiment"] = {  # type: ignore[index]
        "experiment-b": ["reference", "candidate-b"],
    }

    with pytest.raises(ValueError, match="unknown fields.*included_design_ids_by_experiment"):
        FourStateEventWindowAnalysisSpec.from_mapping(payload)


def test_source_accepts_exact_well_exclusions() -> None:
    payload = _payload()
    payload["source"]["well_exclusions"] = [  # type: ignore[index]
        {
            "experiment_id": "experiment-b",
            "design_id": "candidate-b",
            "state": "01",
            "position": "D7",
            "reduction_id": "primary",
            "reason": "insufficient_event_window_coverage",
        }
    ]

    spec = FourStateEventWindowAnalysisSpec.from_mapping(payload)

    assert spec.source.well_exclusions[0].position == "D7"
    assert spec.source.well_exclusions[0].reason == "insufficient_event_window_coverage"
    spec.source.require_known_experiment_ids(("experiment-a", "experiment-b"))


def test_source_accepts_plate_scoped_design_dispositions() -> None:
    payload = _payload()
    payload["source"]["design_dispositions"] = [  # type: ignore[index]
        {
            "experiment_id": "experiment-b",
            "design_id": "candidate-b",
            "state": "01",
            "reduction_id": "primary",
            "reason": "insufficient_state_observations",
        }
    ]

    spec = FourStateEventWindowAnalysisSpec.from_mapping(payload)

    assert spec.source.design_dispositions[0].design_id == "candidate-b"
    assert spec.source.design_dispositions[0].state == "01"
    spec.source.require_known_experiment_ids(("experiment-a", "experiment-b"))


def test_analysis_rejects_design_disposition_for_unknown_reduction() -> None:
    payload = _payload()
    payload["source"]["design_dispositions"] = [  # type: ignore[index]
        {
            "experiment_id": "experiment-b",
            "design_id": "candidate-b",
            "state": "01",
            "reduction_id": "misspelled",
            "reason": "insufficient_state_observations",
        }
    ]

    with pytest.raises(ValueError, match="design disposition names unknown reduction 'misspelled'"):
        FourStateEventWindowAnalysisSpec.from_mapping(payload)


def test_source_rejects_duplicate_well_exclusions() -> None:
    payload = _payload()
    exclusion = {
        "experiment_id": "experiment-b",
        "design_id": "candidate-b",
        "state": "01",
        "position": "D7",
        "reduction_id": "primary",
        "reason": "insufficient_event_window_coverage",
    }
    payload["source"]["well_exclusions"] = [exclusion, exclusion.copy()]  # type: ignore[index]

    with pytest.raises(ValueError, match="well_exclusions must not repeat"):
        FourStateEventWindowAnalysisSpec.from_mapping(payload)


def test_analysis_requires_one_primary_reduction() -> None:
    payload = _payload()
    payload["reductions"][0]["role"] = "sensitivity"  # type: ignore[index]

    with pytest.raises(ValueError, match="exactly one primary reduction"):
        FourStateEventWindowAnalysisSpec.from_mapping(payload)


def test_analysis_rejects_unknown_fields() -> None:
    payload = _payload()
    payload["study_id"] = "belongs-elsewhere"

    with pytest.raises(ValueError, match="unknown fields"):
        FourStateEventWindowAnalysisSpec.from_mapping(payload)


@pytest.mark.parametrize(
    ("section", "retired_key", "replacement_key"),
    [
        ("aggregation", "replicate_stat", "observation_stat"),
        ("aggregation", "bootstrap_samples", "descriptive_resampling_draws"),
        ("aggregation", "confidence_level", "descriptive_interval_mass"),
        ("quality", "min_replicates_per_state", "min_observations_per_state"),
    ],
)
def test_analysis_rejects_retired_replicate_and_inferential_keys(
    section: str,
    retired_key: str,
    replacement_key: str,
) -> None:
    payload = _payload()
    section_payload = payload[section]
    assert isinstance(section_payload, dict)
    section_payload[retired_key] = section_payload.pop(replacement_key)

    with pytest.raises(ValueError, match=rf"unknown fields.*{retired_key}"):
        FourStateEventWindowAnalysisSpec.from_mapping(payload)


def test_analysis_exposes_observation_and_descriptive_resampling_policy() -> None:
    spec = FourStateEventWindowAnalysisSpec.from_mapping(_payload())

    assert spec.aggregation.observation_stat == "median"
    assert spec.aggregation.descriptive_resampling_draws == 200
    assert spec.aggregation.descriptive_interval_mass == 0.9
    assert spec.quality.min_observations_per_state == 2
    assert not hasattr(spec.aggregation, "replicate_stat")
    assert not hasattr(spec.quality, "min_replicates_per_state")


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("aggregation", "observation_stat", "mode", "observation_stat is unsupported"),
        ("aggregation", "descriptive_resampling_draws", 99, "integer of at least 100"),
        ("aggregation", "descriptive_resampling_draws", True, "integer of at least 100"),
        ("aggregation", "descriptive_interval_mass", 0.5, "between 0.5 and 1.0"),
        ("quality", "min_observations_per_state", 1, "integer of at least 2"),
        ("quality", "min_observations_per_state", True, "integer of at least 2"),
    ],
)
def test_analysis_rejects_invalid_observation_reduction_values(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _payload()
    section_payload = payload[section]
    assert isinstance(section_payload, dict)
    section_payload[field] = value

    with pytest.raises(ValueError, match=message):
        FourStateEventWindowAnalysisSpec.from_mapping(payload)


def test_protocol_and_transform_publish_only_observation_named_surfaces() -> None:
    descriptor = builtin_protocol_catalog().resolve("plate_reader/four_state_event_window")
    fields = {field.key: field for field in descriptor.analysis_fields}
    aggregation = fields["aggregation"].default
    quality = fields["quality"].default
    ports = FourStateEventWindowTransform.output_ports()

    assert isinstance(aggregation, dict)
    assert isinstance(quality, dict)
    assert set(aggregation) == {
        "observation_stat",
        "descriptive_resampling_draws",
        "descriptive_interval_mass",
        "random_seed",
    }
    assert "min_observations_per_state" in quality
    assert "descriptive_resampling_draws" in ports
    assert ports["dispositions"].contract == "plate_reader.four_state_event_window.dispositions.v1"
    assert "bootstrap_draws" not in ports


def test_analysis_requires_exact_four_state_mapping() -> None:
    payload = _payload()
    del payload["source"]["state_values"]["01"]  # type: ignore[index]

    with pytest.raises(ValueError, match="exactly 00, 10, 01, and 11"):
        FourStateEventWindowAnalysisSpec.from_mapping(payload)
