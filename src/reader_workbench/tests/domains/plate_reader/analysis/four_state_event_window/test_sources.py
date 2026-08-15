from __future__ import annotations

import pandas as pd
import pytest

from reader_workbench.domains.plate_reader.analysis.four_state_event_window.contracts import (
    EventSpec,
    FourStateEventWindowAnalysisSpec,
    FourStateEventWindowSourceSpec,
)
from reader_workbench.domains.plate_reader.analysis.four_state_event_window.materialize import (
    materialize_experiment,
)
from reader_workbench.domains.plate_reader.analysis.four_state_event_window.sources import (
    _normalize_value_provenance,
    build_experiment_source,
    resolve_event_interval,
)


def _event_spec() -> EventSpec:
    return EventSpec(
        event_id="addition",
        event_kind="intervention",
        segment_column="segment",
        pre_segment_index=0,
        post_segment_index=1,
        estimate_method="segment_gap_midpoint",
        declaration="The event occurred between segments 0 and 1.",
    )


def _signal(channel: str) -> pd.DataFrame:
    rows = []
    for condition in ("none", "a", "b", "a+b"):
        rows.extend(
            {
                "design_id": "reference",
                "position": f"{condition}-{index}",
                "time": time,
                "channel": channel,
                "value": float(index + 1),
                "condition": condition,
                "segment": 0 if index < 2 else 1,
                "value_policy_clipped": False,
                "value_instrument_overflow": False,
                "value_bound_kind": "exact",
            }
            for index, time in enumerate((0.0, 1.0, 2.0, 3.0))
        )
    return pd.DataFrame(rows)


def _signals_for_designs(channel: str, *design_ids: str) -> pd.DataFrame:
    frames = []
    for design_id in design_ids:
        frame = _signal(channel).copy()
        frame["design_id"] = design_id
        frame["position"] = design_id + "-" + frame["position"].astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_build_experiment_source_consumes_resolved_frames_without_workspace_paths() -> None:
    spec = FourStateEventWindowSourceSpec(
        response_channel="response",
        magnitude_channel="magnitude",
        growth_channel="growth",
        reference_design_id="reference",
        state_column="condition",
        state_values={"00": "none", "10": "a", "01": "b", "11": "a+b"},
        state_labels={"00": "None", "10": "A", "01": "B", "11": "A + B"},
    )

    source = build_experiment_source(
        experiment_id="source-a",
        response_frame=_signal("response"),
        magnitude_frame=_signal("magnitude"),
        trajectory_frame=_signal("growth"),
        source_spec=spec,
        event_spec=_event_spec(),
    )

    assert source.experiment_id == "source-a"
    assert set(source.response["state"]) == {"00", "10", "01", "11"}
    assert source.event.estimate_assay_h == 1.5
    assert not hasattr(source, "config_path")
    assert not hasattr(source, "records_path")


def test_resolve_event_interval_rejects_fractional_segment_indexes() -> None:
    frame = pd.DataFrame({"segment": [0.0, 0.0, 1.9, 1.9], "time": [0.0, 1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="finite integers"):
        resolve_event_interval(frame, experiment_id="source-a", event_spec=_event_spec())


def test_four_state_event_window_source_rejects_missing_value_provenance() -> None:
    with pytest.raises(ValueError, match="missing required value provenance columns"):
        _normalize_value_provenance(pd.DataFrame({"value": [1.0]}), context="source:response")


def test_build_experiment_source_preserves_every_measured_design() -> None:
    spec = FourStateEventWindowSourceSpec.from_mapping(
        {
            "response_channel": "response",
            "magnitude_channel": "magnitude",
            "growth_channel": "growth",
            "reference_design_id": "reference",
            "state_column": "condition",
            "state_values": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
        }
    )

    source = build_experiment_source(
        experiment_id="source-a",
        response_frame=_signals_for_designs("response", "reference", "candidate-a", "excluded"),
        magnitude_frame=_signals_for_designs("magnitude", "reference", "candidate-a", "excluded"),
        trajectory_frame=_signals_for_designs("growth", "reference", "candidate-a", "excluded"),
        source_spec=spec,
        event_spec=_event_spec(),
    )

    assert set(source.response["design_id"]) == {"reference", "candidate-a", "excluded"}
    assert set(source.magnitude["design_id"]) == {"reference", "candidate-a", "excluded"}
    assert set(source.trajectory["design_id"]) == {"reference", "candidate-a", "excluded"}


def test_explicit_well_exclusion_allows_the_declared_state_minimum() -> None:
    payload = {
        "source": {
            "response_channel": "response",
            "magnitude_channel": "magnitude",
            "growth_channel": "growth",
            "reference_design_id": "reference",
            "state_column": "condition",
            "state_values": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
            "well_exclusions": [
                {
                    "experiment_id": "source-a",
                    "design_id": "candidate",
                    "state": "01",
                    "position": "candidate-b-0",
                    "reduction_id": "primary",
                    "reason": "insufficient_event_window_coverage",
                }
            ],
        },
        "event": {
            "event_id": "addition",
            "event_kind": "intervention",
            "segment_column": "segment",
            "pre_segment_index": 0,
            "post_segment_index": 1,
            "estimate_method": "segment_gap_midpoint",
            "declaration": "The event occurred between segments 0 and 1.",
        },
        "reductions": [
            {
                "id": "primary",
                "window_start_event_h": 1.0,
                "window_end_event_h": 2.0,
                "method": "geometric_time_mean",
                "response_basis": "post_window",
                "role": "primary",
            }
        ],
        "aggregation": {
            "observation_stat": "median",
            "descriptive_resampling_draws": 100,
            "descriptive_interval_mass": 0.9,
            "random_seed": 17,
        },
        "quality": {
            "positive_floor": 1.0e-12,
            "max_interior_gap_h": 1.0,
            "min_observations_per_state": 2,
        },
    }
    request = FourStateEventWindowAnalysisSpec.from_mapping(payload)
    response = _window_signals("response")
    magnitude = _window_signals("magnitude")
    trajectory = _window_signals("growth")
    source = build_experiment_source(
        experiment_id="source-a",
        response_frame=response,
        magnitude_frame=magnitude,
        trajectory_frame=trajectory,
        source_spec=request.source,
        event_spec=request.event,
    )

    wells, designs, _, traces, _, dispositions = materialize_experiment(source, request=request)

    candidate = designs.loc[designs["design_id"].eq("candidate")].iloc[0]
    assert candidate["n01"] == 2
    assert "candidate-b-0" not in set(wells["position"])
    assert "candidate-b-0" in set(traces["position"])
    assert dispositions[["scope", "position", "reason"]].to_dict("records") == [
        {
            "scope": "well",
            "position": "candidate-b-0",
            "reason": "insufficient_event_window_coverage",
        }
    ]


def test_plate_scoped_design_disposition_preserves_partial_measurements() -> None:
    payload = {
        "source": {
            "response_channel": "response",
            "magnitude_channel": "magnitude",
            "growth_channel": "growth",
            "reference_design_id": "reference",
            "state_column": "condition",
            "state_values": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
            "well_exclusions": [
                {
                    "experiment_id": "source-a",
                    "design_id": "candidate",
                    "state": "01",
                    "position": position,
                    "reduction_id": "primary",
                    "reason": "insufficient_event_window_coverage",
                }
                for position in ("candidate-b-0", "candidate-b-1")
            ],
            "design_dispositions": [
                {
                    "experiment_id": "source-a",
                    "design_id": "candidate",
                    "state": "01",
                    "reduction_id": "primary",
                    "reason": "insufficient_state_observations",
                }
            ],
        },
        "event": {
            "event_id": "addition",
            "event_kind": "intervention",
            "segment_column": "segment",
            "pre_segment_index": 0,
            "post_segment_index": 1,
            "estimate_method": "segment_gap_midpoint",
            "declaration": "The event occurred between segments 0 and 1.",
        },
        "reductions": [
            {
                "id": "primary",
                "window_start_event_h": 1.0,
                "window_end_event_h": 2.0,
                "method": "geometric_time_mean",
                "response_basis": "post_window",
                "role": "primary",
            }
        ],
        "aggregation": {
            "observation_stat": "median",
            "descriptive_resampling_draws": 100,
            "descriptive_interval_mass": 0.9,
            "random_seed": 17,
        },
        "quality": {
            "positive_floor": 1.0e-12,
            "max_interior_gap_h": 1.0,
            "min_observations_per_state": 2,
        },
    }
    request = FourStateEventWindowAnalysisSpec.from_mapping(payload)
    response = _window_signals("response")
    magnitude = _window_signals("magnitude")
    trajectory = _window_signals("growth")
    for frame in (response, magnitude, trajectory):
        frame.drop(frame.index[frame["position"].eq("candidate-b-1") & frame["time"].eq(4.0)], inplace=True)
    source = build_experiment_source(
        experiment_id="source-a",
        response_frame=response,
        magnitude_frame=magnitude,
        trajectory_frame=trajectory,
        source_spec=request.source,
        event_spec=request.event,
    )

    wells, designs, draws, traces, _, dispositions = materialize_experiment(source, request=request)

    assert "candidate" in set(traces["design_id"])
    assert "candidate" in set(wells["design_id"])
    assert "candidate" not in set(designs["design_id"])
    assert "candidate" not in set(draws["design_id"])
    design_disposition = dispositions.loc[dispositions["scope"].eq("design")].iloc[0]
    assert design_disposition["state"] == "01"
    assert design_disposition["observed_count"] == 1
    assert design_disposition["required_count"] == 2


def test_incomplete_well_fails_without_an_explicit_exclusion() -> None:
    response = _window_signals("response")
    magnitude = _window_signals("magnitude")
    trajectory = _window_signals("growth")
    spec = FourStateEventWindowSourceSpec.from_mapping(
        {
            "response_channel": "response",
            "magnitude_channel": "magnitude",
            "growth_channel": "growth",
            "reference_design_id": "reference",
            "state_column": "condition",
            "state_values": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
        }
    )
    source = build_experiment_source(
        experiment_id="source-a",
        response_frame=response,
        magnitude_frame=magnitude,
        trajectory_frame=trajectory,
        source_spec=spec,
        event_spec=_event_spec(),
    )
    payload = {
        "source": {
            "response_channel": "response",
            "magnitude_channel": "magnitude",
            "growth_channel": "growth",
            "reference_design_id": "reference",
            "state_column": "condition",
            "state_values": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
        },
        "event": {
            "event_id": "addition",
            "event_kind": "intervention",
            "segment_column": "segment",
            "pre_segment_index": 0,
            "post_segment_index": 1,
            "estimate_method": "segment_gap_midpoint",
            "declaration": "The event occurred between segments 0 and 1.",
        },
        "reductions": [
            {
                "id": "primary",
                "window_start_event_h": 1.0,
                "window_end_event_h": 2.0,
                "method": "geometric_time_mean",
                "response_basis": "post_window",
                "role": "primary",
            }
        ],
        "aggregation": {
            "observation_stat": "median",
            "descriptive_resampling_draws": 100,
            "descriptive_interval_mass": 0.9,
            "random_seed": 17,
        },
        "quality": {
            "positive_floor": 1.0e-12,
            "max_interior_gap_h": 1.0,
            "min_observations_per_state": 2,
        },
    }

    with pytest.raises(ValueError, match="does not cover"):
        materialize_experiment(source, request=FourStateEventWindowAnalysisSpec.from_mapping(payload))


def _window_signals(channel: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    state_values = {"00": "none", "10": "a", "01": "b", "11": "a+b"}
    for design_id in ("reference", "candidate"):
        for condition in state_values.values():
            for replicate in range(3):
                position = f"{design_id}-{condition}-{replicate}"
                times = (0.0, 1.0, 2.0, 3.0) if position == "candidate-b-0" else (0.0, 1.0, 2.0, 3.0, 4.0)
                for time in times:
                    rows.append(
                        {
                            "design_id": design_id,
                            "position": position,
                            "time": time,
                            "channel": channel,
                            "value": 2.0 + replicate + time,
                            "condition": condition,
                            "segment": 0 if time <= 1.0 else 1,
                            "value_policy_clipped": False,
                            "value_instrument_overflow": False,
                            "value_bound_kind": "exact",
                        }
                    )
    return pd.DataFrame.from_records(rows)
