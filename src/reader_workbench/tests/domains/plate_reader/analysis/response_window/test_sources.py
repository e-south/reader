from __future__ import annotations

import pandas as pd
import pytest

from reader_workbench.domains.plate_reader.analysis.response_window.contracts import EventSpec, ResponseWindowSourceSpec
from reader_workbench.domains.plate_reader.analysis.response_window.sources import (
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


def test_build_experiment_source_consumes_resolved_frames_without_workspace_paths() -> None:
    spec = ResponseWindowSourceSpec(
        response_channel="response",
        magnitude_channel="magnitude",
        growth_channel="growth",
        reference_design_id="reference",
        state_column="condition",
        state_values={"00": "none", "10": "a", "01": "b", "11": "a+b"},
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


def test_response_window_source_rejects_missing_value_provenance() -> None:
    with pytest.raises(ValueError, match="missing required value provenance columns"):
        _normalize_value_provenance(pd.DataFrame({"value": [1.0]}), context="source:response")
