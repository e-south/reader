from __future__ import annotations

import pandas as pd
import pytest

from reader.domains.plate_reader.analysis.response_window.contracts import EventSpec
from reader.domains.plate_reader.analysis.response_window.sources import resolve_event_interval


def _event_spec() -> EventSpec:
    return EventSpec(
        event_id="stress_addition",
        event_kind="perturbation_addition",
        segment_column="sheet_index",
        pre_segment_index=0,
        post_segment_index=1,
        estimate_method="segment_gap_midpoint",
        declaration="Stress was added between acquisition segments 0 and 1.",
    )


def test_resolve_event_interval_rejects_fractional_segment_indexes() -> None:
    frame = pd.DataFrame(
        {
            "sheet_index": [0.0, 0.0, 1.9, 1.9],
            "time": [0.0, 1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="segment indexes must be finite integers"):
        resolve_event_interval(frame, experiment_id="fractional-segment", event_spec=_event_spec())


def test_resolve_event_interval_rejects_missing_nullable_segment_indexes() -> None:
    frame = pd.DataFrame(
        {
            "sheet_index": pd.Series([0, 0, 1, pd.NA], dtype="Int64"),
            "time": [0.0, 1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="segment indexes must be finite integers"):
        resolve_event_interval(frame, experiment_id="missing-segment", event_spec=_event_spec())


def test_resolve_event_interval_accepts_exact_integral_float_indexes() -> None:
    frame = pd.DataFrame(
        {
            "sheet_index": [0.0, 0.0, 1.0, 1.0],
            "time": [0.0, 1.0, 2.0, 3.0],
        }
    )

    event = resolve_event_interval(frame, experiment_id="integral-segment", event_spec=_event_spec())

    assert event.interval_start_assay_h == 1.0
    assert event.interval_end_assay_h == 2.0
