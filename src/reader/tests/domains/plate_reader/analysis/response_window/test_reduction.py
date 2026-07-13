from __future__ import annotations

import numpy as np
import pytest

from reader.domains.plate_reader.analysis.response_window.reduction import summarize_trace


def test_geometric_time_mean_preserves_constant_ratio() -> None:
    summary = summarize_trace(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([4.0, 4.0, 4.0]),
        window_start_h=0.5,
        window_end_h=1.5,
        method="geometric_time_mean",
        positive_floor=1.0e-12,
        max_interior_gap_h=1.0,
        trace_id="constant",
    )

    assert summary.value == pytest.approx(2.0)
    assert summary.observed_point_count == 1
    assert summary.max_interior_gap_h == pytest.approx(1.0)


def test_linear_mean_integrates_in_linear_space_before_log() -> None:
    summary = summarize_trace(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([1.0, 2.0, 4.0]),
        window_start_h=0.0,
        window_end_h=2.0,
        method="integrated_linear_mean",
        positive_floor=1.0e-12,
        max_interior_gap_h=1.0,
        trace_id="linear",
    )

    assert summary.value == pytest.approx(np.log2(2.25))


@pytest.mark.parametrize(
    ("times", "values", "message"),
    [
        ([1.0, 2.0], [1.0, 2.0], "does not cover"),
        ([0.0, 1.0, 2.0], [1.0, 0.0, 2.0], "positive floor"),
        ([0.0, 0.5, 2.0], [1.0, 1.0, 1.0], "interior gap"),
        ([0.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], "duplicate time"),
    ],
)
def test_trace_reduction_fails_on_invalid_support(
    times: list[float],
    values: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_trace(
            np.asarray(times),
            np.asarray(values),
            window_start_h=0.0,
            window_end_h=2.0,
            method="geometric_time_mean",
            positive_floor=1.0e-12,
            max_interior_gap_h=1.0,
            trace_id="invalid",
        )
