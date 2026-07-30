from __future__ import annotations

import numpy as np
import pytest

from reader.domains.plate_reader.analysis.response_window.contracts import QualitySpec, ReductionSpec
from reader.domains.plate_reader.analysis.response_window.reduction import response_window_temporal_spec
from reader.domains.time_series import reduce_temporal_trace


def _spec(method: str, *, start_h: float = 0.0, end_h: float = 2.0):
    reduction = ReductionSpec(
        id="primary",
        window_start_event_h=start_h,
        window_end_event_h=end_h,
        method=method,
        response_basis="post_window",
        role="primary",
    )
    quality = QualitySpec(positive_floor=1.0e-12, max_interior_gap_h=1.0, min_replicates_per_state=2)
    return response_window_temporal_spec(reduction, quality)


def test_geometric_time_mean_preserves_constant_signal() -> None:
    summary = reduce_temporal_trace(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([4.0, 4.0, 4.0]),
        spec=_spec("geometric_time_mean", start_h=0.5, end_h=1.5),
        origin_h=0.0,
        trace_id="constant",
    )

    assert summary.value == pytest.approx(2.0)
    assert summary.observed_point_count == 1
    assert summary.max_interior_gap_h == pytest.approx(1.0)


def test_linear_mean_integrates_in_linear_space_before_log() -> None:
    summary = reduce_temporal_trace(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([1.0, 2.0, 4.0]),
        spec=_spec("integrated_linear_mean"),
        origin_h=0.0,
        trace_id="linear",
    )

    assert summary.value == pytest.approx(np.log2(2.25))


@pytest.mark.parametrize("method", ["geometric_time_mean", "integrated_linear_mean"])
def test_response_window_adapter_matches_locked_legacy_numerical_reference(method: str) -> None:
    relative_times = np.asarray([-0.5, 0.0, 0.75, 1.5, 2.0])
    values = np.asarray([1.0, 2.0, 5.0, 9.0, 12.0])
    selection_start = 0.25
    selection_end = 1.25
    spec = _spec(method, start_h=selection_start, end_h=selection_end)

    summary = reduce_temporal_trace(
        relative_times + 7.0,
        values,
        spec=spec,
        origin_h=7.0,
        trace_id=f"legacy-parity:{method}",
    )

    inside = (relative_times > selection_start) & (relative_times < selection_end)
    window_times = np.concatenate(([selection_start], relative_times[inside], [selection_end]))
    duration = selection_end - selection_start
    if method == "geometric_time_mean":
        transformed = np.log2(values)
        window_values = np.concatenate(
            (
                [np.interp(selection_start, relative_times, transformed)],
                transformed[inside],
                [np.interp(selection_end, relative_times, transformed)],
            )
        )
        locked_reference = float(np.trapezoid(window_values, window_times) / duration)
    else:
        window_values = np.concatenate(
            (
                [np.interp(selection_start, relative_times, values)],
                values[inside],
                [np.interp(selection_end, relative_times, values)],
            )
        )
        locked_reference = float(np.log2(np.trapezoid(window_values, window_times) / duration))

    assert summary.value == locked_reference
    assert summary.evaluation_point_count == len(window_times)


def test_response_window_adapter_preserves_censor_provenance_projection() -> None:
    summary = reduce_temporal_trace(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([1.0, 2.0, 4.0]),
        spec=_spec("geometric_time_mean", start_h=0.5, end_h=1.5),
        origin_h=0.0,
        trace_id="provenance",
        policy_clipped=np.asarray([False, True, False]),
        instrument_overflow=np.asarray([False, False, False]),
        bound_kinds=np.asarray(["exact", "lower", "exact"], dtype=object),
    )

    assert summary.policy_clipped_point_count == 1
    assert summary.instrument_overflow_point_count == 0
    assert summary.bound_kind == "lower"


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
        reduce_temporal_trace(
            np.asarray(times),
            np.asarray(values),
            spec=_spec("geometric_time_mean"),
            origin_h=0.0,
            trace_id="invalid",
        )
