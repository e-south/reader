from __future__ import annotations

import numpy as np
import pytest

from reader_workbench.domains.time_series import (
    EndpointSelection,
    IntervalSelection,
    TemporalReductionSpec,
    TemporalSupportPolicy,
    reduce_temporal_trace,
)


def _support(
    *,
    boundary: str = "observed",
    minimum: int = 2,
    gap: float | None = 1.0,
    floor: float | None = None,
    positive_scope: str = "selected_support",
):
    return TemporalSupportPolicy(
        boundary_support=boundary,
        minimum_observations=minimum,
        maximum_interior_gap_h=gap,
        positive_floor=floor,
        positive_value_scope=positive_scope,
        censored_values="reject",
    )


def test_observed_median_reduction_is_linear_and_inclusive() -> None:
    spec = TemporalReductionSpec(
        selection=IntervalSelection(time_basis="absolute", start_h=1.0, end_h=3.0),
        method="observed_median",
        output_space="linear",
        support=_support(minimum=3),
    )

    result = reduce_temporal_trace(
        np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.asarray([1.0, 3.0, 9.0, 5.0, 11.0]),
        spec=spec,
        trace_id="observed",
    )

    assert result.value == 5.0
    assert result.observed_point_count == 3
    assert result.evaluation_point_count == 3


@pytest.mark.parametrize(
    ("gap_delta_h", "accepted"),
    [
        (0.0, True),
        (0.5e-9, True),
        (2.0e-9, False),
    ],
    ids=["exact", "within-absolute-time-tolerance", "over-absolute-time-tolerance"],
)
def test_maximum_interior_gap_uses_absolute_time_tolerance(gap_delta_h: float, accepted: bool) -> None:
    spec = TemporalReductionSpec(
        selection=IntervalSelection(time_basis="absolute", start_h=0.0, end_h=2.0),
        method="observed_median",
        output_space="linear",
        support=_support(minimum=3, gap=1.0),
    )
    times = np.asarray([0.0, 1.0 + gap_delta_h, 2.0])
    values = np.asarray([1.0, 2.0, 3.0])

    if accepted:
        result = reduce_temporal_trace(times, values, spec=spec, trace_id="gap-tolerance")
        assert result.value == 2.0
    else:
        with pytest.raises(ValueError, match="interior gap"):
            reduce_temporal_trace(times, values, spec=spec, trace_id="gap-tolerance")


def test_endpoint_selection_is_explicit_and_exact() -> None:
    spec = TemporalReductionSpec(
        selection=EndpointSelection(time_basis="absolute", time_h=2.0, mode="exact", tolerance_h=0.0),
        method="identity",
        output_space="linear",
        support=_support(boundary="none", minimum=1, gap=None),
    )

    result = reduce_temporal_trace(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 4.0, 8.0]),
        spec=spec,
        trace_id="endpoint",
    )

    assert result.value == 4.0


@pytest.mark.parametrize("method", ["observed_mean", "observed_median"])
def test_observed_reduction_contract_rejects_zero_minimum_observations(method: str) -> None:
    with pytest.raises(ValueError, match="at least one minimum observation"):
        TemporalReductionSpec(
            selection=IntervalSelection(time_basis="absolute", start_h=1.0, end_h=2.0),
            method=method,
            output_space="linear",
            support=_support(boundary="none", minimum=0, gap=None),
        )


@pytest.mark.parametrize("output_space", ["linear", "log2"])
def test_endpoint_enforces_entire_trace_positive_value_scope(output_space: str) -> None:
    spec = TemporalReductionSpec(
        selection=EndpointSelection(time_basis="absolute", time_h=2.0, mode="exact", tolerance_h=0.0),
        method="identity",
        output_space=output_space,
        support=_support(
            boundary="none",
            minimum=1,
            gap=None,
            floor=1.0e-12,
            positive_scope="entire_trace",
        ),
    )

    with pytest.raises(ValueError, match="positive floor"):
        reduce_temporal_trace(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([0.0, 4.0, 8.0]),
            spec=spec,
            trace_id="endpoint-entire-trace",
        )


def test_temporal_result_uses_neutral_evaluation_vocabulary_only() -> None:
    spec = TemporalReductionSpec(
        selection=EndpointSelection(time_basis="absolute", time_h=2.0, mode="exact", tolerance_h=0.0),
        method="identity",
        output_space="linear",
        support=_support(boundary="none", minimum=1, gap=None),
    )

    result = reduce_temporal_trace(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 4.0, 8.0]),
        spec=spec,
        trace_id="neutral-result",
    )

    assert result.evaluation_point_count == 1
    assert not hasattr(result, "integration_point_count")


@pytest.mark.parametrize(
    ("times", "bound_kinds", "expected_bound"),
    [
        ([-5.0e-10, 1.0, 2.0], ["lower", "exact", "exact"], "lower"),
        ([0.0, 1.0, 2.0 + 5.0e-10], ["exact", "exact", "upper"], "upper"),
    ],
)
def test_observed_boundary_tolerance_includes_boundary_provenance(
    times: list[float],
    bound_kinds: list[str],
    expected_bound: str,
) -> None:
    spec = TemporalReductionSpec(
        selection=IntervalSelection(time_basis="absolute", start_h=0.0, end_h=2.0),
        method="observed_mean",
        output_space="linear",
        support=_support(minimum=3, gap=1.0),
    )

    result = reduce_temporal_trace(
        np.asarray(times),
        np.asarray([1.0, 2.0, 3.0]),
        spec=spec,
        trace_id="tolerant-boundary",
        bound_kinds=np.asarray(bound_kinds, dtype=object),
    )

    assert result.observed_point_count == 3
    assert result.evaluation_point_count == 3
    assert result.bound_kind == expected_bound


def test_observed_boundary_rejects_ambiguous_equivalent_observations() -> None:
    spec = TemporalReductionSpec(
        selection=IntervalSelection(time_basis="absolute", start_h=0.0, end_h=2.0),
        method="observed_mean",
        output_space="linear",
        support=_support(minimum=3, gap=1.0),
    )

    with pytest.raises(ValueError, match="ambiguous observations at the interval start"):
        reduce_temporal_trace(
            np.asarray([-5.0e-10, 0.0, 1.0, 2.0]),
            np.asarray([1.0, 1.0, 2.0, 3.0]),
            spec=spec,
            trace_id="ambiguous-boundary",
        )


def test_observed_boundary_tolerance_includes_boundary_censor_provenance() -> None:
    spec = TemporalReductionSpec(
        selection=IntervalSelection(time_basis="absolute", start_h=0.0, end_h=2.0),
        method="observed_mean",
        output_space="linear",
        support=_support(minimum=3, gap=1.0),
    )

    with pytest.raises(ValueError, match="clipped or overflowed"):
        reduce_temporal_trace(
            np.asarray([-5.0e-10, 1.0, 2.0]),
            np.asarray([1.0, 2.0, 3.0]),
            spec=spec,
            trace_id="tolerant-censor",
            policy_clipped=np.asarray([True, False, False]),
            bound_kinds=np.asarray(["lower", "exact", "exact"], dtype=object),
        )


def test_observed_boundary_tolerance_drives_selected_support_positivity() -> None:
    spec = TemporalReductionSpec(
        selection=IntervalSelection(time_basis="absolute", start_h=0.0, end_h=2.0),
        method="geometric_time_mean",
        output_space="log2",
        support=_support(minimum=2, gap=1.1, floor=1.0e-12),
    )

    with pytest.raises(ValueError, match="positive floor"):
        reduce_temporal_trace(
            np.asarray([-5.0e-10, 1.0, 2.0]),
            np.asarray([-1.0, 2.0, 3.0]),
            spec=spec,
            trace_id="tolerant-positive-support",
        )


def test_temporal_reduction_rejects_nonfinite_numeric_result() -> None:
    maximum = np.finfo(float).max
    spec = TemporalReductionSpec(
        selection=IntervalSelection(time_basis="absolute", start_h=0.0, end_h=1.0),
        method="observed_mean",
        output_space="linear",
        support=_support(minimum=2, gap=1.0),
    )

    with pytest.raises(ValueError, match="produced a non-finite value"):
        reduce_temporal_trace(
            np.asarray([0.0, 1.0]),
            np.asarray([maximum, maximum]),
            spec=spec,
            trace_id="overflowed-reduction",
        )


def test_observed_and_time_weighted_reductions_diverge_on_uneven_sampling() -> None:
    times = np.asarray([0.0, 1.0, 4.0])
    values = np.asarray([1.0, 8.0, 8.0])
    observed = TemporalReductionSpec(
        selection=IntervalSelection(time_basis="absolute", start_h=0.0, end_h=4.0),
        method="observed_mean",
        output_space="linear",
        support=_support(minimum=3, gap=3.0),
    )
    weighted = TemporalReductionSpec(
        selection=IntervalSelection(time_basis="absolute", start_h=0.0, end_h=4.0),
        method="integrated_linear_mean",
        output_space="log2",
        support=_support(minimum=3, gap=3.0, floor=1.0e-12),
    )

    observed_result = reduce_temporal_trace(times, values, spec=observed, trace_id="observed")
    weighted_result = reduce_temporal_trace(times, values, spec=weighted, trace_id="weighted")

    assert observed_result.value == pytest.approx(17.0 / 3.0)
    assert 2**weighted_result.value == pytest.approx(57.0 / 8.0)
    assert observed_result.value != pytest.approx(2**weighted_result.value)


def test_observed_boundary_policy_fails_when_interval_end_is_only_bracketed() -> None:
    spec = TemporalReductionSpec(
        selection=IntervalSelection(time_basis="absolute", start_h=0.0, end_h=2.0),
        method="observed_median",
        output_space="linear",
        support=_support(minimum=2),
    )

    with pytest.raises(ValueError, match="requires observed interval boundaries"):
        reduce_temporal_trace(
            np.asarray([0.0, 1.0, 3.0]),
            np.asarray([1.0, 2.0, 4.0]),
            spec=spec,
            trace_id="strict-boundary",
        )
