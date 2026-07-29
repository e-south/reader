from __future__ import annotations

import numpy as np
import pandas as pd

from reader.domains.plate_reader.analysis.response_window.contracts import ResponseWindowAnalysisSpec
from reader.domains.plate_reader.analysis.response_window.provenance import stable_seed
from reader.domains.plate_reader.analysis.response_window.sources import STATE_ORDER
from reader.domains.plate_reader.analysis.response_window.uncertainty import (
    bootstrap_draw_records,
    joint_state_bootstrap_draws,
    symmetric_event_sensitivity_half_width,
)
from reader.tests.domains.plate_reader.analysis.response_window.test_response_window_contracts import _payload


def test_joint_bootstrap_preserves_paired_well_covariance() -> None:
    values = np.asarray([1.0, 2.0, 5.0])
    response, relative_fluorescence = joint_state_bootstrap_draws(
        values,
        values,
        np.zeros(3),
        samples=200,
        stat="median",
        rng=np.random.default_rng(17),
    )

    np.testing.assert_array_equal(response, relative_fluorescence)


def test_reference_bootstrap_draws_keep_anchored_fluorescence_at_zero() -> None:
    request = ResponseWindowAnalysisSpec.from_mapping(_payload())
    records = []
    for state_index, state in enumerate(STATE_ORDER):
        for replicate_index, magnitude in enumerate((1.0 + state_index, 4.0 + state_index)):
            records.append(
                {
                    "experiment_id": "20260101_example",
                    "design_id": "reference",
                    "state": state,
                    "position": f"{state}-{replicate_index}",
                    "reduction_id": request.primary_reduction.id,
                    "response_well": magnitude / 2.0,
                    "magnitude_well": magnitude,
                }
            )

    draws = bootstrap_draw_records(pd.DataFrame.from_records(records), request=request)

    fluorescence_columns = [f"b{state}" for state in STATE_ORDER]
    np.testing.assert_array_equal(
        draws[fluorescence_columns].to_numpy(dtype=float),
        np.zeros((request.aggregation.bootstrap_samples, len(STATE_ORDER))),
    )


def test_stable_seed_includes_experiment_identity() -> None:
    first = stable_seed(17, "experiment-a", "design", "00", "reduction")
    second = stable_seed(17, "experiment-b", "design", "00", "reduction")

    assert first != second


def test_event_sensitivity_envelope_covers_asymmetric_bound_reductions() -> None:
    half_width = symmetric_event_sensitivity_half_width(
        midpoint=2.0,
        lower_bound=1.9,
        upper_bound=2.7,
    )

    assert np.isclose(half_width, 0.7)
    assert 2.0 - half_width <= 1.9
    assert 2.0 + half_width >= 2.7


def test_event_sensitivity_envelope_rejects_nonfinite_values() -> None:
    with np.testing.assert_raises_regex(ValueError, "must be finite"):
        symmetric_event_sensitivity_half_width(
            midpoint=0.0,
            lower_bound=float("nan"),
            upper_bound=1.0,
        )
