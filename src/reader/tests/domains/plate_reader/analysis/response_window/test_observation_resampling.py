from __future__ import annotations

import numpy as np
import pandas as pd

from reader.domains.plate_reader.analysis.response_window.contracts import ResponseWindowAnalysisSpec
from reader.domains.plate_reader.analysis.response_window.observation_resampling import (
    descriptive_resampling_records,
    joint_state_descriptive_resampling_draws,
)
from reader.domains.plate_reader.analysis.response_window.seeds import stable_seed
from reader.domains.plate_reader.analysis.response_window.sources import STATE_ORDER
from reader.tests.domains.plate_reader.analysis.response_window.test_response_window_contracts import _payload


def test_joint_descriptive_resampling_preserves_paired_observation_covariance() -> None:
    values = np.asarray([1.0, 2.0, 5.0])
    response, anchored_magnitude = joint_state_descriptive_resampling_draws(
        values,
        values,
        np.zeros(3),
        samples=200,
        stat="median",
        rng=np.random.default_rng(17),
    )

    np.testing.assert_array_equal(response, anchored_magnitude)


def test_reference_descriptive_resampling_draws_keep_anchored_magnitude_at_zero() -> None:
    request = ResponseWindowAnalysisSpec.from_mapping(_payload())
    records = []
    for state_index, state in enumerate(STATE_ORDER):
        for observation_index, magnitude in enumerate((1.0 + state_index, 4.0 + state_index)):
            records.append(
                {
                    "experiment_id": "20260101_example",
                    "design_id": "reference",
                    "state": state,
                    "position": f"{state}-{observation_index}",
                    "reduction_id": request.primary_reduction.id,
                    "response_well": magnitude / 2.0,
                    "magnitude_well": magnitude,
                }
            )

    draws = descriptive_resampling_records(pd.DataFrame.from_records(records), request=request)

    magnitude_columns = [f"b{state}" for state in STATE_ORDER]
    np.testing.assert_array_equal(
        draws[magnitude_columns].to_numpy(dtype=float),
        np.zeros((request.aggregation.descriptive_resampling_draws, len(STATE_ORDER))),
    )


def test_stable_seed_includes_experiment_identity() -> None:
    first = stable_seed(17, "experiment-a", "design", "00", "reduction")
    second = stable_seed(17, "experiment-b", "design", "00", "reduction")

    assert first != second
