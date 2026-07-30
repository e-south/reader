from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reader_workbench.domains.plate_reader.analysis.response_window.contracts import ResponseWindowAnalysisSpec
from reader_workbench.domains.plate_reader.analysis.response_window.observation_resampling import (
    descriptive_resampling_records,
    joint_state_descriptive_resampling_draws,
)
from reader_workbench.domains.plate_reader.analysis.response_window.seeds import stable_seed
from reader_workbench.domains.plate_reader.analysis.response_window.sources import STATE_ORDER
from reader_workbench.tests.domains.plate_reader.analysis.response_window.test_response_window_contracts import _payload


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


def test_descriptive_resampling_requires_declared_observation_support() -> None:
    request = ResponseWindowAnalysisSpec.from_mapping(_payload())
    wells = pd.DataFrame.from_records(
        [
            {
                "experiment_id": "20260101_example",
                "design_id": "reference",
                "state": "00",
                "position": "A1",
                "reduction_id": request.primary_reduction.id,
                "response_well": 1.0,
                "magnitude_well": 1.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="lacks support"):
        descriptive_resampling_records(wells, request=request)


@pytest.mark.parametrize(
    ("response", "magnitude", "anchor", "samples", "paired_anchor", "message"),
    [
        (np.ones((1, 2)), np.ones(2), np.ones(2), 10, False, "one-dimensional"),
        (np.ones(2), np.ones(1), np.ones(2), 10, False, "aligned design observations"),
        (np.ones(2), np.ones(2), np.asarray([]), 10, False, "non-empty reference observations"),
        (np.ones(2), np.ones(2), np.ones(2), 1, False, "at least two draws"),
        (np.ones(2), np.ones(2), np.zeros(2), 10, True, "identical ordered magnitude"),
    ],
)
def test_joint_descriptive_resampling_rejects_invalid_inputs(
    response: np.ndarray,
    magnitude: np.ndarray,
    anchor: np.ndarray,
    samples: int,
    paired_anchor: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        joint_state_descriptive_resampling_draws(
            response,
            magnitude,
            anchor,
            samples=samples,
            stat="median",
            rng=np.random.default_rng(0),
            paired_anchor=paired_anchor,
        )


def test_joint_descriptive_resampling_rejects_an_unknown_observation_statistic() -> None:
    with pytest.raises(ValueError, match="stat to be 'mean' or 'median'"):
        joint_state_descriptive_resampling_draws(
            np.ones(2),
            np.ones(2),
            np.ones(2),
            samples=10,
            stat="mode",  # type: ignore[arg-type]
            rng=np.random.default_rng(0),
        )


def test_stable_seed_includes_experiment_identity() -> None:
    first = stable_seed(17, "experiment-a", "design", "00", "reduction")
    second = stable_seed(17, "experiment-b", "design", "00", "reduction")

    assert first != second
