from __future__ import annotations

import numpy as np

from reader.domains.plate_reader.analysis.response_window.provenance import stable_seed
from reader.domains.plate_reader.analysis.response_window.uncertainty import joint_state_bootstrap_draws


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


def test_stable_seed_includes_experiment_identity() -> None:
    first = stable_seed(17, "experiment-a", "design", "00", "reduction")
    second = stable_seed(17, "experiment-b", "design", "00", "reduction")

    assert first != second
