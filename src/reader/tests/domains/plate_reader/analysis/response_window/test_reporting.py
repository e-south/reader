from __future__ import annotations

import numpy as np
import pytest

from reader.domains.plate_reader.analysis.response_window.reporting import (
    _spearman_rank_correlation,
)


def test_spearman_rank_correlation_uses_average_ranks_for_ties() -> None:
    correlation = _spearman_rank_correlation(
        np.asarray([1.0, 1.0, 2.0, 3.0]),
        np.asarray([1.0, 2.0, 2.0, 3.0]),
    )

    assert correlation == pytest.approx(5.0 / 6.0)


def test_spearman_rank_correlation_handles_constant_series_explicitly() -> None:
    constant = np.asarray([2.0, 2.0, 2.0])

    assert _spearman_rank_correlation(constant, constant) == 1.0
    assert np.isnan(_spearman_rank_correlation(constant, np.asarray([1.0, 2.0, 3.0])))
