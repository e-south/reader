from __future__ import annotations

import numpy as np

from reader_workbench.domains.plate_reader.analysis.response_window.event_sensitivity import (
    symmetric_event_sensitivity_half_width,
)


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
