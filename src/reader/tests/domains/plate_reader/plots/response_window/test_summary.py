from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from reader.domains.plate_reader.plots.response_window.summary import (
    COMPONENT_COLUMNS,
    render_response_window_summary,
    response_window_summary_matrix,
)


def _designs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": ["source_b", "source_a", "source_a", "source_c"],
            "design_id": ["design_b", "design_a", "reference", "other_reduction"],
            "reduction_id": ["event", "event", "event", "endpoint"],
            "is_reference": [False, False, True, False],
            **{
                component: [offset + 1.0, offset + 2.0, offset + 20.0, offset + 200.0]
                for offset, component in enumerate(COMPONENT_COLUMNS)
            },
        }
    )


def test_response_window_summary_matrix_selects_and_labels_rows() -> None:
    summary = response_window_summary_matrix(_designs_frame(), primary_reduction_id="event")

    assert summary.row_labels == ("source_a :: design_a", "source_b :: design_b")
    assert summary.component_labels == COMPONENT_COLUMNS
    assert summary.values.tolist() == [
        [offset + 2.0 for offset in range(len(COMPONENT_COLUMNS))],
        [offset + 1.0 for offset in range(len(COMPONENT_COLUMNS))],
    ]


def test_response_window_summary_matrix_applies_explicit_identity_filters() -> None:
    summary = response_window_summary_matrix(
        _designs_frame(),
        primary_reduction_id="event",
        experiment_ids=["source_b"],
        design_ids=["design_b"],
        maximum_rows=1,
    )

    assert summary.row_labels == ("source_b :: design_b",)
    assert summary.values.tolist() == [[offset + 1.0 for offset in range(len(COMPONENT_COLUMNS))]]


def test_response_window_summary_matrix_rejects_selection_over_row_budget() -> None:
    designs = pd.concat(
        [_designs_frame().iloc[[0]].assign(design_id=f"design_{index}") for index in range(3)],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match=r"selected 3 rows.*maximum_rows=2.*experiment_ids.*design_ids"):
        response_window_summary_matrix(
            designs,
            primary_reduction_id="event",
            maximum_rows=2,
        )


def test_response_window_summary_matrix_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="no non-reference rows for reduction 'missing'"):
        response_window_summary_matrix(_designs_frame(), primary_reduction_id="missing")


def test_response_window_summary_matrix_rejects_nonfinite_values() -> None:
    designs = _designs_frame()
    designs.loc[0, "r00"] = np.inf

    with pytest.raises(ValueError, match="requires finite component values"):
        response_window_summary_matrix(designs, primary_reduction_id="event")


def test_response_window_summary_uses_separate_zero_centered_family_scales() -> None:
    figure = render_response_window_summary(
        _designs_frame(),
        primary_reduction_id="event",
        title="Response-window summary",
    )

    response_axis, magnitude_axis = figure.axes[:2]
    assert response_axis.get_title() == "Response"
    assert magnitude_axis.get_title() == "Anchored magnitude"
    assert (
        response_axis.images[0].norm.vmin,
        response_axis.images[0].norm.vcenter,
        response_axis.images[0].norm.vmax,
    ) == (-5.0, 0.0, 5.0)
    assert (
        magnitude_axis.images[0].norm.vmin,
        magnitude_axis.images[0].norm.vcenter,
        magnitude_axis.images[0].norm.vmax,
    ) == (-9.0, 0.0, 9.0)
    assert [axis.get_ylabel() for axis in figure.axes[2:]] == ["response", "anchored magnitude"]
    plt.close(figure)
