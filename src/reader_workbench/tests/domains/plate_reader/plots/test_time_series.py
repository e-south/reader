from __future__ import annotations

from inspect import signature

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reader_workbench.domains.plate_reader.plots.time_series import plot_time_series


def test_time_series_uses_only_explicit_hue_labels() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2"],
            "time": [0.0, 1.0],
            "channel": ["signal", "signal"],
            "value": [1.0, 2.0],
            "state": ["state_a", "state_a"],
            "treatment": ["condition_x", "condition_x"],
        }
    )

    figure = plot_time_series(
        df=frame,
        x="time",
        y=["signal"],
        hue="state",
        channels=None,
        group_on=None,
        pool_sets=None,
        pool_match="exact",
        fig_kwargs={},
        add_sheet_line=False,
        sheet_line_kwargs=None,
        log_transform=False,
        time_window=None,
        palette_book=None,
        hue_label_map={"state_a": "Relevant stress"},
    )[0].fig

    assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == ["Relevant stress"]
    assert "blanks" not in signature(plot_time_series).parameters
    assert "subplots" not in signature(plot_time_series).parameters
    plt.close(figure)


def test_time_series_describes_selected_nonfinite_measurements() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2", "A3", "A4"],
            "time": [0.0, 0.0, 1.0, 1.0],
            "channel": ["signal"] * 4,
            "value": [1.0, np.inf, 2.0, 3.0],
            "state": ["state_a"] * 4,
            "design_id": ["design_a"] * 4,
        }
    )

    rendered = plot_time_series(
        df=frame,
        x="time",
        y=["signal"],
        hue="state",
        channels=None,
        group_on="design_id",
        pool_sets=None,
        pool_match="exact",
        fig_kwargs={},
        add_sheet_line=False,
        sheet_line_kwargs=None,
        log_transform=False,
        time_window=None,
        palette_book=None,
    )

    assert len(rendered) == 1
    assert rendered[0].description == (
        "Selected measurements: 4 observed, 1 omitted as non-finite; affected time-series summaries were withheld."
    )
    plt.close(rendered[0].fig)
