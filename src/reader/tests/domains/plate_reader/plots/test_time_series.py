from __future__ import annotations

from inspect import signature

import matplotlib.pyplot as plt
import pandas as pd

from reader.domains.plate_reader.plots.time_series import plot_time_series


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
