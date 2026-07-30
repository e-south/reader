from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from reader.plugins.plot.response_window_summary import (
    ResponseWindowSummaryCfg,
    ResponseWindowSummaryPlot,
)


def _designs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": ["source_b", "source_a", "source_a"],
            "design_id": ["design_b", "design_a", "reference"],
            "reduction_id": ["event", "event", "event"],
            "is_reference": [False, False, True],
            "r00": [1.0, 2.0, 20.0],
            "r10": [3.0, 4.0, 40.0],
            "r01": [5.0, 6.0, 60.0],
            "r11": [7.0, 8.0, 80.0],
            "b00": [9.0, 10.0, 100.0],
            "b10": [11.0, 12.0, 120.0],
            "b01": [13.0, 14.0, 140.0],
            "b11": [15.0, 16.0, 160.0],
        }
    )


def test_response_window_summary_plot_adapts_figure_metadata() -> None:
    cfg = ResponseWindowSummaryCfg(
        primary_reduction_id="event",
        experiment_ids=["source_b"],
        design_ids=["design_b"],
        maximum_rows=1,
        title="Selected response window",
        filename="summary",
        format=["png", "pdf"],
        dpi=144,
    )

    rendered = ResponseWindowSummaryPlot().render(None, {"designs": _designs_frame()}, cfg)

    assert [(item.filename, item.ext, item.dpi) for item in rendered] == [
        ("summary", "png", 144),
        ("summary", "pdf", 144),
    ]
    assert {item.description for item in rendered} == {
        "Primary event-relative response and anchored-magnitude components by source and design."
    }
    assert rendered[0].fig is rendered[1].fig
    assert rendered[0].fig.get_suptitle() == "Selected response window"
    assert [axis.get_title() for axis in rendered[0].fig.axes[:2]] == ["Response", "Anchored magnitude"]
    assert [tick.get_text() for tick in rendered[0].fig.axes[0].get_yticklabels()] == [
        "source_b :: design_b",
    ]
    plt.close(rendered[0].fig)
