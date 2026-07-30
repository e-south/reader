from __future__ import annotations

from inspect import signature

import matplotlib.pyplot as plt
import pandas as pd

from reader_workbench.domains.plate_reader.plots.snapshot_heatmap import (
    plot_snapshot_heatmap,
    prepare_snapshot_heatmap_inputs,
)


def _snapshot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": ["A1", "A2", "A3"],
            "time": [4.0, 4.0, 4.0],
            "channel": ["signal", "signal", "signal"],
            "value": [1.0, 2.0, 3.0],
            "treatment": ["reference", "negative", "baseline"],
            "design_id": ["design_a", "design_a", "design_a"],
        }
    )


def test_snapshot_heatmap_only_reorders_treatments_when_order_is_explicit() -> None:
    default_figure = plot_snapshot_heatmap(
        df=_snapshot_frame(),
        channel="signal",
        time=4.0,
        order_x=None,
        order_y=None,
        fig_kwargs={},
        filename=None,
    )[0].fig
    explicit_figure = plot_snapshot_heatmap(
        df=_snapshot_frame(),
        channel="signal",
        time=4.0,
        order_x=["reference", "negative", "baseline"],
        order_y=None,
        fig_kwargs={},
        filename=None,
    )[0].fig

    assert [tick.get_text() for tick in default_figure.axes[0].get_xticklabels()] == [
        "baseline",
        "negative",
        "reference",
    ]
    assert [tick.get_text() for tick in explicit_figure.axes[0].get_xticklabels()] == [
        "reference",
        "negative",
        "baseline",
    ]
    plt.close(default_figure)
    plt.close(explicit_figure)


def test_snapshot_heatmap_domain_accepts_explicit_values_not_runtime_models() -> None:
    assert "blanks" not in signature(plot_snapshot_heatmap).parameters
    assert "ctx" not in signature(prepare_snapshot_heatmap_inputs).parameters
    assert "cfg" not in signature(prepare_snapshot_heatmap_inputs).parameters
