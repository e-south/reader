from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from pydantic import ValidationError

from reader.domains.plate_reader.plots.ts_and_snap import plot_ts_and_snap
from reader.plugins.plot.ts_and_snap import TSAndSnapCfg


def _windowed_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for treatment_index, treatment in enumerate(("Null", "IPTG")):
        for replicate in range(3):
            position = f"{treatment_index}-{replicate}"
            for time in (0.0, 8.0, 12.0):
                rows.append(
                    {
                        "position": position,
                        "time": time,
                        "channel": "OD600",
                        "value": treatment_index + time,
                        "treatment": treatment,
                    }
                )
            rows.append(
                {
                    "position": position,
                    "time": 12.0,
                    "channel": "RFP/OD600",
                    "value": 10.0 + treatment_index + replicate,
                    "treatment": treatment,
                }
            )
    return pd.DataFrame(rows)


def test_time_series_window_does_not_change_snapshot_endpoint() -> None:
    figures = plot_ts_and_snap(
        df=_windowed_frame(),
        output_dir=None,
        group_on=None,
        pool_sets=None,
        ts_channel="OD600",
        ts_hue="treatment",
        ts_time_window=[0.0, 8.0],
        snap_x="treatment",
        snap_channel="RFP/OD600",
        snap_time=12.0,
        snap_agg="median",
        snap_err="none",
    )

    ax_ts, ax_snap = figures[0].fig.axes
    assert ax_ts.get_xlim()[1] == pytest.approx(8.0)
    assert "t=12.00 h" in ax_snap.get_title()
    assert sorted(patch.get_height() for patch in ax_snap.patches) == pytest.approx([11.0, 12.0])
    plt.close(figures[0].fig)


def test_paired_row_rejects_heterogeneous_treatment_domains() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2", "B1", "B2"],
            "time": [0.0] * 4,
            "channel": ["OD600"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
            "group": ["G1", "G1", "G2", "G2"],
            "treatment": ["A", "B", "B", "C"],
        }
    )

    with pytest.raises(ValueError, match="identical ts_hue levels"):
        plot_ts_and_snap(
            df=frame,
            output_dir=None,
            group_on="group",
            pool_sets=[{"G1": ["G1"]}, {"G2": ["G2"]}],
            group_layout="paired_row",
            ts_channel="OD600",
            ts_hue="treatment",
            snap_x="treatment",
            snap_channel="OD600",
            snap_time=0.0,
        )


def test_paired_row_uses_one_treatment_color_map() -> None:
    rows = []
    for group in ("G1", "G2"):
        for treatment in ("A", "B"):
            rows.append(
                {
                    "position": f"{group}-{treatment}",
                    "time": 0.0,
                    "channel": "OD600",
                    "value": 1.0,
                    "group": group,
                    "treatment": treatment,
                }
            )

    figures = plot_ts_and_snap(
        df=pd.DataFrame(rows),
        output_dir=None,
        group_on="group",
        pool_sets=[{"G1": ["G1"]}, {"G2": ["G2"]}],
        group_layout="paired_row",
        ts_channel="OD600",
        ts_hue="treatment",
        snap_x="treatment",
        snap_channel="OD600",
        snap_color_by_x=True,
        snap_time=0.0,
    )

    axes = figures[0].fig.axes
    for treatment_index in range(2):
        first_line = axes[0].lines[treatment_index]
        second_line = axes[2].lines[treatment_index]
        first_bar = axes[1].patches[treatment_index]
        second_bar = axes[3].patches[treatment_index]
        expected = to_rgba(first_line.get_color())
        assert to_rgba(second_line.get_color()) == pytest.approx(expected)
        assert first_bar.get_facecolor() == pytest.approx(expected)
        assert second_bar.get_facecolor() == pytest.approx(expected)
    plt.close(figures[0].fig)


def test_composite_figure_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="axis_lable_size"):
        TSAndSnapCfg.model_validate(
            {
                "ts_channel": "OD600",
                "ts_hue": "treatment",
                "snap_time": 12.0,
                "fig": {"axis_lable_size": 12},
            }
        )


def test_composite_figure_config_accepts_scalar_extension() -> None:
    cfg = TSAndSnapCfg.model_validate(
        {
            "ts_channel": "OD600",
            "ts_hue": "treatment",
            "snap_time": 12.0,
            "fig": {"ext": "png"},
        }
    )

    figures = plot_ts_and_snap(
        df=_windowed_frame(),
        output_dir=None,
        group_on=None,
        pool_sets=None,
        ts_channel="OD600",
        ts_hue="treatment",
        snap_x="treatment",
        snap_channel="RFP/OD600",
        snap_time=12.0,
        fig_kwargs=cfg.fig.model_dump(exclude_none=True),
    )

    assert figures[0].ext == "png"
    plt.close(figures[0].fig)


def test_iqr_error_bars_emit_no_deprecation_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        figures = plot_ts_and_snap(
            df=_windowed_frame(),
            output_dir=None,
            group_on=None,
            pool_sets=None,
            ts_channel="OD600",
            ts_hue="treatment",
            snap_x="treatment",
            snap_channel="RFP/OD600",
            snap_time=12.0,
            snap_agg="median",
            snap_err="iqr",
        )
    plt.close(figures[0].fig)
