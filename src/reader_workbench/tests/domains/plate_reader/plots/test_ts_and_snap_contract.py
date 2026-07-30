from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from pydantic import ValidationError

from reader_workbench.domains.plate_reader.plots.ts_and_snap import plot_ts_and_snap
from reader_workbench.plugins.plot.ts_and_snap import TSAndSnapCfg


def _windowed_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for treatment_index, treatment in enumerate(("Null", "IPTG")):
        for observation_index in range(3):
            position = f"{treatment_index}-{observation_index}"
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
                    "value": 10.0 + treatment_index + observation_index,
                    "treatment": treatment,
                }
            )
    return pd.DataFrame(rows)


def test_time_series_window_does_not_change_snapshot_endpoint() -> None:
    figures = plot_ts_and_snap(
        df=_windowed_frame(),
        group_on=None,
        pool_sets=None,
        ts_channel="OD600",
        ts_hue="treatment",
        ts_time_window=[0.0, 8.0],
        snap_x="treatment",
        snap_channel="RFP/OD600",
        snap_time=12.0,
        snap_agg="median",
        snap_dispersion="none",
    )

    ax_ts, ax_snap = figures[0].fig.axes
    assert ax_ts.get_xlim()[1] == pytest.approx(8.0)
    assert "t=12.00 h" in ax_snap.get_title()
    assert sorted(patch.get_height() for patch in ax_snap.patches) == pytest.approx([11.0, 12.0])
    plt.close(figures[0].fig)


def test_time_series_levels_follow_the_windowed_channel_rows() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A1", "B1", "C1"],
            "time": [0.0, 8.0, 0.0, 12.0],
            "channel": ["OD600", "OD600", "RFP/OD600", "OD600"],
            "value": [0.1, 0.3, 5.0, 0.8],
            "treatment": ["visible", "visible", "wrong-channel", "outside-window"],
            "subject": ["tracked", "tracked", "wrong-channel", "outside-window"],
        }
    )

    figures = plot_ts_and_snap(
        df=frame,
        group_on=None,
        pool_sets=None,
        ts_channel="OD600",
        ts_hue="treatment",
        ts_style="subject",
        order_hue=["visible"],
        order_style=["tracked"],
        ts_time_window=[0.0, 8.0],
        snap_x="treatment",
        snap_channel="OD600",
        snap_time=0.0,
    )

    ax_ts = figures[0].fig.axes[0]
    legends = [legend for legend in [ax_ts.get_legend(), *ax_ts.artists] if hasattr(legend, "get_texts")]
    assert len(ax_ts.lines) == 1
    assert {text.get_text() for legend in legends for text in legend.get_texts()} == {"visible", "tracked"}
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
            group_on="group",
            pool_sets=[{"G1": ["G1"]}, {"G2": ["G2"]}],
            group_layout="paired_row",
            ts_channel="OD600",
            ts_hue="treatment",
            snap_x="treatment",
            snap_channel="OD600",
            snap_time=0.0,
        )


def test_paired_row_compares_only_visible_time_series_levels() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2", "B1", "B2"],
            "time": [0.0] * 4,
            "channel": ["OD600", "RFP/OD600", "OD600", "RFP/OD600"],
            "value": [0.1, 4.0, 0.2, 5.0],
            "group": ["G1", "G1", "G2", "G2"],
            "treatment": ["visible", "G1-hidden", "visible", "G2-hidden"],
        }
    )

    figures = plot_ts_and_snap(
        df=frame,
        group_on="group",
        pool_sets=[{"G1": ["G1"]}, {"G2": ["G2"]}],
        group_layout="paired_row",
        ts_channel="OD600",
        ts_hue="treatment",
        order_hue=["visible"],
        snap_x="treatment",
        snap_channel="OD600",
        snap_time=0.0,
    )

    axes = figures[0].fig.axes
    assert len(axes[0].lines) == 1
    assert len(axes[2].lines) == 1
    plt.close(figures[0].fig)


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


def test_snapshot_only_group_uses_snapshot_hue_palette_without_time_series_rows() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A1", "B1"],
            "time": [0.0, 0.0, 0.0],
            "channel": ["OD600", "RFP/OD600", "RFP/OD600"],
            "value": [0.1, 1.0, 2.0],
            "group": ["complete", "complete", "snapshot-only"],
            "treatment": ["A", "A", "B"],
        }
    )

    figures = plot_ts_and_snap(
        df=frame,
        group_on="group",
        pool_sets=[{"complete": ["complete"]}, {"snapshot-only": ["snapshot-only"]}],
        ts_channel="OD600",
        ts_hue="treatment",
        snap_x="position",
        snap_channel="RFP/OD600",
        snap_hue="treatment",
        snap_time=0.0,
    )

    assert len(figures) == 2
    snapshot_only_ts, snapshot_only_snap = figures[1].fig.axes
    assert len(snapshot_only_ts.lines) == 0
    assert len(snapshot_only_snap.patches) == 1
    for figure in figures:
        plt.close(figure.fig)


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


def test_composite_figure_returns_publication_metadata() -> None:
    cfg = TSAndSnapCfg.model_validate(
        {
            "ts_channel": "OD600",
            "ts_hue": "treatment",
            "snap_time": 12.0,
            "fig": {"ext": "png", "dpi": 144},
            "filename": "endpoint_pair",
        }
    )

    figures = plot_ts_and_snap(
        df=_windowed_frame(),
        group_on=None,
        pool_sets=None,
        ts_channel="OD600",
        ts_hue="treatment",
        snap_x="treatment",
        snap_channel="RFP/OD600",
        snap_time=12.0,
        fig_kwargs=cfg.fig.model_dump(exclude_none=True),
        filename=cfg.filename,
    )

    assert (figures[0].filename, figures[0].ext, figures[0].dpi) == ("endpoint_pair", "png", 144)
    plt.close(figures[0].fig)


def test_iqr_error_bars_emit_no_deprecation_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        figures = plot_ts_and_snap(
            df=_windowed_frame(),
            group_on=None,
            pool_sets=None,
            ts_channel="OD600",
            ts_hue="treatment",
            snap_x="treatment",
            snap_channel="RFP/OD600",
            snap_time=12.0,
            snap_agg="median",
            snap_dispersion="iqr",
        )
    plt.close(figures[0].fig)
