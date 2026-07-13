from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from reader.domains.plate_reader.analysis.response_window.review_time_series import time_series_figure


def test_time_series_uses_median_bands_and_shows_the_eight_value_handoff() -> None:
    figure = time_series_figure(
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
        selected=_selected(),
        traces=_traces(),
        events=pd.DataFrame([{"experiment_id": "experiment", "event_time_uncertainty_h": 0.2}]),
        display=_display(),
    )

    try:
        assert len(figure.axes) == 4
        trajectory_axes = figure.axes[:3]
        handoff_axis = figure.axes[3]
        assert [
            len([line for line in axis.lines if line.get_gid() == "response-window-median"]) for axis in trajectory_axes
        ] == [4, 4, 8]
        assert [
            len([collection for collection in axis.collections if collection.get_gid() == "replicate-interval"])
            for axis in trajectory_axes
        ] == [4, 4, 4]
        assert len(handoff_axis.patches) == 8
        assert handoff_axis.get_title() == (
            "The response window reduces the trajectories to eight condition-specific values"
        )
        assert [tick.get_text() for tick in handoff_axis.get_xticklabels()] == [
            "No stress\n(00)",
            "Ethanol\n(10)",
            "Ciprofloxacin\n(01)",
            "Ethanol +\nciprofloxacin\n(11)",
        ]
        legend_labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert "Central 90% of design wells" in legend_labels
        assert "pDual-10 median anchor" in legend_labels
        assert {
            line.get_marker() for line in trajectory_axes[0].lines if line.get_gid() == "response-window-median"
        } == {
            "o",
            "s",
            "^",
            "D",
        }
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        handoff_legend = handoff_axis.get_legend()
        assert handoff_legend is not None
        assert handoff_legend.get_window_extent(renderer).y0 >= handoff_axis.bbox.y1
    finally:
        plt.close(figure)


def _selected() -> pd.Series:
    values: dict[str, object] = {
        "reference_design_id": "pDual-10",
        "confidence_level": 0.9,
        "window_start_event_h": 0.5,
        "window_end_event_h": 1.5,
        "reduction_method": "geometric_time_mean",
        "response_basis": "post_window",
        "reduction_role": "primary",
    }
    for index, state in enumerate(("00", "10", "01", "11")):
        values[f"r{state}"] = -1.0 + index * 0.3
        values[f"b{state}"] = -0.4 + index * 0.2
        values[f"r{state}_bootstrap_sd"] = 0.08
        values[f"b{state}_bootstrap_sd"] = 0.06
        values[f"r{state}_event_half_range"] = 0.03
        values[f"b{state}_event_half_range"] = 0.02
    return pd.Series(values)


def _traces() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    states = ("00", "10", "01", "11")
    times = (-1.0, 0.0, 1.0, 2.0)
    for signal_kind in ("growth", "response", "magnitude"):
        source_designs = ("design", "pDual-10") if signal_kind == "magnitude" else ("design",)
        for source_design in source_designs:
            for state_index, state in enumerate(states):
                for replicate_index, position in enumerate(("A1", "A2", "A3")):
                    for time in times:
                        if signal_kind == "growth":
                            value = 0.2 + state_index * 0.02 + replicate_index * 0.01 + (time + 1.0) * 0.03
                        else:
                            log_value = (
                                -1.5
                                + state_index * 0.2
                                + replicate_index * 0.05
                                + time * 0.1
                                + (0.3 if source_design == "pDual-10" else 0.0)
                            )
                            value = 2.0**log_value
                        records.append(
                            {
                                "experiment_id": "experiment",
                                "design_id": source_design,
                                "position": position,
                                "state": state,
                                "time_from_event_h": time,
                                "value": value,
                                "signal_kind": signal_kind,
                            }
                        )
    return pd.DataFrame.from_records(records)


def _display() -> dict[str, object]:
    return {
        "event_label": "Stress addition",
        "state_labels": {
            "00": "No stress",
            "10": "Ethanol",
            "01": "Ciprofloxacin",
            "11": "Ethanol + ciprofloxacin",
        },
        "channels": {
            "response_ratio": "YFP/CFP",
            "magnitude_ratio": "YFP/OD600",
            "growth": "OD600",
            "reference_design_id": "pDual-10",
        },
    }
