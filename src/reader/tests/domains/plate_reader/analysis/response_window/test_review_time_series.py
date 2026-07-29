from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reader.domains.plate_reader.analysis.response_window.review_time_series import time_series_figure


def test_time_series_uses_median_bands_and_shows_the_eight_value_handoff() -> None:
    figure = time_series_figure(
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
        selected=_selected(),
        wells=_wells(),
        traces=_traces(),
        events=pd.DataFrame([{"experiment_id": "experiment", "event_time_uncertainty_h": 0.2}]),
        display=_display(),
    )

    try:
        assert len(figure.axes) == 6
        trajectory_axes = figure.axes[:3]
        response_axis, fluorescence_axis, support_axis = figure.axes[3:]
        assert [
            len([line for line in axis.lines if line.get_gid() == "response-window-median"]) for axis in trajectory_axes
        ] == [4, 4, 8]
        assert [
            len([collection for collection in axis.collections if collection.get_gid() == "replicate-interval"])
            for axis in trajectory_axes
        ] == [4, 4, 4]
        assert [
            len([collection for collection in axis.collections if collection.get_gid() == "anchor-replicate-interval"])
            for axis in trajectory_axes
        ] == [0, 0, 4]
        assert response_axis.get_title(loc="left") == "D  Response handoff, rᵢ"
        assert fluorescence_axis.get_title(loc="left") == "E  pDual-10-relative fluorescence, bᵢ"
        assert support_axis.get_title(loc="left") == "F  Window and support"
        assert all(axis.get_box_aspect() == 1.0 for axis in figure.axes)
        assert [tick.get_text() for tick in response_axis.get_xticklabels()] == ["00", "10", "01", "11"]
        assert response_axis.findobj(lambda artist: artist.get_gid() == "bootstrap-uncertainty")
        assert response_axis.findobj(lambda artist: artist.get_gid() == "event-time-sensitivity")
        assert fluorescence_axis.findobj(lambda artist: artist.get_gid() == "bootstrap-uncertainty")
        assert fluorescence_axis.findobj(lambda artist: artist.get_gid() == "event-time-sensitivity")
        response_ci = response_axis.findobj(lambda artist: artist.get_gid() == "bootstrap-uncertainty")[0]
        for index, state in enumerate(("00", "10", "01", "11")):
            np.testing.assert_allclose(
                response_ci.get_segments()[index],
                [[index, _selected()[f"r{state}_ci_low"]], [index, _selected()[f"r{state}_ci_high"]]],
            )
        for state in ("00", "10", "01", "11"):
            response_points = response_axis.findobj(
                lambda artist, gid=f"replicate-values-r{state}": artist.get_gid() == gid
            )
            assert len(response_points) == 1
            assert len(response_points[0].get_offsets()) == 3
            assert not fluorescence_axis.findobj(
                lambda artist, gid=f"replicate-values-b{state}": artist.get_gid() == gid
            )
            assert response_axis.findobj(lambda artist, gid=f"handoff-summary-r{state}": artist.get_gid() == gid)
            assert fluorescence_axis.findobj(lambda artist, gid=f"handoff-summary-b{state}": artist.get_gid() == gid)
        support_text = "\n".join(text.get_text() for text in support_axis.texts)
        assert "0.5–1.5 h after stress addition" in support_text
        assert "00 3/3" in support_text
        assert "central 90%" in support_text
        assert "observed rᵢ wells" in support_text
        assert "independent design/reference" in support_text
        assert "dashed/hollow: pDual-10 anchor" in support_text
        assert "central 2/8 bounded" in support_text
        assert "event envelope 1/8" in support_text
        assert response_axis.findobj(lambda artist: artist.get_gid() == "censor-bound-r10")[0].get_text() == "≥"
        assert fluorescence_axis.findobj(lambda artist: artist.get_gid() == "censor-bound-b01")[0].get_text() == "≤"
        legend_labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert legend_labels == ["No stress", "Ethanol", "Ciprofloxacin", "Ethanol + ciprofloxacin"]
        assert "Stress addition interval" not in legend_labels
        assert "Response window" not in legend_labels
        assert {
            line.get_marker() for line in trajectory_axes[0].lines if line.get_gid() == "response-window-median"
        } == {
            "o",
            "s",
            "^",
            "D",
        }
    finally:
        plt.close(figure)


def _selected() -> pd.Series:
    values: dict[str, object] = {
        "reference_design_id": "pDual-10",
        "confidence_level": 0.9,
        "replicate_stat": "median",
        "window_start_event_h": 0.5,
        "window_end_event_h": 1.5,
        "reduction_method": "geometric_time_mean",
        "response_basis": "post_window",
        "reduction_role": "primary",
        "min_replicates_per_state": 3,
        "min_observed_points_per_trace": 4,
        "max_interior_gap_h": 0.5,
    }
    for index, state in enumerate(("00", "10", "01", "11")):
        values[f"r{state}"] = -1.0 + index * 0.3
        values[f"b{state}"] = -0.4 + index * 0.2
        values[f"r{state}_bootstrap_sd"] = 0.08
        values[f"b{state}_bootstrap_sd"] = 0.06
        values[f"r{state}_ci_low"] = float(values[f"r{state}"]) - 0.12
        values[f"r{state}_ci_high"] = float(values[f"r{state}"]) + 0.09
        values[f"b{state}_ci_low"] = float(values[f"b{state}"]) - 0.10
        values[f"b{state}_ci_high"] = float(values[f"b{state}"]) + 0.07
        values[f"r{state}_event_half_range"] = 0.03
        values[f"b{state}_event_half_range"] = 0.02
        for prefix in ("r", "b"):
            values[f"{prefix}{state}_bound_kind"] = "exact"
            values[f"{prefix}{state}_event_sensitivity_has_policy_clipping"] = False
            values[f"{prefix}{state}_event_sensitivity_has_instrument_overflow"] = False
        values[f"n{state}"] = 3
    values["r10_bound_kind"] = "lower"
    values["b01_bound_kind"] = "upper"
    values["r11_event_sensitivity_has_policy_clipping"] = True
    return pd.Series(values)


def _wells() -> pd.DataFrame:
    selected = _selected()
    records: list[dict[str, object]] = []
    for state in ("00", "10", "01", "11"):
        center = float(selected[f"r{state}"])
        for source_design in ("design", "pDual-10"):
            for index, offset in enumerate((-0.1, 0.0, 0.1), start=1):
                records.append(
                    {
                        "experiment_id": "experiment",
                        "design_id": source_design,
                        "reduction_id": "primary",
                        "state": state,
                        "position": f"{source_design}-{index}",
                        "response_well": center + offset,
                        "magnitude_well": 2.0 + offset,
                    }
                )
    return pd.DataFrame.from_records(records)


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
