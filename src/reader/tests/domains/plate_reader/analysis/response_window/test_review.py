from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from reader.domains.plate_reader.analysis.response_window.review import (
    VIEW_LABELS,
    render_review_figure,
    response_summary_options,
)
from reader.domains.plate_reader.analysis.response_window.review_endpoint_plots import (
    measured_response_examples_figure,
    quality_figure,
    reduction_sensitivity_figure,
)


def test_review_vocabulary_distinguishes_examples_from_reference_anchor() -> None:
    assert VIEW_LABELS["Measured response examples"] == "measured_response_examples"
    assert "Reference examples" not in VIEW_LABELS


def test_response_summary_options_use_compact_labels_and_stable_ids() -> None:
    options = response_summary_options(
        pd.DataFrame(
            [
                {
                    "reduction_id": "event_logmean_6_12h_post",
                    "window_start_event_h": 6.0,
                    "window_end_event_h": 12.0,
                    "reduction_method": "geometric_time_mean",
                    "response_basis": "post_window",
                    "reduction_role": "primary",
                }
            ]
        )
    )

    assert options == {"6-12 h log mean (primary)": "event_logmean_6_12h_post"}


def test_state_summary_uses_explicit_assay_labels_on_white_canvas() -> None:
    figure = render_review_figure(
        view_id="state_summary",
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
        designs=_designs(),
        wells=_wells(),
        traces=pd.DataFrame(),
        events=pd.DataFrame(),
        display=_display(),
    )

    try:
        assert mcolors.to_hex(figure.get_facecolor()) == "#ffffff"
        assert all(mcolors.to_hex(axis.get_facecolor()) == "#ffffff" for axis in figure.axes)
        assert [axis.get_ylabel() for axis in figure.axes] == [
            "log2[(YFP / CFP)_design]",
            "log2[(YFP / OD600)_design / (YFP / OD600)_pDual-10]",
        ]
        assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == [
            "No stress\n(00)",
            "Ethanol\n(10)",
            "Ciprofloxacin\n(01)",
            "Ethanol +\nciprofloxacin\n(11)",
        ]
        assert figure._suptitle is not None
        assert figure._suptitle.get_text() == (
            "Observed wells and interval summaries preserve the four-condition handoff"
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        suptitle_box = figure._suptitle.get_window_extent(renderer)
        title_boxes = [axis.title.get_window_extent(renderer) for axis in figure.axes[:2]]
        assert min(suptitle_box.y0 - title_box.y1 for title_box in title_boxes) >= 8.0
        for axis in figure.axes[:2]:
            tick_boxes = [tick.get_window_extent(renderer) for tick in axis.get_xticklabels()]
            assert all(left.x1 < right.x0 for left, right in zip(tick_boxes, tick_boxes[1:], strict=False))
            assert not axis.patches
        for state in ("00", "10", "01", "11"):
            assert figure.axes[0].findobj(lambda artist, gid=f"replicate-values-r{state}": artist.get_gid() == gid)
            assert not figure.axes[1].findobj(lambda artist, gid=f"replicate-values-b{state}": artist.get_gid() == gid)
    finally:
        plt.close(figure)


def test_reduction_heatmaps_use_square_tiles() -> None:
    rows = []
    for index in range(5):
        row: dict[str, object] = {
            "reduction_id": f"reduction_{index}",
            "window_start_event_h": 4.0 + index,
            "window_end_event_h": 8.0 + index,
            "reduction_method": "geometric_time_mean",
            "response_basis": "post_window",
            "reduction_role": "primary" if index == 0 else "sensitivity",
        }
        for state_index, state in enumerate(("00", "10", "01", "11")):
            row[f"r{state}"] = float(index - state_index)
            row[f"b{state}"] = 0.0
        rows.append(row)

    figure = reduction_sensitivity_figure(rows=pd.DataFrame(rows), display=_display())

    try:
        heatmap_axes = [axis for axis in figure.axes if axis.images]
        assert len(heatmap_axes) == 2
        assert all(axis.get_aspect() == 1.0 for axis in heatmap_axes)
        assert (heatmap_axes[1].images[0].norm.vmin, heatmap_axes[1].images[0].norm.vmax) == (-1.0, 1.0)
    finally:
        plt.close(figure)


def test_measured_response_heatmap_condition_labels_do_not_collide() -> None:
    rows = []
    for index in range(6):
        row: dict[str, object] = {
            "experiment_id": f"2026070{index}_experiment",
            "example_label": f"Measured example {index}",
        }
        for state_index, state in enumerate(("00", "10", "01", "11")):
            row[f"r{state}"] = float(index - state_index)
            row[f"b{state}"] = float(state_index - index) / 2.0
        rows.append(row)

    figure = measured_response_examples_figure(rows=pd.DataFrame(rows), display=_display())

    try:
        heatmap_axes = [axis for axis in figure.axes if axis.images]
        assert len(heatmap_axes) == 2
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in heatmap_axes:
            tick_boxes = [tick.get_window_extent(renderer) for tick in axis.get_xticklabels()]
            assert all(left.x1 < right.x0 for left, right in zip(tick_boxes, tick_boxes[1:], strict=False))
    finally:
        plt.close(figure)


def test_quality_figure_labels_do_not_collide_and_grids_stay_behind_bars() -> None:
    wells = pd.DataFrame(
        [
            {"state": state, "position": f"{row}{column}"}
            for state in ("00", "10", "01", "11")
            for row, column in (("A", 1), ("B", 1), ("C", 1))
        ]
    )
    figure = quality_figure(selected=_designs().iloc[0], selected_wells=wells, display=_display())

    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        assert figure._suptitle is not None
        suptitle_box = figure._suptitle.get_window_extent(renderer)
        for axis in figure.axes[:3]:
            tick_boxes = [tick.get_window_extent(renderer) for tick in axis.get_xticklabels()]
            assert all(left.x1 < right.x0 for left, right in zip(tick_boxes, tick_boxes[1:], strict=False))
            assert axis.yaxis.label.get_window_extent(renderer).y1 < suptitle_box.y0
            mark_zorders = [patch.get_zorder() for patch in axis.patches]
            grid_zorders = [line.get_zorder() for line in axis.get_ygridlines() if line.get_visible()]
            assert grid_zorders and max(grid_zorders) < min(mark_zorders)
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Bootstrap SD",
            "Event-time sensitivity (half-range)",
        ]
    finally:
        plt.close(figure)


def _display() -> dict[str, object]:
    return {
        "schema_version": "reader.response_window.display.v1",
        "study_label": "Ethanol and ciprofloxacin response",
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
        "examples": [
            {"design_id": "pDual-10", "label": "pDual-10 fluorescence anchor", "role": "reference_anchor"},
            {"design_id": "design", "label": "Response example", "role": "response_example"},
        ],
    }


def _designs() -> pd.DataFrame:
    row: dict[str, object] = {
        "experiment_id": "experiment",
        "design_id": "design",
        "reduction_id": "primary",
        "reference_design_id": "pDual-10",
        "replicate_stat": "median",
    }
    for index, state in enumerate(("00", "10", "01", "11")):
        row[f"r{state}"] = float(index)
        row[f"b{state}"] = float(index) / 2.0
        row[f"r{state}_bootstrap_sd"] = 0.1
        row[f"b{state}_bootstrap_sd"] = 0.1
        row[f"r{state}_ci_low"] = float(row[f"r{state}"]) - 0.2
        row[f"r{state}_ci_high"] = float(row[f"r{state}"]) + 0.15
        row[f"b{state}_ci_low"] = float(row[f"b{state}"]) - 0.2
        row[f"b{state}_ci_high"] = float(row[f"b{state}"]) + 0.15
        row[f"r{state}_event_half_range"] = 0.05
        row[f"b{state}_event_half_range"] = 0.05
        row[f"n{state}"] = 3
    return pd.DataFrame([row])


def _wells() -> pd.DataFrame:
    selected = _designs().iloc[0]
    return pd.DataFrame.from_records(
        [
            {
                "experiment_id": "experiment",
                "design_id": "design",
                "reduction_id": "primary",
                "state": state,
                "position": f"A{replicate}",
                "response_well": float(selected[f"r{state}"]) + offset,
                "magnitude_well": 2.0 + offset,
            }
            for state in ("00", "10", "01", "11")
            for replicate, offset in enumerate((-0.1, 0.0, 0.1), start=1)
        ]
    )
