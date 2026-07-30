from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from reader.domains.plate_reader.plots.response_window import (
    COMPONENT_COLUMNS,
    prepare_response_window_diagnostic,
    render_response_window_diagnostic,
)
from reader.domains.plate_reader.plots.response_window.diagnostic_render import _aligned_trace_center


def _traces_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_id in ("source-a", "source-b"):
        for design_id, is_reference in (("design-a", False), ("reference", True)):
            for signal_index, signal_kind in enumerate(("growth", "response", "magnitude"), start=1):
                for state_index, state in enumerate(("00", "10", "01", "11")):
                    for position_index, position in enumerate(("A1", "A2")):
                        for time in (0.0, 1.0, 2.0):
                            rows.append(
                                {
                                    "experiment_id": experiment_id,
                                    "design_id": design_id,
                                    "position": position,
                                    "state": state,
                                    "time": time + 4.0,
                                    "time_from_event_h": time,
                                    "value": float(
                                        signal_index + state_index + position_index + time + (is_reference * 0.5)
                                    ),
                                    "value_policy_clipped": False,
                                    "value_instrument_overflow": False,
                                    "value_bound_kind": "exact",
                                    "signal_kind": signal_kind,
                                    "is_reference": is_reference,
                                }
                            )
    return pd.DataFrame.from_records(rows)


def _designs_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_id in ("source-a", "source-b"):
        for design_id, is_reference in (("design-a", False), ("reference", True)):
            for reduction_id, offset in (("primary", 0.0), ("sensitivity", 10.0)):
                row: dict[str, object] = {
                    "experiment_id": experiment_id,
                    "design_id": design_id,
                    "reference_design_id": "reference",
                    "reduction_id": reduction_id,
                    "reduction_method": "geometric_time_mean",
                    "response_basis": "post_window",
                    "replicate_stat": "mean",
                    "bootstrap_samples": 200,
                    "confidence_level": 0.9,
                    "event_id": "addition",
                    "event_time_uncertainty_h": 0.25,
                    "window_start_event_h": 0.5,
                    "window_end_event_h": 1.5,
                    "is_reference": is_reference,
                }
                for index, component in enumerate(COMPONENT_COLUMNS):
                    value = float(index + offset)
                    row[component] = value
                    row[f"{component}_ci_low"] = value - 0.25
                    row[f"{component}_ci_high"] = value + 0.25
                    row[f"{component}_event_half_range"] = 0.1
                    row[f"{component}_bound_kind"] = "exact"
                    row[f"{component}_has_policy_clipping"] = False
                    row[f"{component}_has_instrument_overflow"] = False
                    row[f"{component}_event_sensitivity_has_policy_clipping"] = False
                    row[f"{component}_event_sensitivity_has_instrument_overflow"] = False
                rows.append(row)
    return pd.DataFrame.from_records(rows)


def test_prepare_response_window_diagnostic_selects_one_explicit_subject() -> None:
    diagnostic = prepare_response_window_diagnostic(
        _traces_frame(),
        _designs_frame(),
        source_experiment_id="source-a",
        design_id="design-a",
        reduction_id="primary",
        pre_window_duration_h=None,
    )

    assert diagnostic.source_experiment_id == "source-a"
    assert diagnostic.design_id == "design-a"
    assert diagnostic.reference_design_id == "reference"
    assert diagnostic.reduction_id == "primary"
    assert diagnostic.replicate_stat == "mean"
    assert diagnostic.confidence_level == 0.9
    assert diagnostic.event_time_uncertainty_h == 0.25
    assert diagnostic.window == (0.5, 1.5)
    assert diagnostic.component_values == tuple(float(index) for index in range(8))
    assert set(diagnostic.traces["experiment_id"]) == {"source-a"}
    assert set(diagnostic.traces["design_id"]) == {"design-a", "reference"}
    assert set(diagnostic.traces.loc[diagnostic.traces["design_id"].eq("reference"), "signal_kind"]) == {"magnitude"}


def test_prepare_response_window_diagnostic_rejects_a_missing_subject() -> None:
    with pytest.raises(
        ValueError,
        match="source experiment 'source-a', design 'missing', and reduction 'primary'",
    ):
        prepare_response_window_diagnostic(
            _traces_frame(),
            _designs_frame(),
            source_experiment_id="source-a",
            design_id="missing",
            reduction_id="primary",
            pre_window_duration_h=None,
        )


def test_render_response_window_diagnostic_has_four_neutral_panels() -> None:
    figure = render_response_window_diagnostic(
        _traces_frame(),
        _designs_frame(),
        source_experiment_id="source-a",
        design_id="design-a",
        reduction_id="primary",
        pre_window_duration_h=None,
        title="Selected diagnostic",
    )

    assert figure.get_gid() == "response-window-diagnostic"
    assert figure.get_suptitle().startswith("Selected diagnostic\n")
    assert "mean center" in figure.get_suptitle()
    assert "90% bootstrap CI (200 draws)" in figure.get_suptitle()
    assert [axis.get_title() for axis in figure.axes] == [
        "Growth traces",
        "Response traces",
        "Magnitude traces + reference",
        "Reduced components",
    ]
    assert [tick.get_text() for tick in figure.axes[3].get_yticklabels()] == list(COMPONENT_COLUMNS)
    assert any(line.get_gid() == "response-window-reference-trace" for line in figure.axes[2].lines)
    assert all(line.get_gid() != "response-window-reference-trace" for axis in figure.axes[:2] for line in axis.lines)
    assert all(
        any(patch.get_gid() == "response-window-event-interval" for patch in axis.patches) for axis in figure.axes[:3]
    )
    plt.close(figure)


def test_post_minus_pre_diagnostic_discloses_and_shades_the_pre_window() -> None:
    designs = _designs_frame()
    designs.loc[designs["reduction_id"].eq("primary"), "response_basis"] = "post_minus_pre"

    diagnostic = prepare_response_window_diagnostic(
        _traces_frame(),
        designs,
        source_experiment_id="source-a",
        design_id="design-a",
        reduction_id="primary",
        pre_window_duration_h=1.0,
    )
    figure = render_response_window_diagnostic(
        _traces_frame(),
        designs,
        source_experiment_id="source-a",
        design_id="design-a",
        reduction_id="primary",
        pre_window_duration_h=1.0,
    )

    assert diagnostic.pre_window == (-1.25, -0.25)
    assert "post_minus_pre" in figure.get_suptitle()
    assert any(patch.get_gid() == "response-window-pre-window" for patch in figure.axes[1].patches)
    assert all(
        patch.get_gid() != "response-window-pre-window"
        for axis in (figure.axes[0], figure.axes[2])
        for patch in axis.patches
    )
    plt.close(figure)


def test_diagnostic_renders_component_event_range_and_bound_notes() -> None:
    designs = _designs_frame()
    selected = (
        designs["experiment_id"].eq("source-a")
        & designs["design_id"].eq("design-a")
        & designs["reduction_id"].eq("primary")
    )
    designs.loc[selected, "r00_bound_kind"] = "lower"
    designs.loc[selected, "r00_has_policy_clipping"] = True
    designs.loc[selected, "r00_event_half_range"] = 0.75

    figure = render_response_window_diagnostic(
        _traces_frame(),
        designs,
        source_experiment_id="source-a",
        design_id="design-a",
        reduction_id="primary",
        pre_window_duration_h=None,
    )

    component_axis = figure.axes[3]
    segments = [
        segment
        for collection in component_axis.collections
        if hasattr(collection, "get_segments")
        for segment in collection.get_segments()
    ]
    assert any(tuple(segment[:, 0]) == (-0.75, 0.75) for segment in segments)
    assert any("r00: lower bound, policy clipping" in text.get_text() for text in component_axis.texts)
    plt.close(figure)


def test_aligned_trace_center_honors_stat_and_rejects_unaligned_grids() -> None:
    traces = [
        pd.DataFrame({"time_from_event_h": [0.0, 1.0], "plot_value": [1.0, 2.0]}),
        pd.DataFrame({"time_from_event_h": [0.0, 1.0], "plot_value": [2.0, 3.0]}),
        pd.DataFrame({"time_from_event_h": [0.0, 1.0], "plot_value": [9.0, 10.0]}),
    ]

    _, mean = _aligned_trace_center(traces, replicate_stat="mean")
    _, median = _aligned_trace_center(traces, replicate_stat="median")
    unaligned = [*traces[:2], pd.DataFrame({"time_from_event_h": [0.1, 1.1], "plot_value": [9.0, 10.0]})]

    assert mean.tolist() == pytest.approx([4.0, 5.0])
    assert median.tolist() == pytest.approx([2.0, 3.0])
    assert _aligned_trace_center(unaligned, replicate_stat="mean") is None
