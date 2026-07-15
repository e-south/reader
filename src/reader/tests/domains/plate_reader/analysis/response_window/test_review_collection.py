from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from reader.domains.plate_reader.analysis.response_window.review_collection import (
    cross_experiment_design_rows,
    response_window_review_collection,
)
from reader.domains.plate_reader.analysis.response_window.review_cross_experiment import (
    cross_experiment_state_figure,
)
from reader.domains.plate_reader.analysis.response_window.review_cross_experiment_contract import (
    prepare_cross_experiment_context,
)
from reader.domains.plate_reader.analysis.response_window.review_reduction_options import (
    common_cross_experiment_reductions,
)


def test_response_window_review_collection_uses_exact_primary_nonreference_occurrences() -> None:
    designs = _design_rows()
    singleton = designs.iloc[[0]].copy()
    singleton["experiment_id"] = "experiment_c"
    singleton["design_id"] = "design_beta"
    reference = designs.iloc[[0]].copy()
    reference["experiment_id"] = "experiment_c"
    reference["design_id"] = "pDual-10"
    reference["is_reference"] = True
    rows = pd.concat([designs, singleton, reference], ignore_index=True)

    index = response_window_review_collection(
        rows,
        experiment_ids=("experiment_b", "experiment_a", "experiment_c"),
        experiment_titles={
            "experiment_a": "January plate",
            "experiment_b": "June plate",
            "experiment_c": "Reference plate",
        },
        review_collection_id="response_review.v1",
        review_collection_label="Response review",
    )

    assert index.entity_kind.kind_id == "reader.design_id"
    assert index.multi_experiment_entity_options() == {"design_alpha": "design_alpha"}
    assert [item.experiment_id for item in index.experiments_for_entity("design_alpha")] == [
        "experiment_b",
        "experiment_a",
    ]
    assert index.multi_experiment_entity_ids() == ("design_alpha",)
    assert {item.entity_id for item in index.entities_for_experiment("experiment_c")} == {"design_beta"}


def test_cross_experiment_design_rows_preserve_one_row_per_experiment() -> None:
    selected = cross_experiment_design_rows(
        _design_rows(),
        design_id="design_alpha",
        reduction_id="primary",
    )

    assert selected["experiment_id"].tolist() == ["experiment_a", "experiment_b"]


def test_common_cross_experiment_reductions_omit_incomplete_options_and_reject_drift() -> None:
    primary = _design_rows()
    sensitivity = primary.copy()
    sensitivity["reduction_id"] = "early"
    sensitivity["reduction_role"] = "sensitivity"
    incomplete = sensitivity.iloc[[0]].copy()
    rows = pd.concat([primary, incomplete], ignore_index=True)

    common = common_cross_experiment_reductions(
        rows,
        design_id="design_alpha",
        experiment_ids=("experiment_a", "experiment_b"),
    )

    assert set(common["reduction_id"]) == {"primary"}

    drift = pd.concat([primary, sensitivity], ignore_index=True)
    drift.loc[drift.index[-1], "window_end_event_h"] = 10.0
    with pytest.raises(ValueError, match="shared definitions"):
        common_cross_experiment_reductions(
            drift,
            design_id="design_alpha",
            experiment_ids=("experiment_a", "experiment_b"),
        )


def test_cross_experiment_design_rows_reject_singletons_duplicates_and_semantic_drift() -> None:
    rows = _design_rows()
    with pytest.raises(ValueError, match="at least two experiments"):
        cross_experiment_design_rows(
            rows.iloc[[0]],
            design_id="design_alpha",
            reduction_id="primary",
        )

    with pytest.raises(ValueError, match="one row per experiment"):
        cross_experiment_design_rows(
            pd.concat([rows, rows.iloc[[0]]], ignore_index=True),
            design_id="design_alpha",
            reduction_id="primary",
        )

    drift = rows.copy()
    drift.loc[1, "reference_design_id"] = "other-reference"
    with pytest.raises(ValueError, match="shared reduction semantics"):
        cross_experiment_design_rows(
            drift,
            design_id="design_alpha",
            reduction_id="primary",
        )


def test_cross_experiment_design_rows_reject_reference_and_invalid_interval_values() -> None:
    rows = _design_rows()
    rows["is_reference"] = True
    with pytest.raises(ValueError, match="non-reference Reader design"):
        cross_experiment_design_rows(rows, design_id="design_alpha", reduction_id="primary")

    rows = _design_rows()
    rows.loc[0, "r01_ci_low"] = rows.loc[0, "r01"] + 1.0
    with pytest.raises(ValueError, match="interval does not contain"):
        cross_experiment_design_rows(rows, design_id="design_alpha", reduction_id="primary")


def test_cross_experiment_state_figure_keeps_experiments_separate() -> None:
    selected = cross_experiment_design_rows(
        _design_rows(),
        design_id="design_alpha",
        reduction_id="primary",
    )
    figure = cross_experiment_state_figure(
        selected=selected,
        state="01",
        experiment_labels={
            "experiment_a": "January reference plate",
            "experiment_b": "June panel plate",
        },
        wells=_wells(),
        traces=_traces(),
        events=pd.DataFrame(
            [
                {"experiment_id": "experiment_a", "event_time_uncertainty_h": 0.2},
                {"experiment_id": "experiment_b", "event_time_uncertainty_h": 0.3},
            ]
        ),
        display=_display(),
    )

    try:
        assert len(figure.axes) == 6
        assert all(axis.get_box_aspect() == 1.0 for axis in figure.axes)
        assert [
            len(axis.findobj(lambda artist: artist.get_gid() == "cross-experiment-median")) for axis in figure.axes[:3]
        ] == [2, 2, 4]
        response_axis, fluorescence_axis, support_axis = figure.axes[3:]
        assert figure.axes[2].get_title(loc="left") == "C  Fluorescence + anchor"
        assert response_axis.get_title(loc="left") == "D  Response, rᵢ"
        assert fluorescence_axis.get_title(loc="left") == "E  Anchored fluorescence, bᵢ"
        assert support_axis.get_title(loc="left") == "F  Evidence boundary"
        assert len(response_axis.findobj(lambda artist: artist.get_gid() == "replicate-values-r01")) == 2
        assert not fluorescence_axis.findobj(lambda artist: artist.get_gid() == "replicate-values-b01")
        assert response_axis.findobj(lambda artist: artist.get_gid() == "bootstrap-uncertainty")
        assert response_axis.findobj(lambda artist: artist.get_gid() == "event-time-sensitivity")
        assert fluorescence_axis.findobj(lambda artist: artist.get_gid() == "bootstrap-uncertainty")
        assert fluorescence_axis.findobj(lambda artist: artist.get_gid() == "event-time-sensitivity")
        assert [tick.get_text() for tick in response_axis.get_yticklabels()] == [
            "January reference plate",
            "June panel plate",
        ]
        support = "\n".join(text.get_text() for text in support_axis.texts)
        assert "Reader design  design_alpha" in support
        assert "Ciprofloxacin (01)" in support
        assert "2 Reader experiments" in support
        assert "thin color: central 90% bootstrap" in " ".join(support.split())
        assert "thick gray: event-time sensitivity" in " ".join(support.split())
        assert "No cross-experiment aggregation or comparability decision is made" in " ".join(support.split())
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bottom_title_boxes = [axis.title.get_window_extent(renderer) for axis in figure.axes[3:]]
        assert not any(
            left.overlaps(right)
            for index, left in enumerate(bottom_title_boxes)
            for right in bottom_title_boxes[index + 1 :]
        )
        assert not response_axis.xaxis.label.get_window_extent(renderer).overlaps(
            fluorescence_axis.xaxis.label.get_window_extent(renderer)
        )
        support_box = support_axis.get_window_extent(renderer)
        assert all(
            support_box.contains(*corner)
            for text in support_axis.texts
            for corner in text.get_window_extent(renderer).corners()
        )
    finally:
        plt.close(figure)


def test_cross_experiment_state_figure_rejects_unknown_state_and_incomplete_events() -> None:
    selected = cross_experiment_design_rows(
        _design_rows(),
        design_id="design_alpha",
        reduction_id="primary",
    )
    kwargs = {
        "selected": selected,
        "experiment_labels": {
            "experiment_a": "January reference plate",
            "experiment_b": "June panel plate",
        },
        "wells": _wells(),
        "traces": _traces(),
        "events": pd.DataFrame([{"experiment_id": "experiment_a", "event_time_uncertainty_h": 0.2}]),
        "display": _display(),
    }
    with pytest.raises(ValueError, match="unknown response-window state"):
        cross_experiment_state_figure(state="cipro", **kwargs)
    with pytest.raises(ValueError, match="exactly one event"):
        cross_experiment_state_figure(state="01", **kwargs)


def test_cross_experiment_context_uses_compact_unique_dates_for_plot_labels() -> None:
    selected = _design_rows().copy()
    selected["experiment_id"] = ["20260621_panel", "20260121_reference"]

    _, context = prepare_cross_experiment_context(
        selected=selected,
        state="01",
        experiment_labels={
            "20260121_reference": "January reference plate",
            "20260621_panel": "June panel plate",
        },
        events=pd.DataFrame(
            [
                {"experiment_id": "20260121_reference", "event_time_uncertainty_h": 0.2},
                {"experiment_id": "20260621_panel", "event_time_uncertainty_h": 0.3},
            ]
        ),
        display=_display(),
    )

    assert context.plot_experiment_labels == {
        "20260121_reference": "2026-01-21",
        "20260621_panel": "2026-06-21",
    }


def _design_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_id, offset in (("experiment_b", 0.5), ("experiment_a", -0.5)):
        row: dict[str, object] = {
            "experiment_id": experiment_id,
            "design_id": "design_alpha",
            "reference_design_id": "pDual-10",
            "reduction_id": "primary",
            "reduction_method": "geometric_time_mean",
            "response_basis": "post_window",
            "reduction_role": "primary",
            "replicate_stat": "median",
            "bootstrap_samples": 500,
            "confidence_level": 0.9,
            "window_start_event_h": 6.0,
            "window_end_event_h": 12.0,
            "is_reference": False,
        }
        for index, state in enumerate(("00", "10", "01", "11")):
            response = offset + index * 0.2
            fluorescence = offset / 2.0 + index * 0.1
            row[f"r{state}"] = response
            row[f"b{state}"] = fluorescence
            row[f"r{state}_ci_low"] = response - 0.2
            row[f"r{state}_ci_high"] = response + 0.2
            row[f"b{state}_ci_low"] = fluorescence - 0.15
            row[f"b{state}_ci_high"] = fluorescence + 0.15
            row[f"r{state}_event_half_range"] = 0.05
            row[f"b{state}_event_half_range"] = 0.03
            row[f"n{state}"] = 3
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _wells() -> pd.DataFrame:
    designs = _design_rows().set_index("experiment_id")
    records: list[dict[str, object]] = []
    for experiment_id in ("experiment_a", "experiment_b"):
        for state in ("00", "10", "01", "11"):
            response = float(designs.loc[experiment_id, f"r{state}"])
            for source_design in ("design_alpha", "pDual-10"):
                for index, offset in enumerate((-0.1, 0.0, 0.1), start=1):
                    records.append(
                        {
                            "experiment_id": experiment_id,
                            "design_id": source_design,
                            "reduction_id": "primary",
                            "state": state,
                            "position": f"{source_design}-{index}",
                            "response_well": response + offset,
                            "magnitude_well": 2.0 + offset,
                        }
                    )
    return pd.DataFrame.from_records(records)


def _traces() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    time_by_experiment = {
        "experiment_a": (-1.0, 0.0, 1.0, 2.0),
        "experiment_b": (-0.5, 0.0, 1.5, 3.0),
    }
    for experiment_index, (experiment_id, times) in enumerate(time_by_experiment.items()):
        for signal_kind in ("growth", "response", "magnitude"):
            source_designs = ("design_alpha", "pDual-10") if signal_kind == "magnitude" else ("design_alpha",)
            for source_design in source_designs:
                for replicate_index, position in enumerate(("A1", "A2", "A3")):
                    for time in times:
                        if signal_kind == "growth":
                            value = 0.2 + experiment_index * 0.03 + replicate_index * 0.01 + (time + 1.0) * 0.03
                        else:
                            log_value = (
                                -1.2
                                + experiment_index * 0.4
                                + replicate_index * 0.05
                                + time * 0.08
                                + (0.25 if source_design == "pDual-10" else 0.0)
                            )
                            value = 2.0**log_value
                        records.append(
                            {
                                "experiment_id": experiment_id,
                                "design_id": source_design,
                                "position": position,
                                "state": "01",
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
