from __future__ import annotations

import json
from types import SimpleNamespace

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from reader.domains.plate_reader.analysis.response_window import promoter_evidence_figure as figure_module
from reader.domains.plate_reader.analysis.response_window.promoter_evidence_overlay import (
    load_objective_display_overlay,
)
from reader.domains.promoter import sequence_panel as sequence_panel_module
from reader.domains.promoter.candidate_bindings import (
    PromoterCandidateBinding,
)
from reader.response_window_review import OBJECTIVE_DISPLAY_LABEL_MAX_LENGTH
from reader.tests.domains.plate_reader.analysis.response_window.test_promoter_evidence_overlay import (
    _write_overlay,
)


def test_promoter_evidence_figure_connects_trajectories_handoff_and_sequence_without_static_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sequence_panel_module, "require_baserender_api", lambda: _FakeBaseRender)

    figure, diagnostics = figure_module.promoter_evidence_figure(
        experiment_id="experiment",
        design_id="design",
        reduction_id="event_logmean_6_12h_post",
        selected=_selected(),
        wells=_wells(reduction_id="event_logmean_6_12h_post"),
        traces=_traces(),
        events=pd.DataFrame([{"experiment_id": "experiment", "event_time_uncertainty_h": 0.2}]),
        display=_display(),
        binding=_binding(),
    )

    try:
        assert diagnostics.adapter_kind == "densegen_tfbs"
        assert len(figure.axes) == 7
        header, growth, response, fluorescence, handoff, handoff_families, sequence = figure.axes
        assert header.get_gid() == "promoter-evidence-header"
        assert not header.texts
        assert [axis.get_title() for axis in (growth, response, fluorescence)] == [
            "Growth trajectory across conditions",
            "Reporter response across conditions",
            "Fluorescence relative to pDual-10",
        ]
        assert all(not axis.get_title(loc="left") for axis in (growth, response, fluorescence))
        assert response.get_ylabel() == "log₂(YFP / CFP)"
        assert fluorescence.get_ylabel() == "log₂(YFP / OD600)"
        assert all(axis.get_box_aspect() == 1.0 for axis in (growth, response, fluorescence, handoff))
        assert handoff.get_title() == "Eight-value response-window handoff"
        assert handoff.get_title(loc="left") == ""
        assert handoff_families.get_gid() == "promoter-evidence-handoff-families"
        assert sequence.get_title() == "Measured promoter sequence · spyP promoter"
        assert sequence.get_title(loc="left") == ""
        assert not any("Provenance and QC" in axis.get_title() for axis in figure.axes)
        assert handoff.findobj(lambda artist: artist.get_gid() == "bootstrap-uncertainty")
        assert handoff.findobj(lambda artist: artist.get_gid() == "event-time-sensitivity")
        assert handoff.findobj(lambda artist: artist.get_gid() == "censor-bound-r10")[0].get_text() == "≥"
        assert len(fluorescence.findobj(lambda artist: artist.get_gid() == "anchor-replicate-interval")) == 4
        assert [tick.get_text() for tick in handoff.get_yticklabels()] == [
            "r₀₀",
            "r₁₀",
            "r₀₁",
            "r₁₁",
            "b₀₀",
            "b₁₀",
            "b₀₁",
            "b₁₁",
        ]
        for state in ("00", "10", "01", "11"):
            response_points = handoff.findobj(lambda artist, gid=f"replicate-values-r{state}": artist.get_gid() == gid)
            assert len(response_points) == 1
            assert len(response_points[0].get_offsets()) == 3
            assert not handoff.findobj(lambda artist, gid=f"replicate-values-b{state}": artist.get_gid() == gid)
            for prefix in ("r", "b"):
                assert handoff.findobj(lambda artist, gid=f"handoff-summary-{prefix}{state}": artist.get_gid() == gid)
        assert sequence.get_position().x0 > handoff.get_position().x1
        assert handoff_families.get_position().x0 >= handoff.get_position().x1
        assert sequence.get_position().x0 >= handoff_families.get_position().x1
        assert abs(sequence.get_ylim()[1] - sequence.get_ylim()[0]) == 80
        assert mcolors.to_hex(figure.get_facecolor()) == "#ffffff"
        assert all(mcolors.to_hex(axis.get_facecolor()) == "#ffffff" for axis in figure.axes)
        assert figure._suptitle is not None
        assert figure._suptitle.get_text() == "Promoter response evidence · 6–12 h log mean (primary)"
        assert len(figure._suptitle.get_text()) < 100
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        header_legend = header.get_legend()
        assert header_legend is not None
        assert {text.get_text() for text in header_legend.get_texts()} == {
            "No stress",
            "Ethanol",
            "Ciprofloxacin",
            "Ethanol + ciprofloxacin",
            "Selected design",
            "pDual-10 anchor",
        }
        assert growth.get_legend() is None
        span_legend = response.get_legend()
        assert span_legend is not None
        assert [text.get_text() for text in span_legend.get_texts()] == [
            "Event interval",
            "Selected window",
        ]
        assert span_legend._loc == 4
        assert {
            tick.get_fontsize()
            for axis in (growth, response, fluorescence, handoff)
            for tick in (*axis.get_xticklabels(), *axis.get_yticklabels())
        } == {10.0}
        assert {axis.xaxis.label.get_fontsize() for axis in (growth, response, fluorescence, handoff)} == {10.0}
        response_family = handoff_families.findobj(lambda artist: artist.get_gid() == "handoff-family-response-label")
        fluorescence_family = handoff_families.findobj(
            lambda artist: artist.get_gid() == "handoff-family-fluorescence-label"
        )
        assert [text.get_text() for text in response_family] == ["Response rᵢ\nlog₂(YFP/CFP)"]
        assert [text.get_text() for text in fluorescence_family] == [
            "Fluorescence bᵢ\nlog₂(YFP/OD600)\nrelative to\nsame-state\npDual-10"
        ]
        assert {text.get_fontsize() for text in (*response_family, *fluorescence_family)} == {10.0}
        assert handoff_families.findobj(lambda artist: artist.get_gid() == "handoff-family-response-bracket")
        assert handoff_families.findobj(lambda artist: artist.get_gid() == "handoff-family-fluorescence-bracket")
        assert header_legend.get_window_extent(renderer).y1 < figure._suptitle.get_window_extent(renderer).y0
        assert len(figure.legends) == 1
        handoff_legend = figure.legends[0]
        assert [text.get_text() for text in handoff_legend.get_texts()] == [
            "Observed response wells",
            "Median + 90% bootstrap",
            "Value range covering earliest/latest recorded stress-addition times",
        ]
        assert {text.get_fontsize() for text in handoff_legend.get_texts()} == {8.5}
        legend_box = handoff_legend.get_window_extent(renderer)
        figure_box = figure.get_window_extent(renderer)
        handoff_box = handoff_families.get_window_extent(renderer)
        for family_label in (*response_family, *fluorescence_family):
            family_box = family_label.get_window_extent(renderer)
            assert family_box.x0 >= handoff_box.x0
            assert family_box.x1 <= handoff_box.x1
            assert family_box.y0 >= handoff_box.y0
            assert family_box.y1 <= handoff_box.y1
        data_boxes = [axis.get_window_extent(renderer) for axis in (growth, response, fluorescence, handoff)]
        sequence_height_fraction = sequence.get_window_extent(renderer).height / figure_box.height
        assert 0.15 <= sequence_height_fraction <= 0.20
        assert max(box.width for box in data_boxes) - min(box.width for box in data_boxes) <= 1.0
        assert max(box.height for box in data_boxes) - min(box.height for box in data_boxes) <= 1.0
        handoff_data_box = data_boxes[-1]
        assert 0.0 <= handoff_box.x0 - handoff_data_box.x1 <= 20.0
        assert legend_box.x0 >= figure_box.x0
        assert legend_box.x1 <= figure_box.x1
        assert legend_box.y0 >= figure_box.y0
        assert legend_box.y1 <= figure_box.y1
        assert figure.get_gid() == "reader.response_window.promoter_evidence_bundle.v5"
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("objective_id", "objective_display_label"),
    [
        ("response_magnitude_feasibility_v1", "RMF"),
        ("multistate_response_behavior_v1", "MSRB"),
    ],
)
def test_promoter_evidence_figure_keeps_objective_overlay_out_of_static_plot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    objective_id: str,
    objective_display_label: str,
) -> None:
    monkeypatch.setattr(sequence_panel_module, "require_baserender_api", lambda: _FakeBaseRender)
    overlay_path = _write_overlay(tmp_path)
    payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    payload["objective_id"] = objective_id
    payload["objective_display_label"] = objective_display_label
    overlay_path.write_text(json.dumps(payload), encoding="utf-8")
    overlay = load_objective_display_overlay(overlay_path)

    figure, _diagnostics = figure_module.promoter_evidence_figure(
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
        selected=_selected(),
        wells=_wells(reduction_id="primary"),
        traces=_traces(),
        events=pd.DataFrame([{"experiment_id": "experiment", "event_time_uncertainty_h": 0.2}]),
        display=_display(),
        binding=_binding(),
        objective_overlay=overlay,
    )

    try:
        text = "\n".join(item.get_text() for axis in figure.axes for item in axis.texts)
        assert objective_display_label not in text
        assert objective_id not in text
        assert "Response separation" not in text
        assert "ON fluorescence floor" not in text
    finally:
        plt.close(figure)


def test_promoter_evidence_figure_layout_is_independent_of_overlay_component_count(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sequence_panel_module, "require_baserender_api", lambda: _FakeBaseRender)
    overlay_path = _write_overlay(tmp_path)
    payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    payload["objective_display_label"] = "W" * OBJECTIVE_DISPLAY_LABEL_MAX_LENGTH
    payload["components"].extend(
        {
            "component_id": f"screen_component_{index}",
            "label": f"Screen component {index}",
            "value": float(index),
            "unit": "raw log2 units",
        }
        for index in range(4, 7)
    )
    overlay_path.write_text(json.dumps(payload), encoding="utf-8")
    overlay = load_objective_display_overlay(overlay_path)

    figure, _diagnostics = figure_module.promoter_evidence_figure(
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
        selected=_selected(),
        wells=_wells(reduction_id="primary"),
        traces=_traces(),
        events=pd.DataFrame([{"experiment_id": "experiment", "event_time_uncertainty_h": 0.2}]),
        display=_display(),
        binding=_binding(),
        objective_overlay=overlay,
    )

    try:
        figure.canvas.draw()
        assert len(figure.axes) == 7
        assert figure.axes[-1].get_title() == "Measured promoter sequence · spyP promoter"
    finally:
        plt.close(figure)


class _FakeBaseRender:
    BASERENDER_SEQUENCE_PANEL_CONTRACT_ID = "dnadesign.baserender.sequence_panel.v1"
    BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION = "1"

    @staticmethod
    def render_sequence_panel_image(row, **kwargs):
        assert row == {
            "id": "candidate-spyp",
            "sequence": "ACGTACGT",
            "densegen__used_tfbs_detail": [
                {
                    "part_kind": "tfbs",
                    "sequence": "ACGT",
                    "regulator": "CpxR",
                    "orientation": "fwd",
                    "offset": 0,
                    "length": 4,
                    "end": 4,
                }
            ],
        }
        assert kwargs["adapter_kind"] == "densegen_tfbs"
        assert kwargs["target_height_px"] == 480
        image = np.full((80, 400, 4), 255, dtype=np.uint8)
        image[20:60, 50:350, :3] = 0
        return SimpleNamespace(
            image=image,
            diagnostics=SimpleNamespace(
                contract_id="dnadesign.baserender.sequence_panel.v1",
                contract_version="1",
                adapter_kind="densegen_tfbs",
                style_profile="promoter_compact_slide.v1",
                renderer_name="sequence_rows",
                sequence_length_bp=8,
                feature_count=0,
                strand_count=2,
                legend_entries=(),
                image_width_px=400,
                image_height_px=80,
            ),
        )


class _FakeGenBankBaseRender:
    BASERENDER_SEQUENCE_PANEL_CONTRACT_ID = "dnadesign.baserender.sequence_panel.v1"
    BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION = "1"

    @staticmethod
    def render_sequence_panel_image(row, **kwargs):
        assert row["id"] == "candidate-spyp"
        assert row["sequence"] == "ACGTACGT"
        assert row["seq_annot__source_file"] == "_artifacts/genbank/source.gb"
        assert kwargs["adapter_kind"] == "usr_genbank_annotations_v1"
        return SimpleNamespace(
            image=np.full((80, 400, 4), 255, dtype=np.uint8),
            diagnostics=SimpleNamespace(
                contract_id="dnadesign.baserender.sequence_panel.v1",
                contract_version="1",
                adapter_kind="usr_genbank_annotations_v1",
                style_profile="promoter_compact_slide.v1",
                renderer_name="sequence_rows",
                sequence_length_bp=8,
                feature_count=1,
                strand_count=2,
                legend_entries=(),
                image_width_px=400,
                image_height_px=80,
            ),
        )


def _binding() -> PromoterCandidateBinding:
    return PromoterCandidateBinding(
        reader_design_id="design",
        display_label="spyP promoter",
        candidate_id="candidate-spyp",
        canonical_sequence="ACGTACGT",
        sequence_sha256="0" * 64,
        candidate_table_id="candidate-table",
        candidate_selection_sha256="1" * 64,
        sequence_authority_dataset_id="authority",
        sequence_authority_id="authority-row",
        sequence_authority_sha256="2" * 64,
        source_class="densegen",
        design_family="stress_promoter",
        densegen_plan="plan-v1",
        densegen_run_id="run-v1",
        densegen_sampling_library_hash="library-v1",
        baserender_adapter_kind="densegen_tfbs",
        baserender_record={
            "id": "candidate-spyp",
            "sequence": "ACGTACGT",
            "densegen__used_tfbs_detail": [
                {
                    "part_kind": "tfbs",
                    "sequence": "ACGT",
                    "regulator": "CpxR",
                    "orientation": "fwd",
                    "offset": 0,
                    "length": 4,
                    "end": 4,
                }
            ],
        },
        binding_status="resolved",
        binding_method="exact_alias",
    )


def _selected() -> pd.Series:
    values: dict[str, object] = {
        "reference_design_id": "pDual-10",
        "confidence_level": 0.9,
        "replicate_stat": "median",
        "reduction_method": "geometric_time_mean",
        "response_basis": "post_window",
        "reduction_role": "primary",
        "window_start_event_h": 6.0,
        "window_end_event_h": 12.0,
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
    return pd.Series(values)


def _wells(*, reduction_id: str) -> pd.DataFrame:
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
                        "reduction_id": reduction_id,
                        "state": state,
                        "position": f"{source_design}-{index}",
                        "response_well": center + offset,
                        "magnitude_well": 2.0 + offset,
                    }
                )
    return pd.DataFrame.from_records(records)


def _traces() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for signal_kind in ("growth", "response", "magnitude"):
        source_designs = ("design", "pDual-10") if signal_kind == "magnitude" else ("design",)
        for source_design in source_designs:
            for state_index, state in enumerate(("00", "10", "01", "11")):
                for replicate_index, position in enumerate(("A1", "A2", "A3")):
                    for time in (-1.0, 0.0, 6.0, 12.0):
                        if signal_kind == "growth":
                            value = 0.2 + state_index * 0.02 + replicate_index * 0.01 + (time + 1.0) * 0.01
                        else:
                            log_value = (
                                -1.5
                                + state_index * 0.2
                                + replicate_index * 0.05
                                + time * 0.02
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
