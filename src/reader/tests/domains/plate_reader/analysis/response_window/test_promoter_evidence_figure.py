from __future__ import annotations

from types import SimpleNamespace

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.markers import MarkerStyle

from reader.domains.plate_reader.analysis.response_window import promoter_evidence_figure as figure_module
from reader.domains.plate_reader.analysis.response_window.promoter_evidence_overlay import (
    load_objective_display_overlay,
)
from reader.domains.plate_reader.analysis.response_window.visual_labels import STATE_MARKERS
from reader.domains.promoter import sequence_panel as sequence_panel_module
from reader.domains.promoter.candidate_bindings import (
    PromoterCandidateBinding,
)
from reader.tests.domains.plate_reader.analysis.response_window.test_promoter_evidence_overlay import (
    _write_overlay,
)


def test_promoter_evidence_figure_connects_trajectories_handoff_provenance_and_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sequence_panel_module, "require_baserender_api", lambda: _FakeBaseRender)

    figure, diagnostics = figure_module.promoter_evidence_figure(
        experiment_id="experiment",
        design_id="design",
        reduction_id="event_logmean_6_12h_post",
        selected=_selected(),
        traces=_traces(),
        events=pd.DataFrame([{"experiment_id": "experiment", "event_time_uncertainty_h": 0.2}]),
        display=_display(),
        binding=_binding(),
    )

    try:
        assert diagnostics.adapter_kind == "densegen_tfbs"
        assert len(figure.axes) == 8
        header, growth, response, fluorescence, response_handoff, fluorescence_handoff, provenance, sequence = (
            figure.axes
        )
        assert header.get_gid() == "promoter-evidence-header"
        header_text = "\n".join(text.get_text() for text in header.texts)
        assert "Experiment  experiment" in header_text
        assert "Reduction  event_logmean_6_12h_post" in header_text
        assert [axis.get_title(loc="left") for axis in (growth, response, fluorescence)] == [
            "A  Growth by condition",
            "B  YFP / CFP response",
            "C  YFP / OD600 with pDual-10 anchor",
        ]
        assert response_handoff.get_title(loc="left") == "D1  Response handoff, r_i"
        assert fluorescence_handoff.get_title(loc="left") == "D2  pDual-10-relative fluorescence, b_i"
        assert provenance.get_title(loc="left") == "E  Provenance and QC"
        assert response_handoff.findobj(lambda artist: artist.get_gid() == "bootstrap-uncertainty")
        assert response_handoff.findobj(lambda artist: artist.get_gid() == "event-time-sensitivity")
        assert fluorescence_handoff.findobj(lambda artist: artist.get_gid() == "bootstrap-uncertainty")
        assert fluorescence_handoff.findobj(lambda artist: artist.get_gid() == "event-time-sensitivity")
        for axis in (response_handoff, fluorescence_handoff):
            for state, marker in STATE_MARKERS.items():
                gid = f"handoff-value-{state}"
                state_points = axis.findobj(lambda artist, expected_gid=gid: artist.get_gid() == expected_gid)
                assert len(state_points) == 1
                expected = MarkerStyle(marker).get_path().transformed(MarkerStyle(marker).get_transform()).vertices
                np.testing.assert_allclose(state_points[0].get_paths()[0].vertices, expected)
        provenance_text = "\n".join(text.get_text() for text in provenance.texts)
        assert "Objective-neutral evidence" in provenance_text
        assert "RMF is not calculated by Reader" in provenance_text
        assert "Binding  exact alias" in provenance_text
        assert sequence.get_title(loc="left") == "F  DenseGen TFBS annotation"
        assert sequence.get_subplotspec().colspan.start == 0
        assert sequence.get_subplotspec().colspan.stop == 3
        assert abs(sequence.get_ylim()[1] - sequence.get_ylim()[0]) < 80
        assert mcolors.to_hex(figure.get_facecolor()) == "#ffffff"
        assert all(mcolors.to_hex(axis.get_facecolor()) == "#ffffff" for axis in figure.axes)
        assert figure._suptitle is not None
        assert "design · candidate-spyp" in figure._suptitle.get_text()
        assert len(figure._suptitle.get_text()) < 100
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        header_legend = header.get_legend()
        assert header_legend is not None
        assert {text.get_text() for text in header_legend.get_texts()} >= {
            "Bootstrap SD",
            "Event-time sensitivity",
        }
        assert header_legend.get_window_extent(renderer).y1 < figure._suptitle.get_window_extent(renderer).y0
        assert figure.get_gid() == "reader.response_window.promoter_evidence_bundle.v2"
    finally:
        plt.close(figure)


def test_promoter_evidence_figure_labels_supplied_raw_objective_values_as_screen_only(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sequence_panel_module, "require_baserender_api", lambda: _FakeBaseRender)
    overlay = load_objective_display_overlay(_write_overlay(tmp_path))

    figure, _diagnostics = figure_module.promoter_evidence_figure(
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
        selected=_selected(),
        traces=_traces(),
        events=pd.DataFrame([{"experiment_id": "experiment", "event_time_uncertainty_h": 0.2}]),
        display=_display(),
        binding=_binding(),
        objective_overlay=overlay,
    )

    try:
        text = "\n".join(item.get_text() for item in figure.axes[6].texts)
        assert "RMF raw components · screen only" in text
        assert "Response separation  0.8 raw log2 units" in text
        assert "ON fluorescence floor  0.3 pDual-10-relative log2 fluorescence" in text
        assert "calibrated score" not in text.lower()
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
        assert kwargs["target_height_px"] == 310
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
        "window_start_event_h": 6.0,
        "window_end_event_h": 12.0,
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
