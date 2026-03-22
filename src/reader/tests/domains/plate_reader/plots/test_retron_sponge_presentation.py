from __future__ import annotations

import pandas as pd
import pytest

from reader.domains.plate_reader.plots import _retron_sponge_presentation as retron_presentation


def test_retron_presentation_renders_summary_text_with_window_notes() -> None:
    trace = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": "spyP",
                "stress_condition": "3% EtOH",
                "time": time_value,
                "time_from_stress": time_value,
                "configured_max_post_stress_hours": 12.0,
                "in_primary_post_stress": True,
                "in_endpoint_window": time_value >= 3.0,
            }
            for time_value in (0.5, 3.0, 12.0)
        ]
    )

    text = retron_presentation.render_summary_text(
        retron_presentation.summary_metric_text_spec("C_END"),
        trace=trace,
    )

    assert text == (
        "Endpoint of the matched tetO deviation trace; "
        "Window first 12.0 h after stress addition; "
        "Endpoint uses the last 2 flagged reads inside the summary window"
    )


def test_retron_presentation_primary_window_span_prefers_configured_hours() -> None:
    trace = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": "spyP",
                "stress_condition": "3% EtOH",
                "time_from_stress": time_value,
                "configured_max_post_stress_hours": 12.0,
                "in_primary_post_stress": True,
            }
            for time_value in (0.25, 0.5, 1.0)
        ]
    )

    span = retron_presentation.primary_window_span_bounds(trace, stress_condition="3% EtOH")

    assert span == pytest.approx((0.0, 12.0))


def test_retron_presentation_unknown_metric_fallbacks_are_explicit() -> None:
    spec = retron_presentation.summary_metric_text_spec("mystery_END")

    assert retron_presentation.metric_axis_label("mystery") == "Retron sponge metric (mystery)"
    assert retron_presentation.summary_metric_label("mystery") == "Retron sponge summary metric (mystery)"
    assert retron_presentation.burden_axis_label("mystery") == "Burden summary (mystery)"
    assert retron_presentation.trace_text_spec("mystery").formula == "mystery"
    assert spec.include_primary_window_note is True
    assert spec.include_endpoint_window_note is True


def test_retron_presentation_exposes_decision_card_metric_specs_in_review_order() -> None:
    specs = retron_presentation.decision_card_metric_specs()

    assert [spec.metric for spec in specs] == ["P_pre", "D_abs_AUC", "D_AUC"]
    assert [spec.label for spec in specs] == [
        "Pre-stress shift",
        "Total window effect",
        "Post-stress effect",
    ]
    assert [spec.axis_label for spec in specs] == [
        "Matched ratio shift",
        "AUC[D_abs(t)]",
        "AUC[D(t)]",
    ]


def test_retron_presentation_registers_cross_run_summary_labels() -> None:
    assert retron_presentation.summary_metric_label("O_AUC") == "Expected-direction increment AUC"
    assert retron_presentation.summary_metric_label("T_finalOD") == "tetO endpoint burden"
