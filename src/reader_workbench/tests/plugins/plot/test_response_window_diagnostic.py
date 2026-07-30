from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from reader_workbench.plugins.plot.response_window_diagnostic import (
    ResponseWindowDiagnosticCfg,
    ResponseWindowDiagnosticPlot,
)


def _traces_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for design_id, is_reference in (("selected", False), ("reference", True)):
        for signal_index, signal_kind in enumerate(("growth", "response", "magnitude"), start=1):
            for state_index, state in enumerate(("00", "10", "01", "11")):
                for time in (0.0, 1.0):
                    rows.append(
                        {
                            "experiment_id": "source",
                            "design_id": design_id,
                            "position": "A1",
                            "state": state,
                            "time": time,
                            "time_from_event_h": time,
                            "value": float(signal_index + state_index + time + (is_reference * 0.5)),
                            "value_policy_clipped": False,
                            "value_instrument_overflow": False,
                            "value_bound_kind": "exact",
                            "signal_kind": signal_kind,
                            "is_reference": is_reference,
                        }
                    )
    return pd.DataFrame.from_records(rows)


def _designs_frame() -> pd.DataFrame:
    components = ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")
    values: dict[str, float] = {}
    for index, component in enumerate(components):
        value = float(index)
        values[component] = value
        values[f"{component}_descriptive_interval_low"] = value - 0.25
        values[f"{component}_descriptive_interval_high"] = value + 0.25
        values[f"{component}_event_half_range"] = 0.1
        values[f"{component}_bound_kind"] = "exact"
        values[f"{component}_has_policy_clipping"] = False
        values[f"{component}_has_instrument_overflow"] = False
        values[f"{component}_event_sensitivity_has_policy_clipping"] = False
        values[f"{component}_event_sensitivity_has_instrument_overflow"] = False
    return pd.DataFrame.from_records(
        [
            {
                "experiment_id": "source",
                "design_id": "selected",
                "reference_design_id": "reference",
                "reduction_id": "primary",
                "reduction_method": "geometric_time_mean",
                "response_basis": "post_window",
                "observation_stat": "median",
                "descriptive_resampling_draws": 100,
                "descriptive_interval_mass": 0.95,
                "event_id": "addition",
                "event_time_uncertainty_h": 0.25,
                "window_start_event_h": 0.25,
                "window_end_event_h": 0.75,
                "is_reference": False,
                **values,
            }
        ]
    )


def test_response_window_diagnostic_declares_record_contracts() -> None:
    ports = ResponseWindowDiagnosticPlot.input_ports()

    assert ports["traces"].contract == "plate_reader.response_window.traces.v3"
    assert ports["designs"].contract == "plate_reader.response_window.designs.v4"


def test_response_window_diagnostic_adapts_figure_metadata() -> None:
    cfg = ResponseWindowDiagnosticCfg(
        source_experiment_id="source",
        design_id="selected",
        primary_reduction_id="primary",
        pre_window_duration_h=None,
        title="Selected diagnostic",
        filename="diagnostic",
        format=["png", "pdf"],
        dpi=144,
    )

    rendered = ResponseWindowDiagnosticPlot().render(
        None,
        {"traces": _traces_frame(), "designs": _designs_frame()},
        cfg,
    )

    assert [(item.filename, item.ext, item.dpi) for item in rendered] == [
        ("diagnostic", "png", 144),
        ("diagnostic", "pdf", 144),
    ]
    assert {item.description for item in rendered} == {
        "Event-relative growth, response, magnitude, and reduced components for one source experiment and design."
    }
    assert rendered[0].fig is rendered[1].fig
    assert rendered[0].fig.get_suptitle().startswith("Selected diagnostic\n")
    title = rendered[0].fig.get_suptitle()
    assert "median center across within-experiment observations" in title
    assert "descriptive resampling interval" in title
    assert "replicate" not in title
    assert "confidence" not in title
    assert " CI" not in title
    plt.close(rendered[0].fig)
