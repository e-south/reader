from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import ValidationError

from reader_workbench.plugins.plot.four_state_event_window_diagnostic import (
    FourStateEventWindowDiagnosticCfg,
    FourStateEventWindowDiagnosticPlot,
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


def test_four_state_event_window_diagnostic_declares_record_contracts() -> None:
    ports = FourStateEventWindowDiagnosticPlot.input_ports()

    assert ports["traces"].contract == "plate_reader.four_state_event_window.traces.v3"
    assert ports["designs"].contract == "plate_reader.four_state_event_window.designs.v4"


def test_four_state_event_window_diagnostic_adapts_figure_metadata() -> None:
    cfg = FourStateEventWindowDiagnosticCfg(
        subjects=[
            {
                "source_experiment_id": "source",
                "design_id": "selected",
                "title": "Selected diagnostic",
                "filename": "diagnostic",
            }
        ],
        primary_reduction_id="primary",
        pre_window_duration_h=None,
        format=["png", "pdf"],
        dpi=144,
    )

    rendered = FourStateEventWindowDiagnosticPlot().render(
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
    assert "median across observations" in title
    assert "95% resampling range" in title
    assert "replicate" not in title
    assert "confidence" not in title
    assert " CI" not in title
    plt.close(rendered[0].fig)


def test_four_state_event_window_diagnostic_renders_an_explicit_subject_set() -> None:
    traces = _traces_frame()
    second_traces = traces.loc[traces["design_id"].eq("selected")].copy()
    second_traces["design_id"] = "selected-b"
    traces = pd.concat([traces, second_traces], ignore_index=True)
    designs = _designs_frame()
    second_design = designs.copy()
    second_design["design_id"] = "selected-b"
    designs = pd.concat([designs, second_design], ignore_index=True)
    cfg = FourStateEventWindowDiagnosticCfg(
        subjects=[
            {
                "source_experiment_id": "source",
                "design_id": "selected-b",
                "filename": "selected-a",
                "title": "Selected A",
            },
            {
                "source_experiment_id": "source",
                "design_id": "selected",
                "filename": "selected-b",
                "title": "Selected B",
            },
        ],
        primary_reduction_id="primary",
        format=["svg"],
    )

    rendered = FourStateEventWindowDiagnosticPlot().render(
        None,
        {"traces": traces, "designs": designs},
        cfg,
    )

    assert [(item.filename, item.ext) for item in rendered] == [
        ("selected-a", "svg"),
        ("selected-b", "svg"),
    ]
    assert rendered[0].fig is not rendered[1].fig
    assert rendered[0].fig.get_suptitle().startswith("Selected A\n")
    assert rendered[1].fig.get_suptitle().startswith("Selected B\n")
    for item in rendered:
        plt.close(item.fig)


@pytest.mark.parametrize(
    ("first_filename", "second_filename"),
    [("design a", "design/a"), ("Design-A", "design-a")],
)
def test_four_state_event_window_diagnostic_rejects_portable_filename_collisions(
    first_filename: str,
    second_filename: str,
) -> None:
    with pytest.raises(ValidationError, match="unique filenames"):
        FourStateEventWindowDiagnosticCfg(
            subjects=[
                {"source_experiment_id": "source", "design_id": "a", "filename": first_filename},
                {"source_experiment_id": "source", "design_id": "b", "filename": second_filename},
            ],
            primary_reduction_id="primary",
        )


@pytest.mark.parametrize("filename", [" ", "...", "///"])
def test_four_state_event_window_diagnostic_rejects_empty_filename_slug(filename: str) -> None:
    with pytest.raises(ValidationError, match="filesystem-safe character|must not be blank"):
        FourStateEventWindowDiagnosticCfg(
            subjects=[
                {
                    "source_experiment_id": "source",
                    "design_id": "a",
                    "filename": filename,
                }
            ],
            primary_reduction_id="primary",
        )


def test_four_state_event_window_diagnostic_rejects_unknown_subject_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FourStateEventWindowDiagnosticCfg(
            subjects=[
                {
                    "source_experiment_id": "source",
                    "design_id": "a",
                    "filename": "design-a",
                    "unknown": "value",
                }
            ],
            primary_reduction_id="primary",
        )


def test_four_state_event_window_diagnostic_requires_all_state_labels() -> None:
    with pytest.raises(ValidationError, match="state_labels must define exactly"):
        FourStateEventWindowDiagnosticCfg(
            subjects=[{"source_experiment_id": "source", "design_id": "a", "filename": "design-a"}],
            primary_reduction_id="primary",
            state_labels={"00": "No treatment"},
        )
