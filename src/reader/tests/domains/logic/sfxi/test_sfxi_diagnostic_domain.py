from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from reader.domains.logic.sfxi.diagnostic import (
    prepare_sfxi_diagnostics,
    render_sfxi_diagnostic,
)


def _annotated_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    state_values = {"00": "off", "10": "input-a", "01": "input-b", "11": "both"}
    for design_index, design_id in enumerate(("design-a", "design-b")):
        for state_index, source_value in enumerate(state_values.values()):
            for time in (0.0, 1.0, 2.0):
                for replicate in (1, 2):
                    rows.extend(
                        (
                            {
                                "design_id": design_id,
                                "treatment_alias": source_value,
                                "time": time,
                                "channel": "OD600",
                                "value": 0.2 + design_index + state_index * 0.1 + time + replicate * 0.01,
                                "position": f"A{replicate}",
                            },
                            {
                                "design_id": design_id,
                                "treatment_alias": source_value,
                                "time": time,
                                "channel": "response",
                                "value": 1.0 + design_index + state_index + time + replicate * 0.1,
                                "position": f"A{replicate}",
                            },
                        )
                    )
    return pd.DataFrame.from_records(rows)


def _vec8_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for design_index, design_id in enumerate(("design-a", "design-b")):
        rows.append(
            {
                "design_id": design_id,
                "time_selected_h": float(design_index + 1),
                "reference_design_id": "reference",
                "v00": 0.0,
                "v10": 0.25,
                "v01": 0.75,
                "v11": 1.0,
                "y00_star": -1.0,
                "y10_star": -0.5,
                "y01_star": 0.5,
                "y11_star": 1.0,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_sfxi_diagnostic_uses_each_persisted_vec8_time() -> None:
    prepared = prepare_sfxi_diagnostics(
        _annotated_frame(),
        _vec8_frame(),
        treatment_column="treatment_alias",
        treatment_map={"00": "off", "10": "input-a", "01": "input-b", "11": "both"},
        treatment_case_sensitive=True,
        time_column="time",
        growth_channel="OD600",
        response_channel="response",
    )

    assert [item.design_id for item in prepared] == ["design-a", "design-b"]
    assert [item.selected_time_h for item in prepared] == [1.0, 2.0]
    assert prepared[0].triptych.snapshot_time == 1.0
    assert prepared[1].triptych.snapshot_time == 2.0
    assert prepared[0].logic_components == {"00": 0.0, "10": 0.25, "01": 0.75, "11": 1.0}
    assert prepared[0].intensity_components == {"00": -1.0, "10": -0.5, "01": 0.5, "11": 1.0}


def test_sfxi_diagnostic_can_select_designs_without_reordering_them() -> None:
    prepared = prepare_sfxi_diagnostics(
        _annotated_frame(),
        _vec8_frame(),
        treatment_column="treatment_alias",
        treatment_map={"00": "off", "10": "input-a", "01": "input-b", "11": "both"},
        treatment_case_sensitive=True,
        time_column="time",
        growth_channel="OD600",
        response_channel="response",
        design_ids=["design-b"],
    )

    assert [item.design_id for item in prepared] == ["design-b"]


def test_sfxi_diagnostic_rejects_missing_persisted_selection_time() -> None:
    vec8 = _vec8_frame()
    vec8.loc[vec8["design_id"].eq("design-a"), "time_selected_h"] = pd.NA

    with pytest.raises(ValueError, match="time_selected_h"):
        prepare_sfxi_diagnostics(
            _annotated_frame(),
            vec8,
            treatment_column="treatment_alias",
            treatment_map={"00": "off", "10": "input-a", "01": "input-b", "11": "both"},
            treatment_case_sensitive=True,
            time_column="time",
            growth_channel="OD600",
            response_channel="response",
        )


def test_sfxi_diagnostic_rejects_a_vec8_time_absent_from_traces() -> None:
    vec8 = _vec8_frame()
    vec8.loc[vec8["design_id"].eq("design-a"), "time_selected_h"] = 1.5

    with pytest.raises(ValueError, match="persisted selection time"):
        prepare_sfxi_diagnostics(
            _annotated_frame(),
            vec8,
            treatment_column="treatment_alias",
            treatment_map={"00": "off", "10": "input-a", "01": "input-b", "11": "both"},
            treatment_case_sensitive=True,
            time_column="time",
            growth_channel="OD600",
            response_channel="response",
        )


def test_sfxi_diagnostic_renders_four_explicit_measurement_panels() -> None:
    prepared = prepare_sfxi_diagnostics(
        _annotated_frame(),
        _vec8_frame(),
        treatment_column="treatment_alias",
        treatment_map={"00": "off", "10": "input-a", "01": "input-b", "11": "both"},
        treatment_case_sensitive=True,
        time_column="time",
        growth_channel="OD600",
        response_channel="response",
        design_ids=["design-a"],
    )[0]

    figure = render_sfxi_diagnostic(prepared, figsize=(12.0, 3.5), dpi=72)

    assert len(figure.axes) == 4
    assert [axis.get_title() for axis in figure.axes] == [
        "Growth trajectory",
        "Response trajectory",
        "Logic shape",
        "Relative intensity",
    ]
    assert "design-a" in figure.get_suptitle()
    assert "selected time 1 h" in figure.get_suptitle()
    assert "reference reference" in figure.get_suptitle()
    plt.close(figure)
