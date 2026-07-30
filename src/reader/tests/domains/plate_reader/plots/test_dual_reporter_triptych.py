from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from reader.domains.plate_reader.plots.dual_reporter_triptych import (
    build_dual_reporter_triptych_chart,
    build_triptych_data,
    summarize_design_context,
)
from reader.domains.plate_reader.plots.dual_reporter_triptych_render import (
    render_dual_reporter_triptych,
)


def _dual_reporter_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for treatment in ("water", "EtOH"):
        for time in (0.0, 1.0, 2.0):
            for rep in (1, 2):
                rows.append(
                    {
                        "design_id": "pTest",
                        "treatment": treatment,
                        "time": time,
                        "channel": "OD600",
                        "value": 0.1 + time + rep * 0.01,
                        "position": f"A{rep}",
                    }
                )
                rows.append(
                    {
                        "design_id": "pTest",
                        "treatment": treatment,
                        "time": time,
                        "channel": "YFP/CFP",
                        "value": (2.0 if treatment == "EtOH" else 1.0) + time + rep * 0.1,
                        "position": f"A{rep}",
                    }
                )
    return pd.DataFrame(rows)


def test_dual_reporter_triptych_builds_three_panel_data() -> None:
    result = build_triptych_data(
        _dual_reporter_df(),
        time_col="time",
        treatment_col="treatment",
        growth_channel="OD600",
        ratio_channel="YFP/CFP",
        snapshot_channel="YFP/CFP",
        snapshot_time=1.0,
        treatment_order=["water", "EtOH"],
    )

    assert list(result.treatment_order) == ["water", "EtOH"]
    assert set(result.od600_time["treatment"]) == {"water", "EtOH"}
    assert set(result.ratio_time["treatment"]) == {"water", "EtOH"}
    assert list(result.snapshot_stats["treatment"]) == ["water", "EtOH"]
    assert result.snapshot_points.shape[0] == 4
    assert result.trajectory_interval_mass == 0.95
    first_interval = result.od600_time.iloc[0]
    assert first_interval["y_mean"] - first_interval["y_sd"] < first_interval["y_lo"]
    assert first_interval["y_hi"] < first_interval["y_mean"] + first_interval["y_sd"]


def test_dual_reporter_triptych_offsets_snapshot_points_by_well_not_value() -> None:
    frame = _dual_reporter_df()
    mask_a1 = (
        frame["treatment"].eq("water")
        & frame["time"].eq(1.0)
        & frame["channel"].eq("YFP/CFP")
        & frame["position"].eq("A1")
    )
    mask_a2 = (
        frame["treatment"].eq("water")
        & frame["time"].eq(1.0)
        & frame["channel"].eq("YFP/CFP")
        & frame["position"].eq("A2")
    )
    frame.loc[mask_a1, "value"] = 9.0
    frame.loc[mask_a2, "value"] = 1.0

    result = build_triptych_data(
        frame,
        time_col="time",
        treatment_col="treatment",
        growth_channel="OD600",
        ratio_channel="YFP/CFP",
        snapshot_channel="YFP/CFP",
        snapshot_time=1.0,
        treatment_order=["water", "EtOH"],
    )

    water = result.snapshot_points.loc[result.snapshot_points["treatment"].eq("water")]
    assert water[["position", "value", "observation_index"]].to_dict("records") == [
        {"position": "A1", "value": 9.0, "observation_index": 0},
        {"position": "A2", "value": 1.0, "observation_index": 1},
    ]


def test_dual_reporter_triptych_explicit_treatment_order_is_closed() -> None:
    df = _dual_reporter_df()
    extra_rows = df[df["treatment"] == "water"].copy()
    extra_rows["treatment"] = "unexpected"
    df = pd.concat([df, extra_rows], ignore_index=True)

    result = build_triptych_data(
        df,
        time_col="time",
        treatment_col="treatment",
        growth_channel="OD600",
        ratio_channel="YFP/CFP",
        snapshot_channel="YFP/CFP",
        snapshot_time=1.0,
        treatment_order=["water", "EtOH", "AND"],
    )

    assert list(result.treatment_order) == ["water", "EtOH", "AND"]
    assert list(result.missing_treatments) == ["AND"]
    assert "unexpected" not in set(result.od600_time["treatment"])
    assert "unexpected" not in set(result.ratio_time["treatment"])
    assert "unexpected" not in set(result.snapshot_points["treatment"])


def test_dual_reporter_triptych_chart_uses_square_panels_and_full_treatment_domain() -> None:
    alt = pytest.importorskip("altair")
    result = build_triptych_data(
        _dual_reporter_df(),
        time_col="time",
        treatment_col="treatment",
        growth_channel="OD600",
        ratio_channel="YFP/CFP",
        snapshot_channel="YFP/CFP",
        snapshot_time=1.0,
        treatment_order=["water", "EtOH", "AND"],
    )

    chart = build_dual_reporter_triptych_chart(
        alt=alt,
        pd_module=pd,
        data=result,
        time_col="time",
        treatment_col="treatment",
        treatment_title="Condition state",
    )
    spec = chart.to_dict()

    assert spec["spacing"] == 16
    assert [panel["width"] for panel in spec["hconcat"]] == [260, 260, 260]
    assert [panel["height"] for panel in spec["hconcat"]] == [260, 260, 260]
    for panel in spec["hconcat"][:2]:
        line_layer = next(layer for layer in panel["layer"] if layer["mark"]["type"] == "line")
        assert line_layer["encoding"]["strokeDash"]["legend"]["title"] == "Condition state"
        assert len(line_layer["encoding"]["strokeDash"]["scale"]["range"]) == 3
    assert spec["resolve"]["scale"]["strokeDash"] == "shared"
    snapshot_layers = spec["hconcat"][2]["layer"]
    assert snapshot_layers[0]["encoding"]["x"]["scale"]["domain"] == ["water", "EtOH", "AND"]
    assert snapshot_layers[0]["encoding"]["x"]["axis"]["title"] == "Condition state"
    assert all(layer["mark"]["type"] != "bar" for layer in snapshot_layers)
    mean_layer = next(layer for layer in snapshot_layers if layer["mark"].get("size") == 24)
    assert mean_layer["encoding"]["y"]["field"] == "y_mean"
    assert mean_layer["mark"]["color"] == "#334155"
    assert "color" not in mean_layer["encoding"]
    point_layer = next(layer for layer in snapshot_layers if layer["mark"]["type"] == "point")
    assert point_layer["mark"]["fill"] == "white"
    assert point_layer["mark"]["stroke"] == "#94a3b8"
    assert point_layer["encoding"]["xOffset"]["field"] == "observation_index"


def test_dual_reporter_triptych_renders_static_three_panel_figure() -> None:
    data = build_triptych_data(
        _dual_reporter_df(),
        time_col="time",
        treatment_col="treatment",
        growth_channel="OD600",
        ratio_channel="YFP/CFP",
        snapshot_channel="YFP/CFP",
        snapshot_time=1.0,
        treatment_order=["water", "EtOH"],
    )

    figure = render_dual_reporter_triptych(
        data,
        time_col="time",
        treatment_col="treatment",
        acquisition_transition_time_h=0.5,
        title="pTest",
        colors=["#334155", "#2563eb"],
    )

    assert len(figure.axes) == 3
    assert [axis.get_title() for axis in figure.axes] == [
        "OD600 kinetics",
        "YFP/CFP kinetics",
        "YFP/CFP snapshot at 1 h",
    ]
    assert figure.get_suptitle() == "pTest"
    assert [tick.get_text() for tick in figure.axes[2].get_xticklabels()] == ["water", "EtOH"]
    assert len(figure.axes[2].collections) == 2
    assert {tuple(collection.get_edgecolors()[0]) for collection in figure.axes[2].collections} == {
        (0.2, 0.2549019607843137, 0.3333333333333333, 1.0),
        (0.1450980392156863, 0.38823529411764707, 0.9215686274509803, 1.0),
    }

    plt.close(figure)


def test_dual_reporter_triptych_design_context_summarizes_identity_columns() -> None:
    df = _dual_reporter_df()
    df["design_id_alias"] = "alias-A"
    df["id"] = "uuid-1"
    df["sequence"] = "A" * 120

    rows = summarize_design_context(
        df,
        primary_col="design_id",
        primary_value="pTest",
        preferred_columns=("design_id_alias", "design_id", "id", "sequence"),
    )

    assert rows[0] == ("design_id", "pTest")
    assert ("design_id_alias", "alias-A") in rows
    assert ("id", "uuid-1") in rows
    sequence_value = dict(rows)["sequence"]
    assert len(sequence_value) < 90
    assert sequence_value.startswith("AAAA")
    assert sequence_value.endswith("AAAA")


def test_dual_reporter_triptych_rejects_missing_channels() -> None:
    df = _dual_reporter_df()
    df = df[df["channel"] != "YFP/CFP"]

    try:
        build_triptych_data(
            df,
            time_col="time",
            treatment_col="treatment",
            growth_channel="OD600",
            ratio_channel="YFP/CFP",
            snapshot_channel="YFP/CFP",
            snapshot_time=1.0,
            treatment_order=["water", "EtOH"],
        )
    except ValueError as exc:
        assert "YFP/CFP" in str(exc)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("missing ratio channel should fail fast")
