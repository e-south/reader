from __future__ import annotations

import pandas as pd
import pytest

from reader.protocols import ProtocolBinding, builtin_protocol_catalog
from reader.workbench.notebooks.dual_reporter_triptych import (
    build_dual_reporter_triptych_chart,
    build_triptych_data,
    summarize_design_context,
)
from reader.workbench.templates import (
    compatible_notebook_templates,
    resolve_notebook_template_descriptor,
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
    )
    spec = chart.to_dict()

    assert spec["spacing"] == 16
    assert [panel["width"] for panel in spec["hconcat"]] == [260, 260, 260]
    assert [panel["height"] for panel in spec["hconcat"]] == [260, 260, 260]
    assert spec["hconcat"][2]["layer"][0]["encoding"]["x"]["scale"]["domain"] == ["water", "EtOH", "AND"]


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


def test_dual_reporter_triptych_template_is_protocol_neutral() -> None:
    descriptor = resolve_notebook_template_descriptor("notebook/dual_reporter_triptych")

    assert descriptor.domain == "plate_reader"
    assert "sfxi" not in descriptor.tags
    body = descriptor.load_body()
    assert "Dual-reporter triptych" in body
    assert "debounce=True" in body
    assert "chart_selection=False" in body
    assert "mo.output.replace(_chart_panel)" in body
    assert "Selected design" in body
    assert "Triptych context" not in body
    assert "summarize_design_context" in body
    assert "Export 8-vector" not in body


def test_dual_reporter_screen_allows_triptych_without_sfxi_vec8() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="plate_reader/dual_reporter_screen"))
    templates = [item.template for item in compatible_notebook_templates(protocol=protocol)]

    assert "notebook/dual_reporter_triptych" in templates
    assert "notebook/sfxi_eda" not in templates
