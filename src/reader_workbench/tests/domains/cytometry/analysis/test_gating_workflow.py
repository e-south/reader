from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from reader_workbench.contracts import builtin_contract_catalog, validate_df
from reader_workbench.domains.cytometry.analysis import CytometryAnalysisError, GateSpec, ThresholdSpec
from reader_workbench.domains.cytometry.analysis.workflow import (
    CytometryGatingRequest,
    CytometryQCSpec,
    run_cytometry_gating,
)

CHANNELS = ("FSC-A", "SSC-A", "FSC-H", "reporter")


def tidy_events() -> pd.DataFrame:
    samples = {
        "control": ("negative", ((10.0, 10.0, 10.0, 1.0), (20.0, 20.0, 20.0, 2.0), (200.0, 20.0, 200.0, 3.0))),
        "treated": ("induced", ((10.0, 10.0, 10.0, 10.0), (20.0, 20.0, 5.0, 20.0), (30.0, 30.0, 30.0, 30.0))),
    }
    rows: list[dict[str, object]] = []
    for sample_id, (condition, events) in samples.items():
        for event_index, values in enumerate(events):
            for channel, value in zip(CHANNELS, values, strict=True):
                rows.append(
                    {
                        "position": sample_id,
                        "time": 0.0,
                        "sample_id": sample_id,
                        "event_index": event_index,
                        "condition": condition,
                        "channel": channel,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def request(*, group_column: str | None = "condition") -> CytometryGatingRequest:
    return CytometryGatingRequest(
        gate=GateSpec(
            cells_x_channel="FSC-A",
            cells_y_channel="SSC-A",
            cells_x_range=(0.0, 100.0),
            cells_y_range=(0.0, 100.0),
            singlet_x_channel="FSC-A",
            singlet_y_channel="FSC-H",
            singlet_ratio_range=(0.8, 1.2),
            cells_enabled=True,
            singlets_enabled=True,
        ),
        threshold=ThresholdSpec(
            channel="reporter",
            mode="from_control_quantile",
            group_column="condition",
            control_value="negative",
            quantile=0.5,
        ),
        group_column=group_column,
        qc=CytometryQCSpec(
            minimum_final_events=2,
            minimum_final_percent=50.0,
            maximum_nonpositive_percent=10.0,
            nonpositive_scope="all_events",
        ),
    )


def test_gating_workflow_emits_resolved_normal_lifecycle_tables() -> None:
    result = run_cytometry_gating(tidy_events(), request())

    definition = result.gate_definition.row(0, named=True)
    assert definition["cells_x_channel"] == "FSC-A"
    assert definition["fluorescence_channel"] == "reporter"
    assert definition["threshold_group_column"] == "condition"
    assert definition["group_column"] == "condition"
    assert definition["threshold_value"] == 1.5
    assert definition["nonpositive_scope"] == "all_events"

    assert result.gated_events.select("sample_id", "event_index").rows() == [
        ("control", 0),
        ("control", 1),
        ("treated", 0),
        ("treated", 2),
    ]
    assert result.sample_stats.sort("sample_id").get_column("group_value").to_list() == ["negative", "induced"]
    assert result.group_stats.sort("group_value").get_column("group_value").to_list() == ["induced", "negative"]
    assert result.qc.sort("sample_id").get_column("qc_pass").to_list() == [True, True]
    assert result.qc.get_column("nonpositive_scope").unique().to_list() == ["all_events"]


def test_gating_workflow_does_not_choose_a_group_implicitly() -> None:
    result = run_cytometry_gating(tidy_events(), request(group_column=None))

    assert result.group_stats.is_empty()
    assert result.sample_stats.get_column("group_column").null_count() == 2


def test_gating_workflow_fails_when_explicit_threshold_control_is_absent() -> None:
    configured = request()
    missing_control = CytometryGatingRequest(
        gate=configured.gate,
        threshold=ThresholdSpec(
            channel="reporter",
            mode="from_control_quantile",
            group_column="condition",
            control_value="not-present",
            quantile=0.5,
        ),
        group_column=configured.group_column,
        qc=configured.qc,
    )

    with pytest.raises(CytometryAnalysisError, match="No control events"):
        run_cytometry_gating(tidy_events(), missing_control)


def test_disabled_cells_gate_does_not_require_or_evaluate_cell_axes() -> None:
    configured = request()
    without_cells_gate = replace(
        configured,
        gate=replace(
            configured.gate,
            cells_enabled=False,
            cells_x_channel="absent-cells-x",
            cells_y_channel="absent-cells-y",
            cells_x_range=(2.0, 1.0),
            cells_y_range=(2.0, 1.0),
        ),
    )

    result = run_cytometry_gating(tidy_events(), without_cells_gate)

    counts = result.sample_stats.sort("sample_id")
    assert counts.get_column("n_cells_gate").to_list() == counts.get_column("n_total_events").to_list()
    assert counts.get_column("n_singlets").to_list() == [3, 2]


def test_disabled_singlets_gate_does_not_require_or_evaluate_ratio_axes() -> None:
    configured = request()
    without_singlets_gate = replace(
        configured,
        gate=replace(
            configured.gate,
            singlets_enabled=False,
            singlet_x_channel="absent-singlet-x",
            singlet_y_channel="absent-singlet-y",
            singlet_ratio_range=(2.0, 1.0),
        ),
    )

    result = run_cytometry_gating(tidy_events(), without_singlets_gate)

    counts = result.sample_stats.sort("sample_id")
    assert counts.get_column("n_singlets").to_list() == counts.get_column("n_cells_gate").to_list()
    assert counts.get_column("n_cells_gate").to_list() == [2, 3]


def test_gated_nonpositive_qc_fails_closed_for_a_sample_with_no_retained_events() -> None:
    events = tidy_events()
    excluded = (events["sample_id"] == "treated") & events["channel"].isin(("FSC-A", "SSC-A"))
    events.loc[excluded, "value"] = 200.0
    configured = request()
    configured = replace(
        configured,
        qc=replace(configured.qc, nonpositive_scope="gated_events"),
    )

    result = run_cytometry_gating(events, configured)

    treated_qc = result.qc.filter(result.qc["sample_id"] == "treated").row(0, named=True)
    assert treated_qc["n_singlets"] == 0
    assert treated_qc["pct_nonpositive"] == 100.0
    assert treated_qc["passes_nonpositive"] is False
    assert treated_qc["qc_pass"] is False
    assert treated_qc["qc_status"] == "fail"
    validate_df(
        result.qc.to_pandas(),
        builtin_contract_catalog().require("cytometry.qc.v1"),
        where="cytometry.qc",
    )


def test_singlet_gate_uses_y_over_x_for_an_asymmetric_ratio() -> None:
    configured = request(group_column=None)
    configured = replace(
        configured,
        gate=replace(configured.gate, singlet_ratio_range=(0.2, 0.3)),
        threshold=ThresholdSpec(channel="reporter", value=0.0, mode="manual"),
    )

    result = run_cytometry_gating(tidy_events(), configured)

    assert result.gated_events.select("sample_id", "event_index").rows() == [("treated", 1)]
