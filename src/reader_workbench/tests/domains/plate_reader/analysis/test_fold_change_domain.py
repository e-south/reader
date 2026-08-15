from __future__ import annotations

from inspect import signature

import pandas as pd
import pytest

from reader_workbench.domains.plate_reader.analysis.fold_change import (
    FoldChangeAnalysisSpec,
    compute_fold_change_table,
)


def test_fold_change_domain_uses_an_explicit_analysis_spec() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2"],
            "time": [8.0, 8.0],
            "channel": ["signal", "signal"],
            "value": [2.0, 4.0],
            "design_id": ["design_a", "design_a"],
            "treatment": ["baseline", "induced"],
        }
    )
    spec = FoldChangeAnalysisSpec(
        target="signal",
        report_times=(8.0,),
        group_by=("design_id",),
        use_global_baseline=True,
        global_baseline_value="baseline",
        attach_metadata=(),
    )

    table = compute_fold_change_table(frame, spec=spec)

    assert table.sort_values("treatment")["FC"].tolist() == [1.0, 2.0]
    assert "ctx" not in signature(compute_fold_change_table).parameters
    assert "cfg" not in signature(compute_fold_change_table).parameters


def test_fold_change_attaches_metadata_when_only_alias_treatment_is_present() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2"],
            "time": [8.0, 8.0],
            "channel": ["signal", "signal"],
            "value": [2.0, 4.0],
            "design_id": ["design_a", "design_a"],
            "treatment_alias": ["baseline", "induced"],
            "batch": [7, 7],
        }
    )
    spec = FoldChangeAnalysisSpec(
        target="signal",
        report_times=(8.0,),
        group_by=("design_id",),
        use_global_baseline=True,
        global_baseline_value="baseline",
        attach_metadata=("batch",),
    )

    table = compute_fold_change_table(frame, spec=spec).sort_values("treatment")

    assert table["batch"].tolist() == [7, 7]


def test_fold_change_rejects_rows_without_complete_group_identity() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2", "H9", "H10"],
            "time": [8.0, 8.0, 8.0, 8.0],
            "channel": ["signal", "signal", "signal", "signal"],
            "value": [2.0, 4.0, 10.0, 20.0],
            "design_id": ["design_a", "design_a", None, "  "],
            "treatment": ["baseline", "induced", None, None],
        }
    )
    spec = FoldChangeAnalysisSpec(
        target="signal",
        report_times=(8.0,),
        group_by=("design_id",),
        use_global_baseline=True,
        global_baseline_value="baseline",
        attach_metadata=(),
    )

    with pytest.raises(ValueError, match="requires complete analytical identity.*2 row"):
        compute_fold_change_table(frame, spec=spec)


def test_fold_change_rejects_an_entirely_anonymous_grouping_cohort() -> None:
    frame = pd.DataFrame(
        {
            "position": ["H9", "H10"],
            "time": [8.0, 8.0],
            "channel": ["signal", "signal"],
            "value": [10.0, 20.0],
            "design_id": [None, "nan"],
            "treatment": [None, None],
        }
    )
    spec = FoldChangeAnalysisSpec(target="signal", report_times=(8.0,), attach_metadata=())

    with pytest.raises(ValueError, match="requires complete analytical identity"):
        compute_fold_change_table(frame, spec=spec)


def test_fold_change_rejects_censored_values_instead_of_reporting_exact_fc() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2", "A3"],
            "time": [8.0, 8.0, 8.0],
            "channel": ["signal", "signal", "signal"],
            "value": [2.0, 100.0, 8.0],
            "value_policy_clipped": [False, True, False],
            "value_instrument_overflow": [False, False, False],
            "value_bound_kind": ["exact", "lower", "exact"],
            "design_id": ["design_a", "design_a", "design_a"],
            "treatment": ["baseline", "baseline", "induced"],
        }
    )
    spec = FoldChangeAnalysisSpec(
        target="signal",
        report_times=(8.0,),
        use_global_baseline=True,
        global_baseline_value="baseline",
        attach_metadata=(),
    )

    with pytest.raises(ValueError, match="requires exact finite values"):
        compute_fold_change_table(frame, spec=spec)


def test_fold_change_rejects_incomplete_declared_treatment_cohort() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A2", "B1"],
            "time": [8.0, 8.0, 8.0],
            "channel": ["signal", "signal", "signal"],
            "value": [2.0, 8.0, 4.0],
            "design_id": ["complete", "complete", "incomplete"],
            "treatment": ["baseline", "induced", "baseline"],
        }
    )
    spec = FoldChangeAnalysisSpec(
        target="signal",
        report_times=(8.0,),
        expected_treatments=("baseline", "induced"),
        use_global_baseline=True,
        global_baseline_value="baseline",
        attach_metadata=(),
    )

    with pytest.raises(ValueError, match="incomplete treatment cohort.*incomplete.*induced"):
        compute_fold_change_table(frame, spec=spec)
