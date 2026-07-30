from __future__ import annotations

from inspect import signature

import pandas as pd

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
