"""Common response-summary options for cross-experiment review."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

_IDENTITY_COLUMNS = {"experiment_id", "design_id", "reduction_id"}
_DEFINITION_COLUMNS = (
    "window_start_event_h",
    "window_end_event_h",
    "reduction_method",
    "response_basis",
    "reduction_role",
)


def common_cross_experiment_reductions(
    designs: pd.DataFrame,
    *,
    design_id: str,
    experiment_ids: Sequence[str],
) -> pd.DataFrame:
    """Keep only identically defined reductions available in every experiment."""

    required = _IDENTITY_COLUMNS | set(_DEFINITION_COLUMNS)
    missing = sorted(required - set(designs.columns))
    if missing:
        raise ValueError(f"cross-experiment reduction options are missing columns: {missing}.")
    exact_design_id = _required_text(design_id, field="Reader design")
    exact_experiments = tuple(_required_text(value, field="experiment_id") for value in experiment_ids)
    if len(exact_experiments) < 2 or len(set(exact_experiments)) != len(exact_experiments):
        raise ValueError("cross-experiment reduction options require at least two unique experiments.")
    selected = designs.loc[
        designs["design_id"].astype(str).eq(exact_design_id)
        & designs["experiment_id"].astype(str).isin(exact_experiments)
    ].copy()
    observed = set(selected["experiment_id"].astype(str))
    if observed != set(exact_experiments):
        raise ValueError("cross-experiment reduction options require the Reader design in every selected experiment.")
    if selected.duplicated(["experiment_id", "reduction_id"]).any():
        raise ValueError("cross-experiment reduction options require one row per experiment and reduction ID.")
    complete_ids = {
        str(reduction_id)
        for reduction_id, rows in selected.groupby("reduction_id", sort=False)
        if set(rows["experiment_id"].astype(str)) == set(exact_experiments)
    }
    common = selected.loc[selected["reduction_id"].astype(str).isin(complete_ids)].copy()
    if common.empty:
        raise ValueError("selected experiments share no response-window reduction.")
    drifted = [
        str(reduction_id)
        for reduction_id, rows in common.groupby("reduction_id", sort=False)
        if any(rows[column].nunique(dropna=False) != 1 for column in _DEFINITION_COLUMNS)
    ]
    if drifted:
        raise ValueError(f"cross-experiment reductions require shared definitions; drifted IDs: {drifted}.")
    return common.reset_index(drop=True)


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be an exact non-empty string.")
    return value


__all__ = ["common_cross_experiment_reductions"]
