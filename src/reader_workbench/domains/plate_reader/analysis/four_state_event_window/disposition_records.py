"""Materialize explicit well and design eligibility dispositions."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .design_dispositions import DesignDisposition
from .well_exclusions import WellExclusion

_COLUMNS = (
    "experiment_id",
    "design_id",
    "scope",
    "state",
    "position",
    "reduction_id",
    "reason",
    "observed_count",
    "required_count",
)


def partition_eligible_wells(
    wells: pd.DataFrame,
    *,
    dispositions: Sequence[DesignDisposition],
    experiment_id: str,
    reduction_id: str,
    reference_design_id: str,
    available_design_ids: set[str],
    required_count: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selected = [
        item for item in dispositions if item.experiment_id == experiment_id and item.reduction_id == reduction_id
    ]
    disposed_ids: set[str] = set()
    records: list[dict[str, object]] = []
    for item in selected:
        if item.design_id == reference_design_id:
            raise ValueError(f"{experiment_id}:{reduction_id} cannot dispose the reference design.")
        if item.design_id not in available_design_ids:
            raise ValueError(f"{experiment_id}:{reduction_id} disposition names absent design {item.design_id!r}.")
        observed_count = int(
            wells.loc[
                wells["design_id"].astype(str).eq(item.design_id) & wells["state"].astype(str).eq(item.state)
            ].shape[0]
        )
        if observed_count >= required_count:
            raise ValueError(
                f"{experiment_id}:{item.design_id}:{item.state}:{reduction_id} has {observed_count} observations; "
                f"at least {required_count} are required, so the design disposition is not justified."
            )
        disposed_ids.add(item.design_id)
        records.append(
            _record(
                experiment_id=item.experiment_id,
                design_id=item.design_id,
                scope="design",
                state=item.state,
                position=None,
                reduction_id=item.reduction_id,
                reason=item.reason,
                observed_count=observed_count,
                required_count=required_count,
            )
        )
    eligible = wells.loc[~wells["design_id"].astype(str).isin(disposed_ids)].copy()
    return eligible, records


def partition_reduction_wells(
    midpoint: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    *,
    exclusions: Sequence[WellExclusion],
    dispositions: Sequence[DesignDisposition],
    experiment_id: str,
    reduction_id: str,
    reference_design_id: str,
    available_design_ids: set[str],
    required_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    eligible, records = partition_eligible_wells(
        midpoint,
        dispositions=dispositions,
        experiment_id=experiment_id,
        reduction_id=reduction_id,
        reference_design_id=reference_design_id,
        available_design_ids=available_design_ids,
        required_count=required_count,
    )
    eligible_ids = set(eligible["design_id"].astype(str))
    eligible_lower = lower.loc[lower["design_id"].astype(str).isin(eligible_ids)].copy()
    eligible_upper = upper.loc[upper["design_id"].astype(str).isin(eligible_ids)].copy()
    records[:0] = well_disposition_records(
        exclusions,
        experiment_id=experiment_id,
        reduction_id=reduction_id,
    )
    return eligible, eligible_lower, eligible_upper, records


def well_disposition_records(
    exclusions: Sequence[WellExclusion],
    *,
    experiment_id: str,
    reduction_id: str,
) -> list[dict[str, object]]:
    return [
        _record(
            experiment_id=item.experiment_id,
            design_id=item.design_id,
            scope="well",
            state=item.state,
            position=item.position,
            reduction_id=item.reduction_id,
            reason=item.reason,
            observed_count=None,
            required_count=None,
        )
        for item in exclusions
        if item.experiment_id == experiment_id and item.reduction_id == reduction_id
    ]


def disposition_frame(records: Sequence[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=_COLUMNS)
    frame["observed_count"] = frame["observed_count"].astype("Int64")
    frame["required_count"] = frame["required_count"].astype("Int64")
    return frame


def _record(**values: object) -> dict[str, object]:
    return {column: values[column] for column in _COLUMNS}


__all__ = ["disposition_frame", "partition_reduction_wells"]
