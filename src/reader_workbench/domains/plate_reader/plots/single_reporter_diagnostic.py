"""Study-neutral preparation for the single-reporter diagnostic figure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from reader_workbench.domains.plate_reader.ordering import order_levels
from reader_workbench.domains.time_series import (
    EndpointSelection,
    IntervalSelection,
    ObservationAggregationSpec,
    TemporalReductionSpec,
    reduce_temporal_trace,
)

from ._data import alias_column, require_columns
from .grouping import GroupMatch, resolve_groups

SummaryStat = Literal["mean", "median"]
UnitRole = Literal["declared_replicate", "observation_only"]
_PROVENANCE_COLUMNS = (
    "value_policy_clipped",
    "value_instrument_overflow",
    "value_bound_kind",
)


@dataclass(frozen=True)
class SingleReporterSelection:
    temporal_reduction: TemporalReductionSpec

    @property
    def endpoint_time_h(self) -> float | None:
        selection = self.temporal_reduction.selection
        return selection.time_h if isinstance(selection, EndpointSelection) else None

    @property
    def window_h(self) -> tuple[float, float] | None:
        selection = self.temporal_reduction.selection
        return (selection.start_h, selection.end_h) if isinstance(selection, IntervalSelection) else None

    @property
    def label(self) -> str:
        selection = self.temporal_reduction.selection
        method = {
            "identity": "endpoint",
            "observed_mean": "observed mean",
            "observed_median": "observed median",
            "geometric_time_mean": "time-weighted geometric mean",
            "integrated_linear_mean": "time-weighted linear mean",
        }[self.temporal_reduction.method]
        suffix = "" if self.temporal_reduction.output_space == "linear" else " (log2 output)"
        if isinstance(selection, EndpointSelection):
            return f"{method} at {selection.time_h:g} h{suffix}"
        return f"{method} over {selection.start_h:g}–{selection.end_h:g} h{suffix}"


@dataclass(frozen=True)
class SingleReporterDiagnosticData:
    group_label: str
    condition_column: str
    entity_columns: tuple[str, ...]
    unit_column: str
    unit_role: UnitRole
    time_column: str
    normalizer_channel: str
    reporter_channel: str
    ratio_channel: str
    condition_order: tuple[str, ...]
    kinetics: pd.DataFrame
    reduced_ratio: pd.DataFrame
    reduced_normalizer: pd.DataFrame
    selection: SingleReporterSelection
    observation_aggregation: ObservationAggregationSpec


def prepare_single_reporter_diagnostics(
    frame: pd.DataFrame,
    *,
    group_on: str | None,
    collection_items: list[dict[str, list[str]]] | None,
    group_match: GroupMatch,
    condition_column: str,
    condition_order: list[str] | None,
    entity_columns: list[str],
    unit_column: str,
    observation_column: str,
    unit_role: UnitRole,
    time_column: str,
    normalizer_channel: str,
    reporter_channel: str,
    ratio_channel: str,
    temporal_reduction: TemporalReductionSpec,
    observation_aggregation: ObservationAggregationSpec,
) -> tuple[SingleReporterDiagnosticData, ...]:
    """Prepare one diagnostic contract per resolved presentation partition."""

    if temporal_reduction.selection.time_basis != "absolute":
        raise ValueError("single_reporter_diagnostic: acquisition traces require an absolute temporal reduction")
    if unit_role not in {"declared_replicate", "observation_only"}:
        raise ValueError(f"single_reporter_diagnostic: unsupported unit_role {unit_role!r}")
    resolved_condition = str(condition_column)
    resolved_entities = tuple(str(column).strip() for column in entity_columns)
    if not resolved_entities or any(not column for column in resolved_entities):
        raise ValueError("single_reporter_diagnostic: entity_columns must contain non-empty strings")
    if len(set(resolved_entities)) != len(resolved_entities):
        raise ValueError("single_reporter_diagnostic: entity_columns must not contain duplicates")
    if resolved_condition in resolved_entities:
        raise ValueError("single_reporter_diagnostic: condition_column must not be repeated in entity_columns")
    if unit_column == resolved_condition or unit_column in resolved_entities:
        raise ValueError(
            "single_reporter_diagnostic: unit_column must be distinct from condition_column and entity_columns"
        )
    resolved_group = str(alias_column(frame, group_on)) if group_on else None
    required = [
        time_column,
        "channel",
        "value",
        resolved_condition,
        *resolved_entities,
        unit_column,
        observation_column,
        *([resolved_group] if resolved_group else []),
    ]
    if temporal_reduction.support.censored_values == "reject":
        required.extend(_PROVENANCE_COLUMNS)
    require_columns(frame, required, where="single_reporter_diagnostic")

    work = frame.copy()
    work[time_column] = pd.to_numeric(work[time_column], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    channels = (normalizer_channel, reporter_channel, ratio_channel)
    work = work[work["channel"].astype(str).isin(channels)].copy()
    if work.empty:
        raise ValueError("single_reporter_diagnostic: no rows for the compiler-owned channels")
    finite = np.isfinite(work[time_column].to_numpy(dtype=float)) & np.isfinite(work["value"].to_numpy(dtype=float))
    if not finite.all():
        raise ValueError(
            "single_reporter_diagnostic: compiler-owned channel rows contain "
            f"{int((~finite).sum())} non-finite time or value field(s)"
        )

    for column in (resolved_condition, *resolved_entities, unit_column, observation_column, resolved_group):
        if column is not None:
            _require_nonempty_identity(work, column=column)
    _require_channels(work, channels=channels)

    if resolved_group is None:
        groups = [("all", [None])]
    else:
        universe = order_levels(work[resolved_group].astype(str).unique().tolist())
        groups = (
            resolve_groups(universe, collection_items, match=group_match)
            if collection_items
            else [(value, [value]) for value in universe]
        )

    diagnostics: list[SingleReporterDiagnosticData] = []
    for label, members in groups:
        if resolved_group is not None and not members:
            raise ValueError(f"single_reporter_diagnostic: partition {label!r} selects no rows")
        selected = work
        if resolved_group is not None and members != [None]:
            selected = work[work[resolved_group].astype(str).isin(members)].copy()
        if selected.empty:
            raise ValueError(f"single_reporter_diagnostic: partition {label!r} selects no rows")
        entity_count = len(selected.loc[:, list(resolved_entities)].drop_duplicates())
        if entity_count != 1:
            raise ValueError(
                "single_reporter_diagnostic: "
                f"partition {label!r} spans multiple identity_scope entities ({entity_count}); "
                f"partition by {list(resolved_entities)!r} or use an explicit comparison figure"
            )

        diagnostics.append(
            _prepare_partition(
                selected,
                group_label=str(label),
                condition_column=resolved_condition,
                condition_order=_resolve_condition_order(
                    selected,
                    condition_column=resolved_condition,
                    configured=condition_order,
                ),
                entity_columns=resolved_entities,
                unit_column=unit_column,
                observation_column=observation_column,
                unit_role=unit_role,
                time_column=time_column,
                normalizer_channel=normalizer_channel,
                reporter_channel=reporter_channel,
                ratio_channel=ratio_channel,
                temporal_reduction=temporal_reduction,
                observation_aggregation=observation_aggregation,
            )
        )

    if not diagnostics:
        raise ValueError("single_reporter_diagnostic: no diagnostic partitions were prepared")
    return tuple(diagnostics)


def _prepare_partition(
    frame: pd.DataFrame,
    *,
    group_label: str,
    condition_column: str,
    condition_order: list[str],
    entity_columns: tuple[str, ...],
    unit_column: str,
    observation_column: str,
    unit_role: UnitRole,
    time_column: str,
    normalizer_channel: str,
    reporter_channel: str,
    ratio_channel: str,
    temporal_reduction: TemporalReductionSpec,
    observation_aggregation: ObservationAggregationSpec,
) -> SingleReporterDiagnosticData:
    work = frame.copy()
    work["__condition"] = work[condition_column].astype(str)
    work["__unit"] = work[unit_column].astype(str)
    work["__observation"] = work[observation_column].astype(str)
    work["__segment"] = _segment_identity(work)

    semantic_unit_keys = ["__condition", *entity_columns, "__unit"]
    trace_keys = [*semantic_unit_keys, "__observation", "channel"]
    duplicate_keys = [*trace_keys, time_column]
    if work.duplicated(subset=duplicate_keys).any():
        raise ValueError(f"single_reporter_diagnostic: partition {group_label!r} contains duplicate trace rows")
    _require_aligned_channel_times(
        work,
        trace_identity=[*semantic_unit_keys, "__observation"],
        time_column=time_column,
        channels=(normalizer_channel, reporter_channel, ratio_channel),
        where=f"partition {group_label!r}",
    )

    within_unit_stat = observation_aggregation.within_unit_statistic
    kinetics = _reduce_rows(
        work,
        group_columns=["__segment", time_column, *semantic_unit_keys, "channel"],
        statistic=within_unit_stat,
    )
    _require_complete_channels(
        kinetics,
        key_columns=["__segment", time_column, *semantic_unit_keys],
        channels=(normalizer_channel, reporter_channel, ratio_channel),
        where=f"partition {group_label!r} acquisition rows",
    )

    observation_reductions = _reduce_observation_traces(
        work,
        trace_keys=trace_keys,
        time_column=time_column,
        temporal_reduction=temporal_reduction,
        group_label=group_label,
    )
    reduced = _reduce_rows(
        observation_reductions,
        group_columns=[*semantic_unit_keys, "channel"],
        statistic=within_unit_stat,
    )
    _require_complete_channels(
        reduced,
        key_columns=semantic_unit_keys,
        channels=(normalizer_channel, ratio_channel),
        where=f"partition {group_label!r} reduction",
    )

    return SingleReporterDiagnosticData(
        group_label=group_label,
        condition_column=condition_column,
        entity_columns=entity_columns,
        unit_column=unit_column,
        unit_role=unit_role,
        time_column=time_column,
        normalizer_channel=normalizer_channel,
        reporter_channel=reporter_channel,
        ratio_channel=ratio_channel,
        condition_order=tuple(condition_order),
        kinetics=kinetics.reset_index(drop=True),
        reduced_ratio=reduced[reduced["channel"].astype(str) == ratio_channel].reset_index(drop=True),
        reduced_normalizer=reduced[reduced["channel"].astype(str) == normalizer_channel].reset_index(drop=True),
        selection=SingleReporterSelection(temporal_reduction=temporal_reduction),
        observation_aggregation=observation_aggregation,
    )


def _reduce_observation_traces(
    frame: pd.DataFrame,
    *,
    trace_keys: list[str],
    time_column: str,
    temporal_reduction: TemporalReductionSpec,
    group_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for identity, trace in frame.groupby(trace_keys, sort=True, dropna=False):
        identity_values = identity if isinstance(identity, tuple) else (identity,)
        trace_id = f"single_reporter_diagnostic:{group_label}:" + ":".join(map(str, identity_values))
        provenance = {
            "policy_clipped": (
                trace["value_policy_clipped"].to_numpy(dtype=bool) if "value_policy_clipped" in trace.columns else None
            ),
            "instrument_overflow": (
                trace["value_instrument_overflow"].to_numpy(dtype=bool)
                if "value_instrument_overflow" in trace.columns
                else None
            ),
            "bound_kinds": (
                trace["value_bound_kind"].to_numpy(dtype=object) if "value_bound_kind" in trace.columns else None
            ),
        }
        result = reduce_temporal_trace(
            trace[time_column].to_numpy(dtype=float),
            trace["value"].to_numpy(dtype=float),
            spec=temporal_reduction,
            trace_id=trace_id,
            **provenance,
        )
        row = dict(zip(trace_keys, identity_values, strict=True))
        row["value"] = result.value
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _reduce_rows(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    statistic: SummaryStat,
) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False, sort=True)["value"]
    values = grouped.mean() if statistic == "mean" else grouped.median()
    return values.rename("value").reset_index()


def _segment_identity(frame: pd.DataFrame) -> pd.Series:
    if "acquisition_segment_id" in frame.columns:
        return frame["acquisition_segment_id"].astype(str)
    columns = [column for column in ("source", "sheet_name", "sheet_index") if column in frame.columns]
    if not columns:
        return pd.Series("segment", index=frame.index, dtype="string")
    parts = frame[columns].copy()
    for column in columns:
        parts[column] = parts[column].astype(str)
    return parts.agg("::".join, axis=1)


def _resolve_condition_order(
    frame: pd.DataFrame,
    *,
    condition_column: str,
    configured: list[str] | None,
) -> list[str]:
    observed = order_levels(frame[condition_column].astype(str).unique().tolist())
    if configured is None:
        return observed
    order = [str(value).strip() for value in configured]
    if not order or any(not value for value in order):
        raise ValueError("single_reporter_diagnostic: condition_order must contain non-empty labels")
    if len(set(order)) != len(order):
        raise ValueError("single_reporter_diagnostic: condition_order contains duplicate labels")
    missing = [value for value in order if value not in observed]
    omitted = [value for value in observed if value not in order]
    if missing or omitted:
        raise ValueError(
            "single_reporter_diagnostic: condition_order must exactly match observed conditions "
            f"(missing={missing}, omitted={omitted})"
        )
    return order


def _require_nonempty_identity(frame: pd.DataFrame, *, column: str) -> None:
    values = frame[column]
    invalid = values.isna() | values.astype(str).str.strip().str.casefold().isin({"", "nan", "none"})
    if invalid.any():
        raise ValueError(f"single_reporter_diagnostic: column {column!r} contains missing identities")


def _require_channels(frame: pd.DataFrame, *, channels: tuple[str, ...]) -> None:
    observed = set(frame["channel"].astype(str).unique().tolist())
    missing = [channel for channel in channels if channel not in observed]
    if missing:
        raise ValueError(
            f"single_reporter_diagnostic: compiler-owned channel(s) missing: {missing}; available={sorted(observed)}"
        )


def _require_aligned_channel_times(
    frame: pd.DataFrame,
    *,
    trace_identity: list[str],
    time_column: str,
    channels: tuple[str, ...],
    where: str,
) -> None:
    for identity, group in frame.groupby(trace_identity, sort=True, dropna=False):
        times = {
            channel: tuple(
                sorted(group.loc[group["channel"].astype(str) == channel, time_column].to_numpy(dtype=float))
            )
            for channel in channels
        }
        if any(not values for values in times.values()) or len(set(times.values())) != 1:
            raise ValueError(
                f"single_reporter_diagnostic: {where} trace {identity!r} lacks exactly aligned channel times"
            )


def _require_complete_channels(
    frame: pd.DataFrame,
    *,
    key_columns: list[str],
    channels: tuple[str, ...],
    where: str,
) -> None:
    expected = set(channels)
    observed = frame.groupby(key_columns, dropna=False)["channel"].agg(lambda values: set(map(str, values)))
    incomplete = observed[~observed.map(expected.issubset)]
    if not incomplete.empty:
        example = incomplete.index[0]
        raise ValueError(f"single_reporter_diagnostic: {where} lacks paired channel observations at {example!r}")


__all__ = [
    "SingleReporterDiagnosticData",
    "SingleReporterSelection",
    "prepare_single_reporter_diagnostics",
]
