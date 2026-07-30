from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import polars as pl

from ...ordering import order_levels
from .._data import alias_column, require_columns, warn_if_empty
from ..grouping import GroupMatch, resolve_groups
from ..panels.snapshot_data import select_snapshot_rows, summarize_snapshot_values


@dataclass(frozen=True)
class GroupFrame:
    label: str
    members: tuple[str | None, ...]
    snapshot: pd.DataFrame
    time_series: pd.DataFrame


@dataclass(frozen=True)
class CompositeInputs:
    ts_x_col: str
    group_col: str | None
    ts_hue_col: str
    ts_style_col: str | None
    snap_x_col: str
    snap_hue_col: str | None
    ts_channel: str
    snap_channel: str
    groups: tuple[GroupFrame, ...]


@dataclass(frozen=True)
class TimeSeriesPanelData:
    frame: pd.DataFrame
    sheet_lines: list[float] | None
    segment_col: str | None


@dataclass(frozen=True)
class SnapshotPanelData:
    frame: pd.DataFrame
    stats: pd.DataFrame
    time_used: float
    x_order: list[str]
    hue_order: list[str]
    fell_back: bool
    fallback_times_preview: str
    fallback_delta: float


def resolve_level_order(*, observed: list[str], configured: list[str] | None, name: str) -> list[str]:
    ordered_observed = order_levels([str(value) for value in observed])
    if configured is None:
        return ordered_observed
    resolved = [str(value) for value in configured]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"ts_and_snap: {name} contains duplicate labels")
    observed_set = set(ordered_observed)
    missing = [value for value in resolved if value not in observed_set]
    omitted = [value for value in ordered_observed if value not in resolved]
    if missing:
        raise ValueError(f"ts_and_snap: {name} includes missing label(s): {missing}")
    if omitted:
        raise ValueError(f"ts_and_snap: {name} omits observed label(s): {omitted}")
    return resolved


def prepare_composite_inputs(
    *,
    df: pd.DataFrame,
    group_on: str | None,
    pool_sets: list[dict[str, list[str]]] | None,
    pool_match: GroupMatch,
    ts_x: str,
    ts_channel: str,
    ts_hue: str,
    ts_style: str | None,
    ts_time_window: list[float] | None,
    snap_x: str,
    snap_channel: str | None,
    snap_hue: str | None,
) -> CompositeInputs | None:
    """Normalize and partition one composite request without creating a figure."""

    ts_x_col = alias_column(df, ts_x)
    group_col = alias_column(df, group_on) if group_on else None
    ts_hue_col = alias_column(df, ts_hue)
    ts_style_col = alias_column(df, ts_style) if ts_style else None
    snap_x_col = alias_column(df, snap_x)
    snap_hue_col = alias_column(df, snap_hue) if snap_hue else None
    resolved_ts_channel = str(ts_channel)
    resolved_snap_channel = str(snap_channel if snap_channel else ts_channel)

    required = ["time", "channel", "value", "position", ts_x_col, ts_hue_col, snap_x_col]
    if group_col:
        required.append(group_col)
    if snap_hue_col:
        required.append(snap_hue_col)
    if ts_style_col:
        required.append(ts_style_col)
    require_columns(df, required, where="ts_and_snap")

    work_pl = pl.from_pandas(df)
    value_expr = pl.col("value").cast(pl.Float64, strict=False)
    value_expr = pl.when(value_expr.is_nan()).then(None).otherwise(value_expr)
    time_expr = pl.col("time").cast(pl.Float64, strict=False)
    time_expr = pl.when(time_expr.is_nan()).then(None).otherwise(time_expr)
    work_pl = work_pl.with_columns(value_expr.alias("value"), time_expr.alias("time"))
    work = work_pl.to_pandas(use_pyarrow_extension_array=False)
    if warn_if_empty(work, where="ts_and_snap", detail="after numeric normalization"):
        return None

    ts_work_pl = work_pl
    if ts_time_window:
        lo, hi = float(ts_time_window[0]), float(ts_time_window[1])
        ts_x_num = pl.col(ts_x_col).cast(pl.Float64, strict=False)
        ts_work_pl = ts_work_pl.filter((ts_x_num >= lo) & (ts_x_num <= hi))
    ts_work = ts_work_pl.to_pandas(use_pyarrow_extension_array=False)
    if warn_if_empty(ts_work, where="ts_and_snap", detail="after time-series window filter"):
        return None

    available_ts_channels = sorted(ts_work["channel"].astype(str).unique().tolist())
    available_snap_channels = sorted(work["channel"].astype(str).unique().tolist())
    if resolved_ts_channel not in available_ts_channels:
        raise ValueError(
            f"ts_and_snap: ts_channel {resolved_ts_channel!r} not in data. Available: {available_ts_channels}"
        )
    if resolved_snap_channel not in available_snap_channels:
        raise ValueError(
            f"ts_and_snap: snap_channel {resolved_snap_channel!r} not in data. Available: {available_snap_channels}"
        )
    ts_work = ts_work[ts_work["channel"].astype(str) == resolved_ts_channel].copy()

    if group_col:
        universe = order_levels(work[group_col].astype(str).unique().tolist())
        figure_groups = (
            resolve_groups(universe, pool_sets, match=pool_match)
            if pool_sets
            else [(group, [group]) for group in universe]
        )
    else:
        figure_groups = [("all", [None])]

    groups: list[GroupFrame] = []
    for label, members in figure_groups:
        snapshot_frame = work.copy()
        time_series_frame = ts_work.copy()
        if group_col and members != [None]:
            snapshot_frame = snapshot_frame[snapshot_frame[group_col].astype(str).isin(members)]
            time_series_frame = time_series_frame[time_series_frame[group_col].astype(str).isin(members)]
        if snapshot_frame.empty:
            continue
        groups.append(
            GroupFrame(
                label=str(label),
                members=tuple(members),
                snapshot=snapshot_frame,
                time_series=time_series_frame,
            )
        )

    if not groups:
        return None
    return CompositeInputs(
        ts_x_col=ts_x_col,
        group_col=group_col,
        ts_hue_col=ts_hue_col,
        ts_style_col=ts_style_col,
        snap_x_col=snap_x_col,
        snap_hue_col=snap_hue_col,
        ts_channel=resolved_ts_channel,
        snap_channel=resolved_snap_channel,
        groups=tuple(groups),
    )


def prepare_time_series_panel_data(
    *,
    frame: pd.DataFrame,
    channel: str,
    x_col: str,
    add_sheet_lines: bool,
) -> TimeSeriesPanelData:
    selected = frame[frame["channel"].astype(str) == channel].copy()
    sheet_lines = None
    if add_sheet_lines and "sheet_index" in selected.columns:
        starts = sorted(selected.groupby("sheet_index")[x_col].min().dropna().tolist())
        sheet_lines = starts[1:] if len(starts) > 1 else []

    segment_col = None
    segment_parts = (
        ["acquisition_segment_id"]
        if "acquisition_segment_id" in selected.columns
        else [
            column for column in ("plate_id", "source_file", "sheet_name", "sheet_index") if column in selected.columns
        ]
    )
    if segment_parts:
        segment_col = "__plot_segment"
        segments = selected[segment_parts].copy()
        for column in segment_parts:
            segments[column] = segments[column].astype(str)
        selected[segment_col] = segments.agg("::".join, axis=1)
    return TimeSeriesPanelData(frame=selected, sheet_lines=sheet_lines, segment_col=segment_col)


def prepare_snapshot_panel_data(
    *,
    frame: pd.DataFrame,
    group_col: str | None,
    snap_x_col: str,
    snap_hue_col: str | None,
    snap_channel: str,
    snap_time: float,
    snap_time_tolerance: float,
    snap_dispersion: str,
    order_x: list[str] | None,
    order_snap_hue: list[str] | None,
    order_hue: list[str] | None,
    ts_hue_col: str,
) -> SnapshotPanelData | None:
    key_cols = [column for column in [group_col, snap_x_col, snap_hue_col, "channel", "position"] if column]
    selection = select_snapshot_rows(
        df=frame,
        target_time=float(snap_time),
        keys=key_cols,
        channel=snap_channel,
        tolerance=float(snap_time_tolerance),
    )
    if selection.rows.empty:
        return None

    group_cols = [snap_x_col] + ([snap_hue_col] if snap_hue_col else [])
    stats = summarize_snapshot_values(
        df=selection.rows,
        group_cols=group_cols,
        dispersion=snap_dispersion,
    )
    hue_order = (
        resolve_level_order(
            observed=stats[snap_hue_col].astype(str).unique().tolist(),
            configured=(
                order_snap_hue if order_snap_hue is not None else (order_hue if snap_hue_col == ts_hue_col else None)
            ),
            name=("order_snap_hue" if order_snap_hue is not None or snap_hue_col != ts_hue_col else "order_hue"),
        )
        if snap_hue_col
        else ["_single"]
    )
    x_order = resolve_level_order(
        observed=stats[snap_x_col].astype(str).unique().tolist(),
        configured=order_x,
        name="order_x",
    )
    return SnapshotPanelData(
        frame=selection.rows,
        stats=stats,
        time_used=selection.time_used,
        x_order=x_order,
        hue_order=hue_order,
        fell_back=selection.fell_back,
        fallback_times_preview=selection.fallback_times_preview or "",
        fallback_delta=float(selection.fallback_delta or 0.0),
    )


def resolve_paired_hue_levels(
    *, groups: tuple[GroupFrame, ...], hue_col: str, configured: list[str] | None
) -> list[str]:
    domains = [set(group.time_series[hue_col].astype(str).unique().tolist()) for group in groups]
    if any(domain != domains[0] for domain in domains[1:]):
        raise ValueError(
            "ts_and_snap: group_layout='paired_row' requires identical ts_hue levels in every group "
            "so one legend and color map remain truthful"
        )
    return resolve_level_order(
        observed=groups[0].time_series[hue_col].astype(str).unique().tolist(),
        configured=configured,
        name="order_hue",
    )
