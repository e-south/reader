"""Bounded Polars payload preparation for cytometry plotting backends."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from reader.domains.cytometry.analysis.events import _require_columns


def prepare_plot_events(
    event_table: pl.DataFrame,
    *,
    columns: Sequence[str],
    x_channel: str,
    y_channel: str,
    max_events: int,
    group_columns: Sequence[str] = (),
    low_clip_quantile: float | None = None,
    clip_group_column: str | None = None,
    positive_x: bool = False,
    positive_y: bool = False,
) -> pl.DataFrame:
    """Project, filter, and downsample an event table before plotting conversion."""

    selected_columns = list(dict.fromkeys(columns))
    _require_columns(event_table, (*selected_columns, x_channel, y_channel))
    work = event_table.select(selected_columns)
    x = pl.col(x_channel).cast(pl.Float64, strict=False)
    y = pl.col(y_channel).cast(pl.Float64, strict=False)
    work = work.filter(x.is_finite() & y.is_finite())
    if low_clip_quantile is not None:
        if not 0.0 <= low_clip_quantile < 1.0:
            raise ValueError("low_clip_quantile must be in [0, 1).")
        if clip_group_column is not None:
            _require_columns(work, (clip_group_column,))
            x_low = x.quantile(low_clip_quantile, interpolation="linear").over(clip_group_column)
            y_low = y.quantile(low_clip_quantile, interpolation="linear").over(clip_group_column)
        else:
            x_low = x.quantile(low_clip_quantile, interpolation="linear")
            y_low = y.quantile(low_clip_quantile, interpolation="linear")
        work = work.filter((x >= x_low) & (y >= y_low))
    if positive_x:
        work = work.filter(x > 0)
    if positive_y:
        work = work.filter(y > 0)
    return _downsample(work, max_events=max_events, group_columns=group_columns)


def prepare_plot_payload(
    event_table: pl.DataFrame,
    *,
    columns: Sequence[str],
    max_events: int,
    group_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Project and downsample an event table before conversion to a plotting backend."""

    selected_columns = list(dict.fromkeys(columns))
    _require_columns(event_table, selected_columns)
    return _downsample(
        event_table.select(selected_columns),
        max_events=max_events,
        group_columns=group_columns,
    )


def _downsample(
    frame: pl.DataFrame,
    *,
    max_events: int,
    group_columns: Sequence[str],
) -> pl.DataFrame:
    if max_events <= 0:
        raise ValueError("max_events must be positive.")
    if frame.height <= max_events:
        return frame
    groups = [column for column in dict.fromkeys(group_columns) if column in frame.columns]
    if not groups:
        return _sample_with_stable_seed(frame, max_events)
    partitions = frame.partition_by(groups, maintain_order=True)
    partitions.sort(
        key=lambda partition: tuple(
            (partition.item(0, column) is None, str(partition.item(0, column))) for column in groups
        )
    )
    per_group = max(1, max_events // max(len(partitions), 1))
    sampled = pl.concat(
        [
            partition if partition.height <= per_group else _sample_with_stable_seed(partition, per_group)
            for partition in partitions
        ],
        how="vertical",
    )
    if sampled.height > max_events:
        return _sample_with_stable_seed(sampled, max_events)
    return sampled


def _sample_with_stable_seed(frame: pl.DataFrame, size: int) -> pl.DataFrame:
    positions = np.random.RandomState(0).choice(frame.height, size=size, replace=False)
    return frame[positions.tolist()]
