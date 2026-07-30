"""Polars-native preparation for tidy cytometry event tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

_REQUIRED_TIDY_COLUMNS = ("channel", "value", "sample_id", "event_index")
_METADATA_COLUMNS = ("treatment", "design_id", "sample_label")


class CytometryAnalysisError(ValueError):
    """Raised when a cytometry event table cannot support the requested analysis."""


@dataclass(frozen=True, slots=True)
class GateSpec:
    """Sequential cells and singlets gates for an event-wide table."""

    cells_x_channel: str
    cells_y_channel: str
    cells_x_range: tuple[float, float]
    cells_y_range: tuple[float, float]
    singlet_x_channel: str
    singlet_y_channel: str
    singlet_ratio_range: tuple[float, float]
    cells_enabled: bool = True
    singlets_enabled: bool = True


@dataclass(frozen=True, slots=True)
class ThresholdSpec:
    """Positive-event threshold applied to the selected fluorescence channel."""

    channel: str
    value: float = 0.0
    mode: Literal["manual", "from_control_quantile"] = "manual"
    group_column: str | None = None
    control_value: str | None = None
    quantile: float = 0.99


@dataclass(frozen=True, slots=True)
class CytometryAnalysis:
    """Polars-native event, count, summary, and quality-control tables."""

    gated_events: pl.DataFrame
    gate_counts_sample: pl.DataFrame
    stats_sample: pl.DataFrame
    stats_group: pl.DataFrame | None
    qc_table: pl.DataFrame
    threshold_value: float


EventFrame = pl.DataFrame | pl.LazyFrame


def prepare_event_table(
    frame: EventFrame,
    *,
    channels: Sequence[str],
    metadata_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Pivot declared channels from a tidy event table without inferring policy."""

    available = _frame_columns(frame)
    missing_tidy = [column for column in _REQUIRED_TIDY_COLUMNS if column not in available]
    if missing_tidy:
        raise CytometryAnalysisError(
            "Cytometry event data is missing required column(s): " + ", ".join(missing_tidy) + "."
        )

    selected_channels = tuple(dict.fromkeys(str(channel) for channel in channels if str(channel)))
    if not selected_channels:
        raise CytometryAnalysisError("Select at least one cytometry channel.")

    requested_metadata = tuple(dict.fromkeys(str(column).strip() for column in metadata_columns if str(column).strip()))
    missing_metadata = [column for column in requested_metadata if column not in available]
    if missing_metadata:
        raise CytometryAnalysisError(
            "Missing requested cytometry metadata column(s): " + ", ".join(missing_metadata) + "."
        )
    retained_metadata = list(
        dict.fromkeys((*[column for column in _METADATA_COLUMNS if column in available], *requested_metadata))
    )
    index_columns = ["sample_id", "event_index", *retained_metadata]
    pivot_key_columns = [*index_columns, "channel"]
    projected_columns = [*index_columns, "channel", "value"]
    event_identity_columns = ["sample_id", "event_index"]
    metadata_consistency_columns = {
        column: f"__reader_event_metadata_n_unique__{column}" for column in retained_metadata
    }
    query = _as_lazy(frame).select(projected_columns)
    if metadata_consistency_columns:
        query = query.with_columns(
            *[
                pl.col(column).n_unique().over(event_identity_columns).alias(consistency_column)
                for column, consistency_column in metadata_consistency_columns.items()
            ]
        )
    query = query.filter(pl.col("channel").is_in(selected_channels))

    channel_profile = query.select(
        pl.len().alias("row_count"),
        pl.struct(pivot_key_columns).n_unique().alias("unique_pivot_key_count"),
        pl.col("channel").unique().sort().implode().alias("channels"),
        pl.col("channel").n_unique().over(event_identity_columns).min().alias("minimum_event_selected_channel_count"),
        *[pl.col(consistency_column).max() for consistency_column in metadata_consistency_columns.values()],
    ).collect()
    row_count = int(channel_profile.item(0, "row_count"))
    unique_pivot_key_count = int(channel_profile.item(0, "unique_pivot_key_count"))
    if row_count:
        inconsistent_metadata = [
            column
            for column, consistency_column in metadata_consistency_columns.items()
            if int(channel_profile.item(0, consistency_column)) > 1
        ]
        if inconsistent_metadata:
            identity = ", ".join(event_identity_columns)
            metadata = ", ".join(inconsistent_metadata)
            raise CytometryAnalysisError(
                "Cytometry event data contains inconsistent metadata across channel rows for "
                f"event identity {identity}: {metadata}."
            )
    if row_count > unique_pivot_key_count:
        duplicate_row_count = row_count - unique_pivot_key_count
        keys = ", ".join(pivot_key_columns)
        raise CytometryAnalysisError(
            f"Cytometry event data contains {duplicate_row_count} duplicate pivot key rows across "
            f"{keys}; each event/channel key must be unique."
        )
    if row_count:
        present_channels = set(channel_profile.item(0, "channels"))
        missing_channels = [channel for channel in selected_channels if channel not in present_channels]
        if missing_channels:
            raise CytometryAnalysisError("Missing channels after pivot: " + ", ".join(missing_channels) + ".")
        minimum_channel_count = int(channel_profile.item(0, "minimum_event_selected_channel_count"))
        if minimum_channel_count < len(selected_channels):
            identity = ", ".join(event_identity_columns)
            required_channels = ", ".join(selected_channels)
            raise CytometryAnalysisError(
                "Cytometry event data is missing selected channels within at least one event; "
                f"each {identity} must contain all of: {required_channels}."
            )

    return (
        query.select(projected_columns)
        .with_columns(pl.col("value").cast(pl.Float64))
        .pivot(
            on="channel",
            on_columns=selected_channels,
            values="value",
            index=index_columns,
            maintain_order=True,
        )
        .collect()
    )


def _as_lazy(frame: EventFrame) -> pl.LazyFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame
    if isinstance(frame, pl.DataFrame):
        return frame.lazy()
    raise TypeError(f"Expected a Polars DataFrame or LazyFrame, got {type(frame).__name__}.")


def _frame_columns(frame: EventFrame) -> tuple[str, ...]:
    if isinstance(frame, pl.LazyFrame):
        return tuple(frame.collect_schema().names())
    if isinstance(frame, pl.DataFrame):
        return tuple(frame.columns)
    raise TypeError(f"Expected a Polars DataFrame or LazyFrame, got {type(frame).__name__}.")


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in dict.fromkeys(columns) if column not in frame.columns]
    if missing:
        raise CytometryAnalysisError("Missing cytometry column(s): " + ", ".join(missing) + ".")
