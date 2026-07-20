"""Polars-native preparation for tidy cytometry event tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

_REQUIRED_TIDY_COLUMNS = ("channel", "value", "sample_id", "event_index")
_METADATA_COLUMNS = ("treatment", "design_id", "sample_label")


class CytometryAnalysisError(ValueError):
    """Raised when a cytometry event table cannot support the requested analysis."""


@dataclass(frozen=True, slots=True)
class EventFilters:
    """Optional exact-match filters applied before the long-to-wide pivot."""

    design_id: str | None = None
    treatment: str | None = None
    sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class NumericRange:
    """Observed extent and selected default interval for one numeric signal."""

    minimum: float
    maximum: float
    selected: tuple[float, float]


@dataclass(frozen=True, slots=True)
class GateDefaults:
    """Default cytometry gate ranges derived from finite event values."""

    cells_x: NumericRange
    cells_y: NumericRange
    singlet_ratio: NumericRange


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


def scan_event_table(path: str | Path) -> pl.LazyFrame:
    """Open a parquet event table lazily so filters and projections can be pushed down."""

    return pl.scan_parquet(path)


def frame_columns(frame: EventFrame) -> tuple[str, ...]:
    """Return frame column names without collecting a lazy event table."""

    if isinstance(frame, pl.LazyFrame):
        return tuple(frame.collect_schema().names())
    if isinstance(frame, pl.DataFrame):
        return tuple(frame.columns)
    raise TypeError(f"Expected a Polars DataFrame or LazyFrame, got {type(frame).__name__}.")


def distinct_string_values(frame: EventFrame, column: str) -> list[str]:
    """Collect sorted non-empty strings from one projected column."""

    return distinct_string_values_by_column(frame, (column,))[column]


def distinct_string_values_by_column(
    frame: EventFrame,
    columns: Sequence[str],
) -> dict[str, list[str]]:
    """Collect sorted string values for several columns in one projected query."""

    requested = list(dict.fromkeys(columns))
    available = set(frame_columns(frame))
    present = [column for column in requested if column in available]
    values_by_column = {column: [] for column in requested}
    if not present:
        return values_by_column
    collected = (
        _as_lazy(frame)
        .select(
            *[pl.col(column).drop_nulls().cast(pl.String).unique().sort().implode().alias(column) for column in present]
        )
        .collect()
    )
    row = collected.row(0, named=True)
    for column in present:
        values_by_column[column] = [value for value in row[column] if value]
    return values_by_column


def prepare_event_preview(
    frame: EventFrame,
    *,
    row_limit: int = 10_000,
    column_limit: int = 40,
) -> pl.DataFrame:
    """Collect a bounded table preview for notebook display."""

    if row_limit <= 0:
        raise ValueError("row_limit must be positive.")
    if column_limit <= 0:
        raise ValueError("column_limit must be positive.")
    columns = frame_columns(frame)[:column_limit]
    return _as_lazy(frame).select(columns).head(row_limit).collect()


def prepare_event_table(
    frame: EventFrame,
    *,
    channels: Sequence[str],
    filters: EventFilters | None = None,
) -> pl.DataFrame:
    """Filter and pivot a tidy event table while keeping the preparation in Polars."""

    filters = filters or EventFilters()
    available = frame_columns(frame)
    missing_tidy = [column for column in _REQUIRED_TIDY_COLUMNS if column not in available]
    if missing_tidy:
        raise CytometryAnalysisError(
            "Cytometry event data is missing required column(s): " + ", ".join(missing_tidy) + "."
        )

    selected_channels = tuple(dict.fromkeys(str(channel) for channel in channels if str(channel)))
    if not selected_channels:
        raise CytometryAnalysisError("Select at least one cytometry channel.")

    metadata_columns = [column for column in _METADATA_COLUMNS if column in available]
    index_columns = ["sample_id", "event_index", *metadata_columns]
    pivot_key_columns = [*index_columns, "channel"]
    projected_columns = [*index_columns, "channel", "value"]
    event_identity_columns = ["sample_id", "event_index"]
    metadata_consistency_columns = {
        column: f"__reader_event_metadata_n_unique__{column}" for column in metadata_columns
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
    for column, value in (
        ("design_id", filters.design_id),
        ("treatment", filters.treatment),
        ("sample_id", filters.sample_id),
    ):
        if value is not None:
            if column not in available:
                raise CytometryAnalysisError(f"Cannot filter on missing column `{column}`.")
            query = query.filter(pl.col(column) == value)

    channel_profile = query.select(
        pl.len().alias("row_count"),
        pl.struct(pivot_key_columns).n_unique().alias("unique_pivot_key_count"),
        pl.col("channel").unique().sort().implode().alias("channels"),
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


def gate_defaults(
    event_table: pl.DataFrame,
    *,
    cells_x_channel: str,
    cells_y_channel: str,
    singlet_x_channel: str,
    singlet_y_channel: str,
) -> GateDefaults:
    """Compute finite 1st-to-99th-percentile defaults for cells and singlets gates."""

    _require_columns(
        event_table,
        (cells_x_channel, cells_y_channel, singlet_x_channel, singlet_y_channel),
    )
    ratio = event_table.select(
        (pl.col(singlet_y_channel) / pl.col(singlet_x_channel)).alias("singlet_ratio")
    ).get_column("singlet_ratio")
    return GateDefaults(
        cells_x=_numeric_range(event_table.get_column(cells_x_channel)),
        cells_y=_numeric_range(event_table.get_column(cells_y_channel)),
        singlet_ratio=_numeric_range(ratio),
    )


def _as_lazy(frame: EventFrame) -> pl.LazyFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame
    if isinstance(frame, pl.DataFrame):
        return frame.lazy()
    raise TypeError(f"Expected a Polars DataFrame or LazyFrame, got {type(frame).__name__}.")


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in dict.fromkeys(columns) if column not in frame.columns]
    if missing:
        raise CytometryAnalysisError("Missing cytometry column(s): " + ", ".join(missing) + ".")


def _numeric_range(values: pl.Series, *, low_quantile: float = 0.01, high_quantile: float = 0.99) -> NumericRange:
    numeric = values.cast(pl.Float64, strict=False).drop_nulls()
    numeric = numeric.filter(numeric.is_finite())
    if numeric.is_empty():
        raise CytometryAnalysisError("No finite values are available for gate defaults.")
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if maximum <= minimum:
        maximum = minimum + 1e-6
    selected_low = max(float(numeric.quantile(low_quantile, interpolation="linear")), minimum)
    selected_high = min(float(numeric.quantile(high_quantile, interpolation="linear")), maximum)
    if selected_high <= selected_low:
        selected_low, selected_high = minimum, maximum
    return NumericRange(minimum=minimum, maximum=maximum, selected=(selected_low, selected_high))
