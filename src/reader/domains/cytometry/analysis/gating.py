"""Sequential gating and statistical summaries for cytometry events."""

from __future__ import annotations

import math

import polars as pl

from reader.domains.cytometry.analysis.events import (
    _METADATA_COLUMNS,
    CytometryAnalysis,
    CytometryAnalysisError,
    GateSpec,
    ThresholdSpec,
    _require_columns,
)


def analyze_events(
    event_table: pl.DataFrame,
    *,
    gate: GateSpec,
    threshold: ThresholdSpec,
    group_column: str | None = None,
) -> CytometryAnalysis:
    """Apply sequential gates and compute per-sample and group summaries in Polars."""

    required_columns = [threshold.channel, "sample_id"]
    if gate.cells_enabled:
        required_columns.extend((gate.cells_x_channel, gate.cells_y_channel))
    if gate.singlets_enabled:
        required_columns.extend((gate.singlet_x_channel, gate.singlet_y_channel))
    _require_columns(event_table, required_columns)

    cells_mask = pl.lit(True)
    if gate.cells_enabled:
        _validate_interval("cells X", gate.cells_x_range)
        _validate_interval("cells Y", gate.cells_y_range)
        cells_x = pl.col(gate.cells_x_channel).cast(pl.Float64, strict=False)
        cells_y = pl.col(gate.cells_y_channel).cast(pl.Float64, strict=False)
        cells_mask = (
            cells_x.is_finite()
            & cells_y.is_finite()
            & cells_x.is_between(*gate.cells_x_range, closed="both")
            & cells_y.is_between(*gate.cells_y_range, closed="both")
        )

    singlet_mask = pl.lit(True)
    if gate.singlets_enabled:
        _validate_interval("singlet ratio", gate.singlet_ratio_range)
        singlet_x = pl.col(gate.singlet_x_channel).cast(pl.Float64, strict=False)
        singlet_y = pl.col(gate.singlet_y_channel).cast(pl.Float64, strict=False)
        ratio = singlet_y / singlet_x
        singlet_mask = ratio.is_finite() & ratio.is_between(*gate.singlet_ratio_range, closed="both")

    cells_mask_column = "__reader_cells_mask"
    gate_mask_column = "__reader_gate_mask"
    work = event_table.with_columns(
        cells_mask.alias(cells_mask_column),
        (cells_mask & singlet_mask).alias(gate_mask_column),
    )
    gated_events = work.filter(pl.col(gate_mask_column)).drop(cells_mask_column, gate_mask_column)
    if gated_events.is_empty():
        raise CytometryAnalysisError("No events remain after gating. Adjust ranges.")

    if group_column is not None:
        if not isinstance(group_column, str) or not group_column.strip():
            raise CytometryAnalysisError("Group column must be a non-empty string or null.")
        group_column = group_column.strip()
        _require_columns(event_table, (group_column,))
    metadata_columns = list(
        dict.fromkeys(
            (
                *[column for column in _METADATA_COLUMNS if column in event_table.columns],
                *([group_column] if group_column else []),
            )
        )
    )
    counts = work.group_by("sample_id", maintain_order=True).agg(
        *[pl.col(column).first().alias(column) for column in metadata_columns],
        pl.len().alias("n_total_events"),
        pl.col(cells_mask_column).sum().cast(pl.Int64).alias("n_cells_gate"),
        pl.col(gate_mask_column).sum().cast(pl.Int64).alias("n_singlets"),
    )
    counts = counts.with_columns(
        pl.when(pl.col("n_total_events") > 0)
        .then(100.0 * pl.col("n_cells_gate") / pl.col("n_total_events"))
        .otherwise(float("nan"))
        .alias("pct_cells"),
        pl.when(pl.col("n_cells_gate") > 0)
        .then(100.0 * pl.col("n_singlets") / pl.col("n_cells_gate"))
        .otherwise(float("nan"))
        .alias("pct_singlets_of_cells"),
        pl.when(pl.col("n_total_events") > 0)
        .then(100.0 * pl.col("n_singlets") / pl.col("n_total_events"))
        .otherwise(float("nan"))
        .alias("pct_final"),
    )

    threshold_value = _resolve_threshold(gated_events, threshold)
    fluor_column = "__reader_fluor"
    gated_for_stats = gated_events.with_columns(
        pl.col(threshold.channel).cast(pl.Float64, strict=False).alias(fluor_column)
    )
    finite_fluor = pl.col(fluor_column).filter(pl.col(fluor_column).is_finite())
    positive_fluor = finite_fluor.filter(finite_fluor > 0)
    sample_stats = gated_for_stats.group_by("sample_id", maintain_order=True).agg(
        finite_fluor.median().alias("fluor_median"),
        finite_fluor.mean().alias("fluor_mean"),
        positive_fluor.log().mean().exp().alias("fluor_geomean"),
        finite_fluor.quantile(0.90, interpolation="linear").alias("fluor_p90"),
        finite_fluor.quantile(0.99, interpolation="linear").alias("fluor_p99"),
        (100.0 * (finite_fluor > threshold_value).mean()).alias("pct_positive"),
    )
    stats_sample = counts.join(sample_stats, on="sample_id", how="left")

    stats_group = None
    if group_column is not None:
        stats_group = stats_sample.group_by(group_column, maintain_order=True).agg(
            pl.col("sample_id").n_unique().alias("n_samples"),
            pl.col("fluor_median").mean().alias("fluor_median_mean"),
            pl.col("fluor_median").std().alias("fluor_median_std"),
            pl.col("fluor_geomean").mean().alias("fluor_geomean_mean"),
            pl.col("pct_positive").mean().alias("pct_positive_mean"),
        )

    all_fluor = pl.col(threshold.channel).cast(pl.Float64, strict=False)
    finite_all_fluor = all_fluor.filter(all_fluor.is_finite())
    qc_table = event_table.group_by("sample_id", maintain_order=True).agg(
        pl.when(finite_all_fluor.len() > 0)
        .then(100.0 * (finite_all_fluor <= 0).mean())
        .otherwise(float("nan"))
        .alias("pct_nonpositive")
    )

    return CytometryAnalysis(
        gated_events=gated_events,
        gate_counts_sample=counts,
        stats_sample=stats_sample,
        stats_group=stats_group,
        qc_table=qc_table,
        threshold_value=threshold_value,
    )


def _validate_interval(label: str, interval: tuple[float, float]) -> None:
    low, high = interval
    if not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise CytometryAnalysisError(f"{label} range must contain two finite values in ascending order.")


def _resolve_threshold(gated_events: pl.DataFrame, threshold: ThresholdSpec) -> float:
    if threshold.mode == "manual":
        value = float(threshold.value)
    elif threshold.mode == "from_control_quantile":
        if threshold.group_column is None or threshold.control_value is None:
            raise CytometryAnalysisError("Control thresholding requires a group column and control value.")
        _require_columns(gated_events, (threshold.group_column, threshold.channel))
        if not 0.0 <= threshold.quantile <= 1.0:
            raise CytometryAnalysisError("Control quantile must be between 0 and 1.")
        control_values = (
            gated_events.filter(pl.col(threshold.group_column).cast(pl.String) == threshold.control_value)
            .get_column(threshold.channel)
            .cast(pl.Float64, strict=False)
            .drop_nulls()
        )
        control_values = control_values.filter(control_values.is_finite())
        if control_values.is_empty():
            raise CytometryAnalysisError("No control events are available for thresholding.")
        value = float(control_values.quantile(threshold.quantile, interpolation="linear"))
    else:
        raise CytometryAnalysisError(f"Unknown threshold mode `{threshold.mode}`.")
    if not math.isfinite(value):
        raise CytometryAnalysisError("Threshold value must be finite.")
    return value
