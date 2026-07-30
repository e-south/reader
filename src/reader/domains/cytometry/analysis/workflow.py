"""Explicit normal-lifecycle cytometry gating workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import polars as pl

from .events import CytometryAnalysisError, GateSpec, ThresholdSpec, prepare_event_table
from .gating import analyze_events

_NONPOSITIVE_EVALUABLE_EVENTS_COLUMN = "__reader_nonpositive_evaluable_events"


@dataclass(frozen=True, slots=True)
class CytometryQCSpec:
    minimum_final_events: int
    minimum_final_percent: float
    maximum_nonpositive_percent: float
    nonpositive_scope: Literal["all_events", "gated_events"]


@dataclass(frozen=True, slots=True)
class CytometryGatingRequest:
    gate: GateSpec
    threshold: ThresholdSpec
    group_column: str | None
    qc: CytometryQCSpec


@dataclass(frozen=True, slots=True)
class CytometryGatingResult:
    gate_definition: pl.DataFrame
    gated_events: pl.DataFrame
    sample_stats: pl.DataFrame
    group_stats: pl.DataFrame
    qc: pl.DataFrame


def run_cytometry_gating(events: pd.DataFrame | pl.DataFrame, request: CytometryGatingRequest) -> CytometryGatingResult:
    """Resolve an explicit gating request into typed, persistence-ready tables."""

    _validate_request(request)
    source = pl.from_pandas(events) if isinstance(events, pd.DataFrame) else events
    if not isinstance(source, pl.DataFrame):
        raise TypeError(f"Expected pandas or Polars event data, got {type(events).__name__}.")

    selected_channels: list[str] = []
    if request.gate.cells_enabled:
        selected_channels.extend((request.gate.cells_x_channel, request.gate.cells_y_channel))
    if request.gate.singlets_enabled:
        selected_channels.extend((request.gate.singlet_x_channel, request.gate.singlet_y_channel))
    selected_channels.append(request.threshold.channel)
    metadata_columns = tuple(
        dict.fromkeys(
            column
            for column in (request.group_column, request.threshold.group_column)
            if isinstance(column, str) and column
        )
    )
    wide = prepare_event_table(
        source,
        channels=tuple(dict.fromkeys(selected_channels)),
        metadata_columns=metadata_columns,
    )
    analysis = analyze_events(
        wide,
        gate=request.gate,
        threshold=request.threshold,
        group_column=request.group_column,
    )
    sample_stats = _sample_stats(analysis.stats_sample, request=request, threshold_value=analysis.threshold_value)
    group_stats = _group_stats(analysis.stats_group, request=request)
    nonpositive_source = wide if request.qc.nonpositive_scope == "all_events" else analysis.gated_events
    qc = _qc_table(
        analysis.gate_counts_sample,
        _nonpositive_table(nonpositive_source, channel=request.threshold.channel),
        request=request,
    )
    return CytometryGatingResult(
        gate_definition=_gate_definition(request, threshold_value=analysis.threshold_value),
        gated_events=analysis.gated_events,
        sample_stats=sample_stats,
        group_stats=group_stats,
        qc=qc,
    )


def _validate_request(request: CytometryGatingRequest) -> None:
    if request.group_column is not None and (
        not isinstance(request.group_column, str) or not request.group_column.strip()
    ):
        raise CytometryAnalysisError("group_column must be a non-empty string or null.")
    if request.threshold.mode == "manual":
        if request.threshold.group_column is not None or request.threshold.control_value is not None:
            raise CytometryAnalysisError("Manual thresholding may not declare control-group fields.")
    elif request.threshold.mode == "from_control_quantile":
        if not request.threshold.group_column or not request.threshold.control_value:
            raise CytometryAnalysisError("Control thresholding requires explicit group_column and control_value.")
    else:
        raise CytometryAnalysisError(f"Unknown threshold mode `{request.threshold.mode}`.")
    if request.qc.minimum_final_events < 0:
        raise CytometryAnalysisError("minimum_final_events must be nonnegative.")
    if request.qc.nonpositive_scope not in {"all_events", "gated_events"}:
        raise CytometryAnalysisError("nonpositive_scope must be 'all_events' or 'gated_events'.")
    for name, value in (
        ("minimum_final_percent", request.qc.minimum_final_percent),
        ("maximum_nonpositive_percent", request.qc.maximum_nonpositive_percent),
    ):
        if not 0.0 <= float(value) <= 100.0:
            raise CytometryAnalysisError(f"{name} must be between 0 and 100.")


def _gate_definition(request: CytometryGatingRequest, *, threshold_value: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "definition_id": ["resolved"],
            "cells_enabled": [request.gate.cells_enabled],
            "cells_x_channel": [request.gate.cells_x_channel],
            "cells_x_min": [float(request.gate.cells_x_range[0])],
            "cells_x_max": [float(request.gate.cells_x_range[1])],
            "cells_y_channel": [request.gate.cells_y_channel],
            "cells_y_min": [float(request.gate.cells_y_range[0])],
            "cells_y_max": [float(request.gate.cells_y_range[1])],
            "singlets_enabled": [request.gate.singlets_enabled],
            "singlet_x_channel": [request.gate.singlet_x_channel],
            "singlet_y_channel": [request.gate.singlet_y_channel],
            "singlet_ratio_min": [float(request.gate.singlet_ratio_range[0])],
            "singlet_ratio_max": [float(request.gate.singlet_ratio_range[1])],
            "fluorescence_channel": [request.threshold.channel],
            "threshold_mode": [request.threshold.mode],
            "threshold_value": [float(threshold_value)],
            "threshold_group_column": [request.threshold.group_column],
            "threshold_control_value": [request.threshold.control_value],
            "threshold_quantile": [
                float(request.threshold.quantile) if request.threshold.mode == "from_control_quantile" else None
            ],
            "group_column": [request.group_column],
            "minimum_final_events": [int(request.qc.minimum_final_events)],
            "minimum_final_percent": [float(request.qc.minimum_final_percent)],
            "maximum_nonpositive_percent": [float(request.qc.maximum_nonpositive_percent)],
            "nonpositive_scope": [request.qc.nonpositive_scope],
        },
        schema_overrides={
            "threshold_group_column": pl.String,
            "threshold_control_value": pl.String,
            "threshold_quantile": pl.Float64,
            "group_column": pl.String,
        },
    )


def _sample_stats(stats: pl.DataFrame, *, request: CytometryGatingRequest, threshold_value: float) -> pl.DataFrame:
    group_value = (
        pl.col(request.group_column).cast(pl.String)
        if request.group_column is not None
        else pl.lit(None, dtype=pl.String)
    )
    return stats.select(
        "sample_id",
        pl.lit(request.group_column, dtype=pl.String).alias("group_column"),
        group_value.alias("group_value"),
        "n_total_events",
        "n_cells_gate",
        "n_singlets",
        "pct_cells",
        "pct_singlets_of_cells",
        "pct_final",
        "fluor_median",
        "fluor_mean",
        "fluor_geomean",
        "fluor_p90",
        "fluor_p99",
        "pct_positive",
        pl.lit(request.threshold.channel).alias("fluorescence_channel"),
        pl.lit(float(threshold_value)).alias("threshold_value"),
    )


def _group_stats(stats: pl.DataFrame | None, *, request: CytometryGatingRequest) -> pl.DataFrame:
    schema = {
        "group_column": pl.String,
        "group_value": pl.String,
        "n_samples": pl.Int64,
        "fluor_median_mean": pl.Float64,
        "fluor_median_std": pl.Float64,
        "fluor_geomean_mean": pl.Float64,
        "pct_positive_mean": pl.Float64,
    }
    if request.group_column is None or stats is None:
        return pl.DataFrame(schema=schema)
    return stats.select(
        pl.lit(request.group_column).alias("group_column"),
        pl.col(request.group_column).cast(pl.String).alias("group_value"),
        "n_samples",
        "fluor_median_mean",
        "fluor_median_std",
        "fluor_geomean_mean",
        "pct_positive_mean",
    )


def _qc_table(counts: pl.DataFrame, nonpositive: pl.DataFrame, *, request: CytometryGatingRequest) -> pl.DataFrame:
    joined = counts.select(
        "sample_id",
        "n_total_events",
        "n_cells_gate",
        "n_singlets",
        "pct_final",
    ).join(nonpositive, on="sample_id", how="left")
    # Keep the persisted percentage finite, but require a real denominator so
    # even a permissive 100% ceiling cannot pass an unevaluable sample.
    return (
        joined.with_columns(
            pl.col("pct_nonpositive").fill_nan(100.0).fill_null(100.0),
            pl.col(_NONPOSITIVE_EVALUABLE_EVENTS_COLUMN).fill_null(0).cast(pl.Int64),
            pl.lit(int(request.qc.minimum_final_events)).alias("minimum_final_events"),
            pl.lit(float(request.qc.minimum_final_percent)).alias("minimum_final_percent"),
            pl.lit(float(request.qc.maximum_nonpositive_percent)).alias("maximum_nonpositive_percent"),
            pl.lit(request.qc.nonpositive_scope).alias("nonpositive_scope"),
        )
        .with_columns(
            (pl.col("n_singlets") >= pl.col("minimum_final_events")).alias("passes_final_events"),
            (pl.col("pct_final") >= pl.col("minimum_final_percent")).alias("passes_final_percent"),
            (
                (pl.col(_NONPOSITIVE_EVALUABLE_EVENTS_COLUMN) > 0)
                & (pl.col("pct_nonpositive") <= pl.col("maximum_nonpositive_percent"))
            ).alias("passes_nonpositive"),
        )
        .with_columns(
            (pl.col("passes_final_events") & pl.col("passes_final_percent") & pl.col("passes_nonpositive")).alias(
                "qc_pass"
            )
        )
        .with_columns(pl.when(pl.col("qc_pass")).then(pl.lit("pass")).otherwise(pl.lit("fail")).alias("qc_status"))
        .drop(_NONPOSITIVE_EVALUABLE_EVENTS_COLUMN)
    )


def _nonpositive_table(events: pl.DataFrame, *, channel: str) -> pl.DataFrame:
    fluorescence = pl.col(channel).cast(pl.Float64, strict=False)
    finite = fluorescence.filter(fluorescence.is_finite())
    return events.group_by("sample_id", maintain_order=True).agg(
        finite.len().cast(pl.Int64).alias(_NONPOSITIVE_EVALUABLE_EVENTS_COLUMN),
        pl.when(finite.len() > 0).then(100.0 * (finite <= 0).mean()).otherwise(100.0).alias("pct_nonpositive"),
    )
