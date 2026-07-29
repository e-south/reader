"""Static review tables and plots for response-window bundles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reader.domains.plate_reader.analysis.response_window.display import validate_display_manifest
from reader.domains.plate_reader.analysis.response_window.sources import STATE_ORDER

from .reporting_plots import (
    write_event_plot,
    write_handoff_plot,
    write_stability_plot,
)
from .reporting_quality_plots import write_repeat_plot, write_uncertainty_plot

_RESPONSE_COLUMNS = tuple(f"r{state}" for state in STATE_ORDER)
_FLUORESCENCE_COLUMNS = tuple(f"b{state}" for state in STATE_ORDER)
_VALUE_COLUMNS = _RESPONSE_COLUMNS + _FLUORESCENCE_COLUMNS


def _spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return tie-aware rank correlation, with explicit constant-series behavior."""

    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("Spearman rank correlation requires equally sized one-dimensional arrays.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Spearman rank correlation requires finite values.")
    if np.array_equal(left, right):
        return 1.0

    left_ranks = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_ranks = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    scale = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if scale == 0.0:
        return float("nan")
    correlation = float(np.dot(left_centered, right_centered) / scale)
    return float(np.clip(correlation, -1.0, 1.0))


def write_review_artifacts(
    designs: pd.DataFrame,
    events: pd.DataFrame,
    *,
    primary_reduction_id: str,
    display: dict[str, object],
    out_dir: Path,
) -> pd.DataFrame:
    """Write a small manifest-backed plot set from persisted design records."""

    display = validate_display_manifest(display)
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    primary = designs.loc[
        designs["reduction_id"].astype(str).eq(primary_reduction_id) & ~designs["is_reference"].astype(bool)
    ].copy()
    if primary.empty:
        raise ValueError(f"primary reduction {primary_reduction_id!r} produced no non-reference rows.")
    primary_path = tables_dir / "primary_handoff.csv"
    primary.to_csv(primary_path, index=False)

    stability = _reduction_stability(designs, primary_reduction_id=primary_reduction_id)
    stability_path = tables_dir / "reduction_stability.csv"
    stability.to_csv(stability_path, index=False)

    repeated = _repeated_agreement(primary)
    repeated_path = tables_dir / "repeated_design_agreement.csv"
    repeated.to_csv(repeated_path, index=False)

    uncertainty = _uncertainty_summary(primary)
    uncertainty_path = tables_dir / "uncertainty_summary.csv"
    uncertainty.to_csv(uncertainty_path, index=False)

    plot_rows = [
        write_event_plot(events, display=display, out_dir=out_dir),
        write_handoff_plot(primary, display=display, out_dir=out_dir),
        write_stability_plot(stability, display=display, out_dir=out_dir),
        write_repeat_plot(repeated, display=display, out_dir=out_dir),
        write_uncertainty_plot(uncertainty, display=display, out_dir=out_dir),
    ]
    manifest = pd.DataFrame.from_records(plot_rows)
    required = {
        "plot_id",
        "tier",
        "title",
        "premise",
        "decision_value",
        "rationale",
        "alt_text",
        "non_claim_boundary",
        "data_table",
        "path",
    }
    if set(manifest.columns) != required:
        raise RuntimeError("response-window plot manifest fields drifted from the declared contract.")
    manifest.to_csv(tables_dir / "plot_manifest.csv", index=False)
    return manifest


def _reduction_stability(designs: pd.DataFrame, *, primary_reduction_id: str) -> pd.DataFrame:
    primary = designs.loc[
        designs["reduction_id"].astype(str).eq(primary_reduction_id) & ~designs["is_reference"].astype(bool)
    ].set_index(["experiment_id", "design_id"])
    rows: list[dict[str, object]] = []
    for reduction_id, frame in designs.loc[~designs["is_reference"].astype(bool)].groupby("reduction_id", sort=False):
        aligned = frame.set_index(["experiment_id", "design_id"])
        if set(aligned.index) != set(primary.index):
            raise ValueError(f"reduction {reduction_id!r} does not preserve the primary design universe.")
        aligned = aligned.loc[primary.index]
        metadata = aligned.iloc[0]
        for column in _VALUE_COLUMNS:
            left = primary[column].to_numpy(dtype=float)
            right = aligned[column].to_numpy(dtype=float)
            correlation = _spearman_rank_correlation(left, right)
            rows.append(
                {
                    "reduction_id": str(reduction_id),
                    "reduction_method": str(metadata["reduction_method"]),
                    "response_basis": str(metadata["response_basis"]),
                    "reduction_role": str(metadata["reduction_role"]),
                    "window_start_event_h": float(metadata["window_start_event_h"]),
                    "window_end_event_h": float(metadata["window_end_event_h"]),
                    "component": column,
                    "spearman_to_primary": correlation,
                    "n": len(left),
                }
            )
    return pd.DataFrame.from_records(rows)


def _repeated_agreement(primary: pd.DataFrame) -> pd.DataFrame:
    counts = primary.groupby("design_id")["experiment_id"].nunique()
    repeated_ids = set(counts.loc[counts > 1].index.astype(str))
    rows: list[dict[str, object]] = []
    for design_id, frame in primary.loc[primary["design_id"].astype(str).isin(repeated_ids)].groupby(
        "design_id", sort=True
    ):
        for column in _VALUE_COLUMNS:
            values = frame[column].to_numpy(dtype=float)
            median = float(np.median(values))
            for source_row, value in zip(frame.itertuples(index=False), values, strict=True):
                rows.append(
                    {
                        "design_id": str(design_id),
                        "experiment_id": str(source_row.experiment_id),
                        "component": column,
                        "value": float(value),
                        "cross_experiment_median": median,
                        "absolute_deviation": abs(float(value) - median),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _uncertainty_summary(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state in STATE_ORDER:
        for family, prefix in (("response", "r"), ("anchored_fluorescence", "b")):
            bootstrap = primary[f"{prefix}{state}_bootstrap_sd"].to_numpy(dtype=float)
            event = primary[f"{prefix}{state}_event_half_range"].to_numpy(dtype=float)
            rows.append(
                {
                    "family": family,
                    "state": state,
                    "median_bootstrap_sd": float(np.median(bootstrap)),
                    "p90_bootstrap_sd": float(np.quantile(bootstrap, 0.9)),
                    "median_event_half_range": float(np.median(event)),
                    "p90_event_half_range": float(np.quantile(event, 0.9)),
                    "n": len(primary),
                }
            )
    return pd.DataFrame.from_records(rows)


__all__ = ["write_review_artifacts"]
