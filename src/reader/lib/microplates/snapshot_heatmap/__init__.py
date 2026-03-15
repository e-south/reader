"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/microplates/snapshot_heatmap/__init__.py

Snapshot heatmap with tidy aggregation and square cells.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.colors import Colormap, LinearSegmentedColormap

from reader.core.plot_sinks import PlotFigure

from ..style import new_fig_ax, use_style
from ..support import alias_column, choose_nearest_time, emit_plot_figure, pretty_name, warn_if_empty
from .inputs import prepare_snapshot_heatmap_inputs as prepare_snapshot_heatmap_inputs

__all__ = ["plot_snapshot_heatmap", "prepare_snapshot_heatmap_inputs"]


def _ensure(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise ValueError(f"snapshot_heatmap: required columns missing: {missing}")


def _apply_explicit_axis_order(
    pivot: pd.DataFrame,
    *,
    axis: str,
    labels: list[str] | None,
) -> pd.DataFrame:
    if labels is None:
        return pivot
    if not labels:
        raise ValueError(f"snapshot_heatmap: {axis} order must not be empty")

    available = list(map(str, pivot.columns if axis == "x" else pivot.index))
    missing = [label for label in labels if label not in available]
    if missing:
        raise ValueError(f"snapshot_heatmap: {axis} order includes missing label(s): {missing}")

    if axis == "x":
        return pivot[labels]
    return pivot.loc[labels]


def _resolve_cmap(spec: Any) -> Colormap | None:
    if spec is None:
        return None
    if isinstance(spec, mcolors.Colormap):
        return spec
    if isinstance(spec, list | tuple):
        return LinearSegmentedColormap.from_list("custom", list(spec))
    if isinstance(spec, str):
        try:
            return plt.get_cmap(spec)
        except Exception:
            return LinearSegmentedColormap.from_list("custom", ["#FFFFFF", spec])
    return None


def plot_snapshot_heatmap(
    *,
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    output_dir: Path | None,
    channel: str,
    time: float,
    x: str = "treatment",
    y: str = "design_id",
    order_x: list[str] | None = None,
    order_y: list[str] | None = None,
    square: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    fig_kwargs: dict[str, Any],
    filename: str | None,
) -> list[PlotFigure]:
    del blanks
    x_col = alias_column(df, x)
    y_col = alias_column(df, y)
    _ensure(df, ["time", "channel", "value", x_col, y_col])

    work_pl = pl.from_pandas(df)
    time_expr = pl.col("time").cast(pl.Float64, strict=True)
    value_expr = pl.col("value").cast(pl.Float64, strict=False)
    value_expr = pl.when(value_expr.is_nan()).then(None).otherwise(value_expr)
    work_pl = work_pl.with_columns(time_expr.alias("time"), value_expr.alias("value"))

    rc = (fig_kwargs or {}).get("rc", {})
    tol = (fig_kwargs or {}).get("time_tolerance", rc.get("time_tolerance", None))
    times = (
        work_pl.filter(pl.col("channel").cast(pl.Utf8) == str(channel))
        .select(pl.col("time").fill_null(float("nan")).alias("time"))
        .to_numpy()
        .ravel()
    )
    tsel = choose_nearest_time(
        times,
        target_time=float(time),
        tol=(None if tol is None else float(tol)),
        where="snapshot_heatmap",
        logger=logging.getLogger("reader"),
    )

    snap_pl = work_pl.filter(
        (pl.col("channel").cast(pl.Utf8) == str(channel)) & ((pl.col("time") - float(tsel)).abs() <= 1e-9)
    )
    if snap_pl.is_empty():
        raise ValueError("snapshot_heatmap: no rows after (channel,time) selection")

    pivot_pl = (
        snap_pl.group_by([y_col, x_col])
        .agg(pl.col("value").median().alias("value"))
        .sort([y_col, x_col])
        .pivot(values="value", index=y_col, on=x_col, aggregate_function="first")
    )
    pivot = pivot_pl.to_pandas(use_pyarrow_extension_array=False)
    if y_col in pivot.columns:
        pivot = pivot.set_index(y_col)

    if order_x is None and pretty_name(str(x_col)) == "treatment":
        cols = list(map(str, pivot.columns))
        lower = [column.lower() for column in cols]
        has_hint = any(
            ("pms" in column)
            or ("etoh" in column)
            or ("ethanol" in column)
            or ("ciprofloxacin" in column)
            or ("cipro" in column)
            or (column.strip() == "negative")
            for column in lower
        )
        if has_hint:

            def _rank(column: str) -> tuple[int, str]:
                value = column.lower()
                if "pms" in value:
                    return (0, value)
                if "etoh" in value or "ethanol" in value:
                    return (1, value)
                if "ciprofloxacin" in value or "cipro" in value:
                    return (2, value)
                if value.strip() == "negative" or "negative" in value:
                    return (4, value)
                return (3, value)

            ordered = sorted(cols, key=_rank)
            pivot = pivot[ordered]
            logging.getLogger("reader").info(
                "[dim]snapshot_heatmap[/dim] • treatment order applied: PMS → EtOH → ciprofloxacin → negative "
                "(override with plots.<spec_id>.with.order_x in config.yaml)"
            )

    pivot = _apply_explicit_axis_order(pivot, axis="x", labels=order_x)
    pivot = _apply_explicit_axis_order(pivot, axis="y", labels=order_y)

    if pivot.empty or pivot.shape[0] == 0 or pivot.shape[1] == 0:
        warn_if_empty(pivot, where="snapshot_heatmap", detail="after pivot")
        return []

    with use_style(rc=(fig_kwargs or {}).get("rc")):
        fig, ax = new_fig_ax(fig_kwargs)
        arr = pivot.to_numpy(dtype=float)
        if not np.isfinite(arr).any():
            raise ValueError("snapshot_heatmap: selected snapshot contains no finite values")
        cmap = _resolve_cmap((fig_kwargs or {}).get("cmap"))
        resolved_vmin = float(np.nanmin(arr)) if vmin is None else float(vmin)
        resolved_vmax = float(np.nanmax(arr)) if vmax is None else float(vmax)

        im = ax.imshow(
            arr,
            aspect=("equal" if square else "auto"),
            vmin=resolved_vmin,
            vmax=resolved_vmax,
            cmap=cmap,
        )

        ax.grid(False)
        with suppress(Exception):
            ax.set_facecolor("white")

        ax.set_xticks(range(arr.shape[1]))
        ax.set_xticklabels(list(map(str, pivot.columns)), rotation=45, ha="right")
        ax.set_yticks(range(arr.shape[0]))
        ax.set_yticklabels(list(map(str, pivot.index)))
        ax.set_xlabel(pretty_name(str(x_col)).title())
        ax.set_ylabel(pretty_name(str(y_col)).title())
        ax.set_title(f"{channel} @ t={tsel:g} h")

        ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        shrink = float((fig_kwargs or {}).get("cbar_shrink", 1.0))
        cbar_label: str | None = (fig_kwargs or {}).get("cbar_label")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=shrink, label=cbar_label)

        base = filename or f"snapshot_heatmap__{channel}__t{tsel:g}h"
        y_levels = list(map(str, pivot.index))
        n_geno = len(y_levels)

        def _short_id(value: str) -> str:
            return hashlib.blake2b(value.encode("utf-8"), digest_size=4).hexdigest()

        geno_id = _short_id("|".join(sorted(y_levels)))
        x_levels = list(map(str, pivot.columns))
        cmap_name = (
            cmap.name if isinstance(cmap, Colormap) and hasattr(cmap, "name") else "custom" if cmap else "default"
        )
        tol = (fig_kwargs or {}).get("time_tolerance", None)
        fp_payload = "|".join(
            [
                f"ch={channel}",
                f"t={tsel:g}",
                "x=" + ",".join(x_levels),
                "y=" + ",".join(y_levels),
                f"cmap={cmap_name}",
                f"vmin={'nan' if vmin is None else float(vmin)}",
                f"vmax={'nan' if vmax is None else float(vmax)}",
                f"square={bool(square)}",
                f"tol={'' if tol is None else float(tol)}",
            ]
        )
        fp_id = _short_id(fp_payload)
        stub = f"{base}__gy{n_geno}-{geno_id}__fp{fp_id}"
        return emit_plot_figure(fig=fig, filename=stub, output_dir=output_dir, fig_kwargs=fig_kwargs)
