"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/microplates/snapshot_heatmap.py

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

from .base import alias_column, pretty_name, save_figure, warn_if_empty
from .style import new_fig_ax, use_style


def _ensure(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"snapshot_heatmap: required columns missing: {missing}")


def _resolve_cmap(spec: Any) -> Colormap | None:
    """
    Accept a matplotlib/seaborn colormap name, a single color name, a list of colors,
    or a Colormap object. Returns a Colormap or None (caller may rely on MPL default).
    """
    if spec is None:
        return None
    # Already a colormap?
    if isinstance(spec, mcolors.Colormap):
        return spec
    # List/tuple of colors → make a segmented map
    if isinstance(spec, list | tuple):
        return LinearSegmentedColormap.from_list("custom", list(spec))
    # String: try colormap name first; else treat as a single color
    if isinstance(spec, str):
        try:
            return plt.get_cmap(spec)
        except Exception:
            # not a known cmap → build white→color gradient
            return LinearSegmentedColormap.from_list("custom", ["#FFFFFF", spec])


def _choose_time(times: np.ndarray, target: float, tol: float | None) -> float:
    times = np.asarray(sorted(times), dtype=float)
    if times.size == 0:
        raise ValueError("snapshot_heatmap: dataframe has no time values")
    diffs = np.abs(times - float(target))
    j = int(np.nanargmin(diffs))
    if tol is not None and diffs[j] > float(tol):
        # Pragmatic behavior: inform (do not fail) and proceed with the nearest.
        logging.getLogger("reader").info(
            "[warn]snapshot_heatmap[/warn] • requested t=%.2f h; nearest available t=%.2f h (Δ=%.2f h) — using nearest",
            float(target),
            float(times[j]),
            float(diffs[j]),
        )
    return float(times[j])


def plot_snapshot_heatmap(
    *,
    df: pd.DataFrame,
    blanks: pd.DataFrame,  # accepted for API parity; not used here
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
    """
    Render a heatmap for a single channel at the snapshot time nearest to `time`.

    - Aggregates by median over (y, x) at the chosen time.
    - Orders axes by `order_x`/`order_y` when provided.
    - Asserts required columns and time resolvability (no silent fallbacks).
    """
    x_col = alias_column(df, x)
    y_col = alias_column(df, y)
    _ensure(df, ["time", "channel", "value", x_col, y_col])

    # numeric conversions with explicit coercion (assertive)
    work_pl = pl.from_pandas(df)
    time_expr = pl.col("time").cast(pl.Float64, strict=True)
    value_expr = pl.col("value").cast(pl.Float64, strict=False)
    value_expr = pl.when(value_expr.is_nan()).then(None).otherwise(value_expr)
    work_pl = work_pl.with_columns(
        time_expr.alias("time"),
        value_expr.alias("value"),
    )

    # choose snapshot time (optional tolerance from fig.rc or dedicated key)
    rc = (fig_kwargs or {}).get("rc", {})
    tol = (fig_kwargs or {}).get("time_tolerance", rc.get("time_tolerance", None))
    times = (
        work_pl.filter(pl.col("channel").cast(pl.Utf8) == str(channel))
        .select(pl.col("time").fill_null(float("nan")).alias("time"))
        .to_numpy()
        .ravel()
    )
    tsel = _choose_time(times, float(time), (None if tol is None else float(tol)))

    snap_pl = work_pl.filter(
        (pl.col("channel").cast(pl.Utf8) == str(channel)) & ((pl.col("time") - float(tsel)).abs() <= 1e-9)
    )
    if snap_pl.is_empty():
        raise ValueError("snapshot_heatmap: no rows after (channel,time) selection")

    # aggregate to a dense pivot (median is robust)
    pivot_pl = (
        snap_pl.group_by([y_col, x_col])
        .agg(pl.col("value").median().alias("value"))
        .sort([y_col, x_col])
        .pivot(values="value", index=y_col, columns=x_col, aggregate_function="first")
    )
    pivot = pivot_pl.to_pandas(use_pyarrow_extension_array=False)
    if y_col in pivot.columns:
        pivot = pivot.set_index(y_col)

    # Optional default ordering for treatment columns (if applicable and not overridden)
    if order_x is None and pretty_name(str(x_col)) == "treatment":
        cols = list(map(str, pivot.columns))
        lower = [c.lower() for c in cols]
        has_hint = any(
            ("pms" in c)
            or ("etoh" in c)
            or ("ethanol" in c)
            or ("ciprofloxacin" in c)
            or ("cipro" in c)
            or (c.strip() == "negative")
            for c in lower
        )
        if has_hint:

            def _rank(c: str) -> tuple[int, str]:
                v = c.lower()
                if "pms" in v:
                    return (0, v)
                if "etoh" in v or "ethanol" in v:
                    return (1, v)
                if "ciprofloxacin" in v or "cipro" in v:
                    return (2, v)
                if v.strip() == "negative" or "negative" in v:
                    return (4, v)
                return (3, v)

            ordered = sorted(cols, key=_rank)
            pivot = pivot[ordered]
            logging.getLogger("reader").info(
                "[dim]snapshot_heatmap[/dim] • treatment order applied: PMS → EtOH → ciprofloxacin → negative "
                "(override with plots.<spec_id>.with.order_x in config.yaml)"
            )

    # ordering
    if order_x:
        pivot = pivot[[c for c in order_x if c in pivot.columns]]
    if order_y:
        pivot = pivot.loc[[r for r in order_y if r in pivot.index]]

    # nothing to draw?
    if pivot.empty or pivot.shape[0] == 0 or pivot.shape[1] == 0:
        warn_if_empty(pivot, where="snapshot_heatmap", detail="after pivot")
        return []

    with use_style(rc=(fig_kwargs or {}).get("rc")):
        fig, ax = new_fig_ax(fig_kwargs)

        arr = pivot.to_numpy(dtype=float)
        cmap = _resolve_cmap((fig_kwargs or {}).get("cmap"))
        # if vmin/vmax unspecified, compute from data (ignoring NaN)
        _vmin = float(np.nanmin(arr)) if vmin is None else float(vmin)
        _vmax = float(np.nanmax(arr)) if vmax is None else float(vmax)

        im = ax.imshow(
            arr,
            aspect=("equal" if square else "auto"),
            vmin=_vmin,
            vmax=_vmax,
            cmap=cmap,
        )

        ax.grid(False)
        with suppress(Exception):
            ax.set_facecolor("white")

        # axis ticks and labels
        ax.set_xticks(range(arr.shape[1]))
        ax.set_xticklabels(list(map(str, pivot.columns)), rotation=45, ha="right")
        ax.set_yticks(range(arr.shape[0]))
        ax.set_yticklabels(list(map(str, pivot.index)))

        # Title‑case the axis labels (e.g., 'Treatment', 'Design Id')
        ax.set_xlabel(pretty_name(str(x_col)).title())
        ax.set_ylabel(pretty_name(str(y_col)).title())
        # Exact time in the title (no approximate symbol)
        ax.set_title(f"{channel} @ t={tsel:g} h")

        # grid between cells for readability
        ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # colorbar
        shrink = float((fig_kwargs or {}).get("cbar_shrink", 1.0))
        cbar_label: str | None = (fig_kwargs or {}).get("cbar_label")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=shrink, label=cbar_label)

        # ---------- collision‑resistant filename ----------
        # Base: include the channel and the *actual* time picked (tsel)
        base = filename or f"snapshot_heatmap__{channel}__t{tsel:g}h"

        # Design roster (y-axis)
        y_levels = list(map(str, pivot.index))
        n_geno = len(y_levels)

        def _short_id(s: str) -> str:
            return hashlib.blake2b(s.encode("utf-8"), digest_size=4).hexdigest()

        geno_id = _short_id("|".join(sorted(y_levels)))

        # Fingerprint settings that change the visual: x/y levels, time used, cmap, vmin/vmax, square, tol
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

        # Always append concise tags so tuning doesn’t overwrite previous files
        stub = f"{base}__gy{n_geno}-{geno_id}__fp{fp_id}"
        ext = str((fig_kwargs or {}).get("ext", "pdf")).lower()
        dpi = (fig_kwargs or {}).get("dpi", None)
        if output_dir is None:
            return [PlotFigure(fig=fig, filename=stub, ext=ext, dpi=dpi)]
        save_figure(fig, Path(output_dir), stub, ext=ext, dpi=dpi)
        plt.close(fig)
        return []
