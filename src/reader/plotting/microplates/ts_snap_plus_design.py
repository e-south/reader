"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plotting/microplates/ts_snap_plus_design.py

Compound plot: (top) time series + snapshot, (bottom) baserender design panel.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from PIL import Image

try:
    from dnadesign.baserender.src import api as br_api
    from dnadesign.baserender.src.presets.loader import load_job as br_load_job
except Exception:
    br_api = None
    br_load_job = None

# reuse existing helpers to keep behavior identical to ts_and_snap
from .base import (
    GroupMatch,
    alias_column,
    nearest_time_per_key,
    resolve_groups,
    save_figure,
    smart_grouped_dose_key,
    smart_string_numeric_key,
    summarize_time_usage,
)
from .style import _DEFAULT_RC as _RC
from .style import PaletteBook, use_style

# ---------- baserender adapter (strict & decoupled) ---------------------------


@dataclass(frozen=True)
class BaseRenderDatasetSpec:
    # Exactly one of job_yaml OR dataset must be provided
    job_yaml: Path | None = None
    dataset: Mapping[str, Any] | None = None  # {"path", "format", "columns", "alphabet", "annotations"}
    plugins: Sequence[Mapping[str, Any] | str] = ()
    style: Mapping[str, Any] = None  # optional overrides merged over job.style
    select_by: Literal["id", "sequence"] = "id"
    tidy_key_column: str = "id"
    # bottom panel sizing (in pixels)
    height_px: int = 220
    dpi: int = 150
    fmt: Literal["png", "svg", "pdf"] = "png"


class _BaseRenderProvider:
    """
    Lazy, cached adapter around dnadesign.baserender to render one record by id/sequence.
    Errors are assertive, never silent.
    """

    def __init__(self, spec: BaseRenderDatasetSpec):
        if br_api is None or br_load_job is None:
            raise ImportError(
                "dnadesign is required for design rendering. "
                "Install the optional extra: `uv sync --extra design` or `pip install 'reader[design]'`."
            )

        self._api = br_api
        self._load_job = br_load_job
        self._spec = spec
        self._job = None
        self._records_indexed = False
        self._by_id: dict[str, Any] = {}
        self._by_seq: dict[str, Any] = {}

    def _ensure_job_or_dataset(self):
        s = self._spec
        has_job = s.job_yaml is not None
        has_ds = s.dataset is not None
        if (has_job + has_ds) != 1:
            raise ValueError("Provide exactly one of design.job_yaml or design.dataset.")
        if has_job:
            p = Path(s.job_yaml).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"design.job_yaml not found: {p}")
            if self._job is None:
                self._job = self._load_job(p)
        else:
            # dataset block must be well-formed
            ds = dict(s.dataset or {})
            required = ("path", "format", "columns")
            miss = [k for k in required if k not in ds]
            if miss:
                raise ValueError(f"design.dataset missing keys: {miss}")
            pp = Path(ds["path"]).expanduser().resolve()
            if not pp.exists():
                raise FileNotFoundError(f"design.dataset.path not found: {pp}")

    def _records_iter(self) -> Iterable[Any]:
        """Yield SeqRecord(s) from a job OR from a dataset spec."""
        self._ensure_job_or_dataset()
        if self._job is not None:
            j = self._job
            return self._api.read_records(
                j.input_path,
                format=j.format,
                sequence_col=j.seq_col,
                annotations_col=j.ann_col,
                id_col=j.id_col,
                alphabet=j.alphabet,
                plugins=tuple(j.plugins),
                ann_policy=j.ann_policy,
            )
        ds = dict(self._spec.dataset or {})
        cols = dict(ds["columns"])
        return self._api.read_records(
            Path(ds["path"]),
            format=str(ds.get("format", "parquet")),
            sequence_col=str(cols.get("sequence", "sequence")),
            annotations_col=str(cols.get("annotations", "densegen__used_tfbs_detail")),
            id_col=str(cols.get("id", "id")),
            alphabet=str(ds.get("alphabet", "DNA")),
            plugins=tuple(self._spec.plugins),
            ann_policy=(ds.get("annotations") or None),
        )

    def _effective_style(self) -> Mapping[str, Any]:
        # job.style merged with user override (shallow is fine; Style fills defaults)
        base = {}
        if self._job is not None:
            base = dict(self._job.style or {})
        override = dict(self._spec.style or {})
        out = dict(base)
        out.update(override)
        return out

    def _index_all(self):
        if self._records_indexed:
            return
        # Index both id and sequence to support explicit, controlled fallback at lookup time.
        by_id: dict[str, Any] = {}
        by_seq: dict[str, Any] = {}
        count = 0
        for r in self._records_iter():
            count += 1
            if r.id:
                by_id[str(r.id)] = r
            # Always index by sequence; memory overhead is acceptable for 10^4–10^5 60‑bp rows.
            by_seq[str(r.sequence)] = r

        self._by_id, self._by_seq = by_id, by_seq
        self._records_indexed = True
        if count == 0:
            raise RuntimeError("Baserender dataset produced zero records.")

    def render_image_array(self, key: str) -> np.ndarray:
        """
        Return an RGBA numpy array for the requested record (pixels), ready to imshow.
        """
        self._index_all()
        k = str(key)
        if self._spec.select_by == "id":
            rec = self._by_id.get(k)
            lookup_mode = "id"
        else:
            rec = self._by_seq.get(k)
            lookup_mode = "sequence"
        # Controlled, one-step fallback to the alternate keyspace (assertive: not silent).
        if rec is None:
            if self._spec.select_by == "id":
                rec = self._by_seq.get(k)
                tried = "id→sequence"
            else:
                rec = self._by_id.get(k)
                tried = "sequence→id"
            if rec is None:
                available = list(self._by_id.keys()) if self._spec.select_by == "id" else list(self._by_seq.keys())
                preview = ", ".join(available[:6]) + (" …" if len(available) > 6 else "")
                raise KeyError(
                    f"Baserender record not found by {lookup_mode}={k!r} (also tried {tried}). "
                    f"Check tidy→dataset mapping. Available preview: [{preview}]"
                )
            else:
                lookup_mode = f"{lookup_mode} (fallback)"

        # Render a single still to an in-memory figure, then to a numpy RGBA array.
        fig = self._api.render_image(
            rec,
            out_path=None,
            fmt=self._spec.fmt,  # image backend choice does not affect imshow
            style=self._effective_style(),
        )
        # Force pixel size (height_px); keep aspect by scaling canvas
        target_h = int(self._spec.height_px)
        dpi = int(self._spec.dpi)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())
        # Letterbox to exactly target height (width auto) with white background
        h, w = arr.shape[:2]
        if h != target_h:
            scale = target_h / float(h)
            img = Image.fromarray(arr)
            new_w = max(1, int(round(w * scale)))
            img = img.resize((new_w, target_h), resample=Image.Resampling.LANCZOS)
            arr = np.asarray(img.convert("RGBA"))
        plt.close(fig)
        return arr


# ---------- ts+snap helpers (reuse from ts_and_snap) --------------------------

_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "H"]


def _colors_for(n: int, palette_book: PaletteBook | None) -> list[str]:
    if palette_book:
        if n == 1:
            pal = palette_book.colors(2)
            return [pal[1]] if (pal and str(pal[0]).lower() in {"#000000", "black"}) else [pal[0]]
        return palette_book.colors(n)
    cyc = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cyc:
        raise RuntimeError("No color cycle available; configure Matplotlib rcParams or provide a PaletteBook.")
    if n == 1 and str(cyc[0]).lower() in {"#000000", "black"} and len(cyc) > 1:
        return [cyc[1]]
    return cyc[:n]


def _order_levels(levels: list[str]) -> list[str]:
    try:
        return sorted(levels, key=smart_grouped_dose_key)
    except Exception:
        return sorted(levels, key=smart_string_numeric_key)


# ---------- main entry --------------------------------------------------------


def plot_ts_snap_plus_design(
    *,
    df: pd.DataFrame,
    output_dir: Path | str,
    # grouping
    group_on: str | None,
    pool_sets: list[dict[str, list[str]]] | None,
    pool_match: GroupMatch = "exact",
    # time series (left)
    ts_x: str = "time",
    ts_channel: str,
    ts_hue: str,
    ts_time_window: list[float] | None = None,
    ts_add_sheet_line: bool = False,
    ts_sheet_line_kwargs: dict | None = None,
    ts_mark_snap_time: bool = False,
    ts_snap_line_kwargs: dict | None = None,
    ts_log_transform: bool | list[str] = False,
    ts_ci: float = 95.0,
    ts_ci_alpha: float = 0.15,
    ts_show_replicates: bool = False,
    ts_legend_loc: str = "upper right",
    # snapshot (right)
    snap_x: str = "treatment",
    snap_channel: str | None = None,
    snap_hue: str | None = None,
    snap_time: float = 0.0,
    snap_agg: str = "mean",
    snap_err: str = "sem",
    snap_time_tolerance: float = 0.51,
    snap_show_legend: bool = False,
    snap_legend_loc: str = "upper right",
    # design (bottom)
    design_key_column: str,
    design_provider: _BaseRenderProvider,
    # figure/style
    fig_kwargs: dict | None = None,
    filename: str | None = None,
    palette_book: PaletteBook | None = None,
) -> tuple[list[Path], dict]:
    """
    Top: time series + snapshot (same semantics as plot_ts_and_snap)
    Bottom: one baserender panel resolved via tidy column `design_key_column` (id or sequence).
    """
    if snap_agg not in {"mean", "median"}:
        raise ValueError("snap_agg must be 'mean' or 'median'")
    if snap_err not in {"sem", "iqr", "none"}:
        raise ValueError("snap_err must be 'sem', 'iqr', or 'none'")

    fig_kwargs = dict(fig_kwargs or {})
    out_dir = Path(output_dir)

    # Resolve columns (prefer *_alias)
    ts_x_col = alias_column(df, ts_x)
    group_col = alias_column(df, group_on) if group_on else None
    ts_hue_col = alias_column(df, ts_hue)
    snap_x_col = alias_column(df, snap_x)
    snap_hue_col = alias_column(df, snap_hue) if snap_hue else None
    # --- Robust design key selection: prefer requested → 'id' → 'sequence'
    tried: list[str] = []
    chosen_key_col: str | None = None
    chosen_key_name: str | None = None
    for candidate in [design_key_column, "id", "sequence"]:
        if candidate is None:
            continue
        kc = alias_column(df, candidate)
        if kc not in tried:
            tried.append(kc)
        if kc in df.columns and df[kc].notna().any():
            chosen_key_col = kc
            chosen_key_name = candidate
            break

    for col in [ts_x_col, ts_hue_col]:
        if col not in df.columns:
            raise ValueError(f"Required column not present in tidy data: {col!r}")

    ch_ts = str(ts_channel)
    ch_snap = str(snap_channel if snap_channel else ts_channel)

    work = df.copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["time"] = pd.to_numeric(work["time"], errors="coerce")
    if ts_time_window:
        lo, hi = float(ts_time_window[0]), float(ts_time_window[1])
        work = work[
            (pd.to_numeric(work[ts_x_col], errors="coerce") >= lo)
            & (pd.to_numeric(work[ts_x_col], errors="coerce") <= hi)
        ].copy()
    if work.empty:
        meta = {
            "time_selection": {
                "requested": float(snap_time),
                "tolerance": float(snap_time_tolerance),
                "fallback_used": False,
                "used": None,
            }
        }
        return [], meta

    # Figure iteration over groups
    if group_col:
        universe = work[group_col].astype(str).unique().tolist()
        fig_groups = (
            resolve_groups(universe, pool_sets, match=pool_match) if pool_sets else [(g, [g]) for g in universe]
        )
    else:
        fig_groups = [("all", [None])]

    saved: list[Path] = []
    used_times: list[float] = []
    fallback_used_any = False
    for label, members in fig_groups:
        d = work.copy()
        if group_col and members != [None]:
            d = d[d[group_col].astype(str).isin(members)]
        if d.empty:
            continue

        # Determine the single design key for this figure (assertive when possible).
        design_key: str | None = None
        if chosen_key_col:
            keys = d[chosen_key_col].dropna().astype(str).unique().tolist()
            if len(keys) != 1:
                raise ValueError(
                    f"Expected exactly 1 unique '{chosen_key_name}' per figure, got {len(keys)} "
                    f"({keys[:6]}{' …' if len(keys) > 6 else ''}) for group={label!r}."
                )
            design_key = str(keys[0])
        else:
            warnings.warn(
                "Design panel disabled: none of the key columns are available with values in this tidy data. "
                f"Tried: {tried or ['<none>']}. Top panels will still be rendered.",
                stacklevel=2,
            )

        # -------------------- Layout & axes
        with use_style(rc=fig_kwargs.get("rc"), color_cycle=None):
            # Default base size; widen for 2 columns
            base_w, base_h = _RC["figure_figsize"]
            if "figsize" not in fig_kwargs:
                fig_kwargs["figsize"] = (base_w * 2.0, base_h * 1.35)  # slightly taller for the bottom panel
            fig = plt.figure(**{k: v for k, v in fig_kwargs.items() if k not in {"rc", "ext"}})
            gs = GridSpec(nrows=2, ncols=2, height_ratios=[3.0, 1.4], hspace=0.30, wspace=0.25, figure=fig)
            ax_ts = fig.add_subplot(gs[0, 0])  # top-left
            ax_snap = fig.add_subplot(gs[0, 1])  # top-right
            ax_design = fig.add_subplot(gs[1, :])  # bottom (span both)

            # -------------------- Top-left: time series
            ts = d[d["channel"].astype(str) == ch_ts].copy()
            if not ts.empty:
                hue_levels_ts = list(ts[ts_hue_col].astype(str).unique())
                marker_map = {h: _MARKERS[i % len(_MARKERS)] for i, h in enumerate(hue_levels_ts)}
                colors = _colors_for(len(hue_levels_ts), palette_book)
                color_map = {h: colors[i % len(colors)] for i, h in enumerate(hue_levels_ts)}

                ax = ax_ts
                ax.grid(False)
                ax.yaxis.grid(True, which="major")
                ax.xaxis.grid(True, which="major")

                if ts_show_replicates:
                    for h in hue_levels_ts:
                        rr = ts[ts[ts_hue_col].astype(str) == h]
                        ax.scatter(
                            rr[ts_x_col],
                            rr["value"],
                            s=18,
                            alpha=float(fig_kwargs.get("replicate_alpha", 0.30)),
                            zorder=3,
                            linewidths=0.0,
                            edgecolors="none",
                            marker=marker_map[h],
                            c=color_map[h],
                        )

                sns.lineplot(
                    data=ts,
                    x=ts_x_col,
                    y="value",
                    hue=ts_hue_col,
                    hue_order=hue_levels_ts,
                    estimator="mean",
                    errorbar=("ci", float(ts_ci)),
                    err_style="band",
                    err_kws={"alpha": float(ts_ci_alpha)},
                    lw=1.8,
                    alpha=float(fig_kwargs.get("line_alpha", 0.85)),
                    legend=False,
                    ax=ax,
                    palette=[color_map[h] for h in hue_levels_ts],
                    marker=None,
                    zorder=1,
                )

                means = ts.groupby([ts_hue_col, ts_x_col], dropna=False)["value"].mean().reset_index()
                for h in hue_levels_ts:
                    mm = means[means[ts_hue_col].astype(str) == h]
                    ax.scatter(
                        mm[ts_x_col],
                        mm["value"],
                        s=36,
                        zorder=2.5,
                        marker=marker_map[h],
                        alpha=float(fig_kwargs.get("mean_marker_alpha", 0.75)),
                        edgecolors="none",
                        linewidths=0.0,
                        c=color_map[h],
                    )

                if ts_add_sheet_line and "sheet_index" in ts.columns:
                    starts = sorted(ts.groupby("sheet_index")[ts_x_col].min().dropna().tolist())
                    trans = starts[1:] if len(starts) > 1 else []
                    style = {"color": "#9E9E9E", "linestyle": "--", "linewidth": 0.8, "alpha": 0.9, "zorder": 0.5}
                    style.update(ts_sheet_line_kwargs or {})
                    for sx in trans:
                        ax.axvline(float(sx), **style)

                if ts_mark_snap_time:
                    sstyle = {"color": "#9E9E9E", "linestyle": "--", "linewidth": 0.9, "alpha": 1.0, "zorder": 0.8}
                    sstyle.update(ts_snap_line_kwargs or {})
                    ax.axvline(float(snap_time), **sstyle)

                if isinstance(ts_log_transform, list):
                    ax.set_yscale("log" if (ch_ts in ts_log_transform) else "linear")
                else:
                    if bool(ts_log_transform):
                        ax.set_yscale("log")

                ax.set_xlabel("Time (h)" if str(ts_x).lower() == "time" else ts_x_col)
                ax.set_ylabel(ch_ts)

                handles = [
                    Line2D(
                        [0],
                        [0],
                        color=color_map[h],
                        marker=marker_map[h],
                        markersize=7,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=float(fig_kwargs.get("mean_marker_alpha", 0.75)),
                        label=str(h),
                    )
                    for h in hue_levels_ts
                ]
                ax.legend(handles=handles, loc=str(ts_legend_loc), title=None)

            # -------------------- Top-right: snapshot
            snap = d.copy()
            key_cols = [c for c in [group_col, snap_x_col, snap_hue_col, "channel", "position"] if c]
            fallback_used = False
            snapped = nearest_time_per_key(
                snap, target_time=float(snap_time), keys=key_cols, tol=float(snap_time_tolerance)
            )
            snapped = snapped[snapped["channel"].astype(str) == ch_snap].copy()
            if snapped.empty:
                log = logging.getLogger("reader")
                fb = nearest_time_per_key(snap, target_time=float(snap_time), keys=key_cols, tol=float("inf"))
                fallback_used = True
                snapped = fb[fb["channel"].astype(str) == ch_snap].copy()
                if not snapped.empty:
                    uniq = sorted(pd.to_numeric(snapped["time"], errors="coerce").dropna().unique().tolist())
                    t_rep = uniq[0] if len(uniq) == 1 else float(pd.Series(uniq).median())
                    delta = abs(float(t_rep) - float(snap_time))
                    preview = ", ".join(f"{t:.2f}" for t in uniq[:6]) + (" …" if len(uniq) > 6 else "")
                    log.info(
                        "[warn]ts_snap_plus_design:snapshot[/warn] • requested t=%.2f h; no rows within ±%.2f h — "
                        "using nearest available per key (times=%s; Δ≈%.2f h)",
                        float(snap_time),
                        float(snap_time_tolerance),
                        preview,
                        float(delta),
                    )
            if fallback_used:
                fallback_used_any = True
            if not snapped.empty:
                used_times.extend(pd.to_numeric(snapped["time"], errors="coerce").dropna().tolist())
            if not snapped.empty:
                ax = ax_snap
                base_group_cols: list[str] = [snap_x_col] + ([snap_hue_col] if snap_hue_col else [])
                stats = (
                    snapped.groupby(base_group_cols, dropna=False)["value"]
                    .agg(n="count", mean="mean", median="median", std="std", sem="sem")
                    .reset_index()
                )

                if snap_err == "iqr":
                    q = (
                        snapped.groupby(base_group_cols, dropna=False)["value"]
                        .quantile([0.25, 0.75])
                        .unstack(-1)
                        .reset_index()
                        .rename(columns={0.25: "q1", 0.75: "q3"})
                    )
                    stats = stats.merge(q, on=base_group_cols, how="left")

                x_levels = stats[snap_x_col].astype(str).unique().tolist()
                x_order = _order_levels(x_levels)
                hue_levels_snap = (
                    sorted(stats[snap_hue_col].astype(str).unique().tolist(), key=smart_string_numeric_key)
                    if snap_hue_col
                    else ["_single"]
                )
                colors = _colors_for(len(hue_levels_snap), palette_book)
                color_map_snap = {h: colors[i % len(colors)] for i, h in enumerate(hue_levels_snap)}
                n_x = len(x_order)
                base_pos = np.arange(n_x, dtype=float)
                num_hues = len(hue_levels_snap) if snap_hue_col else 1
                has_hue = snap_hue_col is not None and num_hues > 1
                width = 0.8 if not has_hue else min(0.85 / max(num_hues, 1), 0.8)
                offsets = (np.arange(num_hues) - (num_hues - 1) / 2.0) * width if has_hue else np.array([0.0])
                hue_index = {h: i for i, h in enumerate(hue_levels_snap)}

                ax.grid(False)
                ax.yaxis.grid(True, which="major")
                ax.xaxis.grid(False)
                legend_handles: dict[str, object] = {}

                for j, xv in enumerate(x_order):
                    x_center = base_pos[j]
                    for h in hue_levels_snap:
                        sub = stats[stats[snap_x_col].astype(str) == str(xv)]
                        if snap_hue_col:
                            sub = sub[sub[snap_hue_col].astype(str) == str(h)]
                        if sub.empty:
                            continue
                        row = sub.iloc[0]
                        height = float(row[snap_agg])

                        yerr = None
                        if snap_err == "sem":
                            ev = float(row.get("sem", np.nan))
                            yerr = None if not np.isfinite(ev) else ev
                        elif snap_err == "iqr":
                            q1 = row.get("q1", np.nan)
                            q3 = row.get("q3", np.nan)
                            if np.isfinite(q1) and np.isfinite(q3):
                                if snap_agg == "median":
                                    lo = max(height - float(q1), 0.0)
                                    hi = max(float(q3) - height, 0.0)
                                    yerr = np.vstack([[lo], [hi]])  # asym
                                else:
                                    half = max(0.5 * (float(q3) - float(q1)), 0.0)
                                    yerr = half

                        error_kw = {"capsize": 3, "elinewidth": 1.0, "alpha": 0.9} if yerr is not None else None
                        xpos = x_center + offsets[hue_index[h]]
                        bar_color = color_map_snap[h] if snap_hue_col else "#D9D9D9"
                        bars = ax.bar(
                            [xpos],
                            [height],
                            width=width,
                            color=bar_color,
                            edgecolor="#C0C0C0",
                            zorder=1,
                            yerr=yerr,
                            **({"error_kw": error_kw} if error_kw else {}),
                            label=(str(h) if (snap_show_legend and h not in legend_handles) else None),
                        )
                        if snap_show_legend and h not in legend_handles and len(bars.patches) > 0:
                            legend_handles[h] = bars.patches[0]

                        rr = snapped[snapped[snap_x_col].astype(str) == str(xv)]
                        if snap_hue_col:
                            rr = rr[rr[snap_hue_col].astype(str) == str(h)]
                        if not rr.empty:
                            rng = np.random.default_rng()
                            jitter = float(width) * (0.08 if has_hue else 0.12)
                            xj = xpos + (rng.random(len(rr)) - 0.5) * (2.0 * jitter)
                            ax.scatter(
                                xj,
                                rr["value"],
                                s=34,
                                zorder=3,
                                facecolors="#FFFFFF",
                                edgecolors="#C0C0C0",
                                linewidths=0.7,
                                color=None,
                            )

                ax.set_xticks(base_pos)
                ax.set_xticklabels(x_order, rotation=45, ha="right")
                ax.set_xlabel("")
                ax.set_ylabel(ch_snap)
                ax.set_title(f"t≈{float(snap_time):.2f} h", fontweight="normal")
                if snap_show_legend and snap_hue_col and len(hue_levels_snap) > 1 and legend_handles:
                    ax.legend(
                        handles=list(legend_handles.values()),
                        labels=list(legend_handles.keys()),
                        loc=str(snap_legend_loc),
                        title=None,
                    )

            # -------------------- Bottom: baserender panel
            if design_key is not None:
                try:
                    arr = design_provider.render_image_array(design_key)
                    ax_design.imshow(arr, interpolation="nearest", origin="upper")
                    ax_design.set_axis_off()
                    ax_design.set_title(f"Design: {design_key}", fontsize=10, loc="left")
                except KeyError as e:
                    warnings.warn(f"Design panel skipped for key={design_key!r}: {e}", stacklevel=2)
                    ax_design.set_axis_off()
                    ax_design.text(
                        0.01,
                        0.50,
                        f"Design unavailable for key: {design_key}",
                        transform=ax_design.transAxes,
                        ha="left",
                        va="center",
                        fontsize=10,
                        alpha=0.9,
                    )
            else:
                ax_design.set_axis_off()
                ax_design.text(
                    0.01,
                    0.50,
                    "Design panel disabled (no id/sequence column).",
                    transform=ax_design.transAxes,
                    ha="left",
                    va="center",
                    fontsize=10,
                    alpha=0.9,
                )

            # -------------------- Title + save
            fig.suptitle(f"{label}", y=float(fig_kwargs.get("suptitle_y", 0.995)))
            ext = str(fig_kwargs.get("ext", "pdf")).lower()

            # unique filename (like ts_and_snap)
            group_tag = None
            if group_col and members != [None]:
                group_tag = f"__{str(group_col)}={str(label)}"
            if filename:
                stub = f"{filename}{group_tag}" if group_tag else filename
            else:
                stub = f"ts_snap_plus_design__{ch_snap}{group_tag if group_tag else f'__{label}'}"
            saved.append(save_figure(fig, out_dir, stub, ext=ext))
            plt.close(fig)
    time_meta = summarize_time_usage(pd.DataFrame({"time": used_times})) if used_times else None
    meta = {
        "time_selection": {
            "requested": float(snap_time),
            "tolerance": float(snap_time_tolerance),
            "fallback_used": fallback_used_any,
            "used": time_meta,
        }
    }
    return saved, meta
