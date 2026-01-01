"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/ts_and_snap.py

Two-panel plot (time series + snapshot barplot).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig


class TSAndSnapCfg(PluginConfig):
    # grouping
    group_on: str | None = None
    pool_sets: str | list[dict[str, list[str]]] | None = None
    pool_match: Literal["exact", "contains", "startswith", "endswith", "regex"] = "exact"

    # time series (left)
    ts_x: str = "time"
    ts_channel: str
    ts_hue: str
    ts_time_window: list[float] | None = None
    ts_add_sheet_line: bool = False
    ts_sheet_line_kwargs: dict[str, Any] = Field(default_factory=dict)
    ts_mark_snap_time: bool = False
    ts_snap_line_kwargs: dict[str, Any] = Field(default_factory=dict)
    ts_log_transform: bool | list[str] = False
    ts_ci: float = 95.0
    ts_ci_alpha: float = 0.15
    ts_show_replicates: bool = False
    ts_legend_loc: str = "upper right"

    # snapshot (right)
    snap_x: str = "treatment"
    snap_channel: str | None = None
    snap_hue: str | None = None
    snap_time: float = 0.0
    snap_agg: Literal["mean", "median"] = "mean"
    snap_err: Literal["sem", "iqr", "none"] = "sem"
    snap_time_tolerance: float = 0.51
    snap_show_legend: bool = False
    snap_legend_loc: str = "upper right"

    # figure
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None


class TSAndSnapPlot(Plugin):
    key = "ts_and_snap"
    category = "plot"
    ConfigModel = TSAndSnapCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}  # blanks not required here

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def run(self, ctx, inputs, cfg: TSAndSnapCfg):
        df: pd.DataFrame = inputs["df"]
        from reader.plotting.microplates.ts_and_snap import plot_ts_and_snap

        # --- resolve pool_sets (inline list or "<column>:<set>" reference) ---
        def _resolve_pool_sets_arg(pool_sets, group_on_col: str | None):
            if pool_sets is None:
                return None
            if isinstance(pool_sets, list):
                return pool_sets
            if isinstance(pool_sets, str):
                ref = pool_sets.strip()
                if ":" in ref:
                    col, set_name = [s.strip() for s in ref.split(":", 1)]
                else:
                    if not group_on_col:
                        raise ValueError("pool_sets reference without group_on; use '<column>:<set>'")
                    col, set_name = str(group_on_col), ref
                cat = (ctx.collections or {}).get(col)
                if not isinstance(cat, dict) or set_name not in cat:
                    opts = ", ".join(sorted((cat or {}).keys())) if isinstance(cat, dict) else "—"
                    raise ValueError(
                        f"Unknown pool_sets reference '{ref}'. "
                        "Define it under collections.<column>.<set_name> in config. "
                        f"(available for {col!r}: {opts})"
                    )
                return cat[set_name]
            raise ValueError("pool_sets must be a list[...] or '<column>:<set>' string")

        resolved_pools = _resolve_pool_sets_arg(cfg.pool_sets, cfg.group_on)

        files = plot_ts_and_snap(
            df=df,
            output_dir=ctx.plots_dir,
            group_on=cfg.group_on,
            pool_sets=resolved_pools,
            pool_match=cfg.pool_match,
            # ts (left)
            ts_x=cfg.ts_x,
            ts_channel=cfg.ts_channel,
            ts_hue=cfg.ts_hue,
            ts_time_window=cfg.ts_time_window,
            ts_add_sheet_line=cfg.ts_add_sheet_line,
            ts_sheet_line_kwargs=cfg.ts_sheet_line_kwargs,
            ts_mark_snap_time=cfg.ts_mark_snap_time,
            ts_snap_line_kwargs=cfg.ts_snap_line_kwargs,
            ts_log_transform=cfg.ts_log_transform,
            ts_ci=cfg.ts_ci,
            ts_ci_alpha=cfg.ts_ci_alpha,
            ts_show_replicates=cfg.ts_show_replicates,
            ts_legend_loc=cfg.ts_legend_loc,
            # snap (right)
            snap_x=cfg.snap_x,
            snap_channel=cfg.snap_channel,
            snap_hue=cfg.snap_hue,
            snap_time=cfg.snap_time,
            snap_agg=cfg.snap_agg,
            snap_err=cfg.snap_err,
            snap_time_tolerance=cfg.snap_time_tolerance,
            snap_show_legend=cfg.snap_show_legend,
            snap_legend_loc=cfg.snap_legend_loc,
            # fig
            fig_kwargs=cfg.fig,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
        )
        if not files:
            ctx.logger.warning("plot/ts_and_snap produced no files (empty data after filtering).")
        return {"files": files}
