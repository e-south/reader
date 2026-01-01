"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/snapshot_barplot.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig
from reader.plotting.microplates.snapshot_barplot import plot_snapshot_barplot


class SnapshotBarCfg(PluginConfig):
    x: str
    y: list[str] | str
    hue: str | None = None
    group_on: str | None = None
    pool_sets: str | list[dict[str, list[str]]] | None = None
    time: float = 0.0
    pool_match: Literal["exact", "contains", "startswith", "endswith", "regex"] = "exact"
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None
    agg: str = "mean"  # median|mean
    err: str = "sem"  # iqr|sem|none
    time_tolerance: float = 0.51
    panel_by: Literal["channel", "x", "group"] = "channel"
    channel_select: str | None = None
    file_by: Literal["auto", "group", "channel", "x"] = "auto"
    show_legend: bool = False
    legend_loc: str = "upper right"


class SnapshotBarplot(Plugin):
    key = "snapshot_barplot"
    category = "plot"
    ConfigModel = SnapshotBarCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none", "meta": "none"}

    def run(self, ctx, inputs, cfg: SnapshotBarCfg):
        df: pd.DataFrame = inputs["df"]

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

        files, meta = plot_snapshot_barplot(
            df=df,
            output_dir=ctx.plots_dir,
            x=cfg.x,
            y=cfg.y,
            hue=cfg.hue,
            group_on=cfg.group_on,
            pool_sets=resolved_pools,
            time=cfg.time,
            pool_match=cfg.pool_match,  # type: ignore
            fig_kwargs=cfg.fig,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
            agg=cfg.agg,
            err=cfg.err,
            time_tolerance=cfg.time_tolerance,
            panel_by=cfg.panel_by,
            channel_select=cfg.channel_select,
            file_by=cfg.file_by,
            show_legend=cfg.show_legend,
            legend_loc=cfg.legend_loc,
        )
        if not files:
            ctx.logger.warning("plot/snapshot_barplot produced no files (empty data after filtering).")
        return {"files": files, "meta": meta}
