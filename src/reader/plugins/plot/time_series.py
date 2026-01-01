"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/time_series.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig
from reader.plotting.microplates.time_series import plot_time_series


class TimeSeriesCfg(PluginConfig):
    x: str = "time"
    y: list[str] | None = None
    hue: str = "treatment"
    group_on: str | None = None
    pool_sets: str | list[dict[str, list[str]]] | None = None
    pool_match: Literal["exact", "contains", "startswith", "endswith", "regex"] = "exact"
    fig: dict[str, Any] = Field(default_factory=dict)
    channels: list[str] | None = None
    add_sheet_line: bool = False
    sheet_line_kwargs: dict[str, Any] = Field(default_factory=dict)
    log_transform: bool | list[str] = False
    time_window: list[float] | None = None
    ci: float = 95.0
    ci_alpha: float = 0.15
    legend_loc: str = "upper left"
    show_replicates: bool = False


class TimeSeriesPlot(Plugin):
    key = "time_series"
    category = "plot"
    ConfigModel = TimeSeriesCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1", "blanks?": "tidy.v1"}  # '?' is human hint; engine passes only present keys

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def run(self, ctx, inputs, cfg: TimeSeriesCfg):
        df: pd.DataFrame = inputs["df"]
        blanks = inputs.get("blanks", df.iloc[0:0].copy())

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
                    if not cfg.group_on:
                        raise ValueError("pool_sets reference without group_on; use '<column>:<set>'")
                    col, set_name = str(cfg.group_on), ref
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

        files = plot_time_series(
            df=df,
            blanks=blanks,
            output_dir=ctx.plots_dir,
            x=cfg.x,
            y=cfg.y,
            hue=cfg.hue,
            channels=cfg.channels,
            subplots=None,
            group_on=cfg.group_on,
            pool_sets=resolved_pools,
            pool_match=cfg.pool_match,
            fig_kwargs=cfg.fig,
            add_sheet_line=cfg.add_sheet_line,
            sheet_line_kwargs=cfg.sheet_line_kwargs,
            log_transform=cfg.log_transform,
            time_window=cfg.time_window,
            palette_book=ctx.palette_book,
            ci=cfg.ci,
            ci_alpha=cfg.ci_alpha,
            legend_loc=cfg.legend_loc,
            show_replicates=cfg.show_replicates,
        )
        if not files:
            ctx.logger.warning("plot/time_series produced no files (empty data after filtering).")
        return {"files": files}
