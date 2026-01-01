"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/ts_snap_plus_design.py

Compound plot: ts+snap (top) + baserender design panel (bottom).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig

if TYPE_CHECKING:
    from reader.plotting.microplates.ts_snap_plus_design import BaseRenderDatasetSpec, _BaseRenderProvider


class DesignBlock(PluginConfig):
    # Provide exactly one of:
    job_yaml: str | None = None
    dataset: dict[str, Any] | None = None  # see baserender README: {path,format,columns,alphabet,annotations}
    # Optional plugin specs (only when dataset is provided; job_yaml carries its own)
    plugins: list[str | dict[str, Any]] = Field(default_factory=list)
    # Optional baserender style overrides (merged over job.style)
    style: dict[str, Any] = Field(default_factory=dict)
    select_by: Literal["id", "sequence"] = "id"
    tidy_key_column: str = "id"
    height_px: int = 220
    dpi: int = 150
    fmt: Literal["png", "svg", "pdf"] = "png"


class TSSnapPlusDesignCfg(PluginConfig):
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

    # bottom panel
    design: DesignBlock


class TSSnapPlusDesignPlot(Plugin):
    key = "ts_snap_plus_design"
    category = "plot"
    ConfigModel = TSSnapPlusDesignCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def run(self, ctx, inputs, cfg: TSSnapPlusDesignCfg):
        df: pd.DataFrame = inputs["df"]
        from reader.plotting.microplates.ts_snap_plus_design import (
            BaseRenderDatasetSpec,
            _BaseRenderProvider,
            plot_ts_snap_plus_design,
        )

        # Resolve pool_sets reference (identical policy to ts_and_snap)
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

        # Build strict baserender spec
        br = cfg.design
        # Fail fast: exactly one of job_yaml OR dataset must be provided.
        if bool(br.job_yaml) == bool(br.dataset):
            raise ValueError("design: provide exactly ONE of 'job_yaml' or 'dataset' (not both / not neither).")
        spec = BaseRenderDatasetSpec(
            job_yaml=Path(br.job_yaml).expanduser().resolve() if br.job_yaml else None,
            dataset=br.dataset,
            plugins=tuple(br.plugins),
            style=br.style,
            select_by=br.select_by,
            tidy_key_column=br.tidy_key_column,
            height_px=int(br.height_px),
            dpi=int(br.dpi),
            fmt=br.fmt,
        )
        provider = _BaseRenderProvider(spec)

        files = plot_ts_snap_plus_design(
            df=df,
            output_dir=ctx.plots_dir,
            # group
            group_on=cfg.group_on,
            pool_sets=resolved_pools,
            pool_match=cfg.pool_match,
            # ts
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
            # snap
            snap_x=cfg.snap_x,
            snap_channel=cfg.snap_channel,
            snap_hue=cfg.snap_hue,
            snap_time=cfg.snap_time,
            snap_agg=cfg.snap_agg,
            snap_err=cfg.snap_err,
            snap_time_tolerance=cfg.snap_time_tolerance,
            snap_show_legend=cfg.snap_show_legend,
            snap_legend_loc=cfg.snap_legend_loc,
            # bottom design
            design_key_column=br.tidy_key_column,
            design_provider=provider,
            # figure
            fig_kwargs=cfg.fig,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
        )
        if not files:
            ctx.logger.warning("plot/ts_snap_plus_design produced no files (empty data after filtering).")
        return {"files": files}
