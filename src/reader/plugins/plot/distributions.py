"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/distributions.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig
from reader.plotting.microplates.distributions import plot_distributions


class DistributionsCfg(PluginConfig):
    # what to draw
    channels: list[str]
    # modern grouping
    group_on: str | None = "design_id"
    pool_sets: str | list[str] | list[dict[str, list[str]]] | None = None
    pool_match: Literal["exact", "contains", "startswith", "endswith", "regex"] = "exact"
    # layout
    panel_by: Literal["channel", "group"] = "channel"  # default: per-channel panels
    hue: str | None = None
    legend_loc: Literal["upper left", "upper right", "lower left", "lower right", "center", "best"] = "upper left"
    # style/output
    fig: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None


class DistributionsPlot(Plugin):
    key = "distributions"
    category = "plot"
    ConfigModel = DistributionsCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        # blanks is optional; if absent we’ll draw only data fills
        return {"df": "tidy.v1", "blanks?": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def run(self, ctx, inputs, cfg: DistributionsCfg):
        df: pd.DataFrame = inputs["df"]
        blanks: pd.DataFrame = inputs.get("blanks", df.iloc[0:0])

        # --- Resolve pool_sets (DRY via experiment.collections) ----------------
        # Accept:
        #   • inline list[dict]                       (pass-through)
        #   • "col:set"                               (single reference)
        #   • list[str] of references ["col:set", ...] (union/concatenate)
        def _resolve_pool_sets_arg(pool_sets, group_on_col: str | None):
            if pool_sets is None:
                return None
            # already a concrete list of dicts
            if isinstance(pool_sets, list) and pool_sets and isinstance(pool_sets[0], dict):
                return pool_sets  # type: ignore[return-value]
            # a single string or a list of strings → look up in ctx.collections
            refs: list[str]
            if isinstance(pool_sets, str):
                refs = [pool_sets]
            elif isinstance(pool_sets, list):
                refs = [str(x) for x in pool_sets]
            else:
                raise ValueError("pool_sets must be a list[dict], a 'col:set' string, or list[str] of references")

            out: list[dict[str, list[str]]] = []
            for ref in refs:
                if ":" in ref:
                    col, set_name = [s.strip() for s in ref.split(":", 1)]
                else:
                    if not group_on_col:
                        raise ValueError("pool_sets reference without group_on; use '<column>:<set>'")
                    col, set_name = str(group_on_col), ref.strip()
                cat = (ctx.collections or {}).get(col)
                if not isinstance(cat, dict) or set_name not in cat:
                    opts = ", ".join(sorted((cat or {}).keys())) if isinstance(cat, dict) else "—"
                    raise ValueError(
                        f"Unknown pool_sets reference '{ref}'. "
                        f"Define it under collections.{col}.{set_name} in config. "
                        f"(available for {col!r}: {opts})"
                    )
                # concatenate lists of {label:[members], ...}
                sets_list = cat[set_name]
                if not isinstance(sets_list, list):
                    raise ValueError(f"collections.{col}.{set_name} must be a list of single-key dict objects")
                out.extend(sets_list)
            return out

        resolved_pools = _resolve_pool_sets_arg(cfg.pool_sets, cfg.group_on)

        files = plot_distributions(
            df=df,
            blanks=blanks,
            output_dir=ctx.plots_dir,
            channels=cfg.channels,
            group_on=cfg.group_on,
            pool_sets=resolved_pools,
            pool_match=cfg.pool_match,  # type: ignore[arg-type]
            panel_by=cfg.panel_by,
            hue=cfg.hue,
            legend_loc=cfg.legend_loc,
            fig_kwargs=cfg.fig,
            filename=cfg.filename,
            palette_book=ctx.palette_book,
        )
        if not files:
            ctx.logger.warning("plot/distributions produced no files (empty data after filtering).")
        return {"files": files}
