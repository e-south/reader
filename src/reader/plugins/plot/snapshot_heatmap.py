"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/plot/snapshot_heatmap.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig


class HeatmapCfg(PluginConfig):
    channel: str
    time: float
    x: str = "treatment"
    y: str = "genotype"
    order_x: Optional[list[str]] = None
    order_y: Optional[list[str]] = None
    square: bool = True
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    value_transform: Optional[str] = "none"   # "none" | "log2" | "log10"
    time_tolerance: float = 0.51
    fig: Dict[str, Any] = Field(default_factory=dict)
    filename: Optional[str] = None

class SnapshotHeatmapPlot(Plugin):
    key = "snapshot_heatmap"
    category = "plot"
    ConfigModel = HeatmapCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str,str]:
        # Both are optional; we derive what we need at runtime.
        # - df? : tidy.v1    (OD600, YFP, YFP/OD600, YFP/CFP, …)
        # - fc? : fold_change.v1  (FC/log2FC for specific targets like YFP/CFP)
        return {"df?": "tidy.v1", "fc?": "fold_change.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str,str]:
        return {"files": "none"}

    def run(self, ctx, inputs, cfg: HeatmapCfg):
        import numpy as np

        from reader.lib.microplates.snapshot_heatmap import plot_snapshot_heatmap
        df_in: pd.DataFrame | None = inputs.get("df")
        fc_in: pd.DataFrame | None = inputs.get("fc")

        channel = str(cfg.channel)
        wants_fc = channel.startswith("FC_") or channel.startswith("log2FC_")

        # Make a working copy of fig kwargs; ensure unified tolerance handling.
        fig_kwargs = dict(cfg.fig or {})
        fig_kwargs.setdefault("time_tolerance", cfg.time_tolerance)

        def _auto_cbar_label(ch: str, value_transform: Optional[str]) -> str:
            vtx = (value_transform or "none").lower()
            if ch.startswith("log2FC_"):
                return f"log2FC ({ch.split('_', 1)[1]})"
            if ch.startswith("FC_"):
                return f"FC ({ch.split('_', 1)[1]})"
            if vtx == "log2":
                return f"log2({ch})"
            if vtx == "log10":
                return f"log10({ch})"
            return ch

        if wants_fc:
            if fc_in is None:
                raise ValueError(f"snapshot_heatmap: channel={channel!r} requires a fold_change.v1 input (reads.fc)")
            # Derive tidy-like frame from the FC table
            target = channel.split("_", 1)[1]  # after "FC_" or "log2FC_"
            use_col = "log2FC" if channel.startswith("log2FC_") else "FC"

            tab = fc_in.copy()
            tab = tab[tab["target"].astype(str) == target]
            if tab.empty:
                raise ValueError(f"snapshot_heatmap: fold_change table has no rows for target={target!r}")
            # pick nearest available FC time to cfg.time within tolerance
            times = np.asarray(sorted(pd.to_numeric(tab["time"], errors="coerce").dropna().unique()), float)
            if times.size == 0:
                raise ValueError("snapshot_heatmap: fold_change table has no time values")
            deltas = np.abs(times - float(cfg.time))
            j = int(np.nanargmin(deltas))
            tsel = float(times[j])
            if deltas[j] > float(cfg.time_tolerance):
                # INFO (not WARNING) per request; clearly highlight in styling.
                ctx.logger.info(
                    "[warn]snapshot_heatmap[/warn] • requested t=%.2f h; nearest available t=%.2f h (Δ=%.2f h) — using nearest",
                    float(cfg.time), float(tsel), float(deltas[j])
                )
            sub = tab[pd.to_numeric(tab["time"], errors="coerce") == tsel].copy()
            # Build a tidy-like dataframe that plot_snapshot_heatmap expects:
            # time, channel, value, treatment, genotype(,_alias)…
            sub = sub.rename(columns={use_col: "value"})
            sub["channel"] = channel
            # keep only necessary columns (+ any alias columns)
            keep = ["time", "channel", "value", "treatment", "genotype", "genotype_alias"]
            df = sub[[c for c in keep if c in sub.columns]].copy()

            # Choose a descriptive default filename that includes the time actually used.
            filename = cfg.filename or f"snapshot_heatmap__{channel}__t{tsel:g}h"
            fig_kwargs.setdefault("cbar_label", _auto_cbar_label(channel, None))
        else:
            if df_in is None:
                raise ValueError("snapshot_heatmap: tidy df is required when channel is not FC_/log2FC_-prefixed")
            # Optional: apply log transform to the selected tidy channel
            df = df_in.copy()
            if cfg.value_transform and str(cfg.value_transform).lower() in {"log2", "log10"}:
                mask = df["channel"].astype(str) == channel
                vals = pd.to_numeric(df.loc[mask, "value"], errors="coerce")
                if str(cfg.value_transform).lower() == "log2":
                    df.loc[mask, "value"] = np.where(vals > 0, np.log2(vals), np.nan)
                else:  # log10
                    df.loc[mask, "value"] = np.where(vals > 0, np.log10(vals), np.nan)
            # For tidy mode, the library time chooser now logs & proceeds as well.
            filename = cfg.filename  # let the library default if not provided
            fig_kwargs.setdefault("cbar_label", _auto_cbar_label(channel, cfg.value_transform))
        plot_snapshot_heatmap(
            df=df,
            blanks=df.iloc[0:0] if isinstance(df, pd.DataFrame) else pd.DataFrame(columns=["time","channel","value"]),
            output_dir=ctx.plots_dir,
            channel=channel,
            time=cfg.time,
            x=cfg.x, y=cfg.y,
            order_x=cfg.order_x, order_y=cfg.order_y,
            square=cfg.square, vmin=cfg.vmin, vmax=cfg.vmax,
            fig_kwargs=fig_kwargs,
            filename=filename,
        )
        return {"files": None}
