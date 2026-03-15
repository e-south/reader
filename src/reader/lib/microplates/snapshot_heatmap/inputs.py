from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from reader.lib.microplates.support import choose_nearest_time


def auto_cbar_label(channel: str, value_transform: str | None) -> str:
    transform = (value_transform or "none").lower()
    if channel.startswith("log2FC_"):
        return f"log2FC ({channel.split('_', 1)[1]})"
    if channel.startswith("FC_"):
        return f"FC ({channel.split('_', 1)[1]})"
    if transform == "log2":
        return f"log2({channel})"
    if transform == "log10":
        return f"log10({channel})"
    return channel


def _transform_positive_values(values: pd.Series, *, transform: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    positive = numeric > 0
    if transform == "log2":
        result.loc[positive] = np.log2(numeric.loc[positive])
        return result
    result.loc[positive] = np.log10(numeric.loc[positive])
    return result


def prepare_snapshot_heatmap_inputs(
    *,
    ctx,
    df_in: pd.DataFrame | None,
    fc_in: pd.DataFrame | None,
    cfg: Any,
) -> dict[str, Any]:
    channel = str(cfg.channel)
    wants_fc = channel.startswith("FC_") or channel.startswith("log2FC_")
    fig_kwargs = dict(cfg.fig or {})
    fig_kwargs.setdefault("time_tolerance", cfg.time_tolerance)

    if wants_fc:
        if fc_in is None:
            raise ValueError(f"snapshot_heatmap: channel={channel!r} requires a fold_change.v1 input (reads.fc)")
        target = channel.split("_", 1)[1]
        use_col = "log2FC" if channel.startswith("log2FC_") else "FC"
        table = fc_in.copy()
        table = table[table["target"].astype(str) == target].copy()
        if table.empty:
            raise ValueError(f"snapshot_heatmap: no fold_change rows found for target={target!r}")
        chosen_time = choose_nearest_time(
            table["time"],
            target_time=float(cfg.time),
            tol=float(cfg.time_tolerance),
            where="snapshot_heatmap",
            logger=ctx.logger,
        )
        subset = table[pd.to_numeric(table["time"], errors="coerce") == chosen_time].copy()
        subset = subset.rename(columns={use_col: "value"})
        subset["channel"] = channel
        keep = ["time", "channel", "value", "treatment", "design_id", "design_id_alias"]
        df = subset[[column for column in keep if column in subset.columns]].copy()
        filename = cfg.filename or f"snapshot_heatmap__{channel}__t{chosen_time:g}h"
        fig_kwargs.setdefault("cbar_label", auto_cbar_label(channel, None))
        return {"df": df, "filename": filename, "fig_kwargs": fig_kwargs}

    if df_in is None:
        raise ValueError("snapshot_heatmap: tidy df is required when channel is not FC_/log2FC_-prefixed")

    df = df_in.copy()
    if cfg.value_transform and str(cfg.value_transform).lower() in {"log2", "log10"}:
        mask = df["channel"].astype(str) == channel
        df.loc[mask, "value"] = _transform_positive_values(
            df.loc[mask, "value"],
            transform=str(cfg.value_transform).lower(),
        )
    fig_kwargs.setdefault("cbar_label", auto_cbar_label(channel, cfg.value_transform))
    return {"df": df, "filename": cfg.filename, "fig_kwargs": fig_kwargs}
