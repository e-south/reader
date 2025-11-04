"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/overflow.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional

import numpy as np
import pandas as pd

from reader.core.registry import Plugin, PluginConfig


class OverflowCfg(PluginConfig):
    action: Literal["max","drop","nan","none"] = "max"
    clip_quantile: float = 0.999
    # New: explicit capping strategy
    cap_strategy: Literal["provided","infer","quantile"] = "quantile"
    per_channel_caps: Optional[Mapping[str, float]] = None
    # New: how to detect overflow rows
    flag_column: str = "overflow"
    treat_inf_as_overflow: bool = True

class OverflowHandling(Plugin):
    key = "overflow_handling"
    category = "transform"
    ConfigModel = OverflowCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str,str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str,str]:
        return {"df": "tidy.v1"}

    def run(self, ctx, inputs, cfg: OverflowCfg):
        df = inputs["df"].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        act = cfg.action.lower()
        if act == "none":
            return {"df": df}
        if act == "drop":
            return {"df": df.dropna(subset=["value"])}
        if act == "nan":
            return {"df": df}
        if act == "max":
            # 1) mark which rows are overflowed
            flagged = pd.Series(False, index=df.index)
            if cfg.flag_column in df.columns:
                flagged = flagged | df[cfg.flag_column].astype(bool)
            if cfg.treat_inf_as_overflow:
                flagged = flagged | ~np.isfinite(df["value"])

            # 2) compute per-channel caps explicitly
            if cfg.cap_strategy == "provided":
                if not cfg.per_channel_caps:
                    raise ValueError("overflow_handling: cap_strategy='provided' but per_channel_caps is empty")
                caps = pd.Series({str(k): float(v) for k, v in cfg.per_channel_caps.items()}, name="__cap__")
            elif cfg.cap_strategy == "infer":
                base = df[np.isfinite(df["value"])]
                if base.empty:
                    raise ValueError("overflow_handling: cap_strategy='infer' but no finite values available")
                caps = base.groupby("channel")["value"].max().rename("__cap__")
            elif cfg.cap_strategy == "quantile":
                base = df[np.isfinite(df["value"])]
                if base.empty:
                    raise ValueError("overflow_handling: cap_strategy='quantile' but no finite values available")
                caps = base.groupby("channel")["value"].quantile(float(cfg.clip_quantile)).rename("__cap__")
            else:
                raise ValueError(f"overflow_handling: unknown cap_strategy {cfg.cap_strategy!r}")

            out = df.join(caps, on="channel")
            if out["__cap__"].isna().any():
                missing = sorted(out.loc[out["__cap__"].isna(), "channel"].astype(str).unique())
                raise ValueError(f"overflow_handling: missing cap for channels: {missing}")

            # 3) clamp everything to the cap; overflowed rows land exactly on the cap
            out.loc[flagged, "value"] = np.inf  # ensure clamp hits the cap deterministically
            out["value"] = np.minimum(out["value"], out["__cap__"])

            # 4) concise log
            try:
                counts = out.groupby("channel")[flagged].sum()
                ctx.logger.info(
                    "overflow_handling • strategy=%s • capped_rows=%d • by_channel=%s",
                    cfg.cap_strategy, int(flagged.sum()), dict(counts[counts > 0])
                )
            except Exception:
                pass

            return {"df": out.drop(columns="__cap__")}
        raise ValueError(f"unknown overflow action {cfg.action}")
