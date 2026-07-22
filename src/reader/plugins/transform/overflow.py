"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/overflow.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig


class OverflowCfg(PluginConfig):
    action: Literal["max", "drop", "nan", "none"] = "max"
    clip_quantile: float = 0.999
    # New: explicit capping strategy
    cap_strategy: Literal["provided", "infer", "quantile"] = "quantile"
    per_channel_caps: Mapping[str, float] | None = None
    # New: how to detect overflow rows
    flag_column: str = "overflow"
    treat_inf_as_overflow: bool = True


class OverflowHandling(Plugin):
    ConfigModel = OverflowCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return cls.passthrough_output_ports(
            outputs={"df": dataframe_output("df", "tidy.v1")},
            passthrough={"df": "df"},
            promoted_examples={"df": ("plate_reader.annotated.v1",)},
        )

    def resolve_output_ports(self, *, inputs, outputs, cfg, where):
        del cfg
        return self.inherit_dataframe_output_ports(
            inputs=inputs,
            outputs=outputs,
            passthrough={"df": "df"},
            where=where,
        )

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
                raw_flags = df[cfg.flag_column]
                if raw_flags.isna().any() or not raw_flags.map(lambda value: isinstance(value, (bool, np.bool_))).all():
                    raise ValueError(
                        f"overflow_handling: {cfg.flag_column!r} must contain booleans without missing values"
                    )
                flagged = flagged | raw_flags.astype(bool)
            if cfg.treat_inf_as_overflow:
                flagged = flagged | ~np.isfinite(df["value"])
            elif (~np.isfinite(df["value"]) & ~flagged).any():
                raise ValueError("overflow_handling: non-finite values must be classified as instrument overflow")

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

            # 3) preserve why an observation is no longer exact before clamping.
            # Explicit instrument overflow and finite policy clipping are different
            # evidence states even though both land on the configured upper cap.
            policy_clipped = np.isfinite(out["value"]) & out["value"].gt(out["__cap__"]) & ~flagged
            out["value_policy_clipped"] = policy_clipped.astype(bool)
            out["value_instrument_overflow"] = flagged.astype(bool)
            out["value_bound_kind"] = np.where(policy_clipped | flagged, "lower", "exact")
            out[cfg.flag_column] = flagged.astype(bool)

            # 4) clamp everything to the cap; overflowed rows land exactly on the cap
            out.loc[flagged, "value"] = np.inf  # ensure clamp hits the cap deterministically
            out["value"] = np.minimum(out["value"], out["__cap__"])

            # 5) concise log
            if ctx.logger is not None:
                policy_counts = policy_clipped.groupby(out["channel"]).sum().astype(int)
                overflow_counts = flagged.groupby(out["channel"]).sum().astype(int)
                ctx.logger.info(
                    "overflow_handling • strategy=%s • policy_clipped_rows=%d • "
                    "instrument_overflow_rows=%d • policy_by_channel=%s • instrument_by_channel=%s",
                    cfg.cap_strategy,
                    int(policy_clipped.sum()),
                    int(flagged.sum()),
                    dict(policy_counts[policy_counts > 0]),
                    dict(overflow_counts[overflow_counts > 0]),
                )

            return {"df": out.drop(columns="__cap__")}
        raise ValueError(f"unknown overflow action {cfg.action}")
