"""
--------------------------------------------------------------------------------
<reader project>
plugins/transform/fold_change.py

Fold-change report:
  • Selects the nearest snapshot time(s) per group
  • Computes FC against explicit baselines (global or per-group overrides)
  • Emits a validated artifact (fold_change.v1) — no mutation of the main tidy df
  • Prints a concise, rich stdout summary (via logger) for quick inspection

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig
from reader.lib.microplates.base import (
    nearest_time_per_key,
    smart_string_numeric_key,
)

# ----------------------------- small helpers -----------------------------


def _synonyms_for(col: str) -> list[str]:
    """
    Accept both raw and alias names interchangeably when matching override rules.
    e.g., 'genotype' ↔ 'genotype_alias'
    """
    names = [str(col)]
    if str(col).endswith("_alias"):
        names.append(str(col)[:-6])
    else:
        names.append(str(col) + "_alias")
    return names


def _pick_alias(df: pd.DataFrame, base: str | None) -> str | None:
    """
    Deterministic column resolver (programmatic-first):
      • Prefer the raw '<base>' column
      • Else use '<base>_alias' when present
      • Else return None (assertive; let caller decide)
    """
    if not base:
        return None
    alias = f"{base}_alias"
    if base in df.columns:
        return base
    if alias in df.columns:
        return alias
    return None


# ----------------------------- config model -----------------------------


class FoldChangeCfg(PluginConfig):
    # What to compute
    target: str  # e.g., "YFP/CFP" or "YFP/OD600"
    report_times: list[float]  # e.g., [8.0, 14.0]
    time_tolerance: float = 0.51  # nearest-time selection tolerance (h)
    agg: Literal["median", "mean"] = "median"  # replicate aggregator

    # Grouping and labels
    treatment_column: str = "treatment"  # we will prefer '<col>_alias' when present
    group_by: list[str] = Field(default_factory=lambda: ["design_id"])

    # Baseline policy
    use_global_baseline: bool = False
    global_baseline_value: str | None = None  # used when use_global_baseline==True
    # overrides: list of maps; any keys matching group_by columns define a match; each must
    # include 'baseline_value'. Example:
    #   - { design_id: "araBADp", baseline_value: "0 uM arabinose" }
    overrides: list[dict[str, Any]] = Field(default_factory=list)

    # Output columns (names)
    fc_column: str = "FC"
    log2fc_column: str = "log2FC"

    # Attach extra metadata columns if present (won't be required by contract, just carried through)
    attach_metadata: list[str] = Field(default_factory=lambda: ["batch"])

    # ----- helpers kept here for cohesion (stateless static methods) -----

    @staticmethod
    def _resolve_baseline_label(
        row_like: Mapping[str, Any],
        *,
        use_global: bool,
        global_value: str | None,
        overrides: list[dict[str, Any]],
        group_by_cols: list[str],
        logger=None,
    ) -> str | None:
        """
        Resolve baseline with **override-first** precedence, then global:
          1) If a rule in `overrides` matches the group, return its `baseline_value`.
          2) Else if `use_global` and `global_value` is provided, return it.
          3) Else return None (baseline missing).

        Matching considers BOTH raw and alias keys (e.g., 'design_id' and 'design_id_alias').
        """
        # 1) Try overrides first
        syn_view: dict[str, Any] = {}
        for g in group_by_cols:
            val = row_like.get(g, None)
            for k in _synonyms_for(g):
                if k not in syn_view:
                    syn_view[k] = val
        for rule in overrides or []:
            if "baseline_value" not in rule:
                continue
            ok = True
            for k, v in rule.items():
                if k == "baseline_value":
                    continue
                if str(syn_view.get(k, "")) != str(v):
                    ok = False
                    break
            if ok:
                return str(rule["baseline_value"])
        # 2) Global fallback
        if use_global and global_value is not None:
            return str(global_value)
        # 3) Nothing matched — fine; caller will emit NaN FC
        try:
            if logger is not None and overrides:
                keys = ", ".join(group_by_cols)
                logger.debug(
                    "fold_change: no override matched for {%s}=%s", keys, [row_like.get(k) for k in group_by_cols]
                )
        except Exception:
            pass
        return None

    @staticmethod
    def _log_summary(
        ctx,
        *,
        cfg: FoldChangeCfg,
        target: str,
        t: float,
        stats: pd.DataFrame,
        gcols: list[str],
        tcol: str,
        rows_emitted: int,
        missing_baseline_groups: int,
    ) -> None:
        try:
            n_groups = int(stats[gcols].drop_duplicates().shape[0])
            n_treat = int(stats[tcol].nunique())
            ctx.logger.info(
                "fold_change • target=[accent]%s[/accent] • t≈%.2f h • groups=%d • treatments=%d • rows=%d • missing_baseline=%d",
                target,
                float(t),
                n_groups,
                n_treat,
                rows_emitted,
                missing_baseline_groups,
            )
        except Exception:
            pass


# ----------------------------- plugin class -----------------------------


class FoldChange(Plugin):
    """Contract-driven transform that emits a fold_change.v1 table."""

    key = "fold_change"
    category = "transform"
    ConfigModel = FoldChangeCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        # Accept tidy.v1 — metadata columns are optional but used if present.
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        # One artifact: a validated FC table
        return {"table": "fold_change.v1"}

    # ------------------------------- run --------------------------------

    def run(self, ctx, inputs, cfg: FoldChangeCfg):
        df = inputs["df"].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Prefer alias columns when present (deterministic, no silent fallbacks)
        tcol = _pick_alias(df, cfg.treatment_column)
        gcols = [c for c in (_pick_alias(df, g) for g in cfg.group_by) if c]

        if tcol is None:
            raise ValueError(f"fold_change: treatment column '{cfg.treatment_column}' (or its alias) is missing")
        if not gcols:
            raise ValueError("fold_change: none of the group_by columns are present in the dataframe")

        # Restrict to the target channel
        target = str(cfg.target)
        base = df[df["channel"].astype(str) == target].copy()
        if base.empty:
            ctx.logger.warning("fold_change: target channel %r has no rows; emitting typed empty table", target)
            cols = [
                "target",
                "time",
                "treatment",
                cfg.fc_column,
                cfg.log2fc_column,
                "n",
                "baseline_value",
                "baseline_n",
                "baseline_time",
            ]
            for c in gcols:
                cols.append(c)
            empty = pd.DataFrame(columns=cols)
            return {"table": empty}

        # We will compute per time point, then concat
        out_rows: list[dict[str, Any]] = []

        # "Keys" used to pick nearest-time replicates per (group, treatment, position)
        nearest_keys: list[str] = [c for c in (gcols + [tcol, "position"]) if c in base.columns]

        # Iterate times
        for t in [float(x) for x in cfg.report_times]:
            snapped = nearest_time_per_key(base, target_time=float(t), keys=nearest_keys, tol=float(cfg.time_tolerance))
            if snapped.empty:
                ctx.logger.warning(
                    "fold_change: t≈%.2f h: no rows within ±%.3g h for target=%s", t, cfg.time_tolerance, target
                )
                continue

            # Aggregate replicates per (group_by..., treatment)
            group_cols = [*gcols, tcol]
            extra_aggs = {}
            if "treatment" in snapped.columns:
                extra_aggs["__treatment_raw"] = ("treatment", "first")
            if "treatment_alias" in snapped.columns:
                extra_aggs["__treatment_alias"] = ("treatment_alias", "first")
            grouped = (
                snapped.assign(time_used=pd.to_numeric(snapped["time"], errors="coerce"))
                .groupby(group_cols, dropna=False)
                .agg(
                    val=("value", cfg.agg),
                    n=("value", "count"),
                    time_used=("time_used", "median"),
                    **extra_aggs,
                )
                .reset_index()
            )

            rows_before = len(out_rows)
            fallbacks_used = 0
            missing_baseline_groups = 0

            # Compute baseline per group (group = tuple of gcols)
            for grp_vals, sub in grouped.groupby(gcols, dropna=False):
                # grp_vals can be a scalar if len==1
                if not isinstance(grp_vals, tuple):
                    grp_vals = (grp_vals,)
                grp_map = {gcols[i]: grp_vals[i] for i in range(len(gcols))}

                baseline_label = FoldChangeCfg._resolve_baseline_label(
                    grp_map,
                    use_global=cfg.use_global_baseline,
                    global_value=cfg.global_baseline_value,
                    overrides=cfg.overrides,
                    group_by_cols=gcols,
                    logger=ctx.logger,
                )

                # locate baseline row within this group's aggregated table
                bl = sub[sub[tcol].astype(str) == str(baseline_label)] if baseline_label is not None else pd.DataFrame()
                # Accept baseline labels written in raw or alias form (whichever matches).
                if bl.empty and baseline_label is not None:
                    if "__treatment_raw" in sub.columns:
                        bl = sub[sub["__treatment_raw"].astype(str) == str(baseline_label)]
                    if bl.empty and "__treatment_alias" in sub.columns:
                        bl = sub[sub["__treatment_alias"].astype(str) == str(baseline_label)]

                # If an override was chosen but not present at this time, try the global baseline as a clear fallback.
                if (
                    bl.empty
                    and cfg.use_global_baseline
                    and cfg.global_baseline_value is not None
                    and (baseline_label is not None)
                    and (str(baseline_label) != str(cfg.global_baseline_value))
                ):
                    bl_global = (
                        sub[sub[tcol].astype[str] == str(cfg.global_baseline_value)]
                        if not sub.empty
                        else pd.DataFrame()
                    )
                    if not bl_global.empty:
                        try:
                            grp_desc = " | ".join(f"{c}={grp_map.get(c)}" for c in gcols)
                            ctx.logger.info(
                                "fold_change • t≈%.2f h • %s: override baseline %r not found → using global %r",
                                float(t),
                                grp_desc,
                                str(baseline_label),
                                str(cfg.global_baseline_value),
                            )
                        except Exception:
                            pass
                        baseline_label = str(cfg.global_baseline_value)
                        bl = bl_global
                        fallbacks_used += 1

                if bl.empty:
                    missing_baseline_groups += 1
                    # If neither override nor global is present, make it explicit once per group.
                    try:
                        if cfg.use_global_baseline and cfg.global_baseline_value is not None:
                            grp_desc = " | ".join(f"{c}={grp_map.get(c)}" for c in gcols)
                            ctx.logger.warning(
                                "[warn]fold_change[/warn] • t≈%.2f h • %s: baseline not present "
                                "(override=%r, global=%r) — FC set to NaN",
                                float(t),
                                grp_desc,
                                str(baseline_label),
                                str(cfg.global_baseline_value),
                            )
                    except Exception:
                        pass
                    base_val = math.nan
                    base_n = 0
                    base_time = math.nan
                else:
                    base_val = float(bl["val"].iloc[0])
                    base_n = int(bl["n"].iloc[0])
                    base_time = float(bl["time_used"].iloc[0]) if pd.notna(bl["time_used"].iloc[0]) else float("nan")

                # Emit per treatment
                for _, r in sub.iterrows():
                    v = float(r["val"])
                    fc = float("nan") if not (np.isfinite(base_val) and base_val != 0) else v / base_val
                    log2fc = float(np.log2(fc)) if (np.isfinite(fc) and fc > 0) else float("nan")

                    # Emit raw programmatic treatment label when available
                    trt_out = str(r.get("__treatment_raw", r[tcol]))
                    # Emit raw baseline label when available; otherwise echo provided label
                    base_label_out = (
                        str(bl["__treatment_raw"].iloc[0])
                        if (
                            "__treatment_raw" in bl.columns and not bl.empty and pd.notna(bl["__treatment_raw"].iloc[0])
                        )
                        else (str(baseline_label) if baseline_label is not None else "")
                    )
                    row: dict[str, Any] = {
                        "target": target,
                        "time": float(t),
                        "treatment": trt_out,
                        cfg.fc_column: fc,
                        cfg.log2fc_column: log2fc,
                        "n": int(r["n"]),
                        "baseline_value": base_label_out,
                        "baseline_n": int(base_n),
                        "baseline_time": base_time,
                    }
                    # attach group columns
                    for c in gcols:
                        row[c] = r[c]
                    out_rows.append(row)

            FoldChangeCfg._log_summary(
                ctx,
                cfg=cfg,
                target=target,
                t=t,
                stats=grouped,
                gcols=gcols,
                tcol=tcol,
                rows_emitted=(len(out_rows) - rows_before),
                missing_baseline_groups=missing_baseline_groups,
            )
            try:
                if fallbacks_used:
                    ctx.logger.info(
                        "fold_change • t≈%.2f h • override→global fallbacks: %d", float(t), int(fallbacks_used)
                    )
            except Exception:
                pass

        # Build final table
        if not out_rows:
            # produce a typed empty frame with required columns
            cols = [
                "target",
                "time",
                "treatment",
                cfg.fc_column,
                cfg.log2fc_column,
                "n",
                "baseline_value",
                "baseline_n",
                "baseline_time",
            ]
            for c in gcols:
                cols.append(c)
            out = pd.DataFrame(columns=cols)
        else:
            out = pd.DataFrame(out_rows)

        # (Optional) carry-through extra metadata if they are constant per (group,treatment) — best effort.
        for mcol in cfg.attach_metadata or []:
            if mcol in df.columns and mcol not in out.columns:
                try:
                    base_meta = (
                        base.groupby(gcols + [tcol], dropna=False)[mcol]
                        .agg(lambda s: s.dropna().iloc[0] if s.dropna().nunique() == 1 else np.nan)
                        .reset_index()
                    )
                    out = out.merge(base_meta, on=gcols + [tcol], how="left")
                except Exception:
                    pass

        # Make dtypes clean
        for c in ["time", "baseline_time"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        for c in [cfg.fc_column, cfg.log2fc_column]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        for c in ["n", "baseline_n"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

        # Pretty, one-liner overview with ordered treatments
        try:
            treat_levels = sorted(out["treatment"].astype(str).unique().tolist(), key=smart_string_numeric_key)
            gpreview = ", ".join(gcols)
            ctx.logger.info(
                "fold_change • done • target=[accent]%s[/accent] • times=[%s] • group_by=[%s] • treatments=[%s]",
                target,
                ", ".join(f"{float(x):.2f}" for x in cfg.report_times),
                gpreview or "—",
                ", ".join(treat_levels[:10]) + (" …" if len(treat_levels) > 10 else ""),
            )
        except Exception:
            pass

        # Optional: concise preview of strongest changes per timepoint
        try:
            if not out.empty and cfg.log2fc_column in out.columns:
                good = out[pd.to_numeric(out[cfg.log2fc_column], errors="coerce").notna()].copy()
                if not good.empty:
                    good[cfg.log2fc_column] = pd.to_numeric(good[cfg.log2fc_column], errors="coerce")
                    times = sorted(pd.to_numeric(good["time"], errors="coerce").dropna().unique())
                    for tt in times:
                        sub = good[pd.to_numeric(good["time"], errors="coerce") == float(tt)].copy()
                        if sub.empty:
                            continue
                        sub["__abs_l2fc"] = sub[cfg.log2fc_column].abs()
                        top = sub.sort_values("__abs_l2fc", ascending=False).head(5)

                        def _desc(row) -> str:
                            grp = " | ".join(f"{c}={row[c]}" for c in gcols if c in row.index)
                            base_lbl = str(row.get("baseline_value", "")) or "—"
                            return (
                                f"   • {(grp + ' • ' if grp else '')}treatment={row['treatment']} "
                                f"→ {row[cfg.fc_column]:.3g}x (log2FC={row[cfg.log2fc_column]:.2f}; "
                                f"baseline={base_lbl})"
                            )

                        lines = "\n".join(_desc(r) for _, r in top.iterrows())
                        ctx.logger.info("fold_change • t≈%.2f h • strongest changes:\n%s", float(tt), lines or "   —")
                    # Per‑primary-group succinct readout (top 3 by |log2FC|)
                    if gcols:
                        gkey = gcols[0]
                        for tt in times:
                            sub = good[pd.to_numeric(good["time"], errors="coerce") == float(tt)].copy()
                            if sub.empty:
                                continue
                            ctx.logger.info("fold_change • t=%.2f h • per-%s top changes:", float(tt), gkey)
                            for gv, ss in sub.groupby(gkey, dropna=False):
                                ssg = ss.sort_values("__abs_l2fc", ascending=False).head(3)
                                items = (
                                    ", ".join(
                                        f"treatment={r['treatment']}: {r[cfg.fc_column]:.3g}x (l2FC={r[cfg.log2fc_column]:.2f})"
                                        for _, r in ssg.iterrows()
                                    )
                                    or "—"
                                )
                                ctx.logger.info("   • %s → %s", str(gv), items)
        except Exception:
            # logging must never break the pipeline
            pass

        return {"table": out}
