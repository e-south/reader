"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/analysis/fold_change.py

Fold-change table construction for tidy plate-reader traces.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from reader.domains.plate_reader.analysis.timepoints import nearest_time_per_key
from reader.domains.plate_reader.ordering import smart_string_numeric_key


def synonyms_for(col: str) -> list[str]:
    names = [str(col)]
    if str(col).endswith("_alias"):
        names.append(str(col)[:-6])
    else:
        names.append(str(col) + "_alias")
    return names


def pick_alias_column(df: pd.DataFrame, base: str | None) -> str | None:
    if not base:
        return None
    alias = f"{base}_alias"
    if base in df.columns:
        return base
    if alias in df.columns:
        return alias
    return None


def resolve_baseline_label(
    row_like: Mapping[str, Any],
    *,
    use_global: bool,
    global_value: str | None,
    overrides: list[dict[str, Any]],
    group_by_cols: list[str],
    logger=None,
) -> str | None:
    syn_view: dict[str, Any] = {}
    for group_col in group_by_cols:
        value = row_like.get(group_col, None)
        for key in synonyms_for(group_col):
            if key not in syn_view:
                syn_view[key] = value
    for rule in overrides or []:
        if "baseline_value" not in rule:
            continue
        ok = True
        for key, value in rule.items():
            if key == "baseline_value":
                continue
            if str(syn_view.get(key, "")) != str(value):
                ok = False
                break
        if ok:
            return str(rule["baseline_value"])
    if use_global and global_value is not None:
        return str(global_value)
    try:
        if logger is not None and overrides:
            keys = ", ".join(group_by_cols)
            logger.debug("fold_change: no override matched for {%s}=%s", keys, [row_like.get(k) for k in group_by_cols])
    except Exception:
        pass
    return None


def log_fold_change_summary(
    ctx,
    *,
    target: str,
    timepoint: float,
    stats: pd.DataFrame,
    group_cols: list[str],
    treatment_col: str,
    rows_emitted: int,
    missing_baseline_groups: int,
) -> None:
    try:
        n_groups = int(stats[group_cols].drop_duplicates().shape[0])
        n_treat = int(stats[treatment_col].nunique())
        ctx.logger.info(
            "fold_change • target=[accent]%s[/accent] • t≈%.2f h • groups=%d • treatments=%d • rows=%d • missing_baseline=%d",
            target,
            float(timepoint),
            n_groups,
            n_treat,
            rows_emitted,
            missing_baseline_groups,
        )
    except Exception:
        pass


def compute_fold_change_table(ctx, df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    treatment_col = pick_alias_column(df, cfg.treatment_column)
    group_cols = [col for col in (pick_alias_column(df, group) for group in cfg.group_by) if col]

    if treatment_col is None:
        raise ValueError(f"fold_change: treatment column '{cfg.treatment_column}' (or its alias) is missing")
    if not group_cols:
        raise ValueError("fold_change: none of the group_by columns are present in the dataframe")

    target = str(cfg.target)
    base = df[df["channel"].astype(str) == target].copy()
    if base.empty:
        ctx.logger.warning("fold_change: target channel %r has no rows; emitting typed empty table", target)
        return build_empty_fold_change_table(group_cols=group_cols, cfg=cfg)

    out_rows: list[dict[str, Any]] = []
    nearest_keys: list[str] = [col for col in (group_cols + [treatment_col, "position"]) if col in base.columns]

    for timepoint in [float(item) for item in cfg.report_times]:
        snapped = nearest_time_per_key(
            base, target_time=float(timepoint), keys=nearest_keys, tol=float(cfg.time_tolerance)
        )
        if snapped.empty:
            ctx.logger.warning(
                "fold_change: t≈%.2f h: no rows within ±%.3g h for target=%s",
                timepoint,
                cfg.time_tolerance,
                target,
            )
            continue

        agg_extras = {}
        if "treatment" in snapped.columns:
            agg_extras["__treatment_raw"] = ("treatment", "first")
        if "treatment_alias" in snapped.columns:
            agg_extras["__treatment_alias"] = ("treatment_alias", "first")

        grouped = (
            snapped.assign(time_used=pd.to_numeric(snapped["time"], errors="coerce"))
            .groupby([*group_cols, treatment_col], dropna=False)
            .agg(
                val=("value", cfg.agg),
                n=("value", "count"),
                time_used=("time_used", "median"),
                **agg_extras,
            )
            .reset_index()
        )

        rows_before = len(out_rows)
        fallbacks_used = 0
        missing_baseline_groups = 0

        for group_values, sub in grouped.groupby(group_cols, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group_map = {group_cols[idx]: group_values[idx] for idx in range(len(group_cols))}

            baseline_label = resolve_baseline_label(
                group_map,
                use_global=cfg.use_global_baseline,
                global_value=cfg.global_baseline_value,
                overrides=cfg.overrides,
                group_by_cols=group_cols,
                logger=ctx.logger,
            )

            baseline_rows = (
                sub[sub[treatment_col].astype(str) == str(baseline_label)]
                if baseline_label is not None
                else pd.DataFrame()
            )
            if baseline_rows.empty and baseline_label is not None:
                if "__treatment_raw" in sub.columns:
                    baseline_rows = sub[sub["__treatment_raw"].astype(str) == str(baseline_label)]
                if baseline_rows.empty and "__treatment_alias" in sub.columns:
                    baseline_rows = sub[sub["__treatment_alias"].astype(str) == str(baseline_label)]

            if (
                baseline_rows.empty
                and cfg.use_global_baseline
                and cfg.global_baseline_value is not None
                and baseline_label is not None
                and str(baseline_label) != str(cfg.global_baseline_value)
            ):
                global_baseline_rows = (
                    sub[sub[treatment_col].astype(str) == str(cfg.global_baseline_value)]
                    if not sub.empty
                    else pd.DataFrame()
                )
                if not global_baseline_rows.empty:
                    try:
                        group_desc = " | ".join(f"{col}={group_map.get(col)}" for col in group_cols)
                        ctx.logger.info(
                            "fold_change • t≈%.2f h • %s: override baseline %r not found → using global %r",
                            float(timepoint),
                            group_desc,
                            str(baseline_label),
                            str(cfg.global_baseline_value),
                        )
                    except Exception:
                        pass
                    baseline_label = str(cfg.global_baseline_value)
                    baseline_rows = global_baseline_rows
                    fallbacks_used += 1

            if baseline_rows.empty:
                missing_baseline_groups += 1
                try:
                    if cfg.use_global_baseline and cfg.global_baseline_value is not None:
                        group_desc = " | ".join(f"{col}={group_map.get(col)}" for col in group_cols)
                        ctx.logger.warning(
                            "[warn]fold_change[/warn] • t≈%.2f h • %s: baseline not present "
                            "(override=%r, global=%r) — FC set to NaN",
                            float(timepoint),
                            group_desc,
                            str(baseline_label),
                            str(cfg.global_baseline_value),
                        )
                except Exception:
                    pass
                baseline_value = math.nan
                baseline_n = 0
                baseline_time = math.nan
            else:
                baseline_value = float(baseline_rows["val"].iloc[0])
                baseline_n = int(baseline_rows["n"].iloc[0])
                baseline_time = (
                    float(baseline_rows["time_used"].iloc[0])
                    if pd.notna(baseline_rows["time_used"].iloc[0])
                    else float("nan")
                )

            for _, row in sub.iterrows():
                value = float(row["val"])
                fc = (
                    float("nan")
                    if not (np.isfinite(baseline_value) and baseline_value != 0)
                    else value / baseline_value
                )
                log2fc = float(np.log2(fc)) if (np.isfinite(fc) and fc > 0) else float("nan")

                treatment_out = str(row.get("__treatment_raw", row[treatment_col]))
                baseline_out = (
                    str(baseline_rows["__treatment_raw"].iloc[0])
                    if (
                        "__treatment_raw" in baseline_rows.columns
                        and not baseline_rows.empty
                        and pd.notna(baseline_rows["__treatment_raw"].iloc[0])
                    )
                    else (str(baseline_label) if baseline_label is not None else "")
                )
                emitted = {
                    "target": target,
                    "time": float(timepoint),
                    "treatment": treatment_out,
                    cfg.fc_column: fc,
                    cfg.log2fc_column: log2fc,
                    "n": int(row["n"]),
                    "baseline_value": baseline_out,
                    "baseline_n": int(baseline_n),
                    "baseline_time": baseline_time,
                }
                for group_col in group_cols:
                    emitted[group_col] = row[group_col]
                out_rows.append(emitted)

        log_fold_change_summary(
            ctx,
            target=target,
            timepoint=timepoint,
            stats=grouped,
            group_cols=group_cols,
            treatment_col=treatment_col,
            rows_emitted=(len(out_rows) - rows_before),
            missing_baseline_groups=missing_baseline_groups,
        )
        try:
            if fallbacks_used:
                ctx.logger.info(
                    "fold_change • t≈%.2f h • override→global fallbacks: %d", float(timepoint), int(fallbacks_used)
                )
        except Exception:
            pass

    out = build_empty_fold_change_table(group_cols=group_cols, cfg=cfg) if not out_rows else pd.DataFrame(out_rows)

    for metadata_col in cfg.attach_metadata or []:
        if metadata_col in df.columns and metadata_col not in out.columns:
            try:
                base_meta = (
                    base.groupby(group_cols + [treatment_col], dropna=False)[metadata_col]
                    .agg(lambda series: series.dropna().iloc[0] if series.dropna().nunique() == 1 else np.nan)
                    .reset_index()
                )
                out = out.merge(base_meta, on=group_cols + [treatment_col], how="left")
            except Exception:
                pass

    for column in ["time", "baseline_time"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in [cfg.fc_column, cfg.log2fc_column]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["n", "baseline_n"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    try:
        treatment_levels = sorted(out["treatment"].astype(str).unique().tolist(), key=smart_string_numeric_key)
        group_preview = ", ".join(group_cols)
        ctx.logger.info(
            "fold_change • done • target=[accent]%s[/accent] • times=[%s] • group_by=[%s] • treatments=[%s]",
            target,
            ", ".join(f"{float(item):.2f}" for item in cfg.report_times),
            group_preview or "—",
            ", ".join(treatment_levels[:10]) + (" …" if len(treatment_levels) > 10 else ""),
        )
    except Exception:
        pass

    try:
        if not out.empty and cfg.log2fc_column in out.columns:
            good = out[pd.to_numeric(out[cfg.log2fc_column], errors="coerce").notna()].copy()
            if not good.empty:
                good[cfg.log2fc_column] = pd.to_numeric(good[cfg.log2fc_column], errors="coerce")
                times = sorted(pd.to_numeric(good["time"], errors="coerce").dropna().unique())
                for timepoint in times:
                    sub = good[pd.to_numeric(good["time"], errors="coerce") == float(timepoint)].copy()
                    if sub.empty:
                        continue
                    sub["__abs_l2fc"] = sub[cfg.log2fc_column].abs()
                    top = sub.sort_values("__abs_l2fc", ascending=False).head(5)

                    def _desc(row) -> str:
                        group_desc = " | ".join(f"{col}={row[col]}" for col in group_cols if col in row.index)
                        base_label = str(row.get("baseline_value", "")) or "—"
                        return (
                            f"   • {(group_desc + ' • ' if group_desc else '')}treatment={row['treatment']} "
                            f"→ {row[cfg.fc_column]:.3g}x (log2FC={row[cfg.log2fc_column]:.2f}; baseline={base_label})"
                        )

                    lines = "\n".join(_desc(row) for _, row in top.iterrows())
                    ctx.logger.info(
                        "fold_change • t≈%.2f h • strongest changes:\n%s", float(timepoint), lines or "   —"
                    )
                if group_cols:
                    primary_group = group_cols[0]
                    for timepoint in times:
                        sub = good[pd.to_numeric(good["time"], errors="coerce") == float(timepoint)].copy()
                        if sub.empty:
                            continue
                        ctx.logger.info("fold_change • t=%.2f h • per-%s top changes:", float(timepoint), primary_group)
                        for group_value, subset in sub.groupby(primary_group, dropna=False):
                            top_subset = subset.sort_values("__abs_l2fc", ascending=False).head(3)
                            items = (
                                ", ".join(
                                    f"treatment={row['treatment']}: {row[cfg.fc_column]:.3g}x "
                                    f"(l2FC={row[cfg.log2fc_column]:.2f})"
                                    for _, row in top_subset.iterrows()
                                )
                                or "—"
                            )
                            ctx.logger.info("   • %s → %s", str(group_value), items)
    except Exception:
        pass

    return out


def build_empty_fold_change_table(*, group_cols: list[str], cfg) -> pd.DataFrame:
    columns = [
        "target",
        "time",
        "treatment",
        cfg.fc_column,
        cfg.log2fc_column,
        "n",
        "baseline_value",
        "baseline_n",
        "baseline_time",
        *group_cols,
    ]
    return pd.DataFrame(columns=columns)
