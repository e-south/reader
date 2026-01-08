"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/crosstalk/pairs.py

Crosstalk pairing utilities built on fold-change tables.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

TimeMode = Literal["single", "exact", "nearest", "latest", "all"]
TimePolicy = Literal["per_time", "all"]
MappingMode = Literal["explicit", "column", "top1"]
Top1TiePolicy = Literal["error", "alphabetical"]


@dataclass(frozen=True)
class CrosstalkResult:
    pairs: pd.DataFrame
    designs: pd.DataFrame
    times_used: list[float]
    target_used: str | None
    value_column: str
    value_scale: Literal["log2", "linear"]


def _log(logger, level: str, msg: str, *args) -> None:
    if logger is None:
        return
    try:
        getattr(logger, level)(msg, *args)
    except Exception:
        return


def _select_target(df: pd.DataFrame, target: str | None, logger=None) -> tuple[pd.DataFrame, str | None]:
    if "target" not in df.columns:
        if target is not None:
            raise ValueError("crosstalk_pairs: target specified but 'target' column is missing")
        return df, None
    targets = sorted(df["target"].astype(str).dropna().unique().tolist())
    if target is None:
        if len(targets) > 1:
            raise ValueError(f"crosstalk_pairs: multiple targets present {targets}; set target explicitly")
        if len(targets) == 1:
            return df[df["target"].astype(str) == targets[0]].copy(), targets[0]
        return df.copy(), None
    if target not in targets:
        raise ValueError(f"crosstalk_pairs: target {target!r} not found in table")
    _log(logger, "info", "crosstalk_pairs - target=%s", target)
    return df[df["target"].astype(str) == str(target)].copy(), str(target)


def _select_times(
    df: pd.DataFrame,
    *,
    time_mode: TimeMode,
    time: float | None,
    times: list[float] | None,
    time_column: str,
    tolerance: float,
    logger=None,
) -> list[float]:
    if time_column not in df.columns:
        raise ValueError(f"crosstalk_pairs: time column {time_column!r} not found")
    unique = pd.to_numeric(df[time_column], errors="coerce").dropna().unique()
    if unique.size == 0:
        raise ValueError("crosstalk_pairs: no valid time values")
    unique = np.asarray(sorted(unique.astype(float)), float)

    if time_mode == "single":
        if time is not None or times is not None:
            raise ValueError("crosstalk_pairs: time/time(s) must be omitted when time_mode='single'")
        if len(unique) != 1:
            raise ValueError(f"crosstalk_pairs: expected exactly one time, found {unique.tolist()}")
        return [float(unique[0])]

    if time_mode == "latest":
        if time is not None or times is not None:
            raise ValueError("crosstalk_pairs: time/time(s) must be omitted when time_mode='latest'")
        return [float(unique[-1])]

    if time_mode == "all":
        if time is not None or times is not None:
            raise ValueError("crosstalk_pairs: time/time(s) must be omitted when time_mode='all'")
        return [float(x) for x in unique]

    if time_mode not in {"exact", "nearest"}:
        raise ValueError(f"crosstalk_pairs: unsupported time_mode {time_mode!r}")

    if (time is None and not times) or (time is not None and times is not None):
        raise ValueError("crosstalk_pairs: provide exactly one of time or times")

    requested = [float(time)] if time is not None else [float(x) for x in (times or [])]
    selected: list[float] = []
    for req in requested:
        deltas = np.abs(unique - req)
        idx = int(np.nanargmin(deltas))
        if time_mode == "exact":
            candidates = unique[unique == float(req)]
            if candidates.size == 0:
                raise ValueError(f"crosstalk_pairs: time {req} not found in available times {unique.tolist()}")
            selected.append(float(candidates[0]))
        else:
            # nearest must also be within tolerance
            if float(deltas[idx]) > float(tolerance):
                raise ValueError(
                    f"crosstalk_pairs: nearest time to {req} is {float(unique[idx])} (delta={float(deltas[idx])}),"
                    f" outside tolerance {tolerance}"
                )
            selected.append(float(unique[idx]))

    out = sorted(set(selected))
    _log(logger, "info", "crosstalk_pairs - times selected: %s", out)
    return out


def _mapping_from_column(
    df: pd.DataFrame,
    *,
    design_col: str,
    mapping_col: str,
    overrides: dict[str, str] | None,
    logger=None,
) -> dict[str, str]:
    if mapping_col not in df.columns:
        raise ValueError(f"crosstalk_pairs: mapping column {mapping_col!r} not found")
    mapping: dict[str, str] = {}
    for design, sub in df[[design_col, mapping_col]].dropna().groupby(design_col, dropna=False):
        vals = sorted(sub[mapping_col].astype(str).dropna().unique().tolist())
        if len(vals) == 0:
            continue
        if len(vals) > 1:
            raise ValueError(
                f"crosstalk_pairs: design {design!r} has multiple mapping values in {mapping_col}: {vals}"
            )
        mapping[str(design)] = str(vals[0])

    overrides = overrides or {}
    for k, v in overrides.items():
        if k in mapping and mapping[k] != v:
            _log(logger, "warning", "crosstalk_pairs: mapping override for %r (%r -> %r)", k, mapping[k], v)
        if k not in mapping:
            _log(logger, "warning", "crosstalk_pairs: mapping override for unknown design %r", k)
        mapping[str(k)] = str(v)
    return mapping


def _mapping_from_explicit(
    mapping_map: dict[str, str] | None,
) -> dict[str, str]:
    mapping_map = mapping_map or {}
    if not mapping_map:
        raise ValueError("crosstalk_pairs: explicit mapping_mode requires design_treatment_map")
    return {str(k): str(v) for k, v in mapping_map.items()}


def _mapping_from_top1(
    matrix: pd.DataFrame,
    *,
    tie_policy: Top1TiePolicy,
    tie_tolerance: float,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for design, row in matrix.iterrows():
        vals = row.dropna()
        if vals.empty:
            raise ValueError(f"crosstalk_pairs: design {design!r} has no values for top1 mapping")
        if tie_policy == "alphabetical":
            ordered = sorted(vals.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
            mapping[str(design)] = str(ordered[0][0])
            continue

        sorted_vals = vals.sort_values(ascending=False)
        top1_value = float(sorted_vals.iloc[0])
        if len(sorted_vals) >= 2:
            top2_value = float(sorted_vals.iloc[1])
            if abs(top1_value - top2_value) <= float(tie_tolerance):
                raise ValueError(
                    f"crosstalk_pairs: tie for top1 in design {design!r} (values {top1_value} vs {top2_value})"
                )
        mapping[str(design)] = str(sorted_vals.index[0])
    return mapping


def _aggregate_values(
    df: pd.DataFrame,
    *,
    design_col: str,
    treatment_col: str,
    value_col: str,
    agg: Literal["median", "mean"],
) -> pd.DataFrame:
    if agg not in {"median", "mean"}:
        raise ValueError(f"crosstalk_pairs: agg must be 'median' or 'mean', got {agg!r}")
    work = df[[design_col, treatment_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    grouped = work.groupby([design_col, treatment_col], dropna=False)[value_col].agg(agg).reset_index()
    return grouped


def _compute_design_summary(
    matrix: pd.DataFrame,
    *,
    mapping_mode: MappingMode,
    mapping: dict[str, str] | None,
    value_scale: Literal["log2", "linear"],
    tie_policy: Top1TiePolicy,
    tie_tolerance: float,
    time_value: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for design, row in matrix.iterrows():
        vals = row.dropna()
        if vals.empty:
            raise ValueError(f"crosstalk_pairs: design {design!r} has no values")

        if tie_policy == "alphabetical":
            ordered = sorted(vals.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
            top1_treatment, top1_value = ordered[0][0], float(ordered[0][1])
            if len(ordered) >= 2:
                top2_treatment, top2_value = ordered[1][0], float(ordered[1][1])
            else:
                top2_treatment, top2_value = None, float("nan")
        else:
            sorted_vals = vals.sort_values(ascending=False)
            top1_treatment = sorted_vals.index[0]
            top1_value = float(sorted_vals.iloc[0])
            top2_treatment = sorted_vals.index[1] if len(sorted_vals) >= 2 else None
            top2_value = float(sorted_vals.iloc[1]) if len(sorted_vals) >= 2 else float("nan")
            if len(sorted_vals) >= 2 and abs(top1_value - top2_value) <= float(tie_tolerance):
                raise ValueError(
                    f"crosstalk_pairs: tie for top1 in design {design!r} (values {top1_value} vs {top2_value})"
                )

        delta = float(top1_value) - float(top2_value) if np.isfinite(top2_value) else float("nan")
        if value_scale == "log2":
            ratio = float(2**delta) if np.isfinite(delta) else float("nan")
        else:
            ratio = float(top1_value) / float(top2_value) if np.isfinite(top2_value) and top2_value != 0 else float("nan")

        if mapping_mode == "top1":
            self_treatment = str(top1_treatment)
        else:
            if mapping is None:
                raise ValueError("crosstalk_pairs: mapping is required for mapping_mode 'explicit' or 'column'")
            self_treatment = mapping.get(str(design))

        self_value = float(row.get(self_treatment)) if self_treatment in row.index else float("nan")

        other_vals = vals.drop(index=self_treatment, errors="ignore")
        if other_vals.empty:
            best_other_treatment = None
            best_other_value = float("nan")
        else:
            ordered = sorted(other_vals.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
            best_other_treatment = str(ordered[0][0])
            best_other_value = float(ordered[0][1])

        if np.isfinite(self_value) and np.isfinite(best_other_value):
            self_minus_best_other = float(self_value) - float(best_other_value)
            if value_scale == "log2":
                self_ratio_best_other = float(2**self_minus_best_other)
            else:
                self_ratio_best_other = (
                    float(self_value) / float(best_other_value) if float(best_other_value) != 0 else float("nan")
                )
        else:
            self_minus_best_other = float("nan")
            self_ratio_best_other = float("nan")

        rows.append(
            {
                "design_id": str(design),
                "time": float(time_value),
                "self_treatment": self_treatment,
                "self_value": self_value,
                "best_other_treatment": best_other_treatment,
                "best_other_value": best_other_value,
                "self_minus_best_other": self_minus_best_other,
                "self_ratio_best_other": self_ratio_best_other,
                "self_is_top1": bool(self_treatment == top1_treatment) if self_treatment is not None else False,
                "top1_treatment": str(top1_treatment) if top1_treatment is not None else None,
                "top1_value": top1_value,
                "top2_treatment": str(top2_treatment) if top2_treatment is not None else None,
                "top2_value": top2_value,
                "selectivity_delta": float(delta),
                "selectivity_ratio": float(ratio),
                "n_treatments": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def _empty_pairs_frame() -> pd.DataFrame:
    dtypes = {
        "design_a": "string",
        "design_b": "string",
        "treatment_a": "string",
        "treatment_b": "string",
        "target": "string",
        "time": "float",
        "value_column": "string",
        "value_scale": "string",
        "a_self_value": "float",
        "b_self_value": "float",
        "a_cross_to_b": "float",
        "b_cross_to_a": "float",
        "a_best_other_treatment": "string",
        "b_best_other_treatment": "string",
        "a_best_other_value": "float",
        "b_best_other_value": "float",
        "a_self_minus_best_other": "float",
        "b_self_minus_best_other": "float",
        "a_self_ratio_best_other": "float",
        "b_self_ratio_best_other": "float",
        "a_self_is_top1": "bool",
        "b_self_is_top1": "bool",
        "a_top1_treatment": "string",
        "a_top2_treatment": "string",
        "b_top1_treatment": "string",
        "b_top2_treatment": "string",
        "a_top1_value": "float",
        "a_top2_value": "float",
        "b_top1_value": "float",
        "b_top2_value": "float",
        "a_selectivity_delta": "float",
        "b_selectivity_delta": "float",
        "a_selectivity_ratio": "float",
        "b_selectivity_ratio": "float",
        "pair_score": "float",
        "pair_ratio": "float",
        "passes_filters": "bool",
    }
    return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})


def _ensure_mapping_coverage(
    mapping: dict[str, str] | None,
    *,
    designs: list[str],
    require: bool,
    mapping_mode: MappingMode,
) -> None:
    if not require or mapping_mode == "top1":
        return
    mapping = mapping or {}
    missing = sorted(set(designs) - set(mapping.keys()))
    if missing:
        raise ValueError(f"crosstalk_pairs: mapping missing for designs {missing}")


def compute_crosstalk_pairs(
    df: pd.DataFrame,
    *,
    design_col: str,
    treatment_col: str,
    value_col: str,
    value_scale: Literal["log2", "linear"],
    target: str | None = None,
    time_mode: TimeMode,
    time_policy: TimePolicy = "per_time",
    time: float | None = None,
    times: list[float] | None = None,
    time_column: str = "time",
    time_tolerance: float = 0.51,
    mapping_mode: MappingMode,
    design_treatment_column: str | None = None,
    design_treatment_map: dict[str, str] | None = None,
    design_treatment_overrides: dict[str, str] | None = None,
    top1_tie_policy: Top1TiePolicy = "error",
    top1_tie_tolerance: float = 0.0,
    agg: Literal["median", "mean"] = "median",
    require_self_treatment: bool = True,
    require_self_is_top1: bool = False,
    min_self: float | None = None,
    max_cross: float | None = None,
    max_other: float | None = None,
    min_self_minus_best_other: float | None = None,
    min_self_ratio_best_other: float | None = None,
    min_selectivity_delta: float | None = None,
    min_selectivity_ratio: float | None = None,
    only_passing: bool = True,
    logger=None,
) -> CrosstalkResult:
    if design_col not in df.columns:
        raise ValueError(f"crosstalk_pairs: design column {design_col!r} not found")
    if treatment_col not in df.columns:
        raise ValueError(f"crosstalk_pairs: treatment column {treatment_col!r} not found")
    if value_col not in df.columns:
        raise ValueError(f"crosstalk_pairs: value column {value_col!r} not found")

    work, target_used = _select_target(df, target, logger=logger)
    selected_times = _select_times(
        work,
        time_mode=time_mode,
        time=time,
        times=times,
        time_column=time_column,
        tolerance=time_tolerance,
        logger=logger,
    )
    if time_policy not in {"per_time", "all"}:
        raise ValueError(f"crosstalk_pairs: unsupported time_policy {time_policy!r}")
    _log(
        logger,
        "info",
        "crosstalk_pairs - time_mode=%s - time_policy=%s - mapping_mode=%s",
        time_mode,
        time_policy,
        mapping_mode,
    )

    if mapping_mode not in {"explicit", "column", "top1"}:
        raise ValueError(f"crosstalk_pairs: unsupported mapping_mode {mapping_mode!r}")

    if mapping_mode == "explicit":
        if design_treatment_column is not None or design_treatment_overrides:
            raise ValueError("crosstalk_pairs: mapping_mode='explicit' forbids design_treatment_column/overrides")
        base_mapping = _mapping_from_explicit(design_treatment_map)
    elif mapping_mode == "column":
        if design_treatment_column is None:
            raise ValueError("crosstalk_pairs: mapping_mode='column' requires design_treatment_column")
        base_mapping = _mapping_from_column(
            work,
            design_col=design_col,
            mapping_col=design_treatment_column,
            overrides=design_treatment_overrides,
            logger=logger,
        )
    else:
        if design_treatment_column is not None or design_treatment_map or design_treatment_overrides:
            raise ValueError("crosstalk_pairs: mapping_mode='top1' forbids explicit mapping inputs")
        base_mapping = {}

    if mapping_mode in {"explicit", "column"}:
        treatments_present = set(work[treatment_col].astype(str).dropna().tolist())
        unknown = sorted(set(base_mapping.values()) - treatments_present)
        if unknown:
            raise ValueError(f"crosstalk_pairs: mapping values not present in treatment column: {unknown}")

    all_pairs: list[pd.DataFrame] = []
    all_designs: list[pd.DataFrame] = []

    for tsel in selected_times:
        sub = work[pd.to_numeric(work[time_column], errors="coerce") == float(tsel)].copy()
        grouped = _aggregate_values(sub, design_col=design_col, treatment_col=treatment_col, value_col=value_col, agg=agg)
        if grouped.empty:
            raise ValueError(f"crosstalk_pairs: no rows for time {tsel}")

        matrix = grouped.pivot(index=design_col, columns=treatment_col, values=value_col)
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)

        if mapping_mode == "top1":
            mapping = _mapping_from_top1(matrix, tie_policy=top1_tie_policy, tie_tolerance=top1_tie_tolerance)
        else:
            mapping = base_mapping

        _log(
            logger,
            "info",
            "crosstalk_pairs - t=%.2f - designs=%d - mapping=%d",
            float(tsel),
            int(len(matrix.index)),
            int(len(mapping)) if mapping is not None else 0,
        )

        _ensure_mapping_coverage(mapping, designs=matrix.index.tolist(), require=require_self_treatment, mapping_mode=mapping_mode)

        design_summary = _compute_design_summary(
            matrix,
            mapping_mode=mapping_mode,
            mapping=mapping,
            value_scale=value_scale,
            tie_policy=top1_tie_policy,
            tie_tolerance=top1_tie_tolerance,
            time_value=float(tsel),
        )
        all_designs.append(design_summary)

        designs = design_summary["design_id"].tolist()
        if len(designs) < 2:
            all_pairs.append(_empty_pairs_frame())
            continue

        summary_index = design_summary.set_index("design_id")
        rows: list[dict[str, object]] = []
        for i, a in enumerate(designs):
            for b in designs[i + 1 :]:
                a_row = summary_index.loc[a]
                b_row = summary_index.loc[b]

                treatment_a = a_row["self_treatment"]
                treatment_b = b_row["self_treatment"]

                a_self = float(a_row["self_value"])
                b_self = float(b_row["self_value"])

                a_cross = float("nan")
                b_cross = float("nan")
                if treatment_b in matrix.columns and a in matrix.index:
                    a_cross = float(matrix.loc[a, treatment_b])
                if treatment_a in matrix.columns and b in matrix.index:
                    b_cross = float(matrix.loc[b, treatment_a])

                min_self_val = float(np.nanmin([a_self, b_self])) if np.isfinite([a_self, b_self]).all() else float("nan")
                max_cross_val = float(np.nanmax([a_cross, b_cross])) if np.isfinite([a_cross, b_cross]).all() else float("nan")

                if value_scale == "log2":
                    pair_score = (
                        min_self_val - max_cross_val
                        if np.isfinite(min_self_val) and np.isfinite(max_cross_val)
                        else float("nan")
                    )
                    pair_ratio = float(2**pair_score) if np.isfinite(pair_score) else float("nan")
                else:
                    pair_score = (
                        min_self_val - max_cross_val
                        if np.isfinite(min_self_val) and np.isfinite(max_cross_val)
                        else float("nan")
                    )
                    pair_ratio = (
                        float(min_self_val) / float(max_cross_val)
                        if np.isfinite(min_self_val) and np.isfinite(max_cross_val) and max_cross_val != 0
                        else float("nan")
                    )

                passes = True
                if require_self_treatment:
                    if not (np.isfinite(a_self) and np.isfinite(b_self)):
                        passes = False
                    if not (np.isfinite(a_cross) and np.isfinite(b_cross)):
                        passes = False
                if require_self_is_top1:
                    if not (bool(a_row["self_is_top1"]) and bool(b_row["self_is_top1"])):
                        passes = False
                if min_self is not None:
                    if not (np.isfinite(a_self) and np.isfinite(b_self)) or a_self < float(min_self) or b_self < float(min_self):
                        passes = False
                if max_cross is not None:
                    if not (np.isfinite(a_cross) and np.isfinite(b_cross)) or a_cross > float(max_cross) or b_cross > float(max_cross):
                        passes = False
                if min_selectivity_delta is not None:
                    if (
                        not (np.isfinite(a_row["selectivity_delta"]) and np.isfinite(b_row["selectivity_delta"]))
                        or float(a_row["selectivity_delta"]) < float(min_selectivity_delta)
                        or float(b_row["selectivity_delta"]) < float(min_selectivity_delta)
                    ):
                        passes = False
                if min_selectivity_ratio is not None:
                    if (
                        not (np.isfinite(a_row["selectivity_ratio"]) and np.isfinite(b_row["selectivity_ratio"]))
                        or float(a_row["selectivity_ratio"]) < float(min_selectivity_ratio)
                        or float(b_row["selectivity_ratio"]) < float(min_selectivity_ratio)
                    ):
                        passes = False
                if max_other is not None:
                    if (
                        not (np.isfinite(a_row["best_other_value"]) and np.isfinite(b_row["best_other_value"]))
                        or float(a_row["best_other_value"]) > float(max_other)
                        or float(b_row["best_other_value"]) > float(max_other)
                    ):
                        passes = False
                if min_self_minus_best_other is not None:
                    if (
                        not (
                            np.isfinite(a_row["self_minus_best_other"])
                            and np.isfinite(b_row["self_minus_best_other"])
                        )
                        or float(a_row["self_minus_best_other"]) < float(min_self_minus_best_other)
                        or float(b_row["self_minus_best_other"]) < float(min_self_minus_best_other)
                    ):
                        passes = False
                if min_self_ratio_best_other is not None:
                    if (
                        not (
                            np.isfinite(a_row["self_ratio_best_other"])
                            and np.isfinite(b_row["self_ratio_best_other"])
                        )
                        or float(a_row["self_ratio_best_other"]) < float(min_self_ratio_best_other)
                        or float(b_row["self_ratio_best_other"]) < float(min_self_ratio_best_other)
                    ):
                        passes = False

                row = {
                    "design_a": a,
                    "design_b": b,
                    "treatment_a": treatment_a,
                    "treatment_b": treatment_b,
                    "target": target_used,
                    "time": float(tsel),
                    "value_column": value_col,
                    "value_scale": value_scale,
                    "a_self_value": a_self,
                    "b_self_value": b_self,
                    "a_cross_to_b": a_cross,
                    "b_cross_to_a": b_cross,
                    "a_best_other_treatment": a_row["best_other_treatment"],
                    "b_best_other_treatment": b_row["best_other_treatment"],
                    "a_best_other_value": float(a_row["best_other_value"]),
                    "b_best_other_value": float(b_row["best_other_value"]),
                    "a_self_minus_best_other": float(a_row["self_minus_best_other"]),
                    "b_self_minus_best_other": float(b_row["self_minus_best_other"]),
                    "a_self_ratio_best_other": float(a_row["self_ratio_best_other"]),
                    "b_self_ratio_best_other": float(b_row["self_ratio_best_other"]),
                    "a_self_is_top1": bool(a_row["self_is_top1"]),
                    "b_self_is_top1": bool(b_row["self_is_top1"]),
                    "a_top1_treatment": a_row["top1_treatment"],
                    "a_top2_treatment": a_row["top2_treatment"],
                    "b_top1_treatment": b_row["top1_treatment"],
                    "b_top2_treatment": b_row["top2_treatment"],
                    "a_top1_value": float(a_row["top1_value"]),
                    "a_top2_value": float(a_row["top2_value"]),
                    "b_top1_value": float(b_row["top1_value"]),
                    "b_top2_value": float(b_row["top2_value"]),
                    "a_selectivity_delta": float(a_row["selectivity_delta"]),
                    "b_selectivity_delta": float(b_row["selectivity_delta"]),
                    "a_selectivity_ratio": float(a_row["selectivity_ratio"]),
                    "b_selectivity_ratio": float(b_row["selectivity_ratio"]),
                    "pair_score": pair_score,
                    "pair_ratio": pair_ratio,
                    "passes_filters": bool(passes),
                }
                rows.append(row)

        pairs = pd.DataFrame(rows) if rows else _empty_pairs_frame()
        if not pairs.empty:
            pairs = pairs[_empty_pairs_frame().columns]
        all_pairs.append(pairs)

    pairs_out = pd.concat(all_pairs, ignore_index=True) if all_pairs else _empty_pairs_frame()
    if not pairs_out.empty:
        pairs_out = pairs_out[_empty_pairs_frame().columns]

    if time_policy == "all" and len(selected_times) > 1 and not pairs_out.empty:
        key_cols = ["design_a", "design_b", "treatment_a", "treatment_b", "target", "value_column", "value_scale"]
        grouped = pairs_out.groupby(key_cols, dropna=False)
        summary = grouped.agg(time_count=("time", "nunique"), all_pass=("passes_filters", "all")).reset_index()
        summary = summary[(summary["time_count"] == len(selected_times)) & (summary["all_pass"])]
        if summary.empty:
            pairs_out = _empty_pairs_frame()
        else:
            pairs_out = pairs_out.merge(summary[key_cols], on=key_cols, how="inner")

    if only_passing and not pairs_out.empty:
        pairs_out = pairs_out[pairs_out["passes_filters"]].copy()
    if pairs_out.empty:
        pairs_out = _empty_pairs_frame()

    designs_out = pd.concat(all_designs, ignore_index=True) if all_designs else pd.DataFrame(columns=["design_id"])

    return CrosstalkResult(
        pairs=pairs_out,
        designs=designs_out,
        times_used=[float(x) for x in selected_times],
        target_used=target_used,
        value_column=value_col,
        value_scale=value_scale,
    )
