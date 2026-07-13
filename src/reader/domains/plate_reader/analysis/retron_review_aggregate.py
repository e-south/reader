"""Cross-experiment retron sponge aggregate dataframe preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from reader.domains.plate_reader.analysis import retron_review_semantics

_ALLOWED_AGGREGATE_SCORE_METRICS = ("O_abs_AUC", "S_abs_AUC", "O_AUC", "S_AUC")
_FINGERPRINT_FRAME_COLUMNS = [
    "selected_sponge",
    "sensor",
    "stress_condition",
    "summary_window_start_h",
    "summary_window_end_h",
    "summary_window_duration_h",
    "source_experiment_id",
    "source_label",
    "comparison_group",
    "sponge",
    "sponge_family_size",
    "value",
]
_EXPECTED_VS_OBSERVED_COLUMNS = [
    "sensor",
    "sponge",
    "observed",
    "expected_best_single",
    "expected_sum",
    "relevant_motif_count",
    "sponge_family_size",
]


def available_aggregate_score_metrics(summary_df: pd.DataFrame) -> tuple[str, ...]:
    if "metric" not in summary_df.columns:
        return ()
    available = {str(value) for value in summary_df["metric"].dropna().astype(str)}
    return tuple(metric for metric in _ALLOWED_AGGREGATE_SCORE_METRICS if metric in available)


def build_specificity_matrix(
    summary_df: pd.DataFrame,
    *,
    score_metric: str = "S_AUC",
) -> pd.DataFrame:
    scores = aggregate_on_target_scores(summary_df, score_metric=score_metric)
    if scores.empty:
        return pd.DataFrame()
    pivot = scores.pivot_table(index="sensor", columns="sponge", values="value", aggfunc="mean")
    if pivot.empty:
        return pivot
    row_order = sorted(pivot.index.tolist())
    col_order = sorted(pivot.columns.tolist(), key=retron_review_semantics.sponge_sort_key)
    return pivot.reindex(index=row_order, columns=col_order)


def build_architecture_frame(
    summary_df: pd.DataFrame,
    *,
    sensor_target_map: Mapping[str, tuple[str, ...]],
    score_metric: str = "S_AUC",
) -> pd.DataFrame:
    scores = aggregate_on_target_scores(summary_df, score_metric=score_metric)
    if scores.empty:
        return scores
    frame = scores.copy()
    frame["motif_count"] = frame["sponge"].map(retron_review_semantics.motif_count)
    frame["relevant_motif_count"] = frame.apply(
        lambda row: _relevant_motif_count(
            str(row["sensor"]),
            str(row["sponge"]),
            sensor_target_map=sensor_target_map,
        ),
        axis=1,
    )
    frame["irrelevant_motif_count"] = frame["motif_count"] - frame["relevant_motif_count"]
    return frame.sort_values(["sensor", "motif_count", "sponge"], kind="stable").reset_index(drop=True)


def build_expected_vs_observed_frame(
    summary_df: pd.DataFrame,
    *,
    sensor_target_map: Mapping[str, tuple[str, ...]],
    score_metric: str = "S_AUC",
) -> pd.DataFrame:
    scores = aggregate_on_target_scores(summary_df, score_metric=score_metric)
    if scores.empty:
        return pd.DataFrame(columns=_EXPECTED_VS_OBSERVED_COLUMNS)
    mono_lookup = {
        (str(row["sensor"]), str(row["sponge"])): float(row["value"])
        for _, row in scores[scores["sponge_family_size"].astype(str) == "mono"].iterrows()
    }
    rows: list[dict[str, Any]] = []
    multi = scores[scores["sponge_family_size"].astype(str).isin({"bi", "tri", "quad"})]
    for _, row in multi.iterrows():
        sensor = str(row["sensor"])
        sponge = str(row["sponge"])
        relevant_motifs = _relevant_motifs(sensor=sensor, sponge=sponge, sensor_target_map=sensor_target_map)
        mono_scores = [mono_lookup[(sensor, motif)] for motif in relevant_motifs if (sensor, motif) in mono_lookup]
        if not mono_scores:
            continue
        rows.append(
            {
                "sensor": sensor,
                "sponge": sponge,
                "observed": float(row["value"]),
                "expected_best_single": max(mono_scores),
                "expected_sum": float(sum(mono_scores)),
                "relevant_motif_count": len(relevant_motifs),
                "sponge_family_size": row["sponge_family_size"],
            }
        )
    return (
        pd.DataFrame(rows, columns=_EXPECTED_VS_OBSERVED_COLUMNS)
        .sort_values(["sensor", "sponge"], kind="stable")
        .reset_index(drop=True)
    )


def build_fingerprint_frame(
    summary_df: pd.DataFrame,
    *,
    score_metric: str = "S_AUC",
    fingerprint_sponge: str | None = None,
    control_name: str = "tetO",
) -> pd.DataFrame:
    frame = _normalized_retron_summary_frame(
        summary_df,
        required={"sensor", "sponge", "metric", "value", "is_relevant_stress", "sponge_family_size"},
    )
    sample_rows = _fingerprint_sample_rows(frame, score_metric=score_metric)
    available = sorted(
        {str(value) for value in sample_rows["sponge"].dropna()},
        key=retron_review_semantics.sponge_sort_key,
    )
    if not available:
        return pd.DataFrame(columns=_FINGERPRINT_FRAME_COLUMNS)
    selected_sponges = (
        [_select_fingerprint_sponge(available, fingerprint_sponge=fingerprint_sponge)]
        if fingerprint_sponge is not None
        else list(available)
    )
    frames: list[pd.DataFrame] = []
    for selected_sponge in selected_sponges:
        selected_rows = _group_fingerprint_rows(sample_rows[sample_rows["sponge"] == selected_sponge].copy())
        if selected_rows.empty:
            continue
        control_rows = _group_fingerprint_rows(
            frame[
                (frame["metric"] == str(score_metric))
                & frame["is_relevant_stress"].fillna(False)
                & (frame["sponge"] == str(control_name))
                & frame["sensor"].isin(selected_rows["sensor"].astype(str))
            ].copy()
        ).rename(
            columns={
                "value": "control_value",
                "sponge": "control_sponge",
                "sponge_family_size": "control_family_size",
            }
        )
        paired_rows = _pair_fingerprint_rows(selected_rows, control_rows)
        frames.append(
            _build_fingerprint_long_frame(
                paired_rows,
                selected_sponge=selected_sponge,
                control_name=control_name,
            )
        )
    if not frames:
        return pd.DataFrame(columns=_FINGERPRINT_FRAME_COLUMNS)
    return _sorted_fingerprint_frame(pd.concat(frames, ignore_index=True))


def available_multifunctional_sponges(summary_df: pd.DataFrame) -> list[str]:
    scores = aggregate_on_target_scores(summary_df, score_metric="S_AUC")
    if scores.empty:
        return []
    multi = scores[scores["sponge_family_size"].astype(str).isin({"bi", "tri", "quad"})]
    return sorted(
        {str(value) for value in multi["sponge"].dropna()},
        key=retron_review_semantics.sponge_sort_key,
    )


def aggregate_on_target_scores(summary_df: pd.DataFrame, *, score_metric: str) -> pd.DataFrame:
    frame = _normalized_retron_summary_frame(
        summary_df,
        required={"sensor", "sponge", "metric", "value", "relevant_sensor_pair", "is_relevant_stress"},
    )
    _require_available_score_metric(frame, score_metric=score_metric)
    filtered = frame[
        (frame["metric"] == str(score_metric))
        & frame["relevant_sensor_pair"].fillna(False)
        & frame["is_relevant_stress"].fillna(False)
        & (frame["sponge"] != "tetO")
    ].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "sensor",
                "sponge",
                "sponge_family_size",
                "value",
                "n_rows",
                "n_experiments",
            ]
        )
    agg_kwargs: dict[str, tuple[str, str]] = {
        "value": ("value", "mean"),
        "n_rows": ("value", "size"),
    }
    if "source_experiment_id" in filtered.columns:
        agg_kwargs["n_experiments"] = ("source_experiment_id", "nunique")
    grouped = filtered.groupby(["sensor", "sponge", "sponge_family_size"], dropna=False).agg(**agg_kwargs).reset_index()
    if "n_experiments" not in grouped.columns:
        grouped["n_experiments"] = 1
    return grouped.sort_values(["sensor", "sponge"], kind="stable").reset_index(drop=True)


def build_aggregate_pareto_frame(
    summary_df: pd.DataFrame,
    *,
    score_metric: str = "O_abs_AUC",
    burden_metric: str = "D_growth_AUC",
) -> pd.DataFrame:
    scores = aggregate_on_target_scores(summary_df, score_metric=score_metric)
    if scores.empty:
        return pd.DataFrame(
            columns=[
                "sponge",
                "on_target",
                "burden",
                "preload",
                "leakiness",
                "sponge_family_size",
                "n_experiments",
            ]
        )
    score_by_sponge = scores.groupby("sponge", dropna=False)["value"].mean().rename("on_target")
    experiment_count = scores.groupby("sponge", dropna=False)["n_experiments"].max().rename("n_experiments")
    family = scores.groupby("sponge", dropna=False)["sponge_family_size"].first()
    leak_rows = summary_df[
        (summary_df["metric"].astype(str) == "L_pre")
        & (summary_df["sponge"].astype(str) != "tetO")
        & retron_review_semantics.coerce_optional_bool_series(
            summary_df["relevant_sensor_pair"],
            label="relevant_sensor_pair",
        ).fillna(False)
    ].copy()
    leak_rows["value"] = pd.to_numeric(leak_rows["value"], errors="coerce").abs()
    leakiness = leak_rows.groupby("sponge", dropna=False)["value"].mean().rename("leakiness")
    preload_rows = summary_df[
        (summary_df["metric"].astype(str) == "P_pre")
        & (summary_df["sponge"].astype(str) != "tetO")
        & retron_review_semantics.coerce_optional_bool_series(
            summary_df["relevant_sensor_pair"],
            label="relevant_sensor_pair",
        ).fillna(False)
        & retron_review_semantics.coerce_optional_bool_series(
            summary_df["is_relevant_stress"],
            label="is_relevant_stress",
        ).fillna(False)
    ][["sponge", "value"]].copy()
    preload_rows["value"] = pd.to_numeric(preload_rows["value"], errors="coerce")
    preload = preload_rows.groupby("sponge", dropna=False)["value"].mean().rename("preload")
    burden_rows = summary_df[
        (summary_df["metric"].astype(str) == burden_metric)
        & (summary_df["sponge"].astype(str) != "tetO")
        & retron_review_semantics.coerce_optional_bool_series(
            summary_df["relevant_sensor_pair"],
            label="relevant_sensor_pair",
        ).fillna(False)
    ][["sponge", "value"]].copy()
    burden_rows["value"] = -pd.to_numeric(burden_rows["value"], errors="coerce")
    burden = burden_rows.groupby("sponge", dropna=False)["value"].mean().rename("burden")
    table = (
        pd.concat(
            [score_by_sponge, burden, preload, leakiness, family.rename("sponge_family_size"), experiment_count],
            axis=1,
        )
        .reset_index()
        .dropna(subset=["on_target", "burden"])
    )
    if table.empty:
        return table
    table["__family_order"] = table["sponge_family_size"].map(
        lambda value: retron_review_semantics.FAMILY_ORDER.get(str(value), 99)
    )
    table["__sponge_order"] = table["sponge"].map(retron_review_semantics.sponge_sort_key)
    return (
        table.sort_values(["__family_order", "__sponge_order"], kind="stable")
        .drop(columns=["__family_order", "__sponge_order"])
        .reset_index(drop=True)
    )


def _require_available_score_metric(frame: pd.DataFrame, *, score_metric: str) -> None:
    available = sorted({str(value) for value in frame["metric"].dropna().astype(str)})
    if str(score_metric) in available:
        return
    available_text = ", ".join(available) if available else "none"
    raise ValueError(
        f"retron_review: aggregate score metric {score_metric!r} is unavailable in the loaded semantic summary "
        f"exports. Available metrics: {available_text}. This review bundle is likely backed by stale retron "
        "summary exports. The positive-area aggregate metrics are not backfilled from legacy signed D_abs_* "
        "exports because that would change the statistic. Re-run the source experiments referenced by the review "
        "manifest, then reopen the aggregate notebook."
    )


def _normalized_retron_summary_frame(summary_df: pd.DataFrame, *, required: set[str]) -> pd.DataFrame:
    missing = sorted(required - set(summary_df.columns))
    if missing:
        raise ValueError(f"retron_review: summary dataframe is missing required columns: {missing}")
    frame = summary_df.copy()
    for column in ("metric", "sponge", "sensor", "sponge_family_size"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    for column in ("relevant_sensor_pair", "is_relevant_stress"):
        if column in frame.columns:
            frame[column] = retron_review_semantics.coerce_optional_bool_series(frame[column], label=column)
    return frame


def _fingerprint_sample_rows(frame: pd.DataFrame, *, score_metric: str) -> pd.DataFrame:
    sample_rows = frame[
        (frame["metric"] == str(score_metric))
        & frame["is_relevant_stress"].fillna(False)
        & frame["sponge_family_size"].isin({"bi", "tri", "quad"})
    ].copy()
    if "relevant_sensor_pair" in sample_rows.columns:
        sample_rows = sample_rows[sample_rows["relevant_sensor_pair"].fillna(False)]
    return sample_rows


def _select_fingerprint_sponge(available: Sequence[str], *, fingerprint_sponge: str | None) -> str:
    if fingerprint_sponge is None:
        return str(available[0])
    selected = str(fingerprint_sponge)
    if selected not in set(available):
        raise ValueError(
            f"retron_review: requested fingerprint sponge {selected!r} is not available; available: {list(available)!r}"
        )
    return selected


def _group_fingerprint_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    group_columns = [
        column
        for column in (
            "source_experiment_id",
            "source_label",
            "sensor",
            "stress_condition",
            "summary_window_start_h",
            "summary_window_end_h",
            "summary_window_duration_h",
            "sponge",
            "sponge_family_size",
        )
        if column in frame.columns
    ]
    if not group_columns:
        return frame[["value"]].copy()
    return frame.groupby(group_columns, dropna=False)["value"].mean().reset_index()


def _pair_fingerprint_rows(sample_rows: pd.DataFrame, control_rows: pd.DataFrame) -> pd.DataFrame:
    match_columns = [
        column
        for column in ("source_experiment_id", "source_label", "sensor", "stress_condition")
        if column in sample_rows.columns and column in control_rows.columns
    ]
    control_columns = match_columns + ["control_value", "control_sponge", "control_family_size"]
    return sample_rows.merge(control_rows[control_columns], on=match_columns, how="left")


def _build_fingerprint_long_frame(
    paired_rows: pd.DataFrame,
    *,
    selected_sponge: str,
    control_name: str,
) -> pd.DataFrame:
    long_rows: list[dict[str, Any]] = []
    has_source_experiment_id = "source_experiment_id" in paired_rows.columns
    has_source_label = "source_label" in paired_rows.columns
    has_stress_condition = "stress_condition" in paired_rows.columns
    has_window_start = "summary_window_start_h" in paired_rows.columns
    has_window_end = "summary_window_end_h" in paired_rows.columns
    has_window_duration = "summary_window_duration_h" in paired_rows.columns
    has_sample_family_size = "sponge_family_size" in paired_rows.columns
    has_control_sponge = "control_sponge" in paired_rows.columns
    has_control_family_size = "control_family_size" in paired_rows.columns
    for row in paired_rows.itertuples(index=False):
        source_experiment_id = row.source_experiment_id if has_source_experiment_id else pd.NA
        source_label = row.source_label if has_source_label else pd.NA
        stress_condition = row.stress_condition if has_stress_condition else pd.NA
        summary_window_start_h = row.summary_window_start_h if has_window_start else pd.NA
        summary_window_end_h = row.summary_window_end_h if has_window_end else pd.NA
        summary_window_duration_h = row.summary_window_duration_h if has_window_duration else pd.NA
        sensor = str(row.sensor)
        long_rows.append(
            {
                "selected_sponge": selected_sponge,
                "sensor": sensor,
                "stress_condition": stress_condition,
                "summary_window_start_h": summary_window_start_h,
                "summary_window_end_h": summary_window_end_h,
                "summary_window_duration_h": summary_window_duration_h,
                "source_experiment_id": source_experiment_id,
                "source_label": source_label,
                "comparison_group": "Selected sponge",
                "sponge": str(row.sponge),
                "sponge_family_size": str(row.sponge_family_size) if has_sample_family_size else "other",
                "value": float(row.value),
            }
        )
        control_value = getattr(row, "control_value", pd.NA)
        if pd.notna(control_value):
            long_rows.append(
                {
                    "selected_sponge": selected_sponge,
                    "sensor": sensor,
                    "stress_condition": stress_condition,
                    "summary_window_start_h": summary_window_start_h,
                    "summary_window_end_h": summary_window_end_h,
                    "summary_window_duration_h": summary_window_duration_h,
                    "source_experiment_id": source_experiment_id,
                    "source_label": source_label,
                    "comparison_group": "tetO reference",
                    "sponge": str(row.control_sponge) if has_control_sponge else str(control_name),
                    "sponge_family_size": str(row.control_family_size) if has_control_family_size else "control",
                    "value": float(control_value),
                }
            )
    return pd.DataFrame(long_rows, columns=_FINGERPRINT_FRAME_COLUMNS)


def _sorted_fingerprint_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    sponge_order = sorted(
        out["selected_sponge"].dropna().astype(str).unique(),
        key=retron_review_semantics.sponge_sort_key,
    )
    sponge_order_map = {sponge: idx for idx, sponge in enumerate(sponge_order)}
    sensor_order = sorted(out["sensor"].dropna().astype(str).unique())
    sensor_order_map = {sensor: idx for idx, sensor in enumerate(sensor_order)}
    out["__selected_sponge_order"] = out["selected_sponge"].map(sponge_order_map)
    out["__sensor_order"] = out["sensor"].map(sensor_order_map)
    out["__group_order"] = out["comparison_group"].map({"tetO reference": 0, "Selected sponge": 1}).fillna(99)
    order = [
        "__selected_sponge_order",
        "__sensor_order",
        "__group_order",
        "source_experiment_id",
        "source_label",
        "sponge",
    ]
    return (
        out.sort_values(order, kind="stable")
        .drop(columns=["__selected_sponge_order", "__sensor_order", "__group_order"])
        .reset_index(drop=True)
    )


def _relevant_motifs(
    *,
    sensor: str,
    sponge: str,
    sensor_target_map: Mapping[str, tuple[str, ...]],
) -> list[str]:
    targets = set(sensor_target_map.get(sensor, ()))
    motifs = retron_review_semantics.split_motifs(sponge)
    return [motif for motif in motifs if motif in targets]


def _relevant_motif_count(
    sensor: str,
    sponge: str,
    *,
    sensor_target_map: Mapping[str, tuple[str, ...]],
) -> int:
    return len(_relevant_motifs(sensor=sensor, sponge=sponge, sensor_target_map=sensor_target_map))
