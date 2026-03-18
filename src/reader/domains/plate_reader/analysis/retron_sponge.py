from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd


def compute_retron_sponge_metrics(ctx, df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Materialize matched-control retron sponge metrics from annotated plate-reader traces.

    The contract is intentionally explicit:
    - the primary assay channel must already exist as a tidy channel (for example: ``YFP/CFP``)
    - state decoding comes from the configured 2x2 labels
    - sensor/stress relevance is explicit through config mappings
    - missing pre-stress reads or matched tetO controls are hard errors
    """

    required = {
        "position",
        "time",
        "channel",
        "value",
        cfg.design_column,
        cfg.state_column,
        cfg.raw_treatment_column,
        cfg.plate_column,
    }
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise ValueError(f"retron_sponge_metrics: input dataframe is missing required columns: {missing}")

    relevant_stress_map = {str(key): str(value) for key, value in (cfg.relevant_stress_map or {}).items()}
    if not relevant_stress_map:
        raise ValueError("retron_sponge_metrics: relevant_stress_map must not be empty")
    sensor_target_map = {
        str(key): tuple(str(item) for item in value) for key, value in (cfg.sensor_target_map or {}).items()
    }
    if not sensor_target_map:
        raise ValueError("retron_sponge_metrics: sensor_target_map must not be empty")

    expected_sign_map = _expected_sign_map(ctx, cfg)
    channels = {cfg.measurement_channel, cfg.od_channel}
    if df[df["channel"].isin(channels)].empty:
        available = sorted(df["channel"].dropna().astype(str).unique().tolist())
        raise ValueError(
            "retron_sponge_metrics: none of the required measurement channels are present.\n"
            f"  required: {sorted(channels)}\n"
            f"  available: {available}"
        )
    wide = _wide_channel_frame(df, channels=channels)
    wide = _derive_metadata(
        wide,
        cfg=cfg,
        relevant_stress_map=relevant_stress_map,
        sensor_target_map=sensor_target_map,
        expected_sign_map=expected_sign_map,
    )

    primary_raw = pd.to_numeric(wide[cfg.measurement_channel], errors="coerce")
    bad_primary = int(((~np.isfinite(primary_raw)) | (primary_raw <= 0)).sum())
    if bad_primary:
        raise ValueError(
            "retron_sponge_metrics: primary measurement contains non-positive or non-numeric values; "
            f"cannot compute log2 ratio for {bad_primary} row(s)"
        )
    wide["R"] = np.log2(primary_raw.astype(float))
    wide["R_pre"] = np.nan
    wide["mu"] = np.nan
    wide["in_pre_window"] = False
    wide["in_primary_post_stress"] = False
    wide["in_endpoint_window"] = False

    time_zero = float(cfg.stress_time_zero_h)
    pre_reads = int(cfg.pre_reads)
    endpoint_reads = int(cfg.endpoint_reads)
    if pre_reads <= 0:
        raise ValueError("retron_sponge_metrics: pre_reads must be >= 1")
    if endpoint_reads <= 0:
        raise ValueError("retron_sponge_metrics: endpoint_reads must be >= 1")

    well_group = ["plate_id", "replicate_id"]
    for _, group in wide.groupby(well_group, dropna=False, sort=False):
        ordered = group.sort_values("time").copy()
        pre = ordered[ordered["time"] < time_zero]
        if len(pre) < pre_reads:
            rep = ordered.iloc[0]
            raise ValueError(
                "retron_sponge_metrics: insufficient pre-stress reads for "
                f"plate={rep['plate_id']!r} replicate={rep['replicate_id']!r}; "
                f"need {pre_reads}, found {len(pre)}"
            )
        pre_idx = pre.tail(pre_reads).index
        wide.loc[pre_idx, "in_pre_window"] = True
        wide.loc[ordered.index, "R_pre"] = float(wide.loc[pre_idx, "R"].mean())
        wide.loc[ordered.index, "mu"] = _growth_rate_trace(
            times=ordered["time"].to_numpy(dtype=float),
            od_values=pd.to_numeric(ordered[cfg.od_channel], errors="coerce").to_numpy(dtype=float),
        )

    control_mask = wide["sponge"].astype(str) == str(cfg.control_name)
    if not bool(control_mask.any()):
        raise ValueError(
            f"retron_sponge_metrics: no control rows matched control_name={cfg.control_name!r}; "
            "matched-control normalization cannot proceed"
        )

    cutoff_map = _primary_window_cutoffs(wide, cfg=cfg, control_mask=control_mask)
    for scope, cutoff in cutoff_map.items():
        plate_id, sensor, stress_condition = scope
        scope_mask = (
            (wide["plate_id"] == plate_id) & (wide["sensor"] == sensor) & (wide["stress_condition"] == stress_condition)
        )
        post_mask = scope_mask & (wide["time"] > time_zero) & (wide["time"] <= cutoff)
        if not bool(post_mask.any()):
            raise ValueError(
                "retron_sponge_metrics: primary_post_stress window is empty for "
                f"plate={plate_id!r}, sensor={sensor!r}, stress={stress_condition!r}"
            )
        wide.loc[post_mask, "in_primary_post_stress"] = True
        endpoint_times = sorted(pd.unique(wide.loc[post_mask, "time"]))[-endpoint_reads:]
        if len(endpoint_times) < endpoint_reads:
            raise ValueError(
                "retron_sponge_metrics: endpoint_last_n window is too short for "
                f"plate={plate_id!r}, sensor={sensor!r}, stress={stress_condition!r}; "
                f"need {endpoint_reads} timepoints, found {len(endpoint_times)}"
            )
        endpoint_mask = post_mask & wide["time"].isin(endpoint_times)
        wide.loc[endpoint_mask, "in_endpoint_window"] = True

    wide["time_from_stress"] = wide["time"].astype(float) - time_zero
    wide["B"] = wide["R"] - wide["R_pre"]

    control_means = (
        wide.loc[control_mask]
        .groupby(["plate_id", "sensor", "stress_condition", "IPTG", "time"], dropna=False)["B"]
        .mean()
        .rename("control_B")
        .reset_index()
    )
    wide = wide.merge(
        control_means,
        on=["plate_id", "sensor", "stress_condition", "IPTG", "time"],
        how="left",
        validate="many_to_one",
    )
    missing_control = wide["control_B"].isna()
    if bool(missing_control.any()):
        sample = wide.loc[missing_control, ["plate_id", "sensor", "stress_condition", "IPTG", "time"]].iloc[0]
        raise ValueError(
            "retron_sponge_metrics: missing matched tetO control for "
            f"plate={sample['plate_id']!r}, sensor={sample['sensor']!r}, "
            f"stress={sample['stress_condition']!r}, IPTG={sample['IPTG']!r}, time={sample['time']!r}"
        )
    wide["C"] = wide["B"] - wide["control_B"]

    c_means = _state_metric_means(wide, metric="C")
    b_means = _state_metric_means(wide, metric="B")
    mu_means = _state_metric_means(wide, metric="mu")
    od_means = _state_metric_means(wide, metric=cfg.od_channel)

    d_trace = _difference_by_state(
        c_means,
        value_column="value",
        out_metric="D",
        positive_state="+IPTG",
        negative_state="-IPTG",
    )
    d_trace["O"] = d_trace["expected_decoy_sign"] * d_trace["D"]
    m_trace = _stress_modulation_trace(
        d_trace, relevant_stress_map=relevant_stress_map, no_stress_label=cfg.no_stress_label
    )
    control_burden = _difference_by_state(
        b_means[b_means["sponge"] == str(cfg.control_name)],
        value_column="value",
        out_metric="T_ratio",
        positive_state="+IPTG",
        negative_state="-IPTG",
    )
    control_growth_burden = _difference_by_state(
        mu_means[mu_means["sponge"] == str(cfg.control_name)],
        value_column="value",
        out_metric="T_growth",
        positive_state="+IPTG",
        negative_state="-IPTG",
    )
    control_od_endpoint = _difference_by_state(
        od_means[(od_means["sponge"] == str(cfg.control_name)) & od_means["in_endpoint_window"]],
        value_column="value",
        out_metric="T_finalOD",
        positive_state="+IPTG",
        negative_state="-IPTG",
    )

    trace = _build_trace_table(
        wide=wide,
        d_trace=d_trace,
        m_trace=m_trace,
    )
    summary = _build_summary_table(
        wide=wide,
        c_means=c_means,
        d_trace=d_trace,
        m_trace=m_trace,
        control_burden=control_burden,
        control_growth_burden=control_growth_burden,
        control_od_endpoint=control_od_endpoint,
        relevant_stress_map=relevant_stress_map,
        control_name=str(cfg.control_name),
    )
    return trace, summary


def _wide_channel_frame(df: pd.DataFrame, *, channels: set[str]) -> pd.DataFrame:
    selected = df[df["channel"].astype(str).isin(channels)].copy()
    id_columns = [column for column in selected.columns if column not in {"channel", "value"}]
    duplicate_mask = selected.duplicated(subset=[*id_columns, "channel"], keep=False)
    if bool(duplicate_mask.any()):
        sample = selected.loc[duplicate_mask, [*id_columns, "channel"]].iloc[0].to_dict()
        raise ValueError(f"retron_sponge_metrics: duplicate channel rows prevent wide alignment: {sample}")
    wide = selected.pivot(index=id_columns, columns="channel", values="value").reset_index()
    wide.columns.name = None
    return wide


def _derive_metadata(
    wide: pd.DataFrame,
    *,
    cfg,
    relevant_stress_map: Mapping[str, str],
    sensor_target_map: Mapping[str, tuple[str, ...]],
    expected_sign_map: Mapping[str, int],
) -> pd.DataFrame:
    out = wide.copy()
    out["plate_id"] = out[cfg.plate_column].astype(str)
    out["replicate_id"] = out[cfg.replicate_column].astype(str)
    if cfg.sensor_column and cfg.sponge_column:
        if cfg.sensor_column not in out.columns or cfg.sponge_column not in out.columns:
            raise ValueError(
                "retron_sponge_metrics: configured sensor_column / sponge_column are missing from the dataframe"
            )
        out["sensor"] = out[cfg.sensor_column].astype(str)
        out["sponge"] = out[cfg.sponge_column].astype(str)
        if cfg.genotype_column:
            if cfg.genotype_column not in out.columns:
                raise ValueError(f"retron_sponge_metrics: missing genotype_column={cfg.genotype_column!r}")
            out["genotype_id"] = out[cfg.genotype_column].astype(str)
        else:
            out["genotype_id"] = out["sensor"].astype(str) + "/" + out["sponge"].astype(str)
    else:
        labels = out[cfg.design_column].astype(str)
        if not labels.str.contains(cfg.design_separator, regex=False).all():
            bad = labels[~labels.str.contains(cfg.design_separator, regex=False)].iloc[0]
            raise ValueError(
                "retron_sponge_metrics: design labels must encode 'sensor{sep}sponge' when explicit "
                f"sensor/sponge columns are not configured; bad value: {bad!r}"
            )
        parts = labels.str.split(cfg.design_separator, n=1, expand=True)
        out["sensor"] = parts[0].astype(str)
        out["sponge"] = parts[1].astype(str)
        out["genotype_id"] = labels

    state_map = {
        str(cfg.states.uninduced_unstressed): ("-IPTG", False),
        str(cfg.states.induced_unstressed): ("+IPTG", False),
        str(cfg.states.uninduced_stressed): ("-IPTG", True),
        str(cfg.states.induced_stressed): ("+IPTG", True),
    }
    raw_state = out[cfg.state_column].astype(str)
    unknown_states = sorted(set(raw_state) - set(state_map))
    if unknown_states:
        raise ValueError(
            "retron_sponge_metrics: state_column contains values outside the configured 2x2 state map: "
            f"{unknown_states}"
        )
    decoded = raw_state.map(state_map)
    out["IPTG"] = decoded.map(lambda item: item[0])
    out["is_stressed"] = decoded.map(lambda item: bool(item[1]))
    out["stress_condition"] = out.apply(
        lambda row: _stress_condition_for_row(
            raw_treatment=str(row[cfg.raw_treatment_column]),
            sensor=str(row["sensor"]),
            relevant_stress_map=relevant_stress_map,
            no_stress_label=str(cfg.no_stress_label),
        ),
        axis=1,
    )
    out["is_relevant_stress"] = out.apply(
        lambda row: bool(str(row["stress_condition"]) == str(relevant_stress_map[str(row["sensor"])])),
        axis=1,
    )
    out["expected_decoy_sign"] = out["sensor"].map(expected_sign_map).astype("Int64")
    if out["expected_decoy_sign"].isna().any():
        missing = sorted(set(out.loc[out["expected_decoy_sign"].isna(), "sensor"].astype(str)))
        raise ValueError("retron_sponge_metrics: expected_decoy_sign is undefined for sensor(s): " + ", ".join(missing))
    out["relevant_sensor_pair"] = out.apply(
        lambda row: _relevant_sensor_pair(
            sensor=str(row["sensor"]),
            sponge=str(row["sponge"]),
            control_name=str(cfg.control_name),
            sensor_target_map=sensor_target_map,
        ),
        axis=1,
    )
    out["sponge_family_size"] = out["sponge"].map(
        lambda value: _sponge_family_size(str(value), control_name=str(cfg.control_name))
    )
    out["matched_tetO_group"] = out.apply(
        lambda row: f"{row['plate_id']}::{row['sensor']}::{row['stress_condition']}::{row['IPTG']}",
        axis=1,
    )
    out["time"] = pd.to_numeric(out["time"], errors="raise").astype(float)
    return out


def _expected_sign_map(ctx, cfg) -> dict[str, int]:
    configured = {str(key): int(value) for key, value in (cfg.expected_sign_map or {}).items()}
    descriptor_map: dict[str, int] = {}
    protocol = getattr(ctx, "protocol", None)
    if protocol is not None:
        for item in getattr(protocol.descriptor, "effect_signs", ()):
            if item.expected_sign == "positive":
                descriptor_map[item.target] = 1
            elif item.expected_sign == "negative":
                descriptor_map[item.target] = -1
    return {**descriptor_map, **configured}


def _stress_condition_for_row(
    *,
    raw_treatment: str,
    sensor: str,
    relevant_stress_map: Mapping[str, str],
    no_stress_label: str,
) -> str:
    relevant = str(relevant_stress_map[sensor])
    if relevant and relevant.casefold() in raw_treatment.casefold():
        return relevant
    return str(no_stress_label)


def _relevant_sensor_pair(
    *,
    sensor: str,
    sponge: str,
    control_name: str,
    sensor_target_map: Mapping[str, tuple[str, ...]],
) -> bool:
    if sponge == control_name:
        return False
    targets = sensor_target_map.get(sensor, ())
    sponge_parts = {item.strip() for item in sponge.split("-") if item.strip()}
    return any(target in sponge_parts for target in targets)


def _sponge_family_size(sponge: str, *, control_name: str) -> str:
    if sponge == control_name:
        return "control"
    size = len([item for item in sponge.split("-") if item.strip()])
    if size <= 1:
        return "mono"
    if size == 2:
        return "bi"
    if size == 3:
        return "tri"
    if size == 4:
        return "quad"
    return f"{size}-site"


def _growth_rate_trace(*, times: np.ndarray, od_values: np.ndarray) -> np.ndarray:
    mu = np.full(len(times), np.nan, dtype=float)
    valid = np.isfinite(times) & np.isfinite(od_values) & (od_values > 0)
    if valid.sum() < 2:
        return mu
    mu_valid = np.gradient(np.log(od_values[valid]), times[valid])
    mu[np.where(valid)[0]] = mu_valid
    return mu


def _primary_window_cutoffs(wide: pd.DataFrame, *, cfg, control_mask: pd.Series) -> dict[tuple[str, str, str], float]:
    cutoffs: dict[tuple[str, str, str], float] = {}
    time_zero = float(cfg.stress_time_zero_h)
    control = wide.loc[control_mask]
    for scope, group in control.groupby(["plate_id", "sensor", "stress_condition"], dropna=False):
        post = (
            group[group["time"] > time_zero]
            .groupby("time", dropna=False)[cfg.od_channel]
            .mean()
            .reset_index()
            .sort_values("time")
        )
        if post.empty:
            raise ValueError(
                "retron_sponge_metrics: no post-stress control OD values were found for "
                f"plate={scope[0]!r}, sensor={scope[1]!r}, stress={scope[2]!r}"
            )
        cutoffs[scope] = _post_window_cutoff(
            times=post["time"].to_numpy(dtype=float),
            od_values=pd.to_numeric(post[cfg.od_channel], errors="coerce").to_numpy(dtype=float),
            mode=str(cfg.plateau.mode),
            slope_tolerance=float(cfg.plateau.slope_tolerance),
            min_intervals=int(cfg.plateau.min_intervals),
        )
    return cutoffs


def _post_window_cutoff(
    *,
    times: np.ndarray,
    od_values: np.ndarray,
    mode: Literal["full_post_stress", "control_plateau"],
    slope_tolerance: float,
    min_intervals: int,
) -> float:
    if mode == "full_post_stress":
        return float(times[-1])
    if mode != "control_plateau":
        raise ValueError(f"retron_sponge_metrics: unsupported plateau.mode={mode!r}")
    if len(times) < 2:
        return float(times[-1])
    slopes = np.diff(od_values) / np.diff(times)
    min_intervals = max(int(min_intervals), 1)
    for index in range(0, len(slopes) - min_intervals + 1):
        if np.all(np.abs(slopes[index:]) <= slope_tolerance):
            return float(times[index])
    return float(times[-1])


def _state_metric_means(wide: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    if metric not in wide.columns:
        raise ValueError(f"retron_sponge_metrics: metric column {metric!r} is missing from the analysis frame")
    values = pd.to_numeric(wide[metric], errors="coerce")
    grouped = (
        wide.assign(__metric_value=values)
        .groupby(
            ["plate_id", "sensor", "sponge", "genotype_id", "stress_condition", "IPTG", "time"],
            dropna=False,
        )
        .agg(
            value=("__metric_value", "mean"),
            expected_decoy_sign=("expected_decoy_sign", "first"),
            is_relevant_stress=("is_relevant_stress", "first"),
            relevant_sensor_pair=("relevant_sensor_pair", "first"),
            sponge_family_size=("sponge_family_size", "first"),
            time_from_stress=("time_from_stress", "first"),
            in_primary_post_stress=("in_primary_post_stress", "first"),
            in_endpoint_window=("in_endpoint_window", "first"),
        )
        .reset_index()
    )
    return grouped


def _difference_by_state(
    frame: pd.DataFrame,
    *,
    value_column: str,
    out_metric: str,
    positive_state: str,
    negative_state: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    index_columns = [
        "plate_id",
        "sensor",
        "sponge",
        "genotype_id",
        "stress_condition",
        "time",
        "time_from_stress",
        "expected_decoy_sign",
        "is_relevant_stress",
        "relevant_sensor_pair",
        "sponge_family_size",
        "in_primary_post_stress",
        "in_endpoint_window",
    ]
    pivot = frame.pivot_table(index=index_columns, columns="IPTG", values=value_column, aggfunc="first").reset_index()
    if positive_state not in pivot.columns or negative_state not in pivot.columns:
        return pd.DataFrame(columns=[*index_columns, out_metric])
    pivot["IPTG"] = pd.NA
    pivot[out_metric] = pd.to_numeric(pivot[positive_state], errors="coerce") - pd.to_numeric(
        pivot[negative_state], errors="coerce"
    )
    return pivot


def _stress_modulation_trace(
    d_trace: pd.DataFrame,
    *,
    relevant_stress_map: Mapping[str, str],
    no_stress_label: str,
) -> pd.DataFrame:
    if d_trace.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["plate_id", "sensor", "sponge", "genotype_id", "time"]
    for _, group in d_trace.groupby(group_columns, dropna=False):
        first = group.iloc[0]
        relevant_label = str(relevant_stress_map[str(first["sensor"])])
        relevant = group[group["stress_condition"] == relevant_label]
        baseline = group[group["stress_condition"] == str(no_stress_label)]
        if relevant.empty or baseline.empty:
            continue
        rel_row = relevant.iloc[0]
        base_row = baseline.iloc[0]
        rows.append(
            {
                "plate_id": rel_row["plate_id"],
                "sensor": rel_row["sensor"],
                "sponge": rel_row["sponge"],
                "genotype_id": rel_row["genotype_id"],
                "stress_condition": relevant_label,
                "IPTG": pd.NA,
                "time": float(rel_row["time"]),
                "time_from_stress": float(rel_row["time_from_stress"]),
                "M": float(rel_row["D"]) - float(base_row["D"]),
                "expected_decoy_sign": int(rel_row["expected_decoy_sign"]),
                "is_relevant_stress": True,
                "relevant_sensor_pair": bool(rel_row["relevant_sensor_pair"]),
                "sponge_family_size": rel_row["sponge_family_size"],
                "in_primary_post_stress": bool(
                    rel_row["in_primary_post_stress"] and base_row["in_primary_post_stress"]
                ),
                "in_endpoint_window": bool(rel_row["in_endpoint_window"] and base_row["in_endpoint_window"]),
            }
        )
    return pd.DataFrame(rows)


def _build_trace_table(*, wide: pd.DataFrame, d_trace: pd.DataFrame, m_trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in ("R", "B", "C", "mu"):
        metric_values = pd.to_numeric(wide[metric], errors="coerce")
        for _, row in wide.assign(__value=metric_values).iterrows():
            rows.append(
                {
                    "plate_id": row["plate_id"],
                    "sensor": row["sensor"],
                    "sponge": row["sponge"],
                    "genotype_id": row["genotype_id"],
                    "replicate_id": row["replicate_id"],
                    "stress_condition": row["stress_condition"],
                    "IPTG": row["IPTG"],
                    "time": float(row["time"]),
                    "time_from_stress": float(row["time_from_stress"]),
                    "metric": metric,
                    "value": row["__value"],
                    "expected_decoy_sign": row["expected_decoy_sign"],
                    "is_relevant_stress": row["is_relevant_stress"],
                    "relevant_sensor_pair": row["relevant_sensor_pair"],
                    "sponge_family_size": row["sponge_family_size"],
                    "matched_tetO_group": row["matched_tetO_group"],
                    "in_pre_window": row["in_pre_window"],
                    "in_primary_post_stress": row["in_primary_post_stress"],
                    "in_endpoint_window": row["in_endpoint_window"],
                }
            )
    for _, row in d_trace.iterrows():
        for metric in ("D", "O"):
            rows.append(
                {
                    "plate_id": row["plate_id"],
                    "sensor": row["sensor"],
                    "sponge": row["sponge"],
                    "genotype_id": row["genotype_id"],
                    "replicate_id": pd.NA,
                    "stress_condition": row["stress_condition"],
                    "IPTG": pd.NA,
                    "time": float(row["time"]),
                    "time_from_stress": float(row["time_from_stress"]),
                    "metric": metric,
                    "value": float(row[metric]),
                    "expected_decoy_sign": row["expected_decoy_sign"],
                    "is_relevant_stress": row["is_relevant_stress"],
                    "relevant_sensor_pair": row["relevant_sensor_pair"],
                    "sponge_family_size": row["sponge_family_size"],
                    "matched_tetO_group": pd.NA,
                    "in_pre_window": False,
                    "in_primary_post_stress": row["in_primary_post_stress"],
                    "in_endpoint_window": row["in_endpoint_window"],
                }
            )
    for _, row in m_trace.iterrows():
        rows.append(
            {
                "plate_id": row["plate_id"],
                "sensor": row["sensor"],
                "sponge": row["sponge"],
                "genotype_id": row["genotype_id"],
                "replicate_id": pd.NA,
                "stress_condition": row["stress_condition"],
                "IPTG": pd.NA,
                "time": float(row["time"]),
                "time_from_stress": float(row["time_from_stress"]),
                "metric": "M",
                "value": float(row["M"]),
                "expected_decoy_sign": row["expected_decoy_sign"],
                "is_relevant_stress": row["is_relevant_stress"],
                "relevant_sensor_pair": row["relevant_sensor_pair"],
                "sponge_family_size": row["sponge_family_size"],
                "matched_tetO_group": pd.NA,
                "in_pre_window": False,
                "in_primary_post_stress": row["in_primary_post_stress"],
                "in_endpoint_window": row["in_endpoint_window"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "plate_id",
                "sensor",
                "sponge",
                "genotype_id",
                "replicate_id",
                "stress_condition",
                "IPTG",
                "time",
                "time_from_stress",
                "metric",
                "value",
                "expected_decoy_sign",
                "is_relevant_stress",
                "relevant_sensor_pair",
                "sponge_family_size",
                "matched_tetO_group",
                "in_pre_window",
                "in_primary_post_stress",
                "in_endpoint_window",
            ]
        )
    return out


def _build_summary_table(
    *,
    wide: pd.DataFrame,
    c_means: pd.DataFrame,
    d_trace: pd.DataFrame,
    m_trace: pd.DataFrame,
    control_burden: pd.DataFrame,
    control_growth_burden: pd.DataFrame,
    control_od_endpoint: pd.DataFrame,
    relevant_stress_map: Mapping[str, str],
    control_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    r_pre_groups = (
        wide.groupby(["plate_id", "sensor", "sponge", "genotype_id", "stress_condition", "IPTG"], dropna=False)
        .agg(
            value=("R_pre", "mean"),
            expected_decoy_sign=("expected_decoy_sign", "first"),
            is_relevant_stress=("is_relevant_stress", "first"),
            relevant_sensor_pair=("relevant_sensor_pair", "first"),
            sponge_family_size=("sponge_family_size", "first"),
        )
        .reset_index()
    )
    for _, row in r_pre_groups.iterrows():
        rows.append(_summary_row(metric="R_pre", row=row))

    control_r_pre = (
        r_pre_groups[(r_pre_groups["sponge"] == control_name) & (r_pre_groups["IPTG"] == "-IPTG")]
        .rename(columns={"value": "control_value"})
        .loc[:, ["plate_id", "sensor", "stress_condition", "control_value"]]
    )
    leakiness_pre = r_pre_groups[(r_pre_groups["sponge"] != control_name) & (r_pre_groups["IPTG"] == "-IPTG")].merge(
        control_r_pre,
        on=["plate_id", "sensor", "stress_condition"],
        how="left",
        validate="many_to_one",
    )
    leakiness_pre["value"] = pd.to_numeric(leakiness_pre["value"], errors="coerce") - pd.to_numeric(
        leakiness_pre["control_value"], errors="coerce"
    )
    for _, row in leakiness_pre.iterrows():
        rows.append(_summary_row(metric="L_pre", row=row, iptg=None))

    c_summary = _window_summaries(c_means, value_column="value", metrics=("C_AUC", "C_END"))
    for metric_name, frame in c_summary.items():
        for _, row in frame.iterrows():
            rows.append(_summary_row(metric=metric_name, row=row))

    d_summary = _window_summaries(d_trace, value_column="D", metrics=("D_AUC", "D_END"))
    for metric_name, frame in d_summary.items():
        for _, row in frame.iterrows():
            rows.append(_summary_row(metric=metric_name, row=row, iptg=None))

    o_trace = d_trace.assign(value=pd.to_numeric(d_trace["O"], errors="coerce"))
    o_summary = _window_summaries(o_trace, value_column="value", metrics=("O_AUC",))
    for _, row in o_summary["O_AUC"].iterrows():
        rows.append(_summary_row(metric="O_AUC", row=row, iptg=None))

    m_summary = _window_summaries(m_trace, value_column="M", metrics=("M_AUC", "M_END"))
    for metric_name, frame in m_summary.items():
        for _, row in frame.iterrows():
            rows.append(_summary_row(metric=metric_name, row=row, iptg=None))

    l_post_source = c_means[c_means["IPTG"] == "-IPTG"]
    l_post = _window_summaries(l_post_source, value_column="value", metrics=("L_post_AUC",))
    for _, row in l_post["L_post_AUC"].iterrows():
        rows.append(_summary_row(metric="L_post_AUC", row=row))

    for metric_name, frame in {
        "T_ratio_AUC": _window_summaries(
            control_burden.rename(columns={"T_ratio": "value"}), value_column="value", metrics=("T_ratio_AUC",)
        )["T_ratio_AUC"],
        "T_growth_AUC": _window_summaries(
            control_growth_burden.rename(columns={"T_growth": "value"}), value_column="value", metrics=("T_growth_AUC",)
        )["T_growth_AUC"],
    }.items():
        for _, row in frame.iterrows():
            rows.append(_summary_row(metric=metric_name, row=row, sponge=control_name, genotype_id=None, iptg=None))

    for _, row in control_od_endpoint.iterrows():
        row = row.copy()
        row["value"] = row["T_finalOD"]
        rows.append(_summary_row(metric="T_finalOD", row=row, sponge=control_name, genotype_id=None, iptg=None))

    g_sensor_rows = _native_sensor_response(wide, relevant_stress_map=relevant_stress_map, control_name=control_name)
    for _, row in g_sensor_rows.iterrows():
        rows.append(_summary_row(metric="G_sensor", row=row, sponge=control_name, genotype_id=None, iptg=None))

    g_sensor_lookup = {(row["plate_id"], row["sensor"]): row["value"] for _, row in g_sensor_rows.iterrows()}
    for _, row in o_summary["O_AUC"].iterrows():
        if not bool(row["is_relevant_stress"]):
            continue
        native = g_sensor_lookup.get((row["plate_id"], row["sensor"]))
        scaled = (
            np.nan
            if native is None or not np.isfinite(native) or native == 0
            else float(row["value"]) / abs(float(native))
        )
        row = row.copy()
        row["value"] = scaled
        rows.append(_summary_row(metric="S_AUC", row=row, iptg=None))

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "plate_id",
                "sensor",
                "sponge",
                "genotype_id",
                "stress_condition",
                "IPTG",
                "metric",
                "value",
                "expected_decoy_sign",
                "is_relevant_stress",
                "relevant_sensor_pair",
                "sponge_family_size",
            ]
        )
    return out


def _summary_row(
    *,
    metric: str,
    row: Mapping[str, Any],
    sponge: str | None = None,
    genotype_id: str | None = None,
    iptg: str | None = "__use_row__",
) -> dict[str, Any]:
    return {
        "plate_id": row.get("plate_id"),
        "sensor": row.get("sensor"),
        "sponge": sponge if sponge is not None else row.get("sponge"),
        "genotype_id": genotype_id if genotype_id is not None else row.get("genotype_id"),
        "stress_condition": row.get("stress_condition"),
        "IPTG": (row.get("IPTG") if iptg == "__use_row__" else iptg),
        "metric": metric,
        "value": row.get("value"),
        "expected_decoy_sign": row.get("expected_decoy_sign"),
        "is_relevant_stress": row.get("is_relevant_stress"),
        "relevant_sensor_pair": row.get("relevant_sensor_pair"),
        "sponge_family_size": row.get("sponge_family_size"),
    }


def _window_summaries(
    frame: pd.DataFrame,
    *,
    value_column: str,
    metrics: Iterable[str],
) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {metric: pd.DataFrame() for metric in metrics}
    metric_list = tuple(metrics)
    by_group: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metric_list}
    group_columns = ["plate_id", "sensor", "sponge", "genotype_id", "stress_condition", "IPTG"]
    for _, group in frame.groupby(group_columns, dropna=False):
        ordered = group.sort_values("time")
        values = pd.to_numeric(ordered[value_column], errors="coerce").to_numpy(dtype=float)
        times = pd.to_numeric(ordered["time"], errors="coerce").to_numpy(dtype=float)
        value = ordered.iloc[0]
        base = {
            "plate_id": value["plate_id"],
            "sensor": value["sensor"],
            "sponge": value["sponge"],
            "genotype_id": value["genotype_id"],
            "stress_condition": value["stress_condition"],
            "IPTG": value.get("IPTG"),
            "expected_decoy_sign": value["expected_decoy_sign"],
            "is_relevant_stress": value["is_relevant_stress"],
            "relevant_sensor_pair": value["relevant_sensor_pair"],
            "sponge_family_size": value["sponge_family_size"],
        }
        if any(metric.endswith("_AUC") for metric in metric_list):
            auc_mask = ordered["in_primary_post_stress"].astype(bool).to_numpy()
            base_auc = dict(base)
            base_auc["value"] = _auc(times[auc_mask], values[auc_mask])
            for metric in metric_list:
                if metric.endswith("_AUC"):
                    by_group[metric].append(dict(base_auc))
        if any(metric.endswith("_END") for metric in metric_list):
            end_mask = ordered["in_endpoint_window"].astype(bool).to_numpy()
            end_value = np.nan if not end_mask.any() else float(np.nanmean(values[end_mask]))
            base_end = dict(base)
            base_end["value"] = end_value
            for metric in metric_list:
                if metric.endswith("_END"):
                    by_group[metric].append(dict(base_end))
    return {metric: pd.DataFrame(rows) for metric, rows in by_group.items()}


def _native_sensor_response(
    wide: pd.DataFrame,
    *,
    relevant_stress_map: Mapping[str, str],
    control_name: str,
) -> pd.DataFrame:
    b_means = _state_metric_means(wide[wide["sponge"] == control_name], metric="B")
    baseline = b_means[b_means["IPTG"] == "-IPTG"]
    rows: list[dict[str, Any]] = []
    for (plate_id, sensor), group in baseline.groupby(["plate_id", "sensor"], dropna=False):
        relevant_label = str(relevant_stress_map[str(sensor)])
        relevant = group[group["stress_condition"] == relevant_label].sort_values("time")
        unstressed = group[group["stress_condition"] == "H2O"].sort_values("time")
        if relevant.empty or unstressed.empty:
            continue
        merged = relevant.merge(
            unstressed,
            on=["plate_id", "sensor", "sponge", "genotype_id", "IPTG", "time"],
            how="inner",
            suffixes=("_rel", "_h2o"),
            validate="one_to_one",
        )
        if merged.empty:
            continue
        values = pd.to_numeric(merged["value_rel"], errors="coerce") - pd.to_numeric(
            merged["value_h2o"], errors="coerce"
        )
        mask = (
            merged["in_primary_post_stress_rel"].astype(bool).to_numpy()
            & merged["in_primary_post_stress_h2o"].astype(bool).to_numpy()
        )
        rows.append(
            {
                "plate_id": plate_id,
                "sensor": sensor,
                "stress_condition": relevant_label,
                "value": _auc(
                    pd.to_numeric(merged["time"], errors="coerce").to_numpy(dtype=float)[mask],
                    values.to_numpy(dtype=float)[mask],
                ),
                "expected_decoy_sign": merged["expected_decoy_sign_rel"].iloc[0],
                "is_relevant_stress": True,
                "relevant_sensor_pair": True,
                "sponge_family_size": "control",
            }
        )
    return pd.DataFrame(rows)


def _auc(times: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(times) & np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapezoid(values[valid], times[valid]))
