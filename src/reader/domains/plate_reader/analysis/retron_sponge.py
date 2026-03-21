from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "relevant", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "irrelevant", "off"}


def compute_retron_sponge_metrics(ctx, df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Materialize matched-control retron sponge metrics from annotated plate-reader traces.

    The contract is intentionally explicit:
    - the primary assay channel must already exist as a tidy channel (for example: ``YFP/CFP``)
    - state decoding comes from the configured 2x2 labels
    - sensor/stress relevance is explicit through config mappings
    - missing pre-stress reads or matched tetO controls are hard errors
    """

    required = {"position", "time", "channel", "value", cfg.state_column, cfg.replicate_column}
    if cfg.plate_column:
        required.add(cfg.plate_column)
    if cfg.sensor_column and cfg.sponge_column:
        required.update({cfg.sensor_column, cfg.sponge_column})
        if cfg.genotype_column:
            required.add(cfg.genotype_column)
    else:
        required.add(cfg.design_column)
    if cfg.stress_condition_column:
        required.add(cfg.stress_condition_column)
    else:
        required.add(cfg.raw_treatment_column)
    for optional_column in (
        cfg.relevant_stress_column,
        cfg.expected_sign_column,
        cfg.relevant_sensor_pair_column,
        cfg.matched_control_group_column,
        cfg.sponge_family_size_column,
    ):
        if optional_column:
            required.add(optional_column)
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise ValueError(f"retron_sponge_metrics: input dataframe is missing required columns: {missing}")

    relevant_stress_map = {str(key): str(value) for key, value in (cfg.relevant_stress_map or {}).items()}
    sensor_target_map = {
        str(key): tuple(str(item) for item in value) for key, value in (cfg.sensor_target_map or {}).items()
    }

    expected_sign_map = _expected_sign_map(ctx, cfg)
    channels = {cfg.measurement_channel, cfg.growth_channel}
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
    if not cfg.plate_column:
        segment_count = int(wide["acquisition_segment_id"].astype(str).nunique())
        if segment_count > 1:
            ctx.logger.debug(
                "retron_sponge_metrics: plate_column is unset; treating %d acquisition segment(s) as one logical "
                "plate for normalization and time-zero inference",
                segment_count,
            )
    relevant_stress_map = _resolve_relevant_stress_map(
        wide=wide,
        configured_map=relevant_stress_map,
        no_stress_label=str(cfg.no_stress_label),
    )

    primary_raw = pd.to_numeric(wide[cfg.measurement_channel], errors="coerce")
    valid_primary = np.isfinite(primary_raw) & (primary_raw > 0)
    bad_primary = int((~valid_primary).sum())
    if bad_primary:
        ctx.logger.warning(
            "retron_sponge_metrics: masking %d non-positive or non-numeric %s row(s) before log2 conversion",
            bad_primary,
            cfg.measurement_channel,
        )
    wide["primary_measurement_valid"] = valid_primary
    wide["R"] = np.nan
    wide.loc[valid_primary, "R"] = np.log2(primary_raw.loc[valid_primary].astype(float))
    wide["R_pre"] = np.nan
    wide["mu"] = np.nan
    wide["in_pre_window"] = False
    wide["in_primary_post_stress"] = False
    wide["in_endpoint_window"] = False
    wide["configured_max_post_stress_hours"] = (
        np.nan if cfg.max_post_stress_hours is None else float(cfg.max_post_stress_hours)
    )

    time_zero_by_plate = _stress_time_zero_by_plate(ctx, wide=wide, cfg=cfg)
    wide["stress_time_zero_h"] = wide["plate_id"].map(time_zero_by_plate).astype(float)
    pre_reads = int(cfg.pre_reads)
    endpoint_reads = int(cfg.endpoint_reads)
    if pre_reads <= 0:
        raise ValueError("retron_sponge_metrics: pre_reads must be >= 1")
    if endpoint_reads <= 0:
        raise ValueError("retron_sponge_metrics: endpoint_reads must be >= 1")

    well_group = ["plate_id", "replicate_id"]
    for _, group in wide.groupby(well_group, dropna=False, sort=False):
        ordered = group.sort_values("time").copy()
        time_zero = float(ordered["stress_time_zero_h"].iloc[0])
        pre = ordered[ordered["time"] < time_zero]
        if len(pre) < pre_reads:
            rep = ordered.iloc[0]
            raise ValueError(
                "retron_sponge_metrics: insufficient pre-stress reads for "
                f"plate={rep['plate_id']!r} replicate={rep['replicate_id']!r}; "
                f"need {pre_reads}, found {len(pre)}"
            )
        pre_idx = pre.tail(pre_reads).index
        pre_r = pd.to_numeric(wide.loc[pre_idx, "R"], errors="coerce")
        valid_pre = pre_r[np.isfinite(pre_r)]
        if len(valid_pre) < pre_reads:
            rep = ordered.iloc[0]
            raise ValueError(
                "retron_sponge_metrics: insufficient valid primary measurements in the pre-stress window for "
                f"plate={rep['plate_id']!r} replicate={rep['replicate_id']!r}; "
                f"need {pre_reads}, found {len(valid_pre)}"
            )
        wide.loc[pre_idx, "in_pre_window"] = True
        wide.loc[ordered.index, "R_pre"] = float(valid_pre.mean())
        wide.loc[ordered.index, "mu"] = _growth_rate_trace(
            times=ordered["time"].to_numpy(dtype=float),
            od_values=pd.to_numeric(ordered[cfg.growth_channel], errors="coerce").to_numpy(dtype=float),
        )

    control_mask = wide["sponge"].astype(str) == str(cfg.control_name)
    if not bool(control_mask.any()):
        raise ValueError(
            f"retron_sponge_metrics: no control rows matched control_name={cfg.control_name!r}; "
            "matched-control normalization cannot proceed"
        )

    cutoff_map = _primary_window_cutoffs(
        wide,
        cfg=cfg,
        control_mask=control_mask,
        time_zero_by_plate=time_zero_by_plate,
    )
    for scope, cutoff in cutoff_map.items():
        plate_id, sensor, stress_condition = scope
        time_zero = float(time_zero_by_plate[plate_id])
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

    wide["time_from_stress"] = wide["time"].astype(float) - wide["stress_time_zero_h"].astype(float)
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
    control_r_means = (
        wide.loc[control_mask]
        .groupby(["plate_id", "sensor", "stress_condition", "IPTG", "time"], dropna=False)["R"]
        .mean()
        .rename("control_R")
        .reset_index()
    )
    wide = wide.merge(
        control_r_means,
        on=["plate_id", "sensor", "stress_condition", "IPTG", "time"],
        how="left",
        validate="many_to_one",
    )
    missing_control_r = wide["control_R"].isna()
    if bool(missing_control_r.any()):
        sample = wide.loc[missing_control_r, ["plate_id", "sensor", "stress_condition", "IPTG", "time"]].iloc[0]
        raise ValueError(
            "retron_sponge_metrics: missing matched tetO raw-ratio control for "
            f"plate={sample['plate_id']!r}, sensor={sample['sensor']!r}, "
            f"stress={sample['stress_condition']!r}, IPTG={sample['IPTG']!r}, time={sample['time']!r}"
        )
    wide["C_abs"] = wide["R"] - wide["control_R"]
    control_mu_means = (
        wide.loc[control_mask]
        .groupby(["plate_id", "sensor", "stress_condition", "IPTG", "time"], dropna=False)["mu"]
        .mean()
        .rename("control_mu")
        .reset_index()
    )
    wide = wide.merge(
        control_mu_means,
        on=["plate_id", "sensor", "stress_condition", "IPTG", "time"],
        how="left",
        validate="many_to_one",
    )
    missing_control_mu = wide["control_mu"].isna()
    if bool(missing_control_mu.any()):
        sample = wide.loc[missing_control_mu, ["plate_id", "sensor", "stress_condition", "IPTG", "time"]].iloc[0]
        raise ValueError(
            "retron_sponge_metrics: missing matched tetO growth control for "
            f"plate={sample['plate_id']!r}, sensor={sample['sensor']!r}, "
            f"stress={sample['stress_condition']!r}, IPTG={sample['IPTG']!r}, time={sample['time']!r}"
        )
    wide["C_growth"] = wide["mu"] - wide["control_mu"]

    c_means = _state_metric_means(wide, metric="C")
    c_abs_means = _state_metric_means(wide, metric="C_abs")
    b_means = _state_metric_means(wide, metric="B")
    mu_means = _state_metric_means(wide, metric="mu")
    od_means = _state_metric_means(wide, metric=cfg.growth_channel)
    c_growth_means = _state_metric_means(wide, metric="C_growth")

    d_trace = _difference_by_state(
        c_means,
        value_column="value",
        out_metric="D",
        positive_state="+IPTG",
        negative_state="-IPTG",
    )
    d_abs_trace = _difference_by_state(
        c_abs_means,
        value_column="value",
        out_metric="D_abs",
        positive_state="+IPTG",
        negative_state="-IPTG",
    )
    d_growth_trace = _difference_by_state(
        c_growth_means,
        value_column="value",
        out_metric="D_growth",
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
        d_abs_trace=d_abs_trace,
        d_growth_trace=d_growth_trace,
        m_trace=m_trace,
    )
    summary = _build_summary_table(
        wide=wide,
        c_means=c_means,
        d_trace=d_trace,
        d_abs_trace=d_abs_trace,
        d_growth_trace=d_growth_trace,
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
    if cfg.plate_column:
        out["plate_id"] = out[cfg.plate_column].astype(str)
    else:
        out["plate_id"] = _fallback_plate_id(out)
    out["acquisition_segment_id"] = _fallback_acquisition_segment_id(out)
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
    if cfg.stress_condition_column:
        out["stress_condition"] = out[cfg.stress_condition_column].astype(str)
    else:
        if not relevant_stress_map:
            raise ValueError(
                "retron_sponge_metrics: relevant_stress_map is required when stress_condition_column is not configured"
            )
        out["stress_condition"] = out.apply(
            lambda row: _stress_condition_for_row(
                raw_treatment=str(row[cfg.raw_treatment_column]),
                sensor=str(row["sensor"]),
                relevant_stress_map=relevant_stress_map,
                no_stress_label=str(cfg.no_stress_label),
            ),
            axis=1,
        )
    if cfg.relevant_stress_column:
        out["is_relevant_stress"] = _boolish_series(out[cfg.relevant_stress_column], label=cfg.relevant_stress_column)
    else:
        if not relevant_stress_map:
            raise ValueError(
                "retron_sponge_metrics: relevant_stress_column or relevant_stress_map is required to identify "
                "which stress rows are biologically relevant"
            )
        out["is_relevant_stress"] = out.apply(
            lambda row: bool(str(row["stress_condition"]) == str(relevant_stress_map[str(row["sensor"])])),
            axis=1,
        )
    if cfg.expected_sign_column:
        out["expected_decoy_sign"] = _sign_series(out[cfg.expected_sign_column], label=cfg.expected_sign_column)
    else:
        out["expected_decoy_sign"] = out["sensor"].map(expected_sign_map).astype("Int64")
        if out["expected_decoy_sign"].isna().any():
            missing = sorted(set(out.loc[out["expected_decoy_sign"].isna(), "sensor"].astype(str)))
            raise ValueError(
                "retron_sponge_metrics: expected_decoy_sign is undefined for sensor(s): " + ", ".join(missing)
            )
    if cfg.relevant_sensor_pair_column:
        out["relevant_sensor_pair"] = _boolish_series(
            out[cfg.relevant_sensor_pair_column],
            label=cfg.relevant_sensor_pair_column,
        )
    else:
        if not sensor_target_map:
            raise ValueError(
                "retron_sponge_metrics: relevant_sensor_pair_column or sensor_target_map is required to classify "
                "on-target sensor/sponge pairs"
            )
        out["relevant_sensor_pair"] = out.apply(
            lambda row: _relevant_sensor_pair(
                sensor=str(row["sensor"]),
                sponge=str(row["sponge"]),
                control_name=str(cfg.control_name),
                sensor_target_map=sensor_target_map,
            ),
            axis=1,
        )
    if cfg.sponge_family_size_column:
        out["sponge_family_size"] = out[cfg.sponge_family_size_column].astype(str)
    else:
        out["sponge_family_size"] = out["sponge"].map(
            lambda value: _sponge_family_size(str(value), control_name=str(cfg.control_name))
        )
    if cfg.matched_control_group_column:
        out["matched_tetO_group"] = out[cfg.matched_control_group_column].astype(str)
    else:
        out["matched_tetO_group"] = out.apply(
            lambda row: f"{row['plate_id']}::{row['sensor']}::{row['stress_condition']}::{row['IPTG']}",
            axis=1,
        )
    out["time"] = pd.to_numeric(out["time"], errors="raise").astype(float)
    return out


def _fallback_plate_id(frame: pd.DataFrame) -> pd.Series:
    if "source_file" in frame.columns:
        return frame["source_file"].astype(str)
    return pd.Series("plate", index=frame.index, dtype="object")


def _fallback_acquisition_segment_id(frame: pd.DataFrame) -> pd.Series:
    components = [
        column for column in ("source", "source_file", "sheet_name", "sheet_index") if column in frame.columns
    ]
    if not components:
        return pd.Series("segment", index=frame.index, dtype="object")
    values = frame[components].copy()
    for column in components:
        values[column] = values[column].astype(str)
    return values.agg("::".join, axis=1)


def _resolve_relevant_stress_map(
    *,
    wide: pd.DataFrame,
    configured_map: Mapping[str, str],
    no_stress_label: str,
) -> dict[str, str]:
    resolved = {str(key): str(value) for key, value in configured_map.items()}
    flagged = wide[wide["is_relevant_stress"].astype(bool)]
    for sensor, group in flagged.groupby("sensor", dropna=False):
        labels = sorted({str(value) for value in group["stress_condition"] if str(value) != str(no_stress_label)})
        if not labels:
            continue
        if str(sensor) in resolved:
            if resolved[str(sensor)] not in labels:
                raise ValueError(
                    "retron_sponge_metrics: configured relevant_stress_map does not match explicit relevant-stress "
                    f"rows for sensor={sensor!r}; configured={resolved[str(sensor)]!r}, observed={labels!r}"
                )
            continue
        if len(labels) != 1:
            raise ValueError(
                "retron_sponge_metrics: explicit relevant stress rows are ambiguous for "
                f"sensor={sensor!r}; observed labels={labels!r}"
            )
        resolved[str(sensor)] = labels[0]
    missing = sorted(set(wide["sensor"].astype(str)) - set(resolved))
    if missing:
        raise ValueError(
            "retron_sponge_metrics: relevant stress is undefined for sensor(s): "
            + ", ".join(missing)
            + ". Configure relevant_stress_map or provide relevant_stress_column with stress_condition_column."
        )
    return resolved


def _boolish_series(values: pd.Series, *, label: str) -> pd.Series:
    return values.map(lambda value: _coerce_boolish(value, label=label))


def _coerce_boolish(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().casefold()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"retron_sponge_metrics: {label} contains non-boolean value {value!r}")


def _sign_series(values: pd.Series, *, label: str) -> pd.Series:
    return values.map(lambda value: _coerce_sign(value, label=label)).astype("Int64")


def _coerce_sign(value: Any, *, label: str) -> int:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) in {-1, 1}:
        return int(value)
    text = str(value).strip().casefold()
    if text in {"-1", "negative", "neg", "down"}:
        return -1
    if text in {"1", "positive", "pos", "up"}:
        return 1
    raise ValueError(f"retron_sponge_metrics: {label} contains unsupported sign value {value!r}; use -1 or +1")


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


def _stress_time_zero_by_plate(ctx, *, wide: pd.DataFrame, cfg) -> dict[str, float]:
    policy = str(cfg.stress_time_zero_policy)
    if cfg.stress_time_zero_h is not None and policy == "largest_gap_midpoint":
        policy = "explicit"
    plate_ids = tuple(pd.unique(wide["plate_id"].astype(str)))
    if policy == "explicit":
        if cfg.stress_time_zero_h is None:
            raise ValueError(
                "retron_sponge_metrics: stress_time_zero_h is required when stress_time_zero_policy='explicit'"
            )
        value = float(cfg.stress_time_zero_h)
        return dict.fromkeys(plate_ids, value)
    if policy != "largest_gap_midpoint":
        raise ValueError(f"retron_sponge_metrics: unsupported stress_time_zero_policy={policy!r}")

    resolved: dict[str, float] = {}
    for plate_id, group in wide.groupby("plate_id", dropna=False, sort=False):
        times = np.sort(pd.unique(pd.to_numeric(group["time"], errors="coerce").dropna()))
        if len(times) < 2:
            raise ValueError(
                "retron_sponge_metrics: cannot infer stress_time_zero_h from fewer than two timepoints for "
                f"plate={plate_id!r}; set stress_time_zero_policy='explicit'"
            )
        gaps = np.diff(times)
        if not len(gaps):
            raise ValueError(
                "retron_sponge_metrics: cannot infer stress_time_zero_h for "
                f"plate={plate_id!r}; set stress_time_zero_policy='explicit'"
            )
        max_index = int(np.argmax(gaps))
        max_gap = float(gaps[max_index])
        typical_gap = float(np.median(gaps))
        if max_gap <= 0 or max_gap < max(typical_gap * 1.5, typical_gap + 1e-9):
            raise ValueError(
                "retron_sponge_metrics: could not infer a distinct stress-time boundary for "
                f"plate={plate_id!r}; set semantic_metrics.stress_time_zero_policy='explicit' "
                "and provide stress_time_zero_h"
            )
        resolved[str(plate_id)] = float((times[max_index] + times[max_index + 1]) / 2.0)
    ctx.logger.debug(
        "retron_sponge_metrics: resolved stress_time_zero_h by largest gap midpoint: %s",
        ", ".join(f"{plate_id}={value:.6g}" for plate_id, value in resolved.items()),
    )
    return resolved


def _primary_window_cutoffs(
    wide: pd.DataFrame,
    *,
    cfg,
    control_mask: pd.Series,
    time_zero_by_plate: Mapping[str, float],
) -> dict[tuple[str, str, str], float]:
    cutoffs: dict[tuple[str, str, str], float] = {}
    control = wide.loc[control_mask]
    for scope, group in control.groupby(["plate_id", "sensor", "stress_condition"], dropna=False):
        plate_id = str(scope[0])
        time_zero = float(time_zero_by_plate[plate_id])
        post = (
            group[group["time"] > time_zero]
            .groupby("time", dropna=False)[cfg.growth_channel]
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
            od_values=pd.to_numeric(post[cfg.growth_channel], errors="coerce").to_numpy(dtype=float),
            mode=str(cfg.plateau.mode),
            slope_tolerance=float(cfg.plateau.slope_tolerance),
            min_intervals=int(cfg.plateau.min_intervals),
            time_zero=time_zero,
            max_post_stress_hours=cfg.max_post_stress_hours,
        )
    return cutoffs


def _post_window_cutoff(
    *,
    times: np.ndarray,
    od_values: np.ndarray,
    mode: Literal["full_post_stress", "control_plateau"],
    slope_tolerance: float,
    min_intervals: int,
    time_zero: float,
    max_post_stress_hours: float | None,
) -> float:
    max_cutoff = float(times[-1])
    if mode == "full_post_stress":
        raw_cutoff = float(times[-1])
    elif mode != "control_plateau":
        raise ValueError(f"retron_sponge_metrics: unsupported plateau.mode={mode!r}")
    elif len(times) < 2:
        raw_cutoff = float(times[-1])
    else:
        slopes = np.diff(od_values) / np.diff(times)
        min_intervals = max(int(min_intervals), 1)
        raw_cutoff = float(times[-1])
        for index in range(0, len(slopes) - min_intervals + 1):
            if np.all(np.abs(slopes[index:]) <= slope_tolerance):
                raw_cutoff = float(times[index])
                break
    if max_post_stress_hours is None:
        return raw_cutoff
    capped_cutoff = min(raw_cutoff, float(time_zero) + float(max_post_stress_hours))
    first_post_time = float(times[0])
    if capped_cutoff < first_post_time:
        raise ValueError(
            "retron_sponge_metrics: max_post_stress_hours ends before the first post-stress read; "
            f"first_post_stress_time={first_post_time!r}, time_zero={time_zero!r}, "
            f"max_post_stress_hours={max_post_stress_hours!r}"
        )
    return min(capped_cutoff, max_cutoff)


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
    if "configured_max_post_stress_hours" in frame.columns:
        index_columns.append("configured_max_post_stress_hours")
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
                "configured_max_post_stress_hours": rel_row.get("configured_max_post_stress_hours", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _build_trace_table(
    *,
    wide: pd.DataFrame,
    d_trace: pd.DataFrame,
    d_abs_trace: pd.DataFrame,
    d_growth_trace: pd.DataFrame,
    m_trace: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in ("R", "B", "C", "mu"):
        metric_values = pd.to_numeric(wide[metric], errors="coerce")
        for _, row in wide.assign(__value=metric_values).iterrows():
            rows.append(
                {
                    "plate_id": row["plate_id"],
                    "acquisition_segment_id": row["acquisition_segment_id"],
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
                    "configured_max_post_stress_hours": row.get("configured_max_post_stress_hours", np.nan),
                }
            )
    for _, row in d_trace.iterrows():
        for metric in ("D", "O"):
            rows.append(
                {
                    "plate_id": row["plate_id"],
                    "acquisition_segment_id": pd.NA,
                    "sensor": row["sensor"],
                    "sponge": row["sponge"],
                    "genotype_id": row["genotype_id"],
                    "replicate_id": pd.NA,
                    "stress_condition": row["stress_condition"],
                    "IPTG": pd.NA,
                    "time": float(row["time"]),
                    "time_from_stress": float(row["time_from_stress"]),
                    "metric": metric,
                    "value": _numeric_or_nan(row[metric]),
                    "expected_decoy_sign": row["expected_decoy_sign"],
                    "is_relevant_stress": row["is_relevant_stress"],
                    "relevant_sensor_pair": row["relevant_sensor_pair"],
                    "sponge_family_size": row["sponge_family_size"],
                    "matched_tetO_group": pd.NA,
                    "in_pre_window": False,
                    "in_primary_post_stress": row["in_primary_post_stress"],
                    "in_endpoint_window": row["in_endpoint_window"],
                    "configured_max_post_stress_hours": row.get("configured_max_post_stress_hours", np.nan),
                }
            )
    for _, row in d_abs_trace.iterrows():
        rows.append(
            {
                "plate_id": row["plate_id"],
                "acquisition_segment_id": pd.NA,
                "sensor": row["sensor"],
                "sponge": row["sponge"],
                "genotype_id": row["genotype_id"],
                "replicate_id": pd.NA,
                "stress_condition": row["stress_condition"],
                "IPTG": pd.NA,
                "time": float(row["time"]),
                "time_from_stress": float(row["time_from_stress"]),
                "metric": "D_abs",
                "value": _numeric_or_nan(row["D_abs"]),
                "expected_decoy_sign": row["expected_decoy_sign"],
                "is_relevant_stress": row["is_relevant_stress"],
                "relevant_sensor_pair": row["relevant_sensor_pair"],
                "sponge_family_size": row["sponge_family_size"],
                "matched_tetO_group": pd.NA,
                "in_pre_window": False,
                "in_primary_post_stress": row["in_primary_post_stress"],
                "in_endpoint_window": row["in_endpoint_window"],
                "configured_max_post_stress_hours": row.get("configured_max_post_stress_hours", np.nan),
            }
        )
    for _, row in d_growth_trace.iterrows():
        rows.append(
            {
                "plate_id": row["plate_id"],
                "acquisition_segment_id": pd.NA,
                "sensor": row["sensor"],
                "sponge": row["sponge"],
                "genotype_id": row["genotype_id"],
                "replicate_id": pd.NA,
                "stress_condition": row["stress_condition"],
                "IPTG": pd.NA,
                "time": float(row["time"]),
                "time_from_stress": float(row["time_from_stress"]),
                "metric": "D_growth",
                "value": _numeric_or_nan(row["D_growth"]),
                "expected_decoy_sign": row["expected_decoy_sign"],
                "is_relevant_stress": row["is_relevant_stress"],
                "relevant_sensor_pair": row["relevant_sensor_pair"],
                "sponge_family_size": row["sponge_family_size"],
                "matched_tetO_group": pd.NA,
                "in_pre_window": False,
                "in_primary_post_stress": row["in_primary_post_stress"],
                "in_endpoint_window": row["in_endpoint_window"],
                "configured_max_post_stress_hours": row.get("configured_max_post_stress_hours", np.nan),
            }
        )
    for _, row in m_trace.iterrows():
        rows.append(
            {
                "plate_id": row["plate_id"],
                "acquisition_segment_id": pd.NA,
                "sensor": row["sensor"],
                "sponge": row["sponge"],
                "genotype_id": row["genotype_id"],
                "replicate_id": pd.NA,
                "stress_condition": row["stress_condition"],
                "IPTG": pd.NA,
                "time": float(row["time"]),
                "time_from_stress": float(row["time_from_stress"]),
                "metric": "M",
                "value": _numeric_or_nan(row["M"]),
                "expected_decoy_sign": row["expected_decoy_sign"],
                "is_relevant_stress": row["is_relevant_stress"],
                "relevant_sensor_pair": row["relevant_sensor_pair"],
                "sponge_family_size": row["sponge_family_size"],
                "matched_tetO_group": pd.NA,
                "in_pre_window": False,
                "in_primary_post_stress": row["in_primary_post_stress"],
                "in_endpoint_window": row["in_endpoint_window"],
                "configured_max_post_stress_hours": row.get("configured_max_post_stress_hours", np.nan),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "plate_id",
                "acquisition_segment_id",
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
                "configured_max_post_stress_hours",
            ]
        )
    return out


def _build_summary_table(
    *,
    wide: pd.DataFrame,
    c_means: pd.DataFrame,
    d_trace: pd.DataFrame,
    d_abs_trace: pd.DataFrame,
    d_growth_trace: pd.DataFrame,
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

    d_abs_summary = _window_summaries(d_abs_trace, value_column="D_abs", metrics=("D_abs_AUC", "D_abs_END"))
    for metric_name, frame in d_abs_summary.items():
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

    d_growth_summary = _window_summaries(
        d_growth_trace, value_column="D_growth", metrics=("D_growth_AUC", "D_growth_END")
    )
    for metric_name, frame in d_growth_summary.items():
        for _, row in frame.iterrows():
            rows.append(_summary_row(metric=metric_name, row=row, iptg=None))

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


def _numeric_or_nan(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return np.nan if not np.isfinite(numeric) else float(numeric)
