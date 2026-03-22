from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from reader.plugins.transform.retron_sponge_metrics import RetronSpongeMetrics, RetronSpongeMetricsCfg


def _ctx():
    effect_signs = (
        SimpleNamespace(target="spyP", expected_sign="negative"),
        SimpleNamespace(target="sulAp", expected_sign="positive"),
        SimpleNamespace(target="soxSp", expected_sign="negative"),
    )
    descriptor = SimpleNamespace(effect_signs=effect_signs)
    return SimpleNamespace(
        logger=logging.getLogger("reader.tests.retron_sponge"), protocol=SimpleNamespace(descriptor=descriptor)
    )


def _input_df() -> pd.DataFrame:
    times = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]
    od = {
        "A1": [0.10, 0.15, 0.20, 0.32, 0.38, 0.42],
        "A2": [0.10, 0.15, 0.20, 0.32, 0.38, 0.42],
        "A3": [0.10, 0.15, 0.20, 0.28, 0.33, 0.36],
        "A4": [0.10, 0.15, 0.20, 0.28, 0.33, 0.36],
        "B1": [0.10, 0.15, 0.20, 0.32, 0.38, 0.42],
        "B2": [0.10, 0.15, 0.20, 0.32, 0.38, 0.42],
        "B3": [0.10, 0.15, 0.20, 0.28, 0.33, 0.36],
        "B4": [0.10, 0.15, 0.20, 0.28, 0.33, 0.36],
    }
    ratio = {
        "A1": [1.00, 1.00, 1.00, 1.00, 1.02, 1.03],  # tetO H2O -IPTG
        "A2": [1.00, 1.00, 1.00, 1.03, 1.05, 1.06],  # tetO H2O +IPTG
        "A3": [1.00, 1.00, 1.00, 0.82, 0.80, 0.79],  # tetO EtOH -IPTG
        "A4": [1.00, 1.00, 1.00, 0.84, 0.82, 0.81],  # tetO EtOH +IPTG
        "B1": [1.00, 1.00, 1.00, 0.99, 0.99, 0.99],  # CpxR H2O -IPTG
        "B2": [1.00, 1.00, 1.00, 0.95, 0.94, 0.93],  # CpxR H2O +IPTG
        "B3": [1.00, 1.00, 1.00, 0.80, 0.78, 0.77],  # CpxR EtOH -IPTG
        "B4": [1.00, 1.00, 1.00, 0.62, 0.58, 0.55],  # CpxR EtOH +IPTG
    }
    meta = {
        "A1": ("spyP/tetO", "-IPTG/-stress", "0 uM IPTG"),
        "A2": ("spyP/tetO", "+IPTG/-stress", "500 uM IPTG"),
        "A3": ("spyP/tetO", "-IPTG/+stress", "3% EtOH"),
        "A4": ("spyP/tetO", "+IPTG/+stress", "500 uM IPTG, 3% EtOH"),
        "B1": ("spyP/CpxR", "-IPTG/-stress", "0 uM IPTG"),
        "B2": ("spyP/CpxR", "+IPTG/-stress", "500 uM IPTG"),
        "B3": ("spyP/CpxR", "-IPTG/+stress", "3% EtOH"),
        "B4": ("spyP/CpxR", "+IPTG/+stress", "500 uM IPTG, 3% EtOH"),
    }
    rows: list[dict[str, object]] = []
    for position, (design_id_alias, treatment_alias, treatment) in meta.items():
        for time, od_value, ratio_value in zip(times, od[position], ratio[position], strict=True):
            rows.append(
                {
                    "position": position,
                    "time": time,
                    "channel": "OD600",
                    "value": od_value,
                    "design_id": design_id_alias,
                    "design_id_alias": design_id_alias,
                    "treatment": treatment,
                    "treatment_alias": treatment_alias,
                    "sheet_name": "Plate 1",
                }
            )
            rows.append(
                {
                    "position": position,
                    "time": time,
                    "channel": "YFP/CFP",
                    "value": ratio_value,
                    "design_id": design_id_alias,
                    "design_id_alias": design_id_alias,
                    "treatment": treatment,
                    "treatment_alias": treatment_alias,
                    "sheet_name": "Plate 1",
                }
            )
    return pd.DataFrame(rows)


def _input_df_single_reporter() -> pd.DataFrame:
    times = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]
    od = {
        "A1": [0.10, 0.15, 0.20, 0.31, 0.36, 0.40],
        "A2": [0.10, 0.15, 0.20, 0.31, 0.36, 0.40],
        "A3": [0.10, 0.15, 0.20, 0.29, 0.34, 0.37],
        "A4": [0.10, 0.15, 0.20, 0.29, 0.34, 0.37],
        "B1": [0.10, 0.15, 0.20, 0.31, 0.36, 0.40],
        "B2": [0.10, 0.15, 0.20, 0.31, 0.36, 0.40],
        "B3": [0.10, 0.15, 0.20, 0.29, 0.34, 0.37],
        "B4": [0.10, 0.15, 0.20, 0.29, 0.34, 0.37],
    }
    ratio = {
        "A1": [1.00, 1.00, 1.00, 1.00, 1.01, 1.02],  # tetO H2O -IPTG
        "A2": [1.00, 1.00, 1.00, 1.02, 1.03, 1.04],  # tetO H2O +IPTG
        "A3": [1.00, 1.00, 1.00, 1.10, 1.12, 1.14],  # tetO cipro -IPTG
        "A4": [1.00, 1.00, 1.00, 1.12, 1.15, 1.18],  # tetO cipro +IPTG
        "B1": [1.00, 1.00, 1.00, 1.00, 1.00, 1.01],  # LexA H2O -IPTG
        "B2": [1.00, 1.00, 1.00, 1.02, 1.03, 1.04],  # LexA H2O +IPTG
        "B3": [1.00, 1.00, 1.00, 1.12, 1.14, 1.17],  # LexA cipro -IPTG
        "B4": [1.00, 1.00, 1.00, 1.42, 1.50, 1.58],  # LexA cipro +IPTG
    }
    meta = {
        "A1": ("sulAp/tetO", "-IPTG/-stress", "0 uM IPTG"),
        "A2": ("sulAp/tetO", "+IPTG/-stress", "500 uM IPTG"),
        "A3": ("sulAp/tetO", "-IPTG/+stress", "100 nM ciprofloxacin"),
        "A4": ("sulAp/tetO", "+IPTG/+stress", "500 uM IPTG, 100 nM ciprofloxacin"),
        "B1": ("sulAp/LexA", "-IPTG/-stress", "0 uM IPTG"),
        "B2": ("sulAp/LexA", "+IPTG/-stress", "500 uM IPTG"),
        "B3": ("sulAp/LexA", "-IPTG/+stress", "100 nM ciprofloxacin"),
        "B4": ("sulAp/LexA", "+IPTG/+stress", "500 uM IPTG, 100 nM ciprofloxacin"),
    }
    rows: list[dict[str, object]] = []
    for position, (design_id_alias, treatment_alias, treatment) in meta.items():
        for time, od_value, ratio_value in zip(times, od[position], ratio[position], strict=True):
            rows.append(
                {
                    "position": position,
                    "time": time,
                    "channel": "OD600",
                    "value": od_value,
                    "design_id": design_id_alias,
                    "design_id_alias": design_id_alias,
                    "treatment": treatment,
                    "treatment_alias": treatment_alias,
                    "sheet_name": "Plate 2",
                }
            )
            rows.append(
                {
                    "position": position,
                    "time": time,
                    "channel": "RFP/OD600",
                    "value": ratio_value,
                    "design_id": design_id_alias,
                    "design_id_alias": design_id_alias,
                    "treatment": treatment,
                    "treatment_alias": treatment_alias,
                    "sheet_name": "Plate 2",
                }
            )
    return pd.DataFrame(rows)


def _input_df_single_reporter_preload() -> pd.DataFrame:
    times = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]
    od = {
        position: [0.10, 0.15, 0.20, 0.31, 0.36, 0.40] for position in ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4")
    }
    ratio = {
        "A1": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # tetO H2O -IPTG
        "A2": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # tetO H2O +IPTG
        "A3": [1.00, 1.00, 1.00, 1.30, 1.31, 1.32],  # tetO cipro -IPTG
        "A4": [1.00, 1.00, 1.00, 1.30, 1.31, 1.32],  # tetO cipro +IPTG
        "B1": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # LexA H2O -IPTG
        "B2": [1.30, 1.30, 1.30, 1.30, 1.31, 1.32],  # LexA H2O +IPTG preload
        "B3": [1.00, 1.00, 1.00, 1.60, 1.62, 1.64],  # LexA cipro -IPTG
        "B4": [1.30, 1.30, 1.30, 1.75, 1.76, 1.77],  # LexA cipro +IPTG preload + stress
    }
    meta = {
        "A1": ("sulAp/tetO", "-IPTG/-stress", "0 uM IPTG"),
        "A2": ("sulAp/tetO", "+IPTG/-stress", "500 uM IPTG"),
        "A3": ("sulAp/tetO", "-IPTG/+stress", "100 nM ciprofloxacin"),
        "A4": ("sulAp/tetO", "+IPTG/+stress", "500 uM IPTG, 100 nM ciprofloxacin"),
        "B1": ("sulAp/LexA", "-IPTG/-stress", "0 uM IPTG"),
        "B2": ("sulAp/LexA", "+IPTG/-stress", "500 uM IPTG"),
        "B3": ("sulAp/LexA", "-IPTG/+stress", "100 nM ciprofloxacin"),
        "B4": ("sulAp/LexA", "+IPTG/+stress", "500 uM IPTG, 100 nM ciprofloxacin"),
    }
    rows: list[dict[str, object]] = []
    for position, (design_id_alias, treatment_alias, treatment) in meta.items():
        for time, od_value, ratio_value in zip(times, od[position], ratio[position], strict=True):
            rows.append(
                {
                    "position": position,
                    "time": time,
                    "channel": "OD600",
                    "value": od_value,
                    "design_id": design_id_alias,
                    "design_id_alias": design_id_alias,
                    "treatment": treatment,
                    "treatment_alias": treatment_alias,
                    "sheet_name": "Plate 3",
                }
            )
            rows.append(
                {
                    "position": position,
                    "time": time,
                    "channel": "RFP/OD600",
                    "value": ratio_value,
                    "design_id": design_id_alias,
                    "design_id_alias": design_id_alias,
                    "treatment": treatment,
                    "treatment_alias": treatment_alias,
                    "sheet_name": "Plate 3",
                }
            )
    return pd.DataFrame(rows)


def _input_df_explicit_semantics() -> pd.DataFrame:
    df = _input_df()
    design_parts = df["design_id_alias"].str.split("/", n=1, expand=True)
    df["sensor_id"] = design_parts[0]
    df["sponge_id"] = design_parts[1]
    df["genotype_id"] = df["design_id_alias"]
    df["stress_condition_id"] = df["treatment"].map(lambda value: "3% EtOH" if "EtOH" in str(value) else "H2O")
    df["is_relevant_stress_flag"] = df["stress_condition_id"].eq("3% EtOH")
    df["expected_sign"] = -1
    df["is_relevant_pair"] = df["sponge_id"].eq("CpxR")
    df["matched_control_group_id"] = df.apply(
        lambda row: (
            f"{row['sheet_name']}::{row['sensor_id']}::{row['stress_condition_id']}::"
            f"{'-IPTG' if row['treatment_alias'].startswith('-IPTG') else '+IPTG'}"
        ),
        axis=1,
    )
    df["sponge_family"] = df["sponge_id"].map(lambda value: "control" if value == "tetO" else "mono")
    return df


def test_retron_sponge_metrics_plugin_emits_trace_and_summary_tables():
    plugin = RetronSpongeMetrics()
    cfg = RetronSpongeMetricsCfg(
        stress_time_zero_h=1.5,
        relevant_stress_map={"spyP": "3% EtOH"},
        sensor_target_map={"spyP": ["CpxR", "BaeR"]},
    )

    outputs = plugin.run(_ctx(), {"df": _input_df()}, cfg)
    trace = outputs["trace"]
    summary = outputs["summary"]

    assert {"R", "B", "C", "D", "D_abs", "D_growth", "M", "O", "O_abs", "mu"} <= set(trace["metric"])
    assert {
        "R_pre",
        "P_pre",
        "L_pre",
        "D_AUC",
        "D_abs_AUC",
        "D_growth_AUC",
        "M_AUC",
        "O_AUC",
        "O_abs_AUC",
        "S_AUC",
        "S_abs_AUC",
        "T_ratio_AUC",
        "T_finalOD",
    } <= set(summary["metric"])

    d_auc = summary[
        (summary["metric"] == "D_AUC")
        & (summary["sensor"] == "spyP")
        & (summary["sponge"] == "CpxR")
        & (summary["stress_condition"] == "3% EtOH")
    ]["value"].iloc[0]
    o_auc = summary[
        (summary["metric"] == "O_AUC")
        & (summary["sensor"] == "spyP")
        & (summary["sponge"] == "CpxR")
        & (summary["stress_condition"] == "3% EtOH")
    ]["value"].iloc[0]
    s_auc = summary[(summary["metric"] == "S_AUC") & (summary["sensor"] == "spyP") & (summary["sponge"] == "CpxR")][
        "value"
    ].iloc[0]
    l_pre = summary[(summary["metric"] == "L_pre") & (summary["sensor"] == "spyP") & (summary["sponge"] == "CpxR")][
        "value"
    ].iloc[0]

    assert d_auc < 0
    assert o_auc > 0
    assert s_auc > 0
    assert abs(l_pre) < 1e-9


def test_retron_sponge_metrics_plugin_supports_single_reporter_profile():
    plugin = RetronSpongeMetrics()
    cfg = RetronSpongeMetricsCfg(
        measurement_channel="RFP/OD600",
        stress_time_zero_h=1.5,
        relevant_stress_map={"sulAp": "100 nM ciprofloxacin"},
        sensor_target_map={"sulAp": ["LexA"]},
    )

    outputs = plugin.run(_ctx(), {"df": _input_df_single_reporter()}, cfg)
    trace = outputs["trace"]
    summary = outputs["summary"]

    assert {"R", "B", "C", "D", "D_abs", "D_growth", "M", "O", "O_abs", "mu"} <= set(trace["metric"])
    assert {
        "R_pre",
        "P_pre",
        "D_AUC",
        "D_abs_AUC",
        "D_growth_AUC",
        "M_AUC",
        "O_AUC",
        "O_abs_AUC",
        "S_AUC",
        "S_abs_AUC",
        "L_pre",
    } <= set(summary["metric"])

    d_auc = summary[
        (summary["metric"] == "D_AUC")
        & (summary["sensor"] == "sulAp")
        & (summary["sponge"] == "LexA")
        & (summary["stress_condition"] == "100 nM ciprofloxacin")
    ]["value"].iloc[0]
    o_auc = summary[
        (summary["metric"] == "O_AUC")
        & (summary["sensor"] == "sulAp")
        & (summary["sponge"] == "LexA")
        & (summary["stress_condition"] == "100 nM ciprofloxacin")
    ]["value"].iloc[0]
    m_auc = summary[
        (summary["metric"] == "M_AUC")
        & (summary["sensor"] == "sulAp")
        & (summary["sponge"] == "LexA")
        & (summary["stress_condition"] == "100 nM ciprofloxacin")
    ]["value"].iloc[0]

    assert d_auc > 0
    assert o_auc > 0
    assert m_auc > 0


def test_retron_sponge_metrics_cfg_defaults_match_protocol_surface() -> None:
    cfg = RetronSpongeMetricsCfg()

    assert cfg.plate_column is None
    assert cfg.stress_time_zero_policy == "largest_gap_midpoint"
    assert cfg.stress_time_zero_h is None
    assert cfg.max_post_stress_hours == 12.0


def test_retron_sponge_metrics_absolute_companion_preserves_preload_sensitive_signal() -> None:
    plugin = RetronSpongeMetrics()
    cfg = RetronSpongeMetricsCfg(
        measurement_channel="RFP/OD600",
        stress_time_zero_h=1.5,
        relevant_stress_map={"sulAp": "100 nM ciprofloxacin"},
        sensor_target_map={"sulAp": ["LexA"]},
    )

    outputs = plugin.run(_ctx(), {"df": _input_df_single_reporter_preload()}, cfg)
    summary = outputs["summary"]

    d_auc = summary[
        (summary["metric"] == "D_AUC")
        & (summary["sensor"] == "sulAp")
        & (summary["sponge"] == "LexA")
        & (summary["stress_condition"] == "100 nM ciprofloxacin")
    ]["value"].iloc[0]
    d_abs_auc = summary[
        (summary["metric"] == "D_abs_AUC")
        & (summary["sensor"] == "sulAp")
        & (summary["sponge"] == "LexA")
        & (summary["stress_condition"] == "100 nM ciprofloxacin")
    ]["value"].iloc[0]
    p_pre = summary[
        (summary["metric"] == "P_pre")
        & (summary["sensor"] == "sulAp")
        & (summary["sponge"] == "LexA")
        & (summary["stress_condition"] == "100 nM ciprofloxacin")
    ]["value"].iloc[0]

    assert d_auc < 0
    assert d_abs_auc > 0
    assert p_pre > 0


def test_retron_sponge_metrics_masks_invalid_post_stress_rows_without_crashing():
    plugin = RetronSpongeMetrics()
    cfg = RetronSpongeMetricsCfg(
        stress_time_zero_h=1.5,
        relevant_stress_map={"spyP": "3% EtOH"},
        sensor_target_map={"spyP": ["CpxR", "BaeR"]},
    )
    df = _input_df()
    mask = (df["position"] == "B4") & (df["channel"] == "YFP/CFP") & (df["time"] >= 2.5)
    df.loc[mask, "value"] = 0.0

    outputs = plugin.run(_ctx(), {"df": df}, cfg)
    trace = outputs["trace"]
    summary = outputs["summary"]

    invalid_r = trace[(trace["metric"] == "R") & (trace["replicate_id"] == "B4") & (trace["time"] >= 2.5)]
    d_end = summary[
        (summary["metric"] == "D_END")
        & (summary["sensor"] == "spyP")
        & (summary["sponge"] == "CpxR")
        & (summary["stress_condition"] == "3% EtOH")
    ]["value"].iloc[0]

    assert invalid_r["value"].isna().any()
    assert pd.notna(d_end)


def test_retron_sponge_metrics_keeps_one_logical_plate_and_tracks_acquisition_segments() -> None:
    plugin = RetronSpongeMetrics()
    cfg = RetronSpongeMetricsCfg(
        stress_time_zero_h=1.5,
        relevant_stress_map={"spyP": "3% EtOH"},
        sensor_target_map={"spyP": ["CpxR", "BaeR"]},
    )
    df = pd.concat(
        [
            _input_df(),
            _input_df().assign(
                sheet_name="Plate 2",
                time=lambda frame: frame["time"] + 3.5,
                value=lambda frame: frame["value"] * 1.01,
            ),
        ],
        ignore_index=True,
    )

    outputs = plugin.run(_ctx(), {"df": df}, cfg)
    trace = outputs["trace"]

    raw_trace = trace[trace["metric"].isin({"R", "B", "C", "mu"})]

    assert set(trace["plate_id"].astype(str)) == {"plate"}
    assert set(raw_trace["acquisition_segment_id"].astype(str)) == {"Plate 1", "Plate 2"}


def test_retron_sponge_metrics_plugin_accepts_explicit_semantic_columns():
    plugin = RetronSpongeMetrics()
    cfg = RetronSpongeMetricsCfg(
        stress_time_zero_h=1.5,
        sensor_column="sensor_id",
        sponge_column="sponge_id",
        genotype_column="genotype_id",
        stress_condition_column="stress_condition_id",
        relevant_stress_column="is_relevant_stress_flag",
        expected_sign_column="expected_sign",
        relevant_sensor_pair_column="is_relevant_pair",
        matched_control_group_column="matched_control_group_id",
        sponge_family_size_column="sponge_family",
    )

    outputs = plugin.run(_ctx(), {"df": _input_df_explicit_semantics()}, cfg)
    summary = outputs["summary"]

    d_auc = summary[
        (summary["metric"] == "D_AUC")
        & (summary["sensor"] == "spyP")
        & (summary["sponge"] == "CpxR")
        & (summary["stress_condition"] == "3% EtOH")
    ]["value"].iloc[0]

    assert {"R_pre", "P_pre", "D_AUC", "O_AUC", "O_abs_AUC", "S_AUC", "S_abs_AUC"} <= set(summary["metric"])
    assert d_auc < 0


def test_retron_sponge_metrics_respects_post_stress_time_cap() -> None:
    plugin = RetronSpongeMetrics()
    cfg = RetronSpongeMetricsCfg(
        stress_time_zero_h=1.5,
        max_post_stress_hours=1.0,
        endpoint_reads=2,
        relevant_stress_map={"spyP": "3% EtOH"},
        sensor_target_map={"spyP": ["CpxR", "BaeR"]},
    )

    outputs = plugin.run(_ctx(), {"df": _input_df()}, cfg)
    trace = outputs["trace"]

    capped_rows = trace[
        (trace["metric"] == "C")
        & (trace["sensor"] == "spyP")
        & (trace["sponge"] == "CpxR")
        & (trace["stress_condition"] == "3% EtOH")
        & (trace["time"] == 3.0)
    ]
    uncapped_rows = trace[
        (trace["metric"] == "C")
        & (trace["sensor"] == "spyP")
        & (trace["sponge"] == "CpxR")
        & (trace["stress_condition"] == "3% EtOH")
        & (trace["time"] == 2.5)
    ]

    assert not capped_rows["in_primary_post_stress"].any()
    assert uncapped_rows["in_primary_post_stress"].all()


def test_retron_sponge_metrics_encodes_window_and_matching_metadata() -> None:
    plugin = RetronSpongeMetrics()
    cfg = RetronSpongeMetricsCfg(
        stress_time_zero_h=1.5,
        relevant_stress_map={"spyP": "3% EtOH"},
        sensor_target_map={"spyP": ["CpxR", "BaeR"]},
    )

    outputs = plugin.run(_ctx(), {"df": _input_df()}, cfg)
    trace = outputs["trace"]
    summary = outputs["summary"]

    relevant_trace = trace[
        (trace["sensor"] == "spyP")
        & (trace["sponge"] == "CpxR")
        & (trace["stress_condition"] == "3% EtOH")
        & (trace["metric"] == "R")
    ]
    relevant_summary = summary[
        (summary["sensor"] == "spyP")
        & (summary["sponge"] == "CpxR")
        & (summary["stress_condition"] == "3% EtOH")
        & (summary["metric"] == "D_abs_AUC")
    ]

    assert {
        "matched_control_key",
        "summary_window_start_h",
        "summary_window_end_h",
        "summary_window_duration_h",
    } <= set(trace.columns)
    assert {
        "matched_control_key",
        "summary_window_start_h",
        "summary_window_end_h",
        "summary_window_duration_h",
        "pre_stress_read_count",
        "post_stress_read_count",
        "matched_group_sample_count",
        "stress_addition_gap_h",
    } <= set(summary.columns)
    assert set(relevant_trace["matched_control_key"]) == {"plate::spyP::3% EtOH"}
    assert set(relevant_summary["matched_control_key"]) == {"plate::spyP::3% EtOH"}
    assert set(relevant_trace["summary_window_start_h"]) == {0.5}
    assert set(relevant_trace["summary_window_end_h"]) == {1.5}
    assert set(relevant_summary["summary_window_duration_h"]) == {1.0}
    assert set(relevant_summary["pre_stress_read_count"]) == {3.0}
    assert set(relevant_summary["post_stress_read_count"]) == {3.0}
    assert set(relevant_summary["matched_group_sample_count"]) == {2.0}
    assert set(relevant_summary["stress_addition_gap_h"]) == {1.0}


def test_retron_sponge_metrics_preserves_d_abs_additive_contracts() -> None:
    plugin = RetronSpongeMetrics()
    cfg = RetronSpongeMetricsCfg(
        stress_time_zero_h=1.5,
        relevant_stress_map={"spyP": "3% EtOH"},
        sensor_target_map={"spyP": ["CpxR", "BaeR"]},
    )

    outputs = plugin.run(_ctx(), {"df": _input_df()}, cfg)
    trace = outputs["trace"]
    summary = outputs["summary"]

    d_trace = trace[
        (trace["sensor"] == "spyP")
        & (trace["sponge"] == "CpxR")
        & (trace["stress_condition"] == "3% EtOH")
        & (trace["metric"] == "D")
    ].sort_values("time_from_stress")
    d_abs_trace = trace[
        (trace["sensor"] == "spyP")
        & (trace["sponge"] == "CpxR")
        & (trace["stress_condition"] == "3% EtOH")
        & (trace["metric"] == "D_abs")
    ].sort_values("time_from_stress")
    p_pre = summary[
        (summary["sensor"] == "spyP")
        & (summary["sponge"] == "CpxR")
        & (summary["stress_condition"] == "3% EtOH")
        & (summary["metric"] == "P_pre")
    ]["value"].iloc[0]
    d_auc = summary[
        (summary["sensor"] == "spyP")
        & (summary["sponge"] == "CpxR")
        & (summary["stress_condition"] == "3% EtOH")
        & (summary["metric"] == "D_AUC")
    ]["value"].iloc[0]
    d_abs_auc = summary[
        (summary["sensor"] == "spyP")
        & (summary["sponge"] == "CpxR")
        & (summary["stress_condition"] == "3% EtOH")
        & (summary["metric"] == "D_abs_AUC")
    ]["value"].iloc[0]
    window_duration = summary[
        (summary["sensor"] == "spyP")
        & (summary["sponge"] == "CpxR")
        & (summary["stress_condition"] == "3% EtOH")
        & (summary["metric"] == "D_abs_AUC")
    ]["summary_window_duration_h"].iloc[0]

    pointwise_delta = d_abs_trace["value"].reset_index(drop=True) - d_trace["value"].reset_index(drop=True)

    assert all(value == pytest.approx(p_pre) for value in pointwise_delta)
    assert d_abs_auc == pytest.approx(d_auc + p_pre * window_duration)


def test_retron_sponge_metrics_flags_unstable_scaled_metrics() -> None:
    plugin = RetronSpongeMetrics()
    cfg = RetronSpongeMetricsCfg(
        stress_time_zero_h=1.5,
        relevant_stress_map={"spyP": "3% EtOH"},
        sensor_target_map={"spyP": ["CpxR", "BaeR"]},
        min_abs_g_sensor=0.1,
    )
    df = _input_df()
    mask = (
        (df["design_id_alias"] == "spyP/tetO")
        & (df["treatment_alias"] == "-IPTG/+stress")
        & (df["channel"] == "YFP/CFP")
        & (df["time"] >= 2.0)
    )
    df.loc[mask, "value"] = 1.00

    outputs = plugin.run(_ctx(), {"df": df}, cfg)
    summary = outputs["summary"]

    scaled_row = summary[
        (summary["sensor"] == "spyP")
        & (summary["sponge"] == "CpxR")
        & (summary["stress_condition"] == "3% EtOH")
        & (summary["metric"] == "S_abs_AUC")
    ].iloc[0]

    assert pd.isna(scaled_row["value"])
    assert bool(scaled_row["scaling_available"]) is False
    assert scaled_row["warning_flag"] == "unstable_scaled_metric"
    assert scaled_row["scale_min_abs_g_sensor"] == pytest.approx(0.1)
