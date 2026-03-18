from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd

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

    assert {"R", "B", "C", "D", "M", "O", "mu"} <= set(trace["metric"])
    assert {"R_pre", "L_pre", "D_AUC", "M_AUC", "O_AUC", "S_AUC", "T_ratio_AUC", "T_finalOD"} <= set(summary["metric"])

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

    assert {"R", "B", "C", "D", "M", "O", "mu"} <= set(trace["metric"])
    assert {"R_pre", "D_AUC", "M_AUC", "O_AUC", "S_AUC", "L_pre"} <= set(summary["metric"])

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
