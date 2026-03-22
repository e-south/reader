from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def contextualize_retron_plot_copy(
    *,
    question: str,
    math: str,
    meaning: str,
    supporting_table: pd.DataFrame | None,
    relevant_stress_map: Mapping[str, str] | None = None,
    no_stress_label: str = "H2O",
) -> dict[str, str]:
    stress_phrase = _infer_relevant_stress_phrase(
        supporting_table=supporting_table,
        relevant_stress_map=relevant_stress_map,
        no_stress_label=no_stress_label,
    )
    if stress_phrase is None:
        return {"question": question, "math": math, "meaning": meaning}
    return {
        "question": _replace_relevant_stress_text(question, stress_phrase),
        "math": _replace_relevant_stress_text(math, stress_phrase),
        "meaning": _replace_relevant_stress_text(meaning, stress_phrase),
    }


def _infer_relevant_stress_phrase(
    *,
    supporting_table: pd.DataFrame | None,
    relevant_stress_map: Mapping[str, str] | None,
    no_stress_label: str,
) -> str | None:
    sensor_stress_pairs: dict[str, str] = {}
    if supporting_table is not None and not supporting_table.empty:
        frame = supporting_table.copy()
        if "stress_condition" in frame.columns:
            stress_rows = frame[frame["stress_condition"].notna()].copy()
            stress_rows["stress_condition"] = stress_rows["stress_condition"].astype(str)
            stress_rows = stress_rows[
                stress_rows["stress_condition"].str.strip().ne("")
                & stress_rows["stress_condition"].str.strip().ne(str(no_stress_label))
            ]
            if "sensor" in stress_rows.columns:
                for sensor, values in stress_rows.groupby(stress_rows["sensor"].astype(str), sort=True)[
                    "stress_condition"
                ]:
                    unique_values = sorted(
                        {value for value in values.astype(str) if value and value != str(no_stress_label)}
                    )
                    if len(unique_values) == 1:
                        sensor_stress_pairs[str(sensor)] = unique_values[0]
            unique_stresses = sorted({value for value in stress_rows["stress_condition"].astype(str) if value})
            if len(unique_stresses) == 1:
                return unique_stresses[0]
        if not sensor_stress_pairs and {"treatment", "treatment_alias"}.issubset(frame.columns):
            stress_rows = frame[
                frame["treatment_alias"].astype(str).str.contains(r"\+stress", regex=True, na=False)
            ].copy()
            stress_rows["__stress_label"] = stress_rows["treatment"].map(_extract_stress_from_treatment)
            stress_rows = stress_rows[stress_rows["__stress_label"].notna()].copy()
            if "sensor" in stress_rows.columns:
                for sensor, values in stress_rows.groupby(stress_rows["sensor"].astype(str), sort=True)[
                    "__stress_label"
                ]:
                    unique_values = sorted({str(value) for value in values if pd.notna(value)})
                    if len(unique_values) == 1:
                        sensor_stress_pairs[str(sensor)] = unique_values[0]
            unique_stresses = sorted({str(value) for value in stress_rows["__stress_label"] if pd.notna(value)})
            if len(unique_stresses) == 1:
                return unique_stresses[0]
    if not sensor_stress_pairs and relevant_stress_map:
        sensor_stress_pairs = {
            str(sensor): str(stress) for sensor, stress in relevant_stress_map.items() if str(stress).strip()
        }
    if not sensor_stress_pairs:
        return None
    unique_stresses = sorted(set(sensor_stress_pairs.values()))
    if len(unique_stresses) == 1:
        return unique_stresses[0]
    ordered_pairs = "; ".join(f"{sensor}: {sensor_stress_pairs[sensor]}" for sensor in sorted(sensor_stress_pairs))
    return f"sensor-matched stress ({ordered_pairs})"


def _extract_stress_from_treatment(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split(",")]
    stress_parts = [part for part in parts if "IPTG" not in part]
    if not stress_parts:
        return None
    stress = ", ".join(part for part in stress_parts if part and part != "H2O").strip()
    return stress or None


def _replace_relevant_stress_text(text: str, stress_phrase: str) -> str:
    updated = str(text)
    replacements = (
        ("Relevant-stress", stress_phrase),
        ("relevant-stress", stress_phrase),
        ("Relevant stress", stress_phrase),
        ("relevant stress", stress_phrase),
    )
    for source, target in replacements:
        updated = updated.replace(source, target)
    return updated
