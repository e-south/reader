from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from reader_workbench.errors import FourStateVectorError

from .collection.checks import require_vector_columns
from .collection.constants import VECTOR_CHANNELS
from .collection.render import render_four_state_vector_collection_heatmap

_NUMERIC_COLUMNS = (
    "intensity_log2_offset_delta",
    "r_logic",
    *VECTOR_CHANNELS,
)


def normalize_experiment_heatmap_frame(vector: pd.DataFrame, *, experiment_id: str) -> pd.DataFrame:
    """Adapt one experiment's logic.four_state_vector.v1 table to the heatmap renderer contract."""

    require_vector_columns(vector)
    if vector.empty:
        raise FourStateVectorError("four-state vector heatmap has no rows to plot.")
    normalized_experiment_id = _non_empty_text(experiment_id, field="experiment_id")

    frame = vector.copy().reset_index(drop=True)
    _require_non_empty_text_columns(frame, columns=("design_id", "reference_design_id"))
    for column in _NUMERIC_COLUMNS:
        frame[column] = _finite_numeric(frame[column], column=column)
    if "time_selected_h" in frame.columns:
        frame["time_selected_h"] = _finite_numeric(
            frame["time_selected_h"],
            column="time_selected_h",
            allow_nan=True,
        )
    frame["flat_logic"] = frame["flat_logic"].astype(bool)

    frame["source_index"] = 0
    frame["source_experiment_id"] = normalized_experiment_id
    frame["source_row_index"] = range(len(frame))
    frame["row_label"] = _row_labels(frame, experiment_id=normalized_experiment_id)
    return frame


def render_experiment_four_state_vector_heatmap(
    vector: pd.DataFrame,
    *,
    experiment_id: str,
    title: str | None = None,
    max_y_tick_labels: int = 80,
):
    frame = normalize_experiment_heatmap_frame(vector, experiment_id=experiment_id)
    return render_four_state_vector_collection_heatmap(frame, title=title, max_y_tick_labels=max_y_tick_labels)


def _row_labels(frame: pd.DataFrame, *, experiment_id: str) -> list[str]:
    return [f"{experiment_id}::{design_id}" for design_id in frame["design_id"].astype(str).tolist()]


def _finite_numeric(series: pd.Series, *, column: str, allow_nan: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    coerced_to_nan = series.notna() & values.isna()
    nonfinite = values.notna() & ~values.fillna(0.0).map(lambda value: math.isfinite(float(value)))
    invalid = coerced_to_nan | nonfinite
    if not allow_nan:
        invalid |= values.isna()
    if invalid.any():
        raise FourStateVectorError(f"four-state vector heatmap column {column!r} must contain finite numeric values.")
    return values.astype(float)


def _require_non_empty_text_columns(frame: pd.DataFrame, *, columns: Sequence[str]) -> None:
    for column in columns:
        values = frame[column].astype(str).str.strip()
        if values.eq("").any():
            raise FourStateVectorError(f"four-state vector heatmap column {column!r} must contain non-empty values.")


def _non_empty_text(value: str, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise FourStateVectorError(f"four-state vector heatmap {field} must be a non-empty string.")
    return text
