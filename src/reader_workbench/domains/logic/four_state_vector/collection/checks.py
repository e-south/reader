from __future__ import annotations

import math

import pandas as pd

from reader_workbench.errors import FourStateVectorError

from ..validation import require_intensity_delta_column
from .constants import REQUIRED_VECTOR_COLUMNS


def require_vector_columns(frame: pd.DataFrame) -> None:
    require_intensity_delta_column(frame)
    missing = [column for column in REQUIRED_VECTOR_COLUMNS if column not in frame.columns]
    if missing:
        raise FourStateVectorError(f"four-state vector input requires columns: {', '.join(missing)}.")


def require_normalized_frame(frame: pd.DataFrame) -> None:
    require_vector_columns(frame)
    source_columns = {"source_resource_id", "source_experiment_id"} & set(frame.columns)
    missing = ([] if source_columns else ["source_resource_id or source_experiment_id"]) + (
        [] if "row_label" in frame.columns else ["row_label"]
    )
    if missing:
        raise FourStateVectorError(f"four-state vector collection requires normalized columns: {', '.join(missing)}.")
    if frame.empty:
        raise FourStateVectorError("four-state vector collection has no rows to plot.")


def finite_numeric_column(
    series: pd.Series,
    *,
    column: str,
    source: str,
    allow_nan: bool = False,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    coerced_to_nan = series.notna() & values.isna()
    nonfinite = values.notna() & ~values.fillna(0.0).map(lambda value: math.isfinite(float(value)))
    invalid = coerced_to_nan | nonfinite
    if not allow_nan:
        invalid |= values.isna()
    if invalid.any():
        raise FourStateVectorError(
            f"four-state vector collection column {column!r} must be finite numeric values in {source}."
        )
    return values.astype(float)
