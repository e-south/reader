"""Shared validation for strict four-state vector dataframe contracts."""

from __future__ import annotations

import pandas as pd

from reader_workbench.errors import FourStateVectorError


def require_intensity_delta_column(frame: pd.DataFrame) -> None:
    column = "intensity_log2_offset_delta"
    if column not in frame.columns:
        raise FourStateVectorError(
            f"four-state vector input requires column {column!r}. "
            "Regenerate an logic.four_state_vector.v1 table instead of inferring a default."
        )


__all__ = ["require_intensity_delta_column"]
