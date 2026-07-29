"""Shared validation for strict SFXI dataframe contracts."""

from __future__ import annotations

import pandas as pd

from reader.errors import SFXIError


def require_intensity_delta_column(frame: pd.DataFrame) -> None:
    column = "intensity_log2_offset_delta"
    if column not in frame.columns:
        raise SFXIError(
            f"SFXI vec8 input requires column {column!r}. "
            "Regenerate an sfxi.vec8.v3 table instead of inferring a default."
        )


__all__ = ["require_intensity_delta_column"]
