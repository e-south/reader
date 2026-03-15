"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/contracts/plate_reader.py

Plate-reader specific contracts layered on top of generic tidy measurements.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .model import ColumnRule, DataFrameContract
from .registry import register_contract

register_contract(
    DataFrameContract(
        id="plate_reader.annotated.v1",
        description="Plate-reader tidy table with required mapped metadata: treatment,str | design_id,str | batch,float (optional).",
        columns=[
            ColumnRule("position", "string"),
            ColumnRule("time", "float", nonnegative=True),
            ColumnRule("channel", "string"),
            ColumnRule("value", "float"),
            ColumnRule("treatment", "string"),
            ColumnRule("design_id", "string"),
            ColumnRule("batch", "float", required=False, allow_nan=True),
        ],
        unique_keys=[],
        parents=("tidy.v1",),
        domain="plate_reader",
        kind="annotated-measurement-trace",
    )
)
