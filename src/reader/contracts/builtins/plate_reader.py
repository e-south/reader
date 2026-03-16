"""
--------------------------------------------------------------------------------
<reader project>
src/reader/contracts/builtins/plate_reader.py

Plate-reader specific dataframe contracts layered on top of generic tidy data.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..model import ColumnRule, DataFrameContract

CONTRACTS: tuple[DataFrameContract, ...] = (
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
    ),
    DataFrameContract(
        id="fold_change.v1",
        description="Fold-change summary table per (group..., treatment, time, target).",
        columns=[
            ColumnRule("target", "string"),
            ColumnRule("time", "float", nonnegative=True),
            ColumnRule("treatment", "string"),
            ColumnRule("FC", "float", required=True, allow_nan=True),
            ColumnRule("log2FC", "float", required=True, allow_nan=True),
            ColumnRule("n", "int", required=True, allow_nan=False),
            ColumnRule("baseline_value", "string", required=True, allow_nan=True),
            ColumnRule("baseline_n", "int", required=True, allow_nan=True),
            ColumnRule("baseline_time", "float", required=True, allow_nan=True),
            ColumnRule("genotype", "string", required=False, allow_nan=True),
            ColumnRule("batch", "int", required=False, allow_nan=True),
        ],
        unique_keys=[],
        domain="plate_reader",
        kind="summary-table",
    ),
)
