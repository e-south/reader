"""Generic tabular contracts shared across analysis domains."""

from __future__ import annotations

from ..model import ColumnRule, DataFrameContract

CONTRACTS: tuple[DataFrameContract, ...] = (
    DataFrameContract(
        id="tidy.v1",
        description="Tidy long table: position,str | time,float | channel,str | value,float",
        columns=[
            ColumnRule("position", "string"),
            ColumnRule("time", "float", nonnegative=True),
            ColumnRule("channel", "string"),
            ColumnRule("value", "float"),
        ],
        unique_keys=[],
        domain="tabular",
        kind="measurement-trace",
    ),
)
