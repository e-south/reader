"""Cytometry-domain dataframe contracts."""

from __future__ import annotations

from ..model import ColumnRule, DataFrameContract

CONTRACTS: tuple[DataFrameContract, ...] = (
    DataFrameContract(
        id="cytometer.channels.v1",
        description="Per-sample cytometer channel metadata from FCS headers.",
        columns=[
            ColumnRule("sample_id", "string"),
            ColumnRule("channel_index", "int"),
            ColumnRule("channel_name", "string"),
            ColumnRule("pns", "string", required=False, allow_nan=True),
            ColumnRule("pnn", "string", required=False, allow_nan=True),
            ColumnRule("pnt", "string", required=False, allow_nan=True),
            ColumnRule("pnf", "string", required=False, allow_nan=True),
            ColumnRule("pnl", "string", required=False, allow_nan=True),
            ColumnRule("pnr", "float", required=False, allow_nan=True),
            ColumnRule("pnb", "float", required=False, allow_nan=True),
            ColumnRule("png", "float", required=False, allow_nan=True),
            ColumnRule("pne_decades", "float", required=False, allow_nan=True),
            ColumnRule("pne_zero", "float", required=False, allow_nan=True),
        ],
        unique_keys=[],
        domain="cytometry",
        kind="channel-metadata",
    ),
)
