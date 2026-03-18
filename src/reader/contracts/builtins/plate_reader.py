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
    DataFrameContract(
        id="plate_reader.sponge_trace.v1",
        description="Matched-control sponge-screen trace table with derived kinetics and window membership.",
        columns=[
            ColumnRule("plate_id", "string"),
            ColumnRule("sensor", "string"),
            ColumnRule("sponge", "string"),
            ColumnRule("genotype_id", "string"),
            ColumnRule("replicate_id", "string", required=False, allow_nan=True),
            ColumnRule("stress_condition", "string", required=False, allow_nan=True),
            ColumnRule("IPTG", "string", required=False, allow_nan=True),
            ColumnRule("time", "float", required=True, allow_nan=False, nonnegative=True),
            ColumnRule("time_from_stress", "float", required=True, allow_nan=False),
            ColumnRule("metric", "string"),
            ColumnRule("value", "float", required=True, allow_nan=True),
            ColumnRule("expected_decoy_sign", "int", required=False, allow_nan=True),
            ColumnRule("is_relevant_stress", "bool", required=False, allow_nan=True),
            ColumnRule("relevant_sensor_pair", "bool", required=False, allow_nan=True),
            ColumnRule("sponge_family_size", "string", required=False, allow_nan=True),
            ColumnRule("matched_tetO_group", "string", required=False, allow_nan=True),
            ColumnRule("in_pre_window", "bool", required=False, allow_nan=True),
            ColumnRule("in_primary_post_stress", "bool", required=False, allow_nan=True),
            ColumnRule("in_endpoint_window", "bool", required=False, allow_nan=True),
        ],
        unique_keys=[],
        domain="plate_reader",
        kind="analysis-trace-table",
    ),
    DataFrameContract(
        id="plate_reader.sponge_summary.v1",
        description="Matched-control sponge-screen summary table with AUC, endpoint, burden, and leakiness metrics.",
        columns=[
            ColumnRule("plate_id", "string"),
            ColumnRule("sensor", "string"),
            ColumnRule("sponge", "string", required=False, allow_nan=True),
            ColumnRule("genotype_id", "string", required=False, allow_nan=True),
            ColumnRule("stress_condition", "string", required=False, allow_nan=True),
            ColumnRule("IPTG", "string", required=False, allow_nan=True),
            ColumnRule("metric", "string"),
            ColumnRule("value", "float", required=True, allow_nan=True),
            ColumnRule("expected_decoy_sign", "int", required=False, allow_nan=True),
            ColumnRule("is_relevant_stress", "bool", required=False, allow_nan=True),
            ColumnRule("relevant_sensor_pair", "bool", required=False, allow_nan=True),
            ColumnRule("sponge_family_size", "string", required=False, allow_nan=True),
        ],
        unique_keys=[],
        domain="plate_reader",
        kind="analysis-summary-table",
    ),
)
