"""
--------------------------------------------------------------------------------
<reader project>
src/reader/contracts/builtins/logic.py

Logic-domain dataframe contracts.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..model import ColumnRule, DataFrameContract


def _sfxi_vec8_columns() -> list[ColumnRule]:
    return [
        ColumnRule("design_id", "string"),
        ColumnRule("sequence", "string", required=False, allow_nan=True),
        ColumnRule("id", "string", required=False, allow_nan=True),
        ColumnRule("time_selected_h", "float", required=False, allow_nan=True),
        ColumnRule("reference_design_id", "string"),
        ColumnRule(
            "intensity_log2_offset_delta",
            "float",
            nonnegative=True,
        ),
        ColumnRule("r_logic", "float", nonnegative=True),
        ColumnRule("v00", "float"),
        ColumnRule("v10", "float"),
        ColumnRule("v01", "float"),
        ColumnRule("v11", "float"),
        ColumnRule("y00_star", "float"),
        ColumnRule("y10_star", "float"),
        ColumnRule("y01_star", "float"),
        ColumnRule("y11_star", "float"),
        ColumnRule("flat_logic", "bool"),
    ]


CONTRACTS: tuple[DataFrameContract, ...] = (
    DataFrameContract(
        id="sfxi.vec8.v3",
        description=("Per design vec8 table with explicit intensity transform parameters and reference provenance"),
        columns=_sfxi_vec8_columns(),
        unique_keys=[["design_id"]],
        domain="logic",
        kind="logic-summary",
    ),
    DataFrameContract(
        id="crosstalk_pairs.v1",
        description="Pairwise crosstalk summary for design pairs at a single time/target.",
        columns=[
            ColumnRule("design_a", "string"),
            ColumnRule("design_b", "string"),
            ColumnRule("treatment_a", "string", allow_nan=True),
            ColumnRule("treatment_b", "string", allow_nan=True),
            ColumnRule("target", "string", allow_nan=True),
            ColumnRule("time", "float", nonnegative=True, allow_nan=True),
            ColumnRule("value_column", "string"),
            ColumnRule("value_scale", "string", allowed_values=["log2", "linear"]),
            ColumnRule("a_self_value", "float", allow_nan=True),
            ColumnRule("b_self_value", "float", allow_nan=True),
            ColumnRule("a_cross_to_b", "float", allow_nan=True),
            ColumnRule("b_cross_to_a", "float", allow_nan=True),
            ColumnRule("a_best_other_treatment", "string", required=False, allow_nan=True),
            ColumnRule("b_best_other_treatment", "string", required=False, allow_nan=True),
            ColumnRule("a_best_other_value", "float", required=False, allow_nan=True),
            ColumnRule("b_best_other_value", "float", required=False, allow_nan=True),
            ColumnRule("a_self_minus_best_other", "float", required=False, allow_nan=True),
            ColumnRule("b_self_minus_best_other", "float", required=False, allow_nan=True),
            ColumnRule("a_self_ratio_best_other", "float", required=False, allow_nan=True),
            ColumnRule("b_self_ratio_best_other", "float", required=False, allow_nan=True),
            ColumnRule("a_self_is_top1", "bool"),
            ColumnRule("b_self_is_top1", "bool"),
            ColumnRule("a_top1_treatment", "string", allow_nan=True),
            ColumnRule("a_top2_treatment", "string", allow_nan=True),
            ColumnRule("b_top1_treatment", "string", allow_nan=True),
            ColumnRule("b_top2_treatment", "string", allow_nan=True),
            ColumnRule("a_top1_value", "float", allow_nan=True),
            ColumnRule("a_top2_value", "float", allow_nan=True),
            ColumnRule("b_top1_value", "float", allow_nan=True),
            ColumnRule("b_top2_value", "float", allow_nan=True),
            ColumnRule("a_selectivity_delta", "float", allow_nan=True),
            ColumnRule("b_selectivity_delta", "float", allow_nan=True),
            ColumnRule("a_selectivity_ratio", "float", allow_nan=True),
            ColumnRule("b_selectivity_ratio", "float", allow_nan=True),
            ColumnRule("pair_score", "float", allow_nan=True),
            ColumnRule("pair_ratio", "float", allow_nan=True),
            ColumnRule("passes_filters", "bool"),
        ],
        unique_keys=[],
        domain="logic",
        kind="pairwise-summary",
    ),
    DataFrameContract(
        id="logic_symmetry.v1",
        description="Logic-symmetry per design (and batch, when present) summary (points + metrics + encodings).",
        columns=[
            ColumnRule("genotype", "string", required=False, allow_nan=True),
            ColumnRule("strain", "string", required=False, allow_nan=True),
            ColumnRule("design", "string", required=False, allow_nan=True),
            ColumnRule("construct", "string", required=False, allow_nan=True),
            ColumnRule("batch", "int", required=False, allow_nan=True),
            ColumnRule("n00", "int"),
            ColumnRule("n10", "int"),
            ColumnRule("n01", "int"),
            ColumnRule("n11", "int"),
            ColumnRule("b00", "float"),
            ColumnRule("b10", "float"),
            ColumnRule("b01", "float"),
            ColumnRule("b11", "float"),
            ColumnRule("sd00", "float"),
            ColumnRule("sd10", "float"),
            ColumnRule("sd01", "float"),
            ColumnRule("sd11", "float"),
            ColumnRule("r", "float"),
            ColumnRule("log_r", "float"),
            ColumnRule("cv", "float"),
            ColumnRule("u00", "float"),
            ColumnRule("u10", "float"),
            ColumnRule("u01", "float"),
            ColumnRule("u11", "float"),
            ColumnRule("L", "float"),
            ColumnRule("A", "float"),
            ColumnRule("baseline_corner", "string"),
            ColumnRule("baseline_value", "float"),
            ColumnRule("size_value", "float", required=False, allow_nan=True),
            ColumnRule("hue_value", "string", required=False, allow_nan=True),
            ColumnRule("alpha_value", "float", required=False, allow_nan=True),
            ColumnRule("shape_value", "string", required=False, allow_nan=True),
        ],
        unique_keys=[],
        domain="logic",
        kind="summary-table",
    ),
)
