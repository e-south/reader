"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_crosstalk_pairs.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pandas as pd
import pytest

from reader.lib.crosstalk import compute_crosstalk_pairs


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "design_id": ["A", "A", "B", "B"],
            "treatment": ["tA", "tB", "tA", "tB"],
            "time": [12.0, 12.0, 12.0, 12.0],
            "target": ["YFP/CFP"] * 4,
            "log2FC": [3.0, 0.2, 0.1, 2.5],
            "cognate_treatment": ["tA", "tA", "tB", "tB"],
        }
    )


def test_crosstalk_pairs_basic_log2():
    df = _base_df()
    res = compute_crosstalk_pairs(
        df,
        design_col="design_id",
        treatment_col="treatment",
        value_col="log2FC",
        value_scale="log2",
        target="YFP/CFP",
        time_mode="exact",
        time=12.0,
        mapping_mode="column",
        design_treatment_column="cognate_treatment",
        min_self=1.0,
        max_cross=0.5,
        require_self_is_top1=True,
        only_passing=True,
    )

    assert res.pairs.shape[0] == 1
    row = res.pairs.iloc[0]
    assert row["design_a"] == "A"
    assert row["design_b"] == "B"
    assert bool(row["passes_filters"]) is True
    assert row["a_self_value"] == pytest.approx(3.0)
    assert row["b_self_value"] == pytest.approx(2.5)
    assert row["a_cross_to_b"] == pytest.approx(0.2)
    assert row["b_cross_to_a"] == pytest.approx(0.1)
    assert row["a_best_other_value"] == pytest.approx(0.2)
    assert row["b_best_other_value"] == pytest.approx(0.1)
    assert row["a_self_minus_best_other"] == pytest.approx(2.8)
    assert row["b_self_minus_best_other"] == pytest.approx(2.4)
    assert row["pair_score"] == pytest.approx(2.3)


def test_crosstalk_pairs_requires_self_top1():
    df = pd.DataFrame(
        {
            "design_id": ["A", "A", "B", "B"],
            "treatment": ["tA", "tB", "tA", "tB"],
            "time": [12.0, 12.0, 12.0, 12.0],
            "target": ["YFP/CFP"] * 4,
            "log2FC": [1.0, 2.0, 2.0, 1.0],
            "cognate_treatment": ["tA", "tA", "tB", "tB"],
        }
    )
    res = compute_crosstalk_pairs(
        df,
        design_col="design_id",
        treatment_col="treatment",
        value_col="log2FC",
        value_scale="log2",
        target="YFP/CFP",
        time_mode="exact",
        time=12.0,
        mapping_mode="column",
        design_treatment_column="cognate_treatment",
        require_self_is_top1=True,
        only_passing=True,
    )

    assert res.pairs.empty


def test_crosstalk_pairs_max_other_filter():
    df = _base_df()
    res = compute_crosstalk_pairs(
        df,
        design_col="design_id",
        treatment_col="treatment",
        value_col="log2FC",
        value_scale="log2",
        target="YFP/CFP",
        time_mode="exact",
        time=12.0,
        mapping_mode="column",
        design_treatment_column="cognate_treatment",
        max_other=0.15,
        only_passing=True,
    )

    assert res.pairs.empty


def test_crosstalk_pairs_target_requires_column():
    df = _base_df().drop(columns=["target"])
    with pytest.raises(ValueError, match="target specified"):
        compute_crosstalk_pairs(
            df,
            design_col="design_id",
            treatment_col="treatment",
            value_col="log2FC",
            value_scale="log2",
            target="YFP/CFP",
            time_mode="exact",
            time=12.0,
            mapping_mode="column",
            design_treatment_column="cognate_treatment",
        )


def test_crosstalk_pairs_all_times():
    df = _base_df()
    df2 = df.copy()
    df2["time"] = 8.0
    df_all = pd.concat([df, df2], ignore_index=True)

    res = compute_crosstalk_pairs(
        df_all,
        design_col="design_id",
        treatment_col="treatment",
        value_col="log2FC",
        value_scale="log2",
        target="YFP/CFP",
        time_mode="all",
        mapping_mode="column",
        design_treatment_column="cognate_treatment",
        require_self_is_top1=True,
        only_passing=True,
    )

    assert sorted(res.pairs["time"].unique().tolist()) == [8.0, 12.0]


def test_crosstalk_pairs_time_policy_all():
    df = _base_df()
    df2 = df.copy()
    df2["time"] = 8.0
    df2.loc[(df2["design_id"] == "A") & (df2["treatment"] == "tB"), "log2FC"] = 1.0
    df_all = pd.concat([df, df2], ignore_index=True)

    res = compute_crosstalk_pairs(
        df_all,
        design_col="design_id",
        treatment_col="treatment",
        value_col="log2FC",
        value_scale="log2",
        target="YFP/CFP",
        time_mode="all",
        time_policy="all",
        mapping_mode="column",
        design_treatment_column="cognate_treatment",
        max_cross=0.5,
        only_passing=True,
    )

    assert res.pairs.empty
