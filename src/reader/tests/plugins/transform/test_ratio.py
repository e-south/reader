"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_ratio_plugin.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from reader.plugins.transform.ratio import RatioCfg, RatioTransform


def _ctx():
    return SimpleNamespace(logger=None)


def test_ratio_requires_align_on_columns():
    df = pd.DataFrame(
        {
            "position": ["A1", "A1"],
            "time": [0.0, 0.0],
            "channel": ["YFP", "CFP"],
            "value": [1.0, 2.0],
        }
    )
    cfg = RatioCfg(name="YFP/CFP", numerator="YFP", denominator="CFP", align_on=["sample_id"])
    with pytest.raises(ValueError, match="align_on"):
        RatioTransform().run(_ctx(), {"df": df}, cfg)


def test_ratio_requires_channels_present():
    df = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["YFP"],
            "value": [1.0],
        }
    )
    cfg = RatioCfg(name="YFP/CFP", numerator="YFP", denominator="CFP")
    with pytest.raises(ValueError, match="missing"):
        RatioTransform().run(_ctx(), {"df": df}, cfg)


def test_ratio_preserves_passthrough_rows_for_nonderived_channels():
    df = pd.DataFrame(
        {
            "position": ["A1", "A1", "A1", "A1", "A1", "A1"],
            "time": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "sheet_index": [0, 0, 0, 0, 0, 0],
            "sheet_name": ["Plate 1"] * 6,
            "channel": ["OD600", "CFP", "YFP", "OD600", "CFP", "YFP"],
            "value": [0.20, 8000.0, 1200.0, 0.25, 9000.0, 1500.0],
            "overflow": [False] * 6,
            "design_id_alias": ["spyP/CpxR"] * 6,
            "treatment_alias": ["-IPTG/-stress"] * 6,
        }
    )

    outputs = RatioTransform().run(
        _ctx(),
        {"df": df},
        RatioCfg(name="YFP/OD600", numerator="YFP", denominator="OD600"),
    )
    out = outputs["df"]

    original_passthrough = df[df["channel"].isin(["OD600", "CFP", "YFP"])].reset_index(drop=True)
    observed_passthrough = out[out["channel"].isin(["OD600", "CFP", "YFP"])].reset_index(drop=True)

    pd.testing.assert_frame_equal(observed_passthrough[df.columns], original_passthrough)
    assert "value_policy_clipped" not in out.columns
    assert "value_instrument_overflow" not in out.columns
    assert "value_bound_kind" not in out.columns
    derived = out[out["channel"] == "YFP/OD600"].sort_values("time").reset_index(drop=True)
    assert derived["value"].tolist() == pytest.approx([6000.0, 6000.0])


@pytest.mark.parametrize("denominator", ["CFP", "OD600"])
def test_ratio_combines_numerator_and_denominator_observation_provenance(denominator: str) -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A1", "A1", "A1"],
            "time": [0.0, 0.0, 1.0, 1.0],
            "channel": ["YFP", denominator, "YFP", denominator],
            "value": [50.0, 10.0, 20.0, 5.0],
            "overflow": [False, False, False, True],
            "value_policy_clipped": [True, False, False, False],
            "value_instrument_overflow": [False, False, False, True],
            "value_bound_kind": ["lower", "exact", "exact", "lower"],
        }
    )

    result = RatioTransform().run(
        _ctx(),
        {"df": frame},
        RatioCfg(name=f"YFP/{denominator}", numerator="YFP", denominator=denominator),
    )["df"]

    derived = result.loc[result["channel"].eq(f"YFP/{denominator}")].sort_values("time")
    assert derived["value_policy_clipped"].tolist() == [True, False]
    assert derived["value_instrument_overflow"].tolist() == [False, True]
    assert derived["value_bound_kind"].tolist() == ["lower", "upper"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.assign(value_policy_clipped=["False", False]),
        lambda frame: frame.assign(value_instrument_overflow=[True, False]),
        lambda frame: frame.assign(value_bound_kind=["lower", "exact"]),
        lambda frame: frame.assign(overflow=[True, False]),
    ],
)
def test_ratio_rejects_contradictory_explicit_value_provenance(mutate) -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A1"],
            "time": [0.0, 0.0],
            "channel": ["YFP", "CFP"],
            "value": [10.0, 2.0],
            "overflow": [False, False],
            "value_policy_clipped": [False, False],
            "value_instrument_overflow": [False, False],
            "value_bound_kind": ["exact", "exact"],
        }
    )

    with pytest.raises(ValueError, match="provenance|boolean|overflow"):
        RatioTransform().run(
            _ctx(),
            {"df": mutate(frame)},
            RatioCfg(name="YFP/CFP", numerator="YFP", denominator="CFP"),
        )


def test_ratio_rejects_partial_explicit_value_provenance() -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A1"],
            "time": [0.0, 0.0],
            "channel": ["YFP", "CFP"],
            "value": [10.0, 2.0],
            "value_bound_kind": ["exact", "exact"],
        }
    )

    with pytest.raises(ValueError, match="all three explicit"):
        RatioTransform().run(
            _ctx(),
            {"df": frame},
            RatioCfg(name="YFP/CFP", numerator="YFP", denominator="CFP"),
        )


@pytest.mark.parametrize("values", [[-10.0, 2.0], [10.0, -2.0]])
def test_ratio_rejects_nonpositive_operands_for_bounded_values(values: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "position": ["A1", "A1"],
            "time": [0.0, 0.0],
            "channel": ["YFP", "CFP"],
            "value": values,
            "value_policy_clipped": [True, False],
            "value_instrument_overflow": [False, False],
            "value_bound_kind": ["lower", "exact"],
        }
    )

    with pytest.raises(ValueError, match="positive operands"):
        RatioTransform().run(
            _ctx(),
            {"df": frame},
            RatioCfg(name="YFP/CFP", numerator="YFP", denominator="CFP"),
        )
