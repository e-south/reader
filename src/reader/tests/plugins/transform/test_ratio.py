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

    pd.testing.assert_frame_equal(observed_passthrough, original_passthrough)
    derived = out[out["channel"] == "YFP/OD600"].sort_values("time").reset_index(drop=True)
    assert derived["value"].tolist() == pytest.approx([6000.0, 6000.0])
