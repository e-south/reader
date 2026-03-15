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
