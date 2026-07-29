from types import SimpleNamespace

import pandas as pd
import pytest

from reader.errors import ExecutionError
from reader.plugins.validator.to_tidy_plus_map import PromoteCfg, PromoteToTidyPlusMap


def _ctx():
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None)
    return SimpleNamespace(logger=logger)


def test_promote_preserves_missing_batch_without_synthesizing():
    df = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["YFP"],
            "value": [1.0],
            "treatment": ["H2O"],
            "design_id": ["design_a__condition_a"],
        }
    )
    cfg = PromoteCfg()
    out = PromoteToTidyPlusMap().run(_ctx(), {"df": df}, cfg)["df"]
    assert "batch" not in out.columns


def test_promote_synthesizes_batch_only_when_explicit():
    df = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["YFP"],
            "value": [1.0],
            "treatment": ["H2O"],
            "design_id": ["design_a__condition_a"],
        }
    )
    cfg = PromoteCfg(synthesize_batch=True, synthesized_batch_value=7)
    out = PromoteToTidyPlusMap().run(_ctx(), {"df": df}, cfg)["df"]
    assert out["batch"].tolist() == [7]


def test_promote_requires_explicit_batch_when_requested():
    df = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["YFP"],
            "value": [1.0],
            "treatment": ["H2O"],
            "design_id": ["design_a__condition_a"],
        }
    )
    cfg = PromoteCfg(require_columns=["treatment", "design_id", "batch"])
    with pytest.raises(ExecutionError, match="missing columns"):
        PromoteToTidyPlusMap().run(_ctx(), {"df": df}, cfg)
