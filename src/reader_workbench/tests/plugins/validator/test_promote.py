from types import SimpleNamespace

import pandas as pd
import pytest

from reader_workbench.errors import ExecutionError
from reader_workbench.plugins.validator.to_tidy_plus_map import PromoteCfg, PromoteToTidyPlusMap


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


def test_promote_validates_required_metadata_after_sample_only_filtering():
    df = pd.DataFrame(
        {
            "position": ["A1", "H12"],
            "time": [0.0, 0.0],
            "channel": ["RFP/OD600", "RFP/OD600"],
            "value": [2.0, 0.0],
            "type": ["SAMPLE", "BLANK"],
            "treatment": ["condition-a", None],
            "design_id": ["sample-a", None],
        }
    )
    cfg = PromoteCfg(
        include_types=["SAMPLE"],
        require_columns=["treatment", "design_id"],
        require_non_null=True,
    )

    out = PromoteToTidyPlusMap().run(_ctx(), {"df": df}, cfg)["df"]

    assert out[["position", "type", "treatment", "design_id"]].to_dict("records") == [
        {
            "position": "A1",
            "type": "SAMPLE",
            "treatment": "condition-a",
            "design_id": "sample-a",
        }
    ]


def test_promote_trims_and_requires_non_blank_configured_columns():
    df = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["RFP/OD600"],
            "value": [2.0],
            "type": ["SAMPLE"],
            "treatment": ["  condition-a  "],
            "design_id": ["  sample-a  "],
        }
    )
    cfg = PromoteCfg(
        include_types=["SAMPLE"],
        require_columns=["treatment", "design_id"],
        require_non_null=True,
        trim_and_require_non_blank=["treatment", "design_id"],
        require_finite=["time", "value"],
    )

    out = PromoteToTidyPlusMap().run(_ctx(), {"df": df}, cfg)["df"]

    assert out[["treatment", "design_id"]].to_dict("records") == [{"treatment": "condition-a", "design_id": "sample-a"}]


@pytest.mark.parametrize(
    ("column", "invalid"),
    [
        ("treatment", ""),
        ("treatment", "   "),
        ("design_id", ""),
        ("design_id", "\t"),
    ],
)
def test_promote_rejects_blank_configured_identity(column: str, invalid: str):
    df = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["RFP/OD600"],
            "value": [2.0],
            "type": ["SAMPLE"],
            "treatment": ["condition-a"],
            "design_id": ["sample-a"],
        }
    )
    df.loc[0, column] = invalid
    cfg = PromoteCfg(
        include_types=["SAMPLE"],
        require_columns=["treatment", "design_id"],
        require_non_null=True,
        trim_and_require_non_blank=["treatment", "design_id"],
    )

    with pytest.raises(ExecutionError, match="blank values"):
        PromoteToTidyPlusMap().run(_ctx(), {"df": df}, cfg)


@pytest.mark.parametrize(
    ("column", "invalid"),
    [
        ("time", float("inf")),
        ("value", float("-inf")),
        ("time", float("nan")),
        ("value", float("nan")),
    ],
)
def test_promote_rejects_nonfinite_configured_measurements(column: str, invalid: float):
    df = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["RFP/OD600"],
            "value": [2.0],
            "type": ["SAMPLE"],
            "treatment": ["condition-a"],
            "design_id": ["sample-a"],
        }
    )
    df.loc[0, column] = invalid
    cfg = PromoteCfg(require_finite=["time", "value"])

    with pytest.raises(ExecutionError, match="non-finite values"):
        PromoteToTidyPlusMap().run(_ctx(), {"df": df}, cfg)
