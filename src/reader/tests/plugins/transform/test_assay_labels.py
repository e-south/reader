from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from reader.plugins.transform.assay_labels import AssayLabelsCfg, AssayLabelsTransform


def _ctx(labels):
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None)
    return SimpleNamespace(logger=logger, assay={"labels": labels})


def test_assay_labels_applies_all_labels_by_default():
    df = pd.DataFrame({"design_id": ["a"], "treatment": ["x"]})
    cfg = AssayLabelsCfg()
    out = AssayLabelsTransform().run(
        _ctx(
            {
                "design_id": {"source": "design_id", "output": "design_id_alias", "values": {"a": "A"}},
                "treatment": {"source": "treatment", "output": "treatment_alias", "values": {"x": "X"}},
            }
        ),
        {"df": df},
        cfg,
    )["df"]
    assert out["design_id_alias"].tolist() == ["A"]
    assert out["treatment_alias"].tolist() == ["X"]


def test_assay_labels_can_select_subset_by_ref():
    df = pd.DataFrame({"design_id": ["a"], "treatment": ["x"]})
    cfg = AssayLabelsCfg(refs=["design_id"])
    out = AssayLabelsTransform().run(
        _ctx(
            {
                "design_id": {"source": "design_id", "output": "design_id_alias", "values": {"a": "A"}},
                "treatment": {"source": "treatment", "output": "treatment_alias", "values": {"x": "X"}},
            }
        ),
        {"df": df},
        cfg,
    )["df"]
    assert out["design_id_alias"].tolist() == ["A"]
    assert "treatment_alias" not in out.columns


def test_assay_labels_requires_configured_labels():
    with pytest.raises(ValueError, match="no assay.labels"):
        AssayLabelsTransform().run(_ctx({}), {"df": pd.DataFrame({"design_id": ["a"]})}, AssayLabelsCfg())
