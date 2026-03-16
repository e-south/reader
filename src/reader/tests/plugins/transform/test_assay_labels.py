from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from reader.plugins.transform.assay_labels import AssayLabelsCfg, AssayLabelsTransform
from reader.workbench.experiment import (
    AssayLabels,
    AssayLabelSpec,
    AssaySemantics,
    ExperimentSemantics,
    OutputLayout,
    ResourceCatalog,
)


def _ctx(labels):
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None)
    semantics = ExperimentSemantics(
        assay=AssaySemantics(
            labels=AssayLabels(
                by_id={
                    key: AssayLabelSpec(
                        source=value["source"],
                        values=dict(value.get("values", {})),
                        output=value.get("output"),
                    )
                    for key, value in labels.items()
                }
            )
        ),
        resources=ResourceCatalog(),
        layout=OutputLayout(
            outputs_dir=Path("."), plots_subdir="plots", exports_subdir="exports", notebooks_subdir="notebooks"
        ),
    )
    return SimpleNamespace(logger=logger, experiment=semantics)


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


def test_assay_labels_requires_existing_ref():
    with pytest.raises(ValueError, match="missing key 'missing'"):
        AssayLabelsTransform().run(
            _ctx({"design_id": {"source": "design_id", "values": {"a": "A"}}}),
            {"df": pd.DataFrame({"design_id": ["a"]})},
            AssayLabelsCfg(refs=["missing"]),
        )
