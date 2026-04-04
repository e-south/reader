from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from reader.plugins.transform.assay_labels import AnnotationLabelsCfg, AnnotationLabelsTransform
from reader.protocols import ProtocolBinding
from reader.workbench.experiment import (
    AnnotationLabels,
    AnnotationLabelSpec,
    AnnotationSemantics,
    ExperimentSemantics,
    OutputLayout,
    ResourceCatalog,
)


def _ctx(labels):
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None)
    semantics = ExperimentSemantics(
        protocol=ProtocolBinding(id="workbench/generic"),
        annotations=AnnotationSemantics(
            labels=AnnotationLabels(
                by_id={
                    key: AnnotationLabelSpec(
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
    cfg = AnnotationLabelsCfg()
    out = AnnotationLabelsTransform().run(
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
    cfg = AnnotationLabelsCfg(refs=["design_id"])
    out = AnnotationLabelsTransform().run(
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
    with pytest.raises(ValueError, match="no annotations.labels"):
        AnnotationLabelsTransform().run(_ctx({}), {"df": pd.DataFrame({"design_id": ["a"]})}, AnnotationLabelsCfg())


def test_assay_labels_requires_existing_ref():
    with pytest.raises(ValueError, match="missing key 'missing'"):
        AnnotationLabelsTransform().run(
            _ctx({"design_id": {"source": "design_id", "values": {"a": "A"}}}),
            {"df": pd.DataFrame({"design_id": ["a"]})},
            AnnotationLabelsCfg(refs=["missing"]),
        )
