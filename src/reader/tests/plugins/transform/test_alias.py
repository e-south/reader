"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_alias_plugin.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from types import SimpleNamespace

import pandas as pd

from reader.plugins.transform.alias import AliasCfg, AliasTransform


def _ctx(labels):
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None)
    return SimpleNamespace(logger=logger, assay={"labels": labels})


def test_alias_ref_single_column_map():
    df = pd.DataFrame({"design_id": ["a", "b"], "treatment": ["x", "y"]})
    cfg = AliasCfg(refs=["design_id"], in_place=False, case_insensitive=False)
    out = AliasTransform().run(
        _ctx({"design_id": {"source": "design_id", "output": "design_id_alias", "values": {"a": "A", "b": "B"}}}),
        {"df": df},
        cfg,
    )["df"]
    assert out["design_id_alias"].tolist() == ["A", "B"]


def test_alias_ref_empty_map_preserves_values():
    df = pd.DataFrame({"treatment": ["x", "y"]})
    cfg = AliasCfg(refs=["treatment"], in_place=False, case_insensitive=True)
    out = AliasTransform().run(
        _ctx({"treatment": {"source": "treatment", "output": "treatment_alias", "values": {}}}),
        {"df": df},
        cfg,
    )["df"]
    assert out["treatment_alias"].tolist() == ["x", "y"]
