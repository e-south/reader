from types import SimpleNamespace

import pandas as pd

from reader_workbench.plugins.transform.alias import AliasCfg, AliasTransform


def _ctx(labels=None):
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None)
    return SimpleNamespace(logger=logger, annotations={"labels": labels or {}})


def test_alias_mapping_single_column_map():
    df = pd.DataFrame({"design_id": ["a", "b"], "treatment": ["x", "y"]})
    cfg = AliasCfg(mappings={"design_id": {"a": "A", "b": "B"}}, in_place=False, case_insensitive=False)
    out = AliasTransform().run(_ctx(), {"df": df}, cfg)["df"]
    assert out["design_id_alias"].tolist() == ["A", "B"]


def test_alias_mapping_empty_map_preserves_values():
    df = pd.DataFrame({"treatment": ["x", "y"]})
    cfg = AliasCfg(mappings={"treatment": {}}, in_place=False, case_insensitive=True)
    out = AliasTransform().run(_ctx(), {"df": df}, cfg)["df"]
    assert out["treatment_alias"].tolist() == ["x", "y"]
