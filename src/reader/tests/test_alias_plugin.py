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


def _ctx(aliases):
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None)
    return SimpleNamespace(logger=logger, aliases=aliases)


def test_alias_ref_single_column_map():
    df = pd.DataFrame({"design_id": ["a", "b"], "treatment": ["x", "y"]})
    cfg = AliasCfg(aliases_ref="design_id", in_place=False, case_insensitive=False)
    out = AliasTransform().run(_ctx({"design_id": {"a": "A", "b": "B"}}), {"df": df}, cfg)["df"]
    assert out["design_id_alias"].tolist() == ["A", "B"]


def test_alias_ref_empty_map_preserves_values():
    df = pd.DataFrame({"treatment": ["x", "y"]})
    cfg = AliasCfg(aliases_ref="treatment", in_place=False, case_insensitive=True)
    out = AliasTransform().run(_ctx({"treatment": {}}), {"df": df}, cfg)["df"]
    assert out["treatment_alias"].tolist() == ["x", "y"]
