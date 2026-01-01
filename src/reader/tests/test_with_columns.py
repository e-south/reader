"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_with_columns.py
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from reader.core.context import RunContext
from reader.core.errors import TransformError
from reader.plugins.transform.with_columns import WithColumns, WithColumnsCfg


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        exp_dir=tmp_path,
        outputs_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        plots_dir=tmp_path / "plots",
        manifest_path=tmp_path / "manifest.json",
        logger=logging.getLogger("reader.tests"),
        palette_book=None,
        collections=None,
    )


def test_with_columns_adds_constants(tmp_path: Path):
    df = pd.DataFrame({"position": ["A1"], "value": [1.0]})
    cfg = WithColumnsCfg(columns={"replicate": 0, "note": "x"})
    out = WithColumns().run(_ctx(tmp_path), {"df": df}, cfg)["df"]
    assert "replicate" in out.columns
    assert "note" in out.columns
    assert int(out.loc[0, "replicate"]) == 0
    assert out.loc[0, "note"] == "x"


def test_with_columns_rejects_existing_by_default(tmp_path: Path):
    df = pd.DataFrame({"position": ["A1"], "replicate": [1]})
    cfg = WithColumnsCfg(columns={"replicate": 0})
    with pytest.raises(TransformError, match="already exists"):
        WithColumns().run(_ctx(tmp_path), {"df": df}, cfg)


def test_with_columns_overwrite_mode(tmp_path: Path):
    df = pd.DataFrame({"position": ["A1"], "replicate": [1]})
    cfg = WithColumnsCfg(columns={"replicate": 0}, mode="overwrite")
    out = WithColumns().run(_ctx(tmp_path), {"df": df}, cfg)["df"]
    assert int(out.loc[0, "replicate"]) == 0
