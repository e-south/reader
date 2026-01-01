"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_transform_strict.py
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from reader.core.context import RunContext
from reader.core.errors import TransformError
from reader.plugins.transform.blank import BlankCfg, BlankCorrection
from reader.plugins.transform.ratio import RatioCfg, RatioTransform


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


def test_ratio_requires_channels(tmp_path: Path):
    df = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["YFP"],
            "value": [1.0],
        }
    )
    cfg = RatioCfg(name="YFP/CFP", numerator="YFP", denominator="CFP")
    with pytest.raises(TransformError, match="missing channel"):
        RatioTransform().run(_ctx(tmp_path), {"df": df}, cfg)


def test_ratio_rejects_zero_denominator(tmp_path: Path):
    df = pd.DataFrame(
        {
            "position": ["A1", "A1"],
            "time": [0.0, 0.0],
            "channel": ["YFP", "CFP"],
            "value": [1.0, 0.0],
        }
    )
    cfg = RatioCfg(name="YFP/CFP", numerator="YFP", denominator="CFP")
    with pytest.raises(TransformError, match="zero_den"):
        RatioTransform().run(_ctx(tmp_path), {"df": df}, cfg)


def test_ratio_requires_alignment(tmp_path: Path):
    df = pd.DataFrame(
        {
            "position": ["A1", "A1"],
            "time": [0.0, 1.0],
            "channel": ["YFP", "CFP"],
            "value": [1.0, 1.0],
        }
    )
    cfg = RatioCfg(name="YFP/CFP", numerator="YFP", denominator="CFP")
    with pytest.raises(TransformError, match="no aligned rows"):
        RatioTransform().run(_ctx(tmp_path), {"df": df}, cfg)


def test_blank_subtract_requires_detected_blanks(tmp_path: Path):
    df = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["YFP"],
            "value": [1.0],
            "treatment": ["sample"],
            "design_id": ["X"],
        }
    )
    cfg = BlankCfg(method="subtract")
    with pytest.raises(TransformError, match="requires detected blanks"):
        BlankCorrection().run(_ctx(tmp_path), {"df": df}, cfg)
