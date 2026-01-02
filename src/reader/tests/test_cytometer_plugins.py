"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_cytometer_plugins.py

Lightweight coverage for cytometry ingest/merge plugins.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from reader.plugins.ingest.flow_cytometer import FlowCytometerCfg, FlowCytometerIngest
from reader.plugins.merge.sample_metadata import SampleMetadataCfg, SampleMetadataMerge


def _ctx(exp_dir: Path):
    return SimpleNamespace(exp_dir=exp_dir, logger=logging.getLogger("reader.tests"))


def test_flow_cytometer_ingest_basic():
    pytest.importorskip("flowio")
    fcs_path = Path("experiments/2026/20260101_cytometer_retron/inputs/retron-26-neg_Data Source - 1.fcs")
    if not fcs_path.exists():
        pytest.skip("Cytometer fixture file missing")
    plugin = FlowCytometerIngest()
    cfg = FlowCytometerCfg(print_summary=False)
    outputs = plugin.run(_ctx(fcs_path.parent.parent), {"raw": fcs_path}, cfg)
    df = outputs["df"]
    assert {"position", "time", "channel", "value", "sample_id"} <= set(df.columns)
    assert df["sample_id"].nunique() == 1
    assert df["position"].nunique() == 1


def test_sample_metadata_merge_basic(tmp_path: Path):
    plugin = SampleMetadataMerge()
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "position": ["s1", "s2"],
            "time": [0.0, 0.0],
            "channel": ["A", "A"],
            "value": [1.0, 2.0],
        }
    )
    meta = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "design_id": ["d1", "d2"],
            "treatment": ["t1", "t2"],
        }
    )
    meta_path = tmp_path / "metadata.csv"
    meta.to_csv(meta_path, index=False)
    outputs = plugin.run(
        _ctx(tmp_path),
        {"df": df, "metadata": meta_path},
        SampleMetadataCfg(require_columns=["design_id", "treatment"], require_non_null=True),
    )
    merged = outputs["df"]
    assert {"design_id", "treatment"} <= set(merged.columns)
