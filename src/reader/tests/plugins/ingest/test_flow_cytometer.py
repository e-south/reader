"""Integration coverage for flow-cytometer ingest."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from reader.plugins.ingest.flow_cytometer import FlowCytometerCfg, FlowCytometerIngest

pytestmark = pytest.mark.integration


def _ctx(exp_dir: Path):
    return SimpleNamespace(exp_dir=exp_dir, logger=logging.getLogger("reader.tests"))


def test_flow_cytometer_ingest_basic():
    fcs_path = Path("experiments/2026/20260101_cytometry_example/inputs/example-negative.fcs")
    if not fcs_path.exists():
        pytest.skip("Cytometer fixture file is not available in this checkout")
    plugin = FlowCytometerIngest()
    cfg = FlowCytometerCfg(print_summary=False)
    outputs = plugin.run(_ctx(fcs_path.parent.parent), {"raw": (fcs_path,)}, cfg)
    df = outputs["df"]
    channels = outputs["channels"]
    assert {"position", "time", "channel", "value", "sample_id"} <= set(df.columns)
    assert df["sample_id"].nunique() == 1
    assert df["position"].nunique() == 1
    assert {"sample_id", "channel_index", "channel_name"} <= set(channels.columns)
    assert channels["sample_id"].nunique() == 1
