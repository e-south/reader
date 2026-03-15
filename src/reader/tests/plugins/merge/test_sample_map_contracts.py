"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_sample_map_contracts.py

Tests for sample-map merge contract promotion.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd

from reader.core.engine import _resolve_runtime_output_contracts
from reader.plugins.merge.sample_map import SampleMapCfg, SampleMapMerge


def _ctx():
    return SimpleNamespace(logger=logging.getLogger("reader.tests"))


def _raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["OD600"],
            "value": [1.0],
        }
    )


def test_sample_map_promotes_to_annotated_when_required_metadata_present(tmp_path):
    sample_map = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "position": ["A1"],
            "design_id": ["ctrl"],
            "treatment": ["negative"],
            "batch": [0.0],
        }
    ).to_csv(sample_map, index=False)

    plugin = SampleMapMerge()
    cfg = SampleMapCfg()
    outputs = plugin.run(_ctx(), {"df": _raw_df(), "sample_map": sample_map}, cfg)

    resolved = _resolve_runtime_output_contracts(
        plugin,
        inputs={"df": SimpleNamespace(contract_id="tidy.v1"), "sample_map": sample_map},
        outputs=outputs,
        cfg=cfg,
        where="merge_map",
    )
    assert resolved["df"] == "plate_reader.annotated.v1"


def test_sample_map_stays_tidy_without_required_annotated_columns(tmp_path):
    sample_map = tmp_path / "metadata.csv"
    pd.DataFrame({"position": ["A1"], "strain": ["ecn"]}).to_csv(sample_map, index=False)

    plugin = SampleMapMerge()
    cfg = SampleMapCfg()
    outputs = plugin.run(_ctx(), {"df": _raw_df(), "sample_map": sample_map}, cfg)

    resolved = _resolve_runtime_output_contracts(
        plugin,
        inputs={"df": SimpleNamespace(contract_id="tidy.v1"), "sample_map": sample_map},
        outputs=outputs,
        cfg=cfg,
        where="merge_map",
    )
    assert resolved["df"] == "tidy.v1"
