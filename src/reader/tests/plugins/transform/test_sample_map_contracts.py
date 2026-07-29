"""Tests for sample-map transform contract promotion."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd

from reader.contracts import builtin_contract_catalog
from reader.plugins.transform.sample_map import SampleMapCfg, SampleMapMerge
from reader.workbench import PluginSemantics
from reader.workbench.assets import build_plugin_asset
from reader.workbench.engine import _resolve_runtime_output_ports


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
    plugin.bind_runtime(
        descriptor=build_plugin_asset(
            plugin_id="transform/sample_map",
            semantics=PluginSemantics(domain="plate_reader", family="metadata_merge", summary="test"),
            plugin_cls=SampleMapMerge,
        ),
        contracts=builtin_contract_catalog(),
    )
    cfg = SampleMapCfg()
    outputs = plugin.run(_ctx(), {"df": _raw_df(), "sample_map": sample_map}, cfg)

    resolved = _resolve_runtime_output_ports(
        plugin,
        inputs={"df": SimpleNamespace(contract_id="tidy.v1"), "sample_map": sample_map},
        outputs=outputs,
        cfg=cfg,
        contracts=builtin_contract_catalog(),
        where="merge_map",
    )
    assert resolved["df"].contract == "plate_reader.annotated.v1"


def test_sample_map_stays_tidy_without_required_annotated_columns(tmp_path):
    sample_map = tmp_path / "metadata.csv"
    pd.DataFrame({"position": ["A1"], "strain": ["ecn"]}).to_csv(sample_map, index=False)

    plugin = SampleMapMerge()
    plugin.bind_runtime(
        descriptor=build_plugin_asset(
            plugin_id="transform/sample_map",
            semantics=PluginSemantics(domain="plate_reader", family="metadata_merge", summary="test"),
            plugin_cls=SampleMapMerge,
        ),
        contracts=builtin_contract_catalog(),
    )
    cfg = SampleMapCfg()
    outputs = plugin.run(_ctx(), {"df": _raw_df(), "sample_map": sample_map}, cfg)

    resolved = _resolve_runtime_output_ports(
        plugin,
        inputs={"df": SimpleNamespace(contract_id="tidy.v1"), "sample_map": sample_map},
        outputs=outputs,
        cfg=cfg,
        contracts=builtin_contract_catalog(),
        where="merge_map",
    )
    assert resolved["df"].contract == "tidy.v1"
