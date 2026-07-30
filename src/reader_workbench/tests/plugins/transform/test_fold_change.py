from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from reader_workbench.plugins.transform.fold_change import FoldChange, FoldChangeCfg


def _ctx():
    return SimpleNamespace(logger=logging.getLogger("reader_workbench.tests.fold_change"))


def test_fold_change_plugin_emits_expected_table():
    df = pd.DataFrame(
        {
            "position": ["A1", "A2", "A3", "A4"],
            "time": [8.0, 8.0, 8.0, 8.0],
            "channel": ["YFP/CFP", "YFP/CFP", "YFP/CFP", "YFP/CFP"],
            "value": [2.0, 2.0, 4.0, 6.0],
            "design_id": ["D1", "D1", "D1", "D1"],
            "treatment": ["control", "control", "induced", "induced"],
        }
    )
    plugin = FoldChange()
    cfg = FoldChangeCfg(
        target="YFP/CFP",
        report_times=[8.0],
        group_by=["design_id"],
        use_global_baseline=True,
        global_baseline_value="control",
        attach_metadata=[],
    )

    table = plugin.run(_ctx(), {"df": df}, cfg)["table"].sort_values("treatment").reset_index(drop=True)

    assert table["treatment"].tolist() == ["control", "induced"]
    assert table["baseline_value"].tolist() == ["control", "control"]
    assert table["FC"].tolist() == [1.0, 2.5]
    assert table["design_id"].tolist() == ["D1", "D1"]


def test_fold_change_config_rejects_legacy_aggregation_key() -> None:
    with pytest.raises(ValidationError, match="agg"):
        FoldChangeCfg(target="YFP/CFP", report_times=[8.0], agg="mean")


def test_fold_change_config_names_the_observation_statistic() -> None:
    cfg = FoldChangeCfg(target="YFP/CFP", report_times=[8.0], observation_stat="mean")

    assert cfg.observation_stat == "mean"
