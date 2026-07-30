from __future__ import annotations

import pytest
from pydantic import ValidationError

from reader.contracts import builtin_contract_catalog, validate_df
from reader.plugins.transform.cytometry_gating import CytometryGatingCfg, CytometryGatingTransform
from reader.tests.domains.cytometry.analysis.test_gating_workflow import tidy_events


def config_payload() -> dict[str, object]:
    return {
        "cells_x_channel": "FSC-A",
        "cells_y_channel": "SSC-A",
        "cells_x_range": (0.0, 100.0),
        "cells_y_range": (0.0, 100.0),
        "singlet_x_channel": "FSC-A",
        "singlet_y_channel": "FSC-H",
        "singlet_ratio_range": (0.8, 1.2),
        "cells_enabled": True,
        "singlets_enabled": True,
        "fluorescence_channel": "reporter",
        "threshold_mode": "manual",
        "threshold_value": 5.0,
        "threshold_group_column": None,
        "threshold_control_value": None,
        "threshold_quantile": None,
        "group_column": "condition",
        "minimum_final_events": 2,
        "minimum_final_percent": 50.0,
        "maximum_nonpositive_percent": 10.0,
        "nonpositive_scope": "all_events",
    }


def test_transform_produces_each_typed_cytometry_record() -> None:
    outputs = CytometryGatingTransform().run(None, {"events": tidy_events()}, CytometryGatingCfg(**config_payload()))
    contracts = builtin_contract_catalog()

    expected = {
        "gate_definition": "cytometry.gate_definition.v1",
        "gated_events": "cytometry.gated_events.v1",
        "sample_stats": "cytometry.sample_stats.v1",
        "group_stats": "cytometry.group_stats.v1",
        "qc": "cytometry.qc.v1",
    }
    assert set(outputs) == set(expected)
    for name, contract_id in expected.items():
        validate_df(outputs[name], contracts.require(contract_id), where=name)


def test_transform_config_requires_explicit_group_policy() -> None:
    payload = config_payload()
    payload.pop("group_column")

    with pytest.raises(ValidationError, match="group_column"):
        CytometryGatingCfg(**payload)


def test_transform_emits_a_typed_empty_group_table_when_grouping_is_disabled() -> None:
    payload = config_payload() | {"group_column": None}
    outputs = CytometryGatingTransform().run(None, {"events": tidy_events()}, CytometryGatingCfg(**payload))

    group_stats = outputs["group_stats"]
    assert group_stats.empty
    validate_df(
        group_stats,
        builtin_contract_catalog().require("cytometry.group_stats.v1"),
        where="group_stats",
    )


def test_transform_config_rejects_incomplete_control_threshold_policy() -> None:
    payload = config_payload() | {
        "threshold_mode": "from_control_quantile",
        "threshold_value": None,
        "threshold_group_column": "condition",
        "threshold_control_value": None,
        "threshold_quantile": 0.99,
    }

    with pytest.raises(ValidationError, match="threshold_control_value"):
        CytometryGatingCfg(**payload)


def test_transform_persists_contract_valid_fail_closed_qc_for_zero_retained_sample() -> None:
    events = tidy_events()
    excluded = (events["sample_id"] == "treated") & events["channel"].isin(("FSC-A", "SSC-A"))
    events.loc[excluded, "value"] = 200.0
    payload = config_payload() | {"nonpositive_scope": "gated_events"}

    outputs = CytometryGatingTransform().run(None, {"events": events}, CytometryGatingCfg(**payload))

    qc = outputs["qc"].set_index("sample_id")
    assert qc.loc["treated", "pct_nonpositive"] == 100.0
    assert bool(qc.loc["treated", "passes_nonpositive"]) is False
    assert bool(qc.loc["treated", "qc_pass"]) is False
    assert qc.loc["treated", "qc_status"] == "fail"
    validate_df(
        outputs["qc"],
        builtin_contract_catalog().require("cytometry.qc.v1"),
        where="qc",
    )
