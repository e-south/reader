from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from reader.domains.logic.logic_symmetry.extract_corners import MappingConfig, resolve_and_aggregate
from reader.domains.logic.logic_symmetry.prep import prepare_for_logic_symmetry
from reader.plugins.transform.logic_symmetry import LogicSymmetryCfg


def _logic_df() -> pd.DataFrame:
    tmap = {"00": "EtOH", "10": "PMS", "01": "Cipro", "11": "NEG"}
    rows = []
    for position, corner, value in zip(
        ["A1", "B1", "C1", "D1"], ["00", "10", "01", "11"], [1.0, 2.0, 3.0, 4.0], strict=False
    ):
        rows.append(
            {
                "position": position,
                "time": 1.0,
                "channel": "YFP/CFP",
                "value": value,
                "treatment": f"raw_{corner}",
                "treatment_alias": tmap[corner],
                "design_id": "G1",
                "batch": 0,
            }
        )
    return pd.DataFrame(rows)


def test_prepare_for_logic_symmetry_uses_configured_treatment_column():
    tmap = {"00": "EtOH", "10": "PMS", "01": "Cipro", "11": "NEG"}
    prepared = prepare_for_logic_symmetry(
        _logic_df(),
        response_channel="YFP/CFP",
        design_by=["design_id"],
        batch_col="batch",
        treatment_map=tmap,
        treatment_column="treatment_alias",
        mode="exact",
        target_time=1.0,
        tolerance=0.1,
    )

    assert len(prepared) == 4
    assert set(prepared["treatment_alias"]) == set(tmap.values())


def test_logic_symmetry_mapping_uses_configured_treatment_column():
    tmap = {"00": "EtOH", "10": "PMS", "01": "Cipro", "11": "NEG"}
    points, per_corner = resolve_and_aggregate(
        _logic_df(),
        MappingConfig(
            treatment_map=tmap,
            case_sensitive=True,
            treatment_column="treatment_alias",
            design_by=["design_id"],
            batch_col="batch",
            response_channel="YFP/CFP",
            observation_stat="mean",
        ),
    )

    assert len(per_corner) == 4
    point = points.iloc[0]
    assert point["b00"] == 1.0
    assert point["b10"] == 2.0
    assert point["b01"] == 3.0
    assert point["b11"] == 4.0


def test_logic_symmetry_config_rejects_legacy_replicate_stat() -> None:
    with pytest.raises(ValidationError, match="replicate_stat"):
        LogicSymmetryCfg(
            response_channel="YFP/CFP",
            state_map_ref="induction_logic",
            replicate_stat="mean",
        )


def test_logic_symmetry_counts_and_dispersion_are_observation_summaries() -> None:
    tmap = {"00": "EtOH", "10": "PMS", "01": "Cipro", "11": "NEG"}
    repeated = pd.concat([_logic_df(), _logic_df().assign(position=lambda frame: frame["position"] + "_2")])
    points, _ = resolve_and_aggregate(
        repeated,
        MappingConfig(
            treatment_map=tmap,
            case_sensitive=True,
            treatment_column="treatment_alias",
            design_by=["design_id"],
            batch_col="batch",
            response_channel="YFP/CFP",
            observation_stat="mean",
        ),
    )

    point = points.iloc[0]
    assert point[["n00", "n10", "n01", "n11"]].tolist() == [2, 2, 2, 2]
    assert point[["sd00", "sd10", "sd01", "sd11"]].tolist() == [0.0, 0.0, 0.0, 0.0]
