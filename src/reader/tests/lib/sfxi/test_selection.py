"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_sfxi_selection.py

Aggregate replicates per corner and map treatments -> {00,10,01,11}.
Also ensures we pick time 'nearest' and counts are correct for both channels.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import pytest

from reader.lib.sfxi.api import load_sfxi_config
from reader.lib.sfxi.math import compute_vec8
from reader.lib.sfxi.reference import resolve_reference_design_id
from reader.lib.sfxi.selection import cornerize_and_aggregate


def _tidy(rows):
    return pd.DataFrame(rows)


def test_cornerize_aggregates_replicates_for_logic_and_intensity():
    # two replicates per corner at the same time (12.0 h), one design_id
    TMAP = {
        "00": "EtOH_0_percent_0nM_cipro",
        "10": "EtOH_3_percent_0nM_cipro",
        "01": "EtOH_0_percent_100nM_cipro",
        "11": "EtOH_3_percent_100nM_cipro",
    }
    rows = []
    # logic channel values set so means are 1, 2, 3, 4 over two replicates
    vals_logic = {"00": [0.5, 1.5], "10": [2.0, 2.0], "01": [2.5, 3.5], "11": [4.0, 4.0]}
    vals_int = {"00": [10, 30], "10": [10, 30], "01": [10, 30], "11": [10, 30]}
    corners = ["00", "10", "01", "11"]
    for pos, corner in zip(["A1", "B1", "C1", "D1"], corners, strict=False):
        # two replicates per corner at the same time
        for rep in range(2):
            rows.append(
                {
                    "position": f"{pos}-{rep}",
                    "time": 12.0,
                    "channel": "YFP/CFP",
                    "value": vals_logic[corner][rep],
                    "treatment": TMAP[corner],
                    "design_id": "G1",
                }
            )
            rows.append(
                {
                    "position": f"{pos}-{rep}",
                    "time": 12.0,
                    "channel": "YFP/OD600",
                    "value": vals_int[corner][rep],
                    "treatment": TMAP[corner],
                    "design_id": "G1",
                }
            )
    df = _tidy(rows)

    common = {
        "design_by": ["design_id"],
        "treatment_map": TMAP,
        "case_sensitive": True,
        "time_column": "time",
        "target_time_h": 12.0,
        "time_mode": "nearest",
        "time_tolerance_h": 0.25,
        "require_all_corners_per_design": True,
    }

    sel_L = cornerize_and_aggregate(df, channel="YFP/CFP", **common)
    sel_I = cornerize_and_aggregate(df, channel="YFP/OD600", **common)

    # means per corner
    ptL = sel_L.points.iloc[0]
    ptI = sel_I.points.iloc[0]
    assert np.isclose(ptL["b00"], 1.0) and np.isclose(ptL["b10"], 2.0)
    assert np.isclose(ptL["b01"], 3.0) and np.isclose(ptL["b11"], 4.0)
    assert np.isclose(ptI["b00"], 20.0) and np.isclose(ptI["b10"], 20.0)
    assert np.isclose(ptI["b01"], 20.0) and np.isclose(ptI["b11"], 20.0)

    # replicate counts carried through (two per corner)
    for col in ["n00", "n10", "n01", "n11"]:
        assert int(ptL[col]) == 2
        assert int(ptI[col]) == 2


def test_missing_reference_design_id_raises_in_compute_vec8():
    # a single design with four corners for logic & intensity
    pts_logic = pd.DataFrame([{"design_id": "X", "b00": 1, "b10": 2, "b01": 3, "b11": 4}])
    pts_int = pd.DataFrame([{"design_id": "X", "b00": 10, "b10": 10, "b01": 10, "b11": 10}])

    # anchors table lacks the requested 'REF' design_id entirely
    per_corner = pd.DataFrame(
        [
            {"design_id": "OTHER", "corner": "00", "y_mean": 5},
            {"design_id": "OTHER", "corner": "10", "y_mean": 5},
            {"design_id": "OTHER", "corner": "01", "y_mean": 5},
            {"design_id": "OTHER", "corner": "11", "y_mean": 5},
        ]
    )

    try:
        compute_vec8(
            points_logic=pts_logic,
            points_intensity=pts_int,
            per_corner_intensity=per_corner,
            design_by=["design_id"],
            reference_design_id="REF",
            reference_stat="mean",
            eps_ratio=1e-12,
            eps_range=1e-12,
            eps_ref=1e-12,
            eps_abs=0.0,
            ref_add_alpha=0.0,
            log2_offset_delta=0.0,
        )
        raise AssertionError("Expected failure when reference design_id is missing")
    except ValueError as e:
        # error originates from missing anchors -> y* computation
        assert "missing anchor" in str(e).lower() or "invalid anchor" in str(e).lower()


def _rows_for_times(times, tmap, *, channel, design="G1", value_base=1.0):
    rows = []
    corners = ["00", "10", "01", "11"]
    for t in times:
        for pos, corner in zip(["A1", "B1", "C1", "D1"], corners, strict=False):
            rows.append(
                {
                    "position": f"{pos}-{t}",
                    "time": float(t),
                    "channel": channel,
                    "value": value_base + float(corners.index(corner)),
                    "treatment": tmap[corner],
                    "design_id": design,
                }
            )
    return rows


def test_time_mode_variants_and_tolerance_warnings():
    tmap = {"00": "A", "10": "B", "01": "C", "11": "D"}
    df = _tidy(_rows_for_times([1.0, 3.0], tmap, channel="YFP/CFP"))
    common = {
        "design_by": ["design_id"],
        "treatment_map": tmap,
        "case_sensitive": True,
        "time_column": "time",
        "time_tolerance_h": 0.5,
        "require_all_corners_per_design": True,
    }

    sel_nearest = cornerize_and_aggregate(df, channel="YFP/CFP", target_time_h=2.2, time_mode="nearest", **common)
    assert np.isclose(sel_nearest.chosen_time, 3.0)
    assert sel_nearest.time_warning  # outside tolerance => warning recorded

    sel_last = cornerize_and_aggregate(df, channel="YFP/CFP", target_time_h=2.2, time_mode="last_before", **common)
    assert np.isclose(sel_last.chosen_time, 1.0)

    sel_first = cornerize_and_aggregate(df, channel="YFP/CFP", target_time_h=2.2, time_mode="first_after", **common)
    assert np.isclose(sel_first.chosen_time, 3.0)

    with pytest.raises(ValueError, match="could not choose a global time"):
        cornerize_and_aggregate(df, channel="YFP/CFP", target_time_h=2.0, time_mode="exact", **common)


def test_treatment_alias_selection_and_case_sensitivity():
    tmap = {"00": "EtOH", "10": "PMS", "01": "Cipro", "11": "NEG"}
    rows = []
    for corner, pos in zip(["00", "10", "01", "11"], ["A1", "B1", "C1", "D1"], strict=False):
        rows.append(
            {
                "position": pos,
                "time": 1.0,
                "channel": "YFP/CFP",
                "value": 1.0,
                "treatment": f"raw_{corner}",
                "treatment_alias": tmap[corner],
                "design_id": "G1",
            }
        )
    df = _tidy(rows)

    sel = cornerize_and_aggregate(
        df,
        design_by=["design_id"],
        treatment_map=tmap,
        case_sensitive=True,
        time_column="time",
        channel="YFP/CFP",
        target_time_h=1.0,
        time_mode="exact",
        time_tolerance_h=0.1,
        require_all_corners_per_design=True,
    )
    assert not sel.points.empty  # matched via treatment_alias

    rows_case = []
    for corner, pos in zip(["00", "10", "01", "11"], ["A1", "B1", "C1", "D1"], strict=False):
        rows_case.append(
            {
                "position": f"{pos}-case",
                "time": 1.0,
                "channel": "YFP/CFP",
                "value": 1.0,
                "treatment": tmap[corner].lower(),
                "design_id": "G1",
            }
        )
    df_case = _tidy(rows_case)
    with pytest.raises(ValueError, match="no rows matched"):
        cornerize_and_aggregate(
            df_case,
            design_by=["design_id"],
            treatment_map=tmap,
            case_sensitive=True,
            time_column="time",
            channel="YFP/CFP",
            target_time_h=1.0,
            time_mode="exact",
            time_tolerance_h=0.1,
            require_all_corners_per_design=True,
        )
    # case-insensitive should pass
    sel_ci = cornerize_and_aggregate(
        df_case,
        design_by=["design_id"],
        treatment_map=tmap,
        case_sensitive=False,
        time_column="time",
        channel="YFP/CFP",
        target_time_h=1.0,
        time_mode="exact",
        time_tolerance_h=0.1,
        require_all_corners_per_design=True,
    )
    assert not sel_ci.points.empty


def test_duplicate_treatment_map_values_and_missing_corners():
    tmap_dup = {"00": "A", "10": "A", "01": "B", "11": "C"}
    df = _tidy(_rows_for_times([1.0], tmap_dup, channel="YFP/CFP"))
    with pytest.raises(ValueError, match="duplicate treatment_map value"):
        cornerize_and_aggregate(
            df,
            design_by=["design_id"],
            treatment_map=tmap_dup,
            case_sensitive=True,
            time_column="time",
            channel="YFP/CFP",
            target_time_h=1.0,
            time_mode="exact",
            time_tolerance_h=0.1,
            require_all_corners_per_design=True,
        )

    tmap = {"00": "A", "10": "B", "01": "C", "11": "D"}
    rows = _rows_for_times([1.0], tmap, channel="YFP/CFP")
    rows = [r for r in rows if r["treatment"] != "D"]  # drop corner 11
    df_missing = _tidy(rows)
    with pytest.raises(ValueError, match="missing corners"):
        cornerize_and_aggregate(
            df_missing,
            design_by=["design_id"],
            treatment_map=tmap,
            case_sensitive=True,
            time_column="time",
            channel="YFP/CFP",
            target_time_h=1.0,
            time_mode="exact",
            time_tolerance_h=0.1,
            require_all_corners_per_design=True,
        )


def test_reference_design_id_resolution_raw_and_alias():
    df_unique = pd.DataFrame(
        {
            "design_id": ["REF_RAW", "G1"],
            "design_id_alias": ["REF", "G1"],
        }
    )
    assert resolve_reference_design_id(df_unique, design_by=["design_id"], ref_label="REF") == "REF_RAW"

    df = pd.DataFrame(
        {
            "design_id": ["REF_RAW", "G1", "G2"],
            "design_id_alias": ["REF", "G1", "REF"],
        }
    )
    assert resolve_reference_design_id(df, design_by=["design_id"], ref_label="REF_RAW") == "REF_RAW"
    assert resolve_reference_design_id(df, design_by=["design_id"], ref_label="G1") == "G1"
    with pytest.raises(ValueError, match="resolves via"):
        resolve_reference_design_id(df, design_by=["design_id"], ref_label="REF")


def _base_sfxi_cfg():
    return {
        "response": {"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        "treatment_map": {"00": "A", "10": "B", "01": "C", "11": "D"},
        "reference": {"design_id": "REF"},
    }


def test_load_sfxi_config_requires_design_id():
    cfg = _base_sfxi_cfg()
    cfg["design_by"] = ["genotype"]
    with pytest.raises(ValueError, match="design_by must start with 'design_id'"):
        load_sfxi_config(cfg)


def test_load_sfxi_config_rejects_reference_genotype():
    cfg = _base_sfxi_cfg()
    cfg["reference"] = {"genotype": "REF"}
    with pytest.raises(ValueError, match="reference.genotype"):
        load_sfxi_config(cfg)
