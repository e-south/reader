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

from reader.lib.sfxi.selection import cornerize_and_aggregate


def _tidy(rows):
    return pd.DataFrame(rows)

def test_cornerize_aggregates_replicates_for_logic_and_intensity():
    # two replicates per corner at the same time (12.0 h), one genotype, one batch
    TMAP = {
        "00": "EtOH_0_percent_0nM_cipro",
        "10": "EtOH_3_percent_0nM_cipro",
        "01": "EtOH_0_percent_100nM_cipro",
        "11": "EtOH_3_percent_100nM_cipro",
    }
    rows = []
    # logic channel values set so means are 1, 2, 3, 4 over two replicates
    vals_logic = {"00": [0.5, 1.5], "10": [2.0, 2.0], "01": [2.5, 3.5], "11": [4.0, 4.0]}
    vals_int   = {"00": [10, 30], "10": [10, 30], "01": [10, 30], "11": [10, 30]}
    corners = ["00","10","01","11"]
    for pos, corner in zip(["A1","B1","C1","D1"], corners):
        # two replicates per corner at the same time
        for rep in range(2):
            rows.append({"position": f"{pos}-{rep}", "time": 12.0, "channel": "YFP/CFP",
                         "value": vals_logic[corner][rep],
                         "treatment": TMAP[corner], "genotype": "G1", "batch": 0})
            rows.append({"position": f"{pos}-{rep}", "time": 12.0, "channel": "YFP/OD600",
                         "value": vals_int[corner][rep],
                         "treatment": TMAP[corner], "genotype": "G1", "batch": 0})
    df = _tidy(rows)

    common = dict(
        design_by=["genotype"], batch_col="batch", treatment_map=TMAP,
        case_sensitive=True, time_column="time",
        target_time_h=12.0, time_mode="nearest", time_tolerance_h=0.25,
        time_per_batch=True, on_missing_time="error",
        require_all_corners_per_design=True,
    )

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
    for col in ["n00","n10","n01","n11"]:
        assert int(ptL[col]) == 2
        assert int(ptI[col]) == 2

def test_missing_reference_genotype_raises_in_compute_vec8():
    import pandas as pd

    from reader.lib.sfxi.math import compute_vec8

    # a single design×batch with four corners for logic & intensity
    pts_logic = pd.DataFrame([{"genotype":"X","batch":0,"b00":1,"b10":2,"b01":3,"b11":4}])
    pts_int   = pd.DataFrame([{"genotype":"X","batch":0,"b00":10,"b10":10,"b01":10,"b11":10}])

    # anchors table lacks the requested 'REF' genotype entirely
    per_corner = pd.DataFrame([
        {"genotype":"OTHER","batch":0,"corner":"00","y_mean":5},
        {"genotype":"OTHER","batch":0,"corner":"10","y_mean":5},
        {"genotype":"OTHER","batch":0,"corner":"01","y_mean":5},
        {"genotype":"OTHER","batch":0,"corner":"11","y_mean":5},
    ])

    try:
        compute_vec8(
            points_logic=pts_logic, points_intensity=pts_int, per_corner_intensity=per_corner,
            design_by=["genotype"], batch_col="batch",
            reference_genotype="REF", reference_scope="batch", reference_stat="mean",
            eps_ratio=1e-12, eps_range=1e-12, eps_ref=1e-12, eps_abs=0.0,
            ref_add_alpha=0.0, log2_offset_delta=0.0,
        )
        assert False, "Expected failure when reference genotype is missing"
    except ValueError as e:
        # error originates from missing anchors -> y* computation
        assert "missing anchor" in str(e).lower() or "invalid anchor" in str(e).lower()
