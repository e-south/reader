"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_sfxi_math.py

Tests for SFXI compute_vec8 (up to the 8-vector).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import math

import numpy as np
import pandas as pd

from reader.lib.sfxi.math import compute_vec8


def _mk(df_rows):
    return pd.DataFrame(df_rows)


# ------------------------ logic: mapping + invariants ------------------------


def test_logic_minmax_basic_and_r_logic():
    # logic values 1,2,4,8 ⇒ u = 0,1,2,3 ⇒ v = 0, 1/3, 2/3, 1
    pts_logic = _mk([{"genotype": "X", "batch": 0, "b00": 1, "b10": 2, "b01": 4, "b11": 8}])
    pts_int = _mk([{"genotype": "X", "batch": 0, "b00": 10, "b10": 10, "b01": 10, "b11": 10}])
    per_corner = _mk([{"genotype": "REF", "batch": 0, "corner": c, "y_mean": 5} for c in ("00", "10", "01", "11")])

    out = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-15,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).iloc[0]

    assert np.isclose(out["v00"], 0.0, atol=1e-12)
    assert np.isclose(out["v10"], 1 / 3, atol=1e-12)
    assert np.isclose(out["v01"], 2 / 3, atol=1e-12)
    assert np.isclose(out["v11"], 1.0, atol=1e-12)
    # dynamic range computed on linear values
    assert np.isclose(out["r_logic"], 8.0 / 1.0, atol=1e-12)
    assert bool(out["flat_logic"]) is False


def test_logic_scale_invariance():
    # Rescale logic by constant factor; v's must be identical
    base_logic = {"genotype": "X", "batch": 0, "b00": 1, "b10": 2, "b01": 4, "b11": 8}
    scaled_logic = {**base_logic, "b00": 10, "b10": 20, "b01": 40, "b11": 80}

    pts_logic_a = _mk([base_logic])
    pts_logic_b = _mk([scaled_logic])

    pts_int = _mk([{"genotype": "X", "batch": 0, "b00": 10, "b10": 10, "b01": 10, "b11": 10}])
    per_corner = _mk([{"genotype": "REF", "batch": 0, "corner": c, "y_mean": 5} for c in ("00", "10", "01", "11")])

    kwargs = {
        "design_by": ["genotype"],
        "batch_col": "batch",
        "reference_genotype": "REF",
        "reference_scope": "batch",
        "reference_stat": "mean",
        "eps_ratio": 1e-12,
        "eps_range": 1e-15,
        "eps_ref": 1e-12,
        "eps_abs": 0.0,
        "ref_add_alpha": 0.0,
        "log2_offset_delta": 0.0,
    }

    va = compute_vec8(points_logic=pts_logic_a, points_intensity=pts_int, per_corner_intensity=per_corner, **kwargs)
    vb = compute_vec8(points_logic=pts_logic_b, points_intensity=pts_int, per_corner_intensity=per_corner, **kwargs)
    for col in ["v00", "v10", "v01", "v11"]:
        assert np.allclose(va[col].to_numpy(), vb[col].to_numpy())


def test_logic_flat_detection_exact():
    pts_logic = _mk([{"genotype": "X", "batch": 0, "b00": 3, "b10": 3, "b01": 3, "b11": 3}])
    pts_int = _mk([{"genotype": "X", "batch": 0, "b00": 1, "b10": 1, "b01": 1, "b11": 1}])
    per_corner = _mk([{"genotype": "REF", "batch": 0, "corner": c, "y_mean": 1} for c in ("00", "10", "01", "11")])

    out = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-9,
        eps_range=1e-12,
        eps_ref=1e-9,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).iloc[0]

    assert bool(out["flat_logic"]) is True
    assert np.allclose([out["v00"], out["v10"], out["v01"], out["v11"]], 0.25)
    assert np.isclose(out["r_logic"], 1.0, atol=1e-12)


def test_logic_flat_detection_within_eps_range():
    # Differences below eps_range → flat
    pts_logic = _mk([{"genotype": "X", "batch": 0, "b00": 1.0, "b10": 1.0 * (1 + 1e-13), "b01": 1.0, "b11": 1.0}])
    pts_int = _mk([{"genotype": "X", "batch": 0, "b00": 1, "b10": 1, "b01": 1, "b11": 1}])
    per_corner = _mk([{"genotype": "REF", "batch": 0, "corner": c, "y_mean": 1} for c in ("00", "10", "01", "11")])

    out = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-10,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).iloc[0]
    assert bool(out["flat_logic"]) is True
    assert np.allclose([out["v00"], out["v10"], out["v01"], out["v11"]], 0.25)


def test_logic_zeros_guard_and_expected_mapping():
    # Zeros in logic channel are clamped by eps_ratio before log2
    pts_logic = _mk([{"genotype": "X", "batch": 0, "b00": 0.0, "b10": 5.0, "b01": 0.0, "b11": 5.0}])
    pts_int = _mk([{"genotype": "X", "batch": 0, "b00": 1, "b10": 1, "b01": 1, "b11": 1}])
    per_corner = _mk([{"genotype": "REF", "batch": 0, "corner": c, "y_mean": 1} for c in ("00", "10", "01", "11")])

    out = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-3,
        eps_range=1e-12,
        eps_ref=1e-9,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).iloc[0]

    # u00 ~ log2(1e-3), u10 ~ log2(5), span > 0, so v ∈ {0,1}
    assert np.allclose([out["v00"], out["v01"]], [0.0, 0.0], atol=1e-12)
    assert np.allclose([out["v10"], out["v11"]], [1.0, 1.0], atol=1e-12)
    # r_logic = 5 / 1e-3 = 5000
    assert np.isclose(out["r_logic"], 5000.0, rtol=1e-12, atol=1e-12)


# ------------------------ intensity: anchor/knobs/guards ------------------------


def test_intensity_alpha_delta_eps_abs_and_log_guard():
    pts_logic = _mk([{"genotype": "X", "batch": 0, "b00": 1, "b10": 1, "b01": 1, "b11": 1}])
    # All zero intensity to exercise guards
    pts_int = _mk([{"genotype": "X", "batch": 0, "b00": 0.0, "b10": 0.0, "b01": 0.0, "b11": 0.0}])
    per_corner = _mk([{"genotype": "REF", "batch": 0, "corner": c, "y_mean": 1.0} for c in ("00", "10", "01", "11")])

    # Case A: no eps_abs, delta=0 -> log2(max(0, eps_ratio)) = log2(eps_ratio)
    outA = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-9,
        eps_range=1e-12,
        eps_ref=1e-9,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).iloc[0]
    assert np.isclose(outA["y00_star"], math.log2(1e-9), atol=1e-12)

    # Case B: tiny eps_abs lifts the numerator → finite negative log value
    outB = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-12,
        eps_ref=1e-12,
        eps_abs=1e-6,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).iloc[0]
    assert outB["y00_star"] > math.log2(1e-12)  # less negative than pure guard

    # Case C: delta shifts log argument upward: y_linear(=0) + delta(=0.5) => log2(0.5)=-1
    outC = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-12,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.5,
    ).iloc[0]
    assert np.isclose(outC["y00_star"], -1.0, atol=1e-12)

    # Case D: alpha increases denominator; with nonzero numerator this would lower y*
    # Use a non‑zero numerator for just one corner to check the effect deterministically
    pts_int_nonzero = _mk([{"genotype": "X", "batch": 0, "b00": 10.0, "b10": 0.0, "b01": 0.0, "b11": 0.0}])
    outD0 = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int_nonzero,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-12,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).iloc[0]  # log2(10/1)=~3.321928
    outD1 = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int_nonzero,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-12,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=9.0,
        log2_offset_delta=0.0,
    ).iloc[0]  # denom = max(1+9, eps_ref) = 10 → log2(10/10)=0
    assert outD0["y00_star"] > outD1["y00_star"]
    assert np.isclose(outD1["y00_star"], 0.0, atol=1e-12)


def test_intensity_batch_vs_global_reference_scope():
    # Two batches with different REF anchors; global scope averages them.
    pts_logic = _mk(
        [
            {"genotype": "X", "batch": 0, "b00": 1, "b10": 1, "b01": 1, "b11": 1},
            {"genotype": "X", "batch": 1, "b00": 1, "b10": 1, "b01": 1, "b11": 1},
        ]
    )
    pts_int = _mk(
        [
            {"genotype": "X", "batch": 0, "b00": 3, "b10": 3, "b01": 3, "b11": 3},
            {"genotype": "X", "batch": 1, "b00": 3, "b10": 3, "b01": 3, "b11": 3},
        ]
    )
    per_corner = _mk(
        [{"genotype": "REF", "batch": 0, "corner": c, "y_mean": 4} for c in ("00", "10", "01", "11")]
        + [{"genotype": "REF", "batch": 1, "corner": c, "y_mean": 2} for c in ("00", "10", "01", "11")]
    )

    # Global: anchor=mean(4,2)=3 ⇒ y_linear=3/3=1 ⇒ y*=0
    out_global = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="global",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-12,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    )
    assert np.allclose(out_global[["y00_star", "y10_star", "y01_star", "y11_star"]].to_numpy(), 0.0)

    # Batch: batch0 anchor=4 ⇒ y*=log2(3/4)<0; batch1 anchor=2 ⇒ y*=log2(3/2)>0
    out_batch = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-12,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).sort_values("batch")
    y0 = out_batch[out_batch["batch"] == 0]["y00_star"].iloc[0]
    y1 = out_batch[out_batch["batch"] == 1]["y00_star"].iloc[0]
    assert y0 < 0 and y1 > 0


def test_intensity_reference_stat_mean_vs_median():
    # Same (batch,corner) repeated for REF to force aggregation difference.
    pts_logic = _mk([{"genotype": "X", "batch": 0, "b00": 1, "b10": 1, "b01": 1, "b11": 1}])
    pts_int = _mk([{"genotype": "X", "batch": 0, "b00": 2, "b10": 2, "b01": 2, "b11": 2}])
    # Three rows for the reference anchor at corner 00: 2, 2, 100
    per_corner = _mk(
        [
            {"genotype": "REF", "batch": 0, "corner": "00", "y_mean": 2},
            {"genotype": "REF", "batch": 0, "corner": "00", "y_mean": 2},
            {"genotype": "REF", "batch": 0, "corner": "00", "y_mean": 100},
            # other corners use 2 to keep things simple
            {"genotype": "REF", "batch": 0, "corner": "10", "y_mean": 2},
            {"genotype": "REF", "batch": 0, "corner": "01", "y_mean": 2},
            {"genotype": "REF", "batch": 0, "corner": "11", "y_mean": 2},
        ]
    )

    # median(anchor00)=2 -> y00*=log2(2/2)=0
    out_median = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="median",
        eps_ratio=1e-12,
        eps_range=1e-12,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).iloc[0]

    # mean(anchor00)=(2+2+100)/3=34.666.. -> y00*=log2(2/34.666..)<0
    out_mean = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-12,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).iloc[0]

    assert np.isclose(out_median["y00_star"], 0.0, atol=1e-12)
    assert out_mean["y00_star"] < 0.0


def test_missing_anchor_raises():
    pts_logic = _mk([{"genotype": "X", "batch": 0, "b00": 1, "b10": 1, "b01": 1, "b11": 1}])
    pts_int = _mk([{"genotype": "X", "batch": 0, "b00": 1, "b10": 1, "b01": 1, "b11": 1}])
    per_corner = _mk(
        [
            {"genotype": "REF", "batch": 0, "corner": "00", "y_mean": 1},
            {"genotype": "REF", "batch": 0, "corner": "10", "y_mean": 1},
            {"genotype": "REF", "batch": 0, "corner": "01", "y_mean": 1},
            # missing "11"
        ]
    )

    try:
        compute_vec8(
            points_logic=pts_logic,
            points_intensity=pts_int,
            per_corner_intensity=per_corner,
            design_by=["genotype"],
            batch_col="batch",
            reference_genotype="REF",
            reference_scope="batch",
            reference_stat="mean",
            eps_ratio=1e-12,
            eps_range=1e-12,
            eps_ref=1e-12,
            eps_abs=0.0,
            ref_add_alpha=0.0,
            log2_offset_delta=0.0,
        )
        raise AssertionError("Expected ValueError for missing anchor")
    except ValueError as e:
        assert "missing anchor" in str(e).lower()


def test_corner_order_respected_in_y_stars():
    # Different anchors per corner: 1,2,4,8 with fixed sample intensity=8
    pts_logic = _mk([{"genotype": "X", "batch": 0, "b00": 1, "b10": 1, "b01": 1, "b11": 1}])
    pts_int = _mk([{"genotype": "X", "batch": 0, "b00": 8, "b10": 8, "b01": 8, "b11": 8}])
    per_corner = _mk(
        [
            {"genotype": "REF", "batch": 0, "corner": "00", "y_mean": 1},
            {"genotype": "REF", "batch": 0, "corner": "10", "y_mean": 2},
            {"genotype": "REF", "batch": 0, "corner": "01", "y_mean": 4},
            {"genotype": "REF", "batch": 0, "corner": "11", "y_mean": 8},
        ]
    )
    out = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-12,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    ).iloc[0]
    # Expect y* = log2(8/anchor) = 3, 2, 1, 0 per [00,10,01,11]
    assert np.isclose(out["y00_star"], 3.0, atol=1e-12)
    assert np.isclose(out["y10_star"], 2.0, atol=1e-12)
    assert np.isclose(out["y01_star"], 1.0, atol=1e-12)
    assert np.isclose(out["y11_star"], 0.0, atol=1e-12)


def test_multiple_genotypes_and_batches():
    pts_logic = _mk(
        [
            {"genotype": "X", "batch": 0, "b00": 1, "b10": 2, "b01": 3, "b11": 4},
            {"genotype": "Y", "batch": 1, "b00": 10, "b10": 10, "b01": 10, "b11": 10},  # flat on log scale
        ]
    )
    pts_int = _mk(
        [
            {"genotype": "X", "batch": 0, "b00": 10, "b10": 10, "b01": 10, "b11": 10},
            {"genotype": "Y", "batch": 1, "b00": 5, "b10": 5, "b01": 5, "b11": 5},
        ]
    )
    per_corner = _mk(
        [{"genotype": "REF", "batch": 0, "corner": c, "y_mean": 5} for c in ("00", "10", "01", "11")]
        + [{"genotype": "REF", "batch": 1, "corner": c, "y_mean": 5} for c in ("00", "10", "01", "11")]
    )

    out = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-12,
        eps_range=1e-12,
        eps_ref=1e-12,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    )

    # Two rows (X@0 and Y@1); check shapes and flatness flag for Y row.
    assert out.shape[0] == 2
    y_row = out[(out["genotype"] == "Y") & (out["batch"] == 1)].iloc[0]
    assert bool(y_row["flat_logic"]) is True
    # Intensity normalization: 5/5 => log2(1) = 0 for Y
    assert np.allclose(y_row[["y00_star", "y10_star", "y01_star", "y11_star"]].to_numpy(), 0.0)


def _mk(df_rows):
    return pd.DataFrame(df_rows)


def test_compute_vec8_applies_eps_and_alpha():
    pts_logic = _mk([{"genotype": "X", "batch": 0, "b00": 1, "b10": 2, "b01": 3, "b11": 4}])
    pts_int = _mk([{"genotype": "X", "batch": 0, "b00": 10, "b10": 10, "b01": 10, "b11": 10}])
    per_corner = _mk(
        [
            {"genotype": "REF", "batch": 0, "corner": "00", "y_mean": 5},
            {"genotype": "REF", "batch": 0, "corner": "10", "y_mean": 5},
            {"genotype": "REF", "batch": 0, "corner": "01", "y_mean": 5},
            {"genotype": "REF", "batch": 0, "corner": "11", "y_mean": 5},
        ]
    )
    out = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-9,
        eps_range=1e-12,
        eps_ref=1e-9,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    )
    # intensity: (10 / 5) = 2 → log2(2) = 1.0 for all corners
    assert np.allclose(out[["y00_star", "y10_star", "y01_star", "y11_star"]].values, 1.0)


def test_flat_logic_is_boolean_dtype():
    import pandas as pd
    from pandas.api.types import is_bool_dtype

    pts_logic = pd.DataFrame([{"genotype": "X", "batch": 0, "b00": 1, "b10": 2, "b01": 4, "b11": 8}])
    pts_int = pd.DataFrame([{"genotype": "X", "batch": 0, "b00": 10, "b10": 10, "b01": 10, "b11": 10}])
    per_corner = pd.DataFrame(
        [{"genotype": "REF", "batch": 0, "corner": c, "y_mean": 5} for c in ("00", "10", "01", "11")]
    )

    out = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-9,
        eps_range=1e-12,
        eps_ref=1e-9,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    )
    assert "flat_logic" in out.columns
    assert is_bool_dtype(out["flat_logic"])


def test_vec8_contract_critical_dtypes_are_numeric():
    import pandas as pd
    from pandas.api.types import is_float_dtype, is_integer_dtype

    pts_logic = pd.DataFrame([{"genotype": "X", "batch": 0, "b00": 1, "b10": 2, "b01": 4, "b11": 8}])
    pts_int = pd.DataFrame([{"genotype": "X", "batch": 0, "b00": 10, "b10": 10, "b01": 10, "b11": 10}])
    per_corner = pd.DataFrame(
        [{"genotype": "REF", "batch": 0, "corner": c, "y_mean": 5} for c in ("00", "10", "01", "11")]
    )

    out = compute_vec8(
        points_logic=pts_logic,
        points_intensity=pts_int,
        per_corner_intensity=per_corner,
        design_by=["genotype"],
        batch_col="batch",
        reference_genotype="REF",
        reference_scope="batch",
        reference_stat="mean",
        eps_ratio=1e-9,
        eps_range=1e-12,
        eps_ref=1e-9,
        eps_abs=0.0,
        ref_add_alpha=0.0,
        log2_offset_delta=0.0,
    )
    # 'batch' should be integer-like
    assert is_integer_dtype(out["batch"])
    # v*, y* and r_logic should be float-like
    for c in ["v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star", "r_logic"]:
        assert is_float_dtype(out[c])
