from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import ValidationError

from reader.plugins.plot.dual_reporter_triptych import (
    DualReporterTriptychCfg,
    DualReporterTriptychPlot,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for design in ("pA", "pB"):
        for treatment in ("control", "induced"):
            for time in (0.0, 1.0, 2.0):
                for replicate in (1, 2):
                    for channel, base in (("OD600", 0.2), ("YFP/CFP", 1.0)):
                        rows.append(
                            {
                                "design_id": design,
                                "treatment": treatment,
                                "time": time,
                                "channel": channel,
                                "value": base + time + replicate / 10.0,
                                "position": f"A{replicate}",
                                "sheet_index": 0 if time == 0.0 else 1,
                            }
                        )
    return pd.DataFrame.from_records(rows)


def test_dual_reporter_triptych_plugin_renders_one_artifact_per_design() -> None:
    cfg = DualReporterTriptychCfg(
        snapshot_time_h=1.1,
        snapshot_time_mode="nearest",
        snapshot_time_tolerance_h=0.2,
        treatment_order=["control", "induced"],
        trajectory_resamples=25,
        filename_prefix="triptych",
        format=["png", "pdf"],
        dpi=144,
    )

    rendered = DualReporterTriptychPlot().render(
        SimpleNamespace(palette_book=None, experiment=None),
        {"df": _frame()},
        cfg,
    )

    assert [(item.filename, item.ext, item.dpi) for item in rendered] == [
        ("triptych__pA", "png", 144),
        ("triptych__pA", "pdf", 144),
        ("triptych__pB", "png", 144),
        ("triptych__pB", "pdf", 144),
    ]
    assert rendered[0].fig is rendered[1].fig
    assert rendered[2].fig is rendered[3].fig
    assert rendered[0].fig is not rendered[2].fig
    assert {item.description for item in rendered} == {
        "Growth and reporter-ratio kinetics with descriptive within-experiment resampling bands, plus observed endpoint values and mean with sample standard deviation."
    }
    for figure in {item.fig for item in rendered}:
        plt.close(figure)


def test_dual_reporter_triptych_plugin_declares_tidy_record_input() -> None:
    assert DualReporterTriptychPlot.input_ports()["df"].contract == "tidy.v1"


def test_dual_reporter_triptych_normalizes_design_labels_for_partitioning() -> None:
    frame = _frame()
    frame["design_id"] = frame["design_id"].map(lambda value: f" {value} ")

    rendered = DualReporterTriptychPlot().render(
        SimpleNamespace(palette_book=None, experiment=None),
        {"df": frame},
        DualReporterTriptychCfg(snapshot_time_h=1.0, trajectory_resamples=5),
    )

    assert [item.filename for item in rendered] == [
        "dual_reporter_triptych__pA",
        "dual_reporter_triptych__pB",
    ]
    for figure in {item.fig for item in rendered}:
        plt.close(figure)


def test_dual_reporter_triptych_rejects_colliding_design_artifact_names() -> None:
    frame = _frame()
    frame.loc[frame["design_id"] == "pA", "design_id"] = "p A"
    frame.loc[frame["design_id"] == "pB", "design_id"] = "p_A"

    with pytest.raises(ValueError, match="same artifact name"):
        DualReporterTriptychPlot().render(
            SimpleNamespace(palette_book=None, experiment=None),
            {"df": frame},
            DualReporterTriptychCfg(snapshot_time_h=1.0, trajectory_resamples=5),
        )


def test_dual_reporter_triptych_config_rejects_competing_treatment_orders() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        DualReporterTriptychCfg(
            snapshot_time_h=1.0,
            treatment_order=["control", "induced"],
            treatment_order_ref="condition_order",
        )


def test_dual_reporter_triptych_rejects_snapshot_outside_tolerance() -> None:
    cfg = DualReporterTriptychCfg(
        snapshot_time_h=14.0,
        snapshot_time_mode="nearest",
        snapshot_time_tolerance_h=0.2,
    )

    with pytest.raises(ValueError, match="outside snapshot_time_tolerance_h"):
        DualReporterTriptychPlot().render(
            SimpleNamespace(palette_book=None, experiment=None),
            {"df": _frame()},
            cfg,
        )
