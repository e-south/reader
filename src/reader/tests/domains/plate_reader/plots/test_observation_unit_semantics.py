from __future__ import annotations

from inspect import signature

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from reader.domains.plate_reader.plots.common import descriptive_mean_resampling_interval
from reader.domains.plate_reader.plots.dual_reporter_triptych import build_triptych_data
from reader.domains.plate_reader.plots.snapshot_barplot import plot_snapshot_barplot
from reader.domains.plate_reader.plots.snapshot_barplot.planning import compute_shared_ylim
from reader.domains.time_series import ObservationAggregationSpec
from reader.plugins.plot.dual_reporter_triptych import DualReporterTriptychCfg
from reader.plugins.plot.logic_symmetry import LogicSymmetryPlotCfg
from reader.plugins.plot.single_reporter_diagnostic import SingleReporterDiagnosticCfg
from reader.plugins.plot.snapshot_barplot import SnapshotBarCfg
from reader.plugins.plot.time_series import TimeSeriesCfg
from reader.plugins.plot.ts_and_snap import TSAndSnapCfg


def _temporal_reduction() -> dict[str, object]:
    return {
        "selection": {
            "kind": "endpoint",
            "time_basis": "absolute",
            "time_h": 1.0,
            "mode": "exact",
            "tolerance_h": 0.0,
        },
        "method": "identity",
        "output_space": "linear",
        "support": {
            "boundary_support": "none",
            "minimum_observations": 1,
            "maximum_interior_gap_h": None,
            "positive_floor": None,
            "positive_value_scope": "selected_support",
            "censored_values": "allow",
        },
    }


def test_observation_aggregation_retires_replicate_named_contract() -> None:
    policy = ObservationAggregationSpec.from_mapping(
        {"within_unit_statistic": "median", "across_unit_statistic": "median"}
    )

    assert policy.within_unit_statistic == "median"
    assert "ReplicateAggregationSpec" not in __import__("reader.domains.time_series", fromlist=["*"]).__all__


def test_single_reporter_config_rejects_retired_replicate_aggregation_key() -> None:
    common = {
        "temporal_reduction": _temporal_reduction(),
        "normalizer_channel": "OD",
        "reporter_channel": "reporter",
        "ratio_channel": "reporter/OD",
    }

    SingleReporterDiagnosticCfg(
        observation_aggregation={"within_unit_statistic": "median", "across_unit_statistic": "median"},
        **common,
    )
    with pytest.raises(ValidationError, match="replicate_aggregation"):
        SingleReporterDiagnosticCfg(
            replicate_aggregation={"within_unit_statistic": "median", "across_unit_statistic": "median"},
            **common,
        )


def test_plot_configs_reject_replicate_and_inferential_interval_keys() -> None:
    DualReporterTriptychCfg(snapshot_time_h=1.0, trajectory_interval_mass=0.95, trajectory_resamples=25)
    with pytest.raises(ValidationError, match="trajectory_interval_mass"):
        DualReporterTriptychCfg(snapshot_time_h=1.0, trajectory_interval_mass=95.0)
    with pytest.raises(ValidationError, match="trajectory_ci|trajectory_bootstraps"):
        DualReporterTriptychCfg(snapshot_time_h=1.0, trajectory_ci=95.0, trajectory_bootstraps=25)

    TimeSeriesCfg(
        observation_interval_mass=0.95,
        observation_interval_alpha=0.15,
        observation_resamples=25,
        observation_seed=1,
        show_observations=True,
    )
    with pytest.raises(ValidationError, match="observation_interval_mass"):
        TimeSeriesCfg(observation_interval_mass=95.0)
    with pytest.raises(ValidationError, match="ci|show_replicates"):
        TimeSeriesCfg(ci=95.0, ci_boot=25, show_replicates=True)
    with pytest.raises(ValidationError, match="replicate_alpha|replicate_marker_size"):
        TimeSeriesCfg(fig={"replicate_alpha": 0.2, "replicate_marker_size": 8.0})

    TSAndSnapCfg(
        ts_channel="OD",
        ts_hue="condition",
        snap_time=1.0,
        ts_observation_interval_mass=0.95,
        ts_observation_resamples=25,
        ts_show_observations=True,
        snap_dispersion="sd",
        fig={"observation_marker_size": 8.0, "observation_alpha": 0.4},
    )
    with pytest.raises(ValidationError, match="ts_ci|ts_show_replicates|snap_err"):
        TSAndSnapCfg(
            ts_channel="OD",
            ts_hue="condition",
            snap_time=1.0,
            ts_ci=95.0,
            ts_show_replicates=True,
            snap_err="sem",
        )

    LogicSymmetryPlotCfg(dispersion="bars")
    with pytest.raises(ValidationError, match="uncertainty"):
        LogicSymmetryPlotCfg(uncertainty="errorbars")

    SnapshotBarCfg(x="condition", y="signal", time=1.0, dispersion="sd")
    with pytest.raises(ValidationError, match="err"):
        SnapshotBarCfg(x="condition", y="signal", time=1.0, err="sem")


def test_dual_reporter_points_use_observation_identity_language() -> None:
    rows: list[dict[str, object]] = []
    for position, value in (("A1", 1.0), ("A2", 2.0)):
        for channel in ("OD600", "YFP/CFP"):
            rows.append(
                {
                    "position": position,
                    "time": 1.0,
                    "channel": channel,
                    "value": value,
                    "treatment": "control",
                }
            )

    data = build_triptych_data(
        pd.DataFrame.from_records(rows),
        time_col="time",
        treatment_col="treatment",
        growth_channel="OD600",
        ratio_channel="YFP/CFP",
        snapshot_channel=None,
        snapshot_time=1.0,
        treatment_order=None,
        trajectory_interval_mass=0.95,
        trajectory_resamples=10,
    )

    assert "observation_index" in data.snapshot_points.columns
    assert "replicate_index" not in data.snapshot_points.columns


def test_snapshot_plot_uses_descriptive_dispersion_contract() -> None:
    params = signature(plot_snapshot_barplot).parameters

    assert "dispersion" in params
    assert "err" not in params


def test_descriptive_resampling_rejects_percentage_scale_interval_mass() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        descriptive_mean_resampling_interval(
            np.asarray([1.0, 2.0]),
            interval_mass=95.0,
            resamples=10,
            rng=np.random.default_rng(0),
        )


def test_snapshot_shared_limits_include_lower_standard_deviation_extent() -> None:
    stats = pd.DataFrame.from_records(
        [
            {"channel": "signal", "design": "a", "condition": "on", "mean": 1.0, "std": 2.0},
            {"channel": "signal", "design": "b", "condition": "on", "mean": 2.0, "std": 0.5},
        ]
    )
    observations = pd.DataFrame.from_records(
        [
            {"channel": "signal", "design": "a", "condition": "on", "value": 0.5},
            {"channel": "signal", "design": "b", "condition": "on", "value": 1.5},
        ]
    )

    lower, upper = compute_shared_ylim(
        stats=stats,
        snapped=observations,
        panels=["a", "b"],
        panel_by="group",
        selected_channel="signal",
        group_col="design",
        x_col="condition",
        agg="mean",
        dispersion="sd",
    )

    assert lower is not None and lower < -1.0
    assert upper is not None and upper > 3.0
