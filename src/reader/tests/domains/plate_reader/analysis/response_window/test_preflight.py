from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from reader.domains.plate_reader.analysis.response_window.contracts import load_response_window_request
from reader.domains.plate_reader.analysis.response_window.materialize import materialize_experiment
from reader.domains.plate_reader.analysis.response_window.preflight import preflight_response_window_request
from reader.domains.plate_reader.analysis.response_window.sources import EventInterval, ExperimentSource
from reader.tests.domains.plate_reader.analysis.response_window.test_response_window_contracts import _payload


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("coverage", "does not cover"),
        ("gap", "interior gap"),
        ("floor", "positive floor"),
        ("replicates", "replicate support"),
    ],
)
def test_preflight_rejects_unbuildable_reduction_and_quality_inputs(
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    request_path = _write_request(tmp_path, coverage_end=4.0 if drift == "coverage" else 2.0)
    source = _source(tmp_path)
    if drift == "gap":
        for frame in (source.response, source.magnitude):
            frame.drop(frame.loc[frame["time"].eq(2.0)].index, inplace=True)
    elif drift == "floor":
        source.response.loc[source.response.index[0], "value"] = 1.0e-13
    elif drift == "replicates":
        for frame in (source.response, source.magnitude):
            frame.drop(
                frame.loc[frame["design_id"].eq("design") & frame["state"].eq("00") & frame["position"].eq("A2")].index,
                inplace=True,
            )

    with pytest.raises(ValueError, match=message):
        preflight_response_window_request(
            request_path=request_path,
            source_loader=lambda *_args: source,
        )


def test_preflight_accepts_fully_buildable_request(tmp_path: Path) -> None:
    request_path = _write_request(tmp_path, coverage_end=2.0)
    source = _source(tmp_path)

    result = preflight_response_window_request(
        request_path=request_path,
        source_loader=lambda *_args: source,
    )

    assert result.ready is True
    assert result.reduction_ids == ("primary",)


def test_materialization_propagates_trace_bounds_to_well_and_state_summaries(tmp_path: Path) -> None:
    request_path = _write_request(tmp_path, coverage_end=2.0)
    request = load_response_window_request(request_path)
    source = _source(tmp_path)
    response_mask = (
        source.response["design_id"].eq("design")
        & source.response["state"].eq("00")
        & source.response["position"].eq("A1")
        & source.response["time"].eq(2.0)
    )
    source.response.loc[response_mask, "value_policy_clipped"] = True
    source.response.loc[response_mask, "value_bound_kind"] = "lower"
    reference_mask = (
        source.magnitude["design_id"].eq("reference")
        & source.magnitude["state"].eq("00")
        & source.magnitude["position"].eq("A1")
        & source.magnitude["time"].eq(2.0)
    )
    source.magnitude.loc[reference_mask, "value_instrument_overflow"] = True
    source.magnitude.loc[reference_mask, "value_bound_kind"] = "lower"
    event_only_mask = (
        source.response["design_id"].eq("design")
        & source.response["state"].eq("10")
        & source.response["position"].eq("A1")
        & source.response["time"].eq(1.0)
    )
    source.response.loc[event_only_mask, "value_policy_clipped"] = True
    source.response.loc[event_only_mask, "value_bound_kind"] = "lower"

    wells, designs, _draws, traces, _events = materialize_experiment(source, request=request)

    affected_trace = traces.loc[
        traces["design_id"].eq("design")
        & traces["state"].eq("00")
        & traces["position"].eq("A1")
        & traces["signal_kind"].eq("response")
        & traces["time"].eq(2.0)
    ].iloc[0]
    assert bool(affected_trace["value_policy_clipped"])
    assert affected_trace["value_bound_kind"] == "lower"
    affected_well = wells.loc[
        wells["design_id"].eq("design") & wells["state"].eq("00") & wells["position"].eq("A1")
    ].iloc[0]
    assert affected_well["response_policy_clipped_point_count"] == 1
    assert affected_well["response_instrument_overflow_point_count"] == 0
    assert affected_well["response_bound_kind"] == "lower"
    design = designs.loc[designs["design_id"].eq("design")].iloc[0]
    assert bool(design["r00_has_policy_clipping"])
    assert not bool(design["r00_has_instrument_overflow"])
    assert design["r00_bound_kind"] == "lower"
    assert bool(design["b00_has_instrument_overflow"])
    assert design["b00_bound_kind"] == "upper"
    assert design["r10_bound_kind"] == "exact"
    assert bool(design["r10_event_sensitivity_has_policy_clipping"])
    assert not bool(design["r10_event_sensitivity_has_instrument_overflow"])
    assert design["b10_bound_kind"] == "exact"
    reference = designs.loc[designs["design_id"].eq("reference")].iloc[0]
    assert bool(reference["b00_has_instrument_overflow"])
    assert reference["b00"] == 0.0
    assert reference["b00_bound_kind"] == "exact"


def _write_request(tmp_path: Path, *, coverage_end: float) -> Path:
    request = _payload()
    request["experiment_ids"] = ["20260101_example"]
    request["display"]["examples"] = [
        {"design_id": "reference", "label": "Reference", "role": "reference_anchor"},
        {"design_id": "design", "label": "Design", "role": "response_example"},
    ]
    request["reductions"] = [
        {
            "id": "primary",
            "window_start_event_h": 1.0,
            "window_end_event_h": coverage_end,
            "method": "geometric_time_mean",
            "response_basis": "post_window",
            "role": "primary",
        }
    ]
    request["aggregation"] = {
        "replicate_stat": "median",
        "bootstrap_samples": 100,
        "confidence_level": 0.9,
        "random_seed": 17,
    }
    request["quality"] = {
        "positive_floor": 1.0e-12,
        "max_interior_gap_h": 1.1,
        "min_replicates_per_state": 2,
    }
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")
    return path


def _source(tmp_path: Path) -> ExperimentSource:
    rows: list[dict[str, object]] = []
    for design_id in ("reference", "design"):
        for state in ("00", "10", "01", "11"):
            for position in ("A1", "A2"):
                for time in (0.0, 1.0, 2.0, 3.0, 4.0):
                    rows.append(
                        {
                            "experiment_id": "20260101_example",
                            "design_id": design_id,
                            "state": state,
                            "position": position,
                            "time": time,
                            "time_from_event_h": time - 1.0,
                            "value": 2.0 + 0.1 * time,
                            "value_policy_clipped": False,
                            "value_instrument_overflow": False,
                            "value_bound_kind": "exact",
                        }
                    )
    ratios = pd.DataFrame.from_records(rows)
    return ExperimentSource(
        experiment_id="20260101_example",
        experiment_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        records_path=tmp_path / "records.json",
        response_path=tmp_path / "response.parquet",
        magnitude_path=tmp_path / "magnitude.parquet",
        trajectory_path=tmp_path / "growth.parquet",
        response=ratios.copy(),
        magnitude=ratios.copy(),
        trajectory=ratios.copy(),
        event=EventInterval(
            experiment_id="20260101_example",
            event_id="induction",
            event_kind="perturbation_addition",
            interval_start_assay_h=0.75,
            interval_end_assay_h=1.25,
            estimate_assay_h=1.0,
            estimate_method="segment_gap_midpoint",
            uncertainty_h=0.25,
            post_event_coverage_h=2.75,
            declaration="The intervention occurred between acquisition segments 0 and 1.",
        ),
        record_digests={
            "annotated/df": "sha256:" + "1" * 64,
            "ratio_magnitude/df": "sha256:" + "2" * 64,
            "ratio_response/df": "sha256:" + "3" * 64,
        },
    )
