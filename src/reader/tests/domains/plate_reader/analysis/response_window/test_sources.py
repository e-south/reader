from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reader.domains.plate_reader.analysis.response_window.contracts import EventSpec, ResponseSourceSpec
from reader.domains.plate_reader.analysis.response_window.provenance import sha256_file
from reader.domains.plate_reader.analysis.response_window.sources import (
    ResolvedExperimentSource,
    _normalize_value_provenance,
    load_experiment_source,
    resolve_event_interval,
)
from reader.runtime import builtin_runtime


def _event_spec() -> EventSpec:
    return EventSpec(
        event_id="stress_addition",
        event_kind="perturbation_addition",
        segment_column="sheet_index",
        pre_segment_index=0,
        post_segment_index=1,
        estimate_method="segment_gap_midpoint",
        declaration="Stress was added between acquisition segments 0 and 1.",
    )


def test_resolve_event_interval_rejects_fractional_segment_indexes() -> None:
    frame = pd.DataFrame(
        {
            "sheet_index": [0.0, 0.0, 1.9, 1.9],
            "time": [0.0, 1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="segment indexes must be finite integers"):
        resolve_event_interval(frame, experiment_id="fractional-segment", event_spec=_event_spec())


def test_resolve_event_interval_rejects_missing_nullable_segment_indexes() -> None:
    frame = pd.DataFrame(
        {
            "sheet_index": pd.Series([0, 0, 1, pd.NA], dtype="Int64"),
            "time": [0.0, 1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="segment indexes must be finite integers"):
        resolve_event_interval(frame, experiment_id="missing-segment", event_spec=_event_spec())


def test_resolve_event_interval_accepts_exact_integral_float_indexes() -> None:
    frame = pd.DataFrame(
        {
            "sheet_index": [0.0, 0.0, 1.0, 1.0],
            "time": [0.0, 1.0, 2.0, 3.0],
        }
    )

    event = resolve_event_interval(frame, experiment_id="integral-segment", event_spec=_event_spec())

    assert event.interval_start_assay_h == 1.0
    assert event.interval_end_assay_h == 2.0


def test_load_experiment_source_preserves_value_bound_provenance(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for record_id, channel in (
        ("ratio_response/df", "YFP/CFP"),
        ("ratio_magnitude/df", "YFP/OD600"),
        ("annotated/df", "OD600"),
    ):
        frame = pd.DataFrame(
            {
                "design_id": ["reference"] * 4,
                "position": ["A1"] * 4,
                "time": [0.0, 1.0, 2.0, 3.0],
                "channel": [channel] * 4,
                "value": [1.0, 2.0, 3.0, 4.0],
                "treatment": ["none"] * 4,
                "sheet_index": [0, 0, 1, 1],
                "value_policy_clipped": [False] * 4,
                "value_instrument_overflow": [False] * 4,
                "value_bound_kind": ["exact"] * 4,
            }
        )
        if record_id == "ratio_response/df":
            frame["value_policy_clipped"] = [False, False, True, False]
            frame["value_bound_kind"] = ["exact", "exact", "lower", "exact"]
        elif record_id == "ratio_magnitude/df":
            frame["value_instrument_overflow"] = [False, False, True, False]
            frame["value_bound_kind"] = ["exact", "exact", "upper", "exact"]
        path = tmp_path / f"{record_id.replace('/', '_')}.parquet"
        frame.to_parquet(path, index=False)
        paths[record_id] = path
        digests[record_id] = sha256_file(path)

    config_path = tmp_path / "config.yaml"
    records_path = tmp_path / "records.json"
    config_path.write_text("schema: reader/v8\n", encoding="utf-8")
    records_path.write_text("{}\n", encoding="utf-8")
    resolved = ResolvedExperimentSource(
        experiment_id="experiment",
        experiment_dir=tmp_path,
        config_path=config_path,
        records_path=records_path,
        record_paths=paths,
        record_contracts=dict.fromkeys(paths, "plate_reader.annotated.v1"),
        record_digests=digests,
        state_column="treatment",
        treatment_map={"00": "none", "10": "ethanol", "01": "cipro", "11": "both"},
        state_values_case_sensitive=True,
    )
    source_spec = ResponseSourceSpec(
        response_record_id="ratio_response/df",
        magnitude_record_id="ratio_magnitude/df",
        trajectory_record_id="annotated/df",
        response_channel="YFP/CFP",
        magnitude_channel="YFP/OD600",
        growth_channel="OD600",
        reference_design_id="reference",
        state_map_ref="states",
    )

    source = load_experiment_source(
        resolved,
        source_spec=source_spec,
        event_spec=_event_spec(),
        contracts=builtin_runtime().contracts,
    )

    assert source.response["value_policy_clipped"].tolist() == [False, False, True, False]
    assert source.response["value_bound_kind"].tolist() == ["exact", "exact", "lower", "exact"]
    assert source.magnitude["value_instrument_overflow"].tolist() == [False, False, True, False]
    assert source.magnitude["value_bound_kind"].tolist() == ["exact", "exact", "upper", "exact"]
    assert not source.trajectory["value_policy_clipped"].any()
    assert not source.trajectory["value_instrument_overflow"].any()
    assert set(source.trajectory["value_bound_kind"]) == {"exact"}


def test_response_window_source_rejects_missing_value_provenance() -> None:
    frame = pd.DataFrame({"value": [1.0]})

    with pytest.raises(ValueError, match="missing required value provenance columns"):
        _normalize_value_provenance(frame, context="experiment:response")
