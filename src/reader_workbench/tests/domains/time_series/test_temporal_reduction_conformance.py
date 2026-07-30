from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from reader_workbench.domains.time_series import TemporalReductionSpec, reduce_temporal_trace

_VECTOR_PATH = Path(__file__).with_name("fixtures") / "temporal_reduction_conformance_v1.json"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def test_temporal_reduction_conformance_vector_executes_reader_reducer() -> None:
    vector_text = _VECTOR_PATH.read_text(encoding="utf-8")
    assert "technical" not in vector_text.lower()
    vector = json.loads(vector_text)

    assert vector["schema"] == "reader.temporal_reduction_conformance.v1"
    assert vector["contract"] == "reader.domains.time_series.temporal_reduction.v1"
    assert vector["time_unit"] == "hour"
    assert len({case["id"] for case in vector["cases"]}) == len(vector["cases"])

    for case in vector["cases"]:
        assert case["case_payload_digest"] == _canonical_digest(case["payload"]), case["id"]
        assert case["kind"] in {"ratio_then_reduce_observation_aggregation", "single_trace"}
        if case["kind"] == "ratio_then_reduce_observation_aggregation":
            _assert_ratio_then_reduce_case(case)
        else:
            _assert_single_trace_case(case)


def _assert_ratio_then_reduce_case(case: dict[str, object]) -> None:
    payload = case["payload"]
    expected = case["expected"]
    assert isinstance(payload, dict)
    assert isinstance(expected, dict)
    assert payload["operation_order"] == "ratio_then_reduce"
    assert payload["observation_aggregation"] == "median"
    assert payload["declared_ratio_channel"] == {
        "numerator": "signal",
        "denominator": "reference",
        "zero_denominator": "reject",
    }
    spec = TemporalReductionSpec.from_mapping(payload["temporal_reduction"])
    per_well: dict[str, float] = {}
    alternative_per_well: dict[str, float] = {}
    for well in payload["wells"]:
        well_id = well["well_id"]
        rows = np.asarray(well["rows"], dtype=float)
        assert rows.shape == (25, 3)
        if np.any(rows[:, 2] == 0.0):
            raise AssertionError(f"{well_id} contains a zero denominator")
        result = reduce_temporal_trace(
            rows[:, 0],
            rows[:, 1] / rows[:, 2],
            spec=spec,
            trace_id=well_id,
        )
        assert result.observed_point_count == expected["observed_point_count"]
        per_well[well_id] = result.value
        alternative_per_well[well_id] = float(np.median(rows[:, 1]) / np.median(rows[:, 2]))

    assert per_well == pytest.approx(expected["per_well"])
    observation_median = float(np.median(list(per_well.values())))
    assert observation_median == pytest.approx(expected["observation_median"])
    alternative = expected["alternative_reduce_then_ratio"]
    assert alternative_per_well == pytest.approx(alternative["per_well"])
    alternative_median = float(np.median(list(alternative_per_well.values())))
    assert alternative_median == pytest.approx(alternative["observation_median"])
    assert alternative_median != pytest.approx(observation_median)


def _assert_single_trace_case(case: dict[str, object]) -> None:
    payload = case["payload"]
    expected = case["expected"]
    assert isinstance(payload, dict)
    assert isinstance(expected, dict)
    spec = TemporalReductionSpec.from_mapping(payload["temporal_reduction"])
    kwargs = {
        name: np.asarray(payload[name])
        for name in ("policy_clipped", "instrument_overflow", "bound_kinds")
        if name in payload
    }

    if expected["status"] == "error":
        with pytest.raises(ValueError, match=str(expected["message_contains"])):
            reduce_temporal_trace(
                np.asarray(payload["times_h"], dtype=float),
                np.asarray(payload["values"], dtype=float),
                spec=spec,
                trace_id=str(case["id"]),
                **kwargs,
            )
        return

    result = reduce_temporal_trace(
        np.asarray(payload["times_h"], dtype=float),
        np.asarray(payload["values"], dtype=float),
        spec=spec,
        trace_id=str(case["id"]),
        **kwargs,
    )
    assert result.value == pytest.approx(expected["value"])
