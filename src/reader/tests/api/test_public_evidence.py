from __future__ import annotations

from pathlib import Path

import pytest

from reader.api import inspect, open_experiment
from reader.tests.support.configs import base_reader_config, write_config


@pytest.mark.parametrize("replicate_identity_field", ["colony_id", None], ids=["within-record", "experiment"])
def test_experiment_evidence_is_queryable_through_identity_and_inspect(
    tmp_path: Path,
    replicate_identity_field: str | None,
) -> None:
    payload = base_reader_config(
        experiment_id="evidence_api",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [10.0]}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    payload["evidence"] = {
        "data_class": "plate_reader_screen",
        "data_class_reason": "The source is a well-level plate-reader assay.",
        "replicate_kind": "biological",
        "replicate_identity_field": replicate_identity_field,
    }
    config_path = write_config(tmp_path, payload)

    experiment = open_experiment(config_path)
    inspection = inspect(experiment)

    expected = {
        "data_class": "plate_reader_screen",
        "data_class_reason": "The source is a well-level plate-reader assay.",
        "replicate_kind": "biological",
        "replicate_identity_field": replicate_identity_field,
    }
    assert experiment.identity.evidence is not None
    assert experiment.identity.evidence.to_dict() == expected
    assert inspection.experiment.evidence == experiment.identity.evidence
    assert inspection.to_dict()["experiment"]["evidence"] == expected
