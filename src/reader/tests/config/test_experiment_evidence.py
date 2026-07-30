from __future__ import annotations

from pathlib import Path

import pytest

from reader.errors import ConfigError
from reader.tests.support.configs import base_reader_config, load_models, write_config
from reader.workbench.config import ReaderSpec, reader_spec_digest
from reader.workbench.experiment import ExperimentEvidence


def _config_with_evidence() -> dict:
    payload = base_reader_config(
        experiment_id="evidence_example",
        protocol_id="plate_reader/dual_reporter_screen",
    )
    payload["evidence"] = {
        "data_class": "plate_reader_screen",
        "data_class_reason": "Well-level plate-reader measurements with an explicit sample map.",
        "replicate_kind": "biological",
        "replicate_identity_field": "colony_id",
    }
    return payload


def test_evidence_is_optional_and_bound_into_experiment_semantics(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, _config_with_evidence())

    spec, declaration = load_models(config_path)

    assert spec.evidence is not None
    assert spec.evidence.data_class == "plate_reader_screen"
    assert declaration.experiment_semantics.evidence is not None
    assert declaration.experiment_semantics.evidence.to_payload() == {
        "data_class": "plate_reader_screen",
        "data_class_reason": "Well-level plate-reader measurements with an explicit sample map.",
        "replicate_kind": "biological",
        "replicate_identity_field": "colony_id",
    }

    without_evidence = base_reader_config(experiment_id="no_evidence")
    no_evidence_path = write_config(tmp_path / "without.yaml", without_evidence)
    _, no_evidence_decl = load_models(no_evidence_path)
    assert no_evidence_decl.experiment_semantics.evidence is None


def test_evidence_data_class_must_be_registered_with_the_dop(tmp_path: Path) -> None:
    payload = _config_with_evidence()
    payload["evidence"]["data_class"] = "invented_assay"

    with pytest.raises(ConfigError, match=r"Unknown DOP data class 'invented_assay'.*plate_reader_screen"):
        ReaderSpec.load(write_config(tmp_path, payload))


def test_evidence_data_class_must_admit_the_bound_protocol(tmp_path: Path) -> None:
    payload = _config_with_evidence()
    payload["evidence"]["data_class"] = "flow_cytometry_panel"

    with pytest.raises(ConfigError) as exc_info:
        ReaderSpec.load(write_config(tmp_path, payload))

    message = str(exc_info.value)
    assert "flow_cytometry_panel" in message
    assert "plate_reader/dual_reporter_screen" in message
    assert "cytometry/flow_panel" in message


def test_unsupported_long_tail_evidence_accepts_generic_protocol(tmp_path: Path) -> None:
    payload = _config_with_evidence()
    payload["protocol"]["id"] = "workbench/generic"
    payload["evidence"]["data_class"] = "unsupported_long_tail_assay"

    spec = ReaderSpec.load(write_config(tmp_path, payload))

    assert spec.evidence is not None
    assert spec.evidence.data_class == "unsupported_long_tail_assay"


@pytest.mark.parametrize("replicate_kind", ["biological", "technical", "mixed", "unknown", "not_applicable"])
def test_evidence_accepts_the_declared_replicate_kinds(tmp_path: Path, replicate_kind: str) -> None:
    payload = _config_with_evidence()
    payload["evidence"]["replicate_kind"] = replicate_kind
    if replicate_kind == "not_applicable":
        payload["evidence"].pop("replicate_identity_field")

    spec = ReaderSpec.load(write_config(tmp_path / f"{replicate_kind}.yaml", payload))

    assert spec.evidence is not None
    assert spec.evidence.replicate_kind == replicate_kind


def test_evidence_preserves_unknown_replication_without_an_invented_identity(tmp_path: Path) -> None:
    payload = _config_with_evidence()
    payload["evidence"]["replicate_kind"] = "unknown"
    payload["evidence"].pop("replicate_identity_field")

    spec, declaration = load_models(write_config(tmp_path / "unknown-without-identity.yaml", payload))

    assert spec.evidence is not None
    assert spec.evidence.replicate_kind == "unknown"
    assert spec.evidence.replicate_identity_field is None
    assert declaration.experiment_semantics.evidence is not None
    assert declaration.experiment_semantics.evidence.to_payload() == {
        "data_class": "plate_reader_screen",
        "data_class_reason": "Well-level plate-reader measurements with an explicit sample map.",
        "replicate_kind": "unknown",
        "replicate_identity_field": None,
    }


@pytest.mark.parametrize("replicate_kind", ["biological", "technical", "mixed"])
def test_evidence_requires_identity_for_declared_replicate_relationships(
    tmp_path: Path,
    replicate_kind: str,
) -> None:
    payload = _config_with_evidence()
    payload["evidence"]["replicate_kind"] = replicate_kind
    payload["evidence"].pop("replicate_identity_field")

    with pytest.raises(ConfigError, match="replicate_identity_field.*required"):
        ReaderSpec.load(write_config(tmp_path / f"{replicate_kind}-without-identity.yaml", payload))


def test_evidence_rejects_unknown_fields_and_invalid_identity_combinations(tmp_path: Path) -> None:
    payload = _config_with_evidence()
    payload["evidence"]["comment"] = "free-form schema drift"
    with pytest.raises(ConfigError, match="comment"):
        ReaderSpec.load(write_config(tmp_path / "unknown.yaml", payload))

    payload = _config_with_evidence()
    payload["evidence"]["replicate_kind"] = "not_applicable"
    with pytest.raises(ConfigError, match="replicate_identity_field.*not_applicable"):
        ReaderSpec.load(write_config(tmp_path / "not-applicable.yaml", payload))


@pytest.mark.parametrize("field", ["data_class_reason", "replicate_identity_field"])
def test_evidence_rejects_blank_explanatory_fields(tmp_path: Path, field: str) -> None:
    payload = _config_with_evidence()
    payload["evidence"][field] = "   "

    with pytest.raises(ConfigError, match=field):
        ReaderSpec.load(write_config(tmp_path / f"{field}.yaml", payload))


def test_evidence_changes_the_normalized_config_identity() -> None:
    baseline_payload = base_reader_config(
        experiment_id="identity",
        protocol_id="plate_reader/dual_reporter_screen",
    )
    baseline = ReaderSpec.model_validate(baseline_payload)
    with_evidence_payload = base_reader_config(
        experiment_id="identity",
        protocol_id="plate_reader/dual_reporter_screen",
    )
    with_evidence_payload["evidence"] = _config_with_evidence()["evidence"]
    with_evidence = ReaderSpec.model_validate(with_evidence_payload)

    assert reader_spec_digest(with_evidence) != reader_spec_digest(baseline)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"replicate_kind": "biological", "replicate_identity_field": None}, "replicate_identity_field.*required"),
        ({"replicate_kind": "invented"}, "unsupported replicate_kind"),
        ({"replicate_identity_field": "   "}, "replicate_identity_field.*non-empty"),
    ],
)
def test_experiment_evidence_direct_construction_matches_config_invariants(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "data_class": "plate_reader_screen",
        "data_class_reason": "Declared source evidence.",
        "replicate_kind": "unknown",
        "replicate_identity_field": None,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        ExperimentEvidence(**values)  # type: ignore[arg-type]
