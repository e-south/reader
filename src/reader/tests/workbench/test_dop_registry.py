from __future__ import annotations

from typer.testing import CliRunner

from reader.protocols import builtin_protocol_catalog
from reader.tests.support import cli_success_data
from reader.workbench import cli
from reader.workbench.dop import builtin_dop_registry
from reader.workbench.inspection.readiness import READINESS_CAPABILITY_KEYS, READINESS_STATES


def test_dop_registry_protocol_candidates_resolve_against_builtin_catalog() -> None:
    registry = builtin_dop_registry()
    protocols = builtin_protocol_catalog()

    registry.validate_protocol_refs(descriptor.protocol for descriptor in protocols.all())

    covered_protocols = {
        protocol_id for data_class in registry.data_classes() for protocol_id in data_class.protocol_candidates
    }
    assert {descriptor.protocol for descriptor in protocols.all()} <= covered_protocols


def test_dop_registry_ready_specs_match_reader_readiness_contract() -> None:
    registry = builtin_dop_registry()

    registry.validate_ready_refs(
        readiness_states=READINESS_STATES,
        capability_keys=READINESS_CAPABILITY_KEYS,
    )

    assert "catalog_ready" in READINESS_STATES
    assert "verify" in READINESS_CAPABILITY_KEYS

    assert [spec.id for spec in registry.ready_specs()] == [
        "classified",
        "metadata_ready",
        "staged",
        "preflight_ok",
        "runnable",
        "records_ready",
        "review_ready",
    ]


def test_dop_classes_cli_emits_json() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["dop", "classes", "--format", "json"])

    assert result.exit_code == 0
    payload = cli_success_data(result.output)
    ids = [item["id"] for item in payload["data_classes"]]
    assert ids[0] == "plate_reader_screen"
    assert "unsupported_long_tail_assay" in ids
    assert "ready_specs" not in payload


def test_dop_classes_cli_filters_by_protocol() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["dop", "classes", "--protocol", "plate_reader/retron_sponge_screen", "--format", "json"],
    )

    assert result.exit_code == 0
    payload = cli_success_data(result.output)
    assert [item["id"] for item in payload["data_classes"]] == ["plate_reader_screen"]


def test_dop_ready_specs_cli_emits_json() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["dop", "ready-specs", "--format", "json"])

    assert result.exit_code == 0
    payload = cli_success_data(result.output)
    assert [item["id"] for item in payload["ready_specs"]] == [
        "classified",
        "metadata_ready",
        "staged",
        "preflight_ok",
        "runnable",
        "records_ready",
        "review_ready",
    ]
    assert payload["ready_specs"][-1]["required_capabilities"] == [
        "records",
        "plot",
        "notebook_scaffold",
    ]
