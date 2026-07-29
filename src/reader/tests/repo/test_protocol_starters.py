from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from reader.protocols import BUILTIN_PROTOCOLS, ProtocolBinding
from reader.runtime import builtin_runtime
from reader.workbench.cli import app
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import ResourceInputDecl


@pytest.mark.parametrize("protocol_id", [item.protocol for item in BUILTIN_PROTOCOLS])
def test_every_builtin_protocol_starter_survives_the_public_onboarding_path(
    tmp_path: Path,
    protocol_id: str,
) -> None:
    target = tmp_path / protocol_id.replace("/", "__")
    runner = CliRunner()

    init_result = runner.invoke(app, ["init", str(target), "--protocol", protocol_id])

    assert init_result.exit_code == 0, init_result.output
    config_path = target / "config.yaml"
    spec = ReaderSpec.load(config_path)
    assert spec.protocol.id == protocol_id
    assert set(spec.resources.by_id) == {
        item.id for item in next(d for d in BUILTIN_PROTOCOLS if d.protocol == protocol_id).resources
    }

    for command in ("inspect", "explain"):
        result = runner.invoke(app, [command, str(config_path), "--format", "json"])
        assert result.exit_code == 0, f"{protocol_id} {command}: {result.output}"

    validate_result = runner.invoke(app, ["validate", str(config_path), "--no-files", "--format", "json"])
    assert validate_result.exit_code == 0, f"{protocol_id} validate: {validate_result.output}"


def test_generated_starters_declare_only_executable_plot_outputs() -> None:
    for descriptor in BUILTIN_PROTOCOLS:
        if descriptor.figures:
            assert descriptor.execution.compiler is not None


def test_protocol_resource_contract_matches_the_default_compiled_plan() -> None:
    runtime = builtin_runtime()
    for descriptor in BUILTIN_PROTOCOLS:
        plan = runtime.bind_protocol(ProtocolBinding(id=descriptor.protocol)).compile()
        compiled_resource_ids = {
            ref.resource_id
            for step in (*plan.pipeline, *plan.plots, *plan.exports)
            for ref in (step.reads or {}).values()
            if isinstance(ref, ResourceInputDecl)
        }
        assert {item.id for item in descriptor.resources} == compiled_resource_ids
