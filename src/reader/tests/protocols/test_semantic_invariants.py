from __future__ import annotations

import pytest

from reader.errors import ConfigError
from reader.protocols import ProtocolBinding, builtin_protocol_catalog
from reader.protocols.compiler import _semantic_program
from reader.protocols.model import (
    CompiledProtocolPlan,
    ProtocolDescriptor,
    ProtocolExecutionPlan,
    ProtocolMetricSpec,
    ProtocolNotebookPolicy,
    ProtocolSemanticExecution,
    ProtocolSemanticProfileSpec,
)


def test_semantic_program_rejects_profile_scoped_missing_dependencies() -> None:
    descriptor = ProtocolDescriptor(
        protocol="test/profile_scoped_dependency",
        domain="generic",
        family="test_protocol",
        summary="Profile-scoped semantic dependency validation.",
        semantic_profiles=(
            ProtocolSemanticProfileSpec(id="profile_a", family="test", summary="Profile A"),
            ProtocolSemanticProfileSpec(id="profile_b", family="test", summary="Profile B"),
        ),
        metrics=(
            ProtocolMetricSpec(
                id="A",
                stage="raw",
                summary="Metric A.",
                formula="a",
                profiles=("profile_a",),
            ),
            ProtocolMetricSpec(
                id="B",
                stage="summary",
                summary="Metric B.",
                formula="b",
                depends_on=("A",),
                profiles=("profile_b",),
            ),
        ),
        execution=ProtocolExecutionPlan(
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/basic",
                allowed_templates=("notebook/basic",),
                summary="Test notebook policy.",
            ),
            compiler=lambda protocol: CompiledProtocolPlan(),
        ),
    )

    with pytest.raises(ValueError, match="depends on unknown node"):
        descriptor.semantic_program(active_profile="profile_b")


def test_compiler_rejects_unknown_semantic_override_ids() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="plate_reader/dual_reporter_screen"))

    with pytest.raises(ConfigError, match="unknown ids"):
        _semantic_program(
            protocol,
            overrides={"missing_metric": ProtocolSemanticExecution(status="compiled")},
            active_profile="yfp_cfp_fold_change",
        )


def test_bound_protocol_semantic_program_applies_execution_overrides_without_changing_structure() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="plate_reader/dual_reporter_screen"))

    authored = protocol.semantic_program(active_profile="yfp_cfp_fold_change")
    compiled = protocol.semantic_program(
        active_profile="yfp_cfp_fold_change",
        execution_overrides={
            "OD": ProtocolSemanticExecution(
                status="compiled",
                step_ids=("ingest",),
                record_ids=("ingest/df",),
            )
        },
    )
    authored_metrics = {node.id: node for node in authored.metrics}
    compiled_metrics = {node.id: node for node in compiled.metrics}

    assert compiled.active_profile == authored.active_profile
    assert [node.id for node in compiled.metrics] == [node.id for node in authored.metrics]
    assert compiled_metrics["OD"].summary == authored_metrics["OD"].summary
    assert compiled_metrics["OD"].formula == authored_metrics["OD"].formula
    assert authored_metrics["OD"].execution.status == "descriptive_only"
    assert compiled_metrics["OD"].execution.status == "compiled"
    assert compiled_metrics["OD"].execution.step_ids == ("ingest",)
