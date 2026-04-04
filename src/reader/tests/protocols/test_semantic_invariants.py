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
