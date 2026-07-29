from __future__ import annotations

from pathlib import Path

import pytest

from reader.errors import ConfigError
from reader.protocols import BUILTIN_PROTOCOLS, BoundProtocol, ProtocolBinding, builtin_protocol_catalog
from reader.protocols.model import (
    CompiledProtocolPlan,
    ProtocolDescriptor,
    ProtocolExecutionPlan,
    ProtocolMetricSpec,
    ProtocolNotebookPolicy,
    ProtocolSemanticExecution,
    ProtocolSemanticNode,
    ProtocolSemanticProfileSpec,
    ProtocolSemanticProgram,
)
from reader.protocols.semantic_coverage import _semantic_program
from reader.workbench.decl.model import NotebookTemplateCallDecl
from reader.workbench.experiment import AnnotationSemantics, ExperimentSemantics, OutputLayout, ResourceCatalog


def test_builtin_protocol_tuple_keeps_public_order_stable() -> None:
    assert [descriptor.protocol for descriptor in BUILTIN_PROTOCOLS] == [
        "workbench/generic",
        "plate_reader/dual_reporter_screen",
        "plate_reader/single_reporter_screen",
        "plate_reader/response_window",
        "logic/sfxi_screen",
        "logic/sfxi_vec8_collection",
        "cytometry/flow_panel",
    ]


def test_builtin_protocols_do_not_expose_relaxed_runtime_contracts() -> None:
    for descriptor in BUILTIN_PROTOCOLS:
        assert all(field.key != "strict" for field in descriptor.analysis_fields)
        compiled = builtin_protocol_catalog().bind(ProtocolBinding(id=descriptor.protocol)).compile()
        assert "strict" not in compiled.runtime


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
            compiler=lambda protocol: CompiledProtocolPlan(semantic_program=protocol.semantic_program()),
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


def test_bound_protocol_compile_injects_default_notebook_when_compiler_omits_notebooks() -> None:
    descriptor = ProtocolDescriptor(
        protocol="test/default_notebook",
        domain="generic",
        family="test_protocol",
        summary="Compiler notebook fallback contract.",
        execution=ProtocolExecutionPlan(
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/basic",
                allowed_templates=("notebook/basic", "notebook/eda"),
                summary="Notebook policy.",
            ),
            compiler=lambda protocol: CompiledProtocolPlan(semantic_program=protocol.semantic_program()),
        ),
    )

    compiled = BoundProtocol(descriptor=descriptor).compile()

    assert compiled.notebooks == (NotebookTemplateCallDecl(id="default", template="notebook/basic"),)


def test_logic_sfxi_screen_can_compile_vec8_heatmap_plot() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="logic/sfxi_screen",
            outputs={"plots": {"include": ["sfxi_vec8_heatmap"]}},
        )
    )

    compiled = protocol.compile()
    plot = next(step for step in compiled.plots if step.id == "sfxi_vec8_heatmap")

    assert plot.plugin == "plot/sfxi_vec8_heatmap"
    assert plot.reads["vec8"].record_id == "sfxi_vec8/vec8"
    assert any(step.id == "sfxi_vec8" for step in compiled.pipeline)


def test_logic_sfxi_screen_names_typed_vec8_channels() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="logic/sfxi_screen"))

    vec8 = next(metric for metric in protocol.descriptor.metrics if metric.id == "vec8")

    assert vec8.formula == "v00,v10,v01,v11,y00_star,y10_star,y01_star,y11_star"
    assert protocol.descriptor.ranking is None


def test_protocol_semantic_execution_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="must be 'compiled' or 'descriptive_only'"):
        ProtocolSemanticExecution(status="typo")


def test_semantic_program_rejects_profiles_without_declared_catalog() -> None:
    with pytest.raises(ValueError, match="references unknown semantic profiles"):
        ProtocolSemanticProgram(
            protocol="test/missing_profiles",
            metrics=(
                ProtocolSemanticNode(
                    id="M",
                    kind="metric",
                    summary="Metric with undeclared profile.",
                    profiles=("profile_a",),
                    stage="raw",
                    formula="value",
                ),
            ),
        )


def test_experiment_semantics_rejects_mismatched_protocol_program() -> None:
    compiled_program = (
        builtin_protocol_catalog()
        .bind(ProtocolBinding(id="plate_reader/dual_reporter_screen"))
        .compile()
        .semantic_program
    )

    with pytest.raises(ValueError, match="must target the bound protocol"):
        ExperimentSemantics(
            protocol=ProtocolBinding(id="logic/sfxi_screen"),
            annotations=AnnotationSemantics(),
            resources=ResourceCatalog(),
            layout=OutputLayout(
                outputs_dir=Path("outputs"),
                plots_subdir="plots",
                exports_subdir="exports",
                notebooks_subdir="notebooks",
            ),
            protocol_program=compiled_program,
        )
