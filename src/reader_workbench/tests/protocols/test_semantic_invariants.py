from __future__ import annotations

from pathlib import Path

import pytest

from reader_workbench.errors import ConfigError
from reader_workbench.protocols import BUILTIN_PROTOCOLS, ProtocolBinding, builtin_protocol_catalog
from reader_workbench.protocols.model import (
    CompiledProtocolPlan,
    ProtocolDescriptor,
    ProtocolExecutionPlan,
    ProtocolMetricSpec,
    ProtocolSemanticExecution,
    ProtocolSemanticNode,
    ProtocolSemanticProfileSpec,
    ProtocolSemanticProgram,
)
from reader_workbench.protocols.semantic_coverage import _semantic_program
from reader_workbench.tests.support import cytometry_test_gating_policy
from reader_workbench.workbench.experiment import (
    AnnotationSemantics,
    ExperimentSemantics,
    OutputLayout,
    ResourceCatalog,
)


def test_builtin_protocol_tuple_keeps_public_order_stable() -> None:
    assert [descriptor.protocol for descriptor in BUILTIN_PROTOCOLS] == [
        "workbench/generic",
        "plate_reader/dual_reporter_screen",
        "plate_reader/single_reporter_screen",
        "plate_reader/response_window",
        "logic/four_state_vector_screen",
        "logic/four_state_vector_collection",
        "cytometry/flow_panel",
    ]


def test_builtin_protocols_do_not_expose_relaxed_runtime_contracts() -> None:
    for descriptor in BUILTIN_PROTOCOLS:
        assert all(field.key != "strict" for field in descriptor.analysis_fields)
        inputs = {"gating": cytometry_test_gating_policy()} if descriptor.protocol == "cytometry/flow_panel" else {}
        compiled = builtin_protocol_catalog().bind(ProtocolBinding(id=descriptor.protocol, inputs=inputs)).compile()
        assert "strict" not in compiled.runtime


def test_cytometry_singlet_ratio_authoring_describes_y_over_x() -> None:
    descriptor = builtin_protocol_catalog().resolve("cytometry/flow_panel")
    gating = next(field for field in descriptor.input_fields if field.key == "gating")
    fields = {field.key: field for field in gating.children}

    assert fields["singlet_x_channel"].summary == "Denominator channel for the singlet ratio (Y / X)."
    assert fields["singlet_y_channel"].summary == "Numerator channel for the singlet ratio (Y / X)."


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


def test_logic_four_state_vector_screen_can_compile_heatmap_plot() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="logic/four_state_vector_screen",
            outputs={"plots": {"include": ["four_state_vector_heatmap"]}},
        )
    )

    compiled = protocol.compile()
    plot = next(step for step in compiled.plots if step.id == "four_state_vector_heatmap")

    assert plot.plugin == "plot/four_state_vector_heatmap"
    assert plot.reads["vector"].record_id == "four_state_vector/vector"
    assert any(step.id == "four_state_vector" for step in compiled.pipeline)


def test_logic_four_state_vector_screen_compiles_a_record_driven_diagnostic() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="logic/four_state_vector_screen",
            inputs={"state_map_ref": "states", "time_column": "elapsed_h"},
            outputs={
                "plots": {
                    "profile": "none",
                    "include": ["four_state_vector_diagnostic"],
                    "views": {"four_state_vector_diagnostic": {"design_ids": ["design-a"], "format": ["png"]}},
                }
            },
        )
    )

    compiled = protocol.compile()
    diagnostic = next(step for step in compiled.plots if step.id == "four_state_vector_diagnostic")

    assert diagnostic.plugin == "plot/four_state_vector_diagnostic"
    assert diagnostic.reads["df"].record_id == "promote_to_tidy_plus_map/df"
    assert diagnostic.reads["vector"].record_id == "four_state_vector/vector"
    assert diagnostic.with_["state_map_ref"] == "states"
    assert diagnostic.with_["time_column"] == "elapsed_h"
    assert diagnostic.with_["response_channel"] == "YFP/CFP"
    assert diagnostic.with_["design_ids"] == ["design-a"]
    assert any(step.id == "four_state_vector" for step in compiled.pipeline)


@pytest.mark.parametrize("key", ["growth_channel", "response_channel", "state_map_ref", "time_column"])
def test_logic_four_state_vector_diagnostic_rejects_compiler_owned_overrides(key: str) -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="logic/four_state_vector_screen",
            outputs={
                "plots": {
                    "profile": "none",
                    "include": ["four_state_vector_diagnostic"],
                    "views": {"four_state_vector_diagnostic": {key: "override"}},
                }
            },
        )
    )

    with pytest.raises(ConfigError, match="cannot override compiler-owned settings"):
        protocol.compile()


def test_response_window_can_compile_a_focused_diagnostic_as_a_normal_plot() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/response_window",
            outputs={
                "plots": {
                    "include": ["response_window_diagnostic"],
                    "views": {
                        "response_window_diagnostic": {
                            "source_experiment_id": "trace-source",
                            "design_id": "design-a",
                        }
                    },
                }
            },
        )
    )

    compiled = protocol.compile()
    diagnostic = next(step for step in compiled.plots if step.id == "response_window_diagnostic")

    assert diagnostic.plugin == "plot/response_window_diagnostic"
    assert diagnostic.reads["designs"].record_id == "response_window/designs"
    assert diagnostic.reads["traces"].record_id == "response_window/traces"
    assert diagnostic.with_["primary_reduction_id"] == "primary"
    assert diagnostic.with_["pre_window_duration_h"] is None
    assert diagnostic.with_["source_experiment_id"] == "trace-source"
    assert diagnostic.with_["design_id"] == "design-a"


def test_dual_reporter_screen_compiles_triptych_from_persisted_ratio_record() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/dual_reporter_screen",
            outputs={
                "plots": {
                    "profile": "none",
                    "include": ["dual_reporter_triptych"],
                    "views": {
                        "dual_reporter_triptych": {
                            "snapshot_time_h": 8.0,
                            "snapshot_time_tolerance_h": 0.25,
                            "treatment_order_ref": "conditions",
                            "format": ["png", "pdf"],
                        }
                    },
                }
            },
        )
    )

    compiled = protocol.compile()
    triptych = next(step for step in compiled.plots if step.id == "dual_reporter_triptych")

    assert triptych.plugin == "plot/dual_reporter_triptych"
    assert triptych.reads["df"].record_id == "ratio_yfp_od600/df"
    assert triptych.with_ == {
        "design_column": "design_id",
        "treatment_column": "treatment",
        "time_column": "time",
        "growth_channel": "OD600",
        "ratio_channel": "YFP/CFP",
        "snapshot_channel": "YFP/CFP",
        "snapshot_time_h": 8.0,
        "snapshot_time_mode": "nearest",
        "snapshot_time_tolerance_h": 0.25,
        "treatment_order_ref": "conditions",
        "format": ["png", "pdf"],
    }


def test_dual_reporter_screen_requires_explicit_triptych_snapshot_time() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/dual_reporter_screen",
            outputs={"plots": {"profile": "none", "include": ["dual_reporter_triptych"]}},
        )
    )

    with pytest.raises(
        ConfigError,
        match=r"protocol\.outputs\.plots\.views\.dual_reporter_triptych\.snapshot_time_h must be explicit",
    ):
        protocol.compile()


def _single_reporter_interval_policy() -> dict:
    return {
        "selection": {
            "kind": "interval",
            "time_basis": "absolute",
            "start_h": 8.0,
            "end_h": 12.0,
            "boundary": "inclusive",
        },
        "method": "observed_median",
        "output_space": "linear",
        "support": {
            "boundary_support": "observed",
            "minimum_observations": 25,
            "maximum_interior_gap_h": 0.2,
            "positive_floor": None,
            "positive_value_scope": "selected_support",
            "censored_values": "reject",
        },
    }


def _single_reporter_aggregation_policy() -> dict:
    return {
        "within_unit_statistic": "median",
        "across_unit_statistic": "median",
    }


def test_single_reporter_screen_compiles_record_driven_four_panel_diagnostic() -> None:
    temporal_reduction = _single_reporter_interval_policy()
    observation_aggregation = _single_reporter_aggregation_policy()
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/single_reporter_screen",
            analysis={
                "reporter_channel": "mScarlet",
                "normalizer_channel": "absorbance",
                "temporal_reduction": temporal_reduction,
                "observation_aggregation": observation_aggregation,
            },
            outputs={
                "plots": {
                    "profile": "none",
                    "include": ["single_reporter_diagnostic"],
                    "views": {
                        "single_reporter_diagnostic": {
                            "partition": {"collection_ref": "subjects"},
                            "condition_column": "condition_alias",
                            "condition_order_ref": "conditions",
                            "format": ["png", "pdf"],
                        }
                    },
                }
            },
        )
    )

    diagnostic = next(step for step in protocol.compile().plots if step.id == "single_reporter_diagnostic")

    assert diagnostic.plugin == "plot/single_reporter_diagnostic"
    assert diagnostic.reads["df"].record_id == "sample_measurements/df"
    assert diagnostic.with_ == {
        "partition": {"collection_ref": "subjects"},
        "condition_column": "condition_alias",
        "temporal_reduction": temporal_reduction,
        "observation_aggregation": observation_aggregation,
        "time_column": "time",
        "normalizer_channel": "absorbance",
        "reporter_channel": "mScarlet",
        "ratio_channel": "mScarlet/absorbance",
        "condition_order_ref": "conditions",
        "format": ["png", "pdf"],
    }


@pytest.mark.parametrize(
    "temporal_reduction",
    [
        None,
        {**_single_reporter_interval_policy(), "unknown": True},
        {
            **_single_reporter_interval_policy(),
            "selection": {
                **_single_reporter_interval_policy()["selection"],
                "start_h": 12.0,
                "end_h": 8.0,
            },
        },
    ],
)
def test_single_reporter_diagnostic_requires_one_valid_temporal_reduction(
    temporal_reduction: dict | None,
) -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/single_reporter_screen",
            analysis={
                "temporal_reduction": temporal_reduction,
                "observation_aggregation": _single_reporter_aggregation_policy(),
            },
            outputs={
                "plots": {
                    "profile": "none",
                    "include": ["single_reporter_diagnostic"],
                }
            },
        )
    )

    with pytest.raises(ConfigError, match="diagnostic"):
        protocol.compile()


def test_single_reporter_diagnostic_rejects_retired_technical_aggregation_fields() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/single_reporter_screen",
            analysis={
                "temporal_reduction": _single_reporter_interval_policy(),
                "observation_aggregation": {
                    "technical_replicate_statistic": "median",
                    "replicate_center_statistic": "median",
                },
            },
            outputs={"plots": {"profile": "none", "include": ["single_reporter_diagnostic"]}},
        )
    )

    with pytest.raises(ConfigError, match="unknown fields"):
        protocol.compile()


def test_single_reporter_screen_rejects_retired_aggregation_fields_without_diagnostic() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/single_reporter_screen",
            analysis={
                "temporal_reduction": _single_reporter_interval_policy(),
                "observation_aggregation": {
                    "technical_replicate_statistic": "median",
                    "replicate_center_statistic": "median",
                },
            },
            outputs={"plots": {"profile": "none", "include": []}},
        )
    )

    with pytest.raises(ConfigError, match="unknown fields"):
        protocol.compile()


@pytest.mark.parametrize(
    "key",
    [
        "time_column",
        "normalizer_channel",
        "reporter_channel",
        "ratio_channel",
        "temporal_reduction",
        "observation_aggregation",
        "endpoint_time_h",
        "window_h",
        "summary_stat",
    ],
)
def test_single_reporter_diagnostic_rejects_compiler_owned_channel_overrides(key: str) -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/single_reporter_screen",
            analysis={
                "temporal_reduction": _single_reporter_interval_policy(),
                "observation_aggregation": _single_reporter_aggregation_policy(),
            },
            outputs={
                "plots": {
                    "profile": "none",
                    "include": ["single_reporter_diagnostic"],
                    "views": {"single_reporter_diagnostic": {key: "override"}},
                }
            },
        )
    )

    with pytest.raises(ConfigError, match="cannot override compiler-owned fields"):
        protocol.compile()


def test_response_window_diagnostic_requires_an_explicit_record_identity() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/response_window",
            outputs={"plots": {"include": ["response_window_diagnostic"]}},
        )
    )

    with pytest.raises(ConfigError, match="source_experiment_id must be a non-empty string"):
        protocol.compile()


def test_response_window_plot_cannot_override_the_primary_reduction() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/response_window",
            outputs={
                "plots": {
                    "views": {
                        "response_window_summary": {"primary_reduction_id": "secondary"},
                    }
                }
            },
        )
    )

    with pytest.raises(ConfigError, match="cannot override compiler-owned fields"):
        protocol.compile()


def test_response_window_diagnostic_receives_the_compiler_owned_pre_window() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="plate_reader/response_window",
            analysis={
                "reductions": [
                    {
                        "id": "delta",
                        "window_start_event_h": 0.5,
                        "window_end_event_h": 1.0,
                        "method": "geometric_time_mean",
                        "response_basis": "post_minus_pre",
                        "pre_window_duration_h": 1.5,
                        "role": "primary",
                    }
                ]
            },
            outputs={
                "plots": {
                    "include": ["response_window_diagnostic"],
                    "views": {
                        "response_window_diagnostic": {
                            "source_experiment_id": "trace-source",
                            "design_id": "design-a",
                        }
                    },
                }
            },
        )
    )

    diagnostic = next(step for step in protocol.compile().plots if step.id == "response_window_diagnostic")

    assert diagnostic.with_["primary_reduction_id"] == "delta"
    assert diagnostic.with_["pre_window_duration_h"] == 1.5


def test_logic_four_state_vector_screen_names_typed_vector_channels() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="logic/four_state_vector_screen"))

    vector = next(metric for metric in protocol.descriptor.metrics if metric.id == "four_state_vector")

    assert vector.formula == "v00,v10,v01,v11,y00_star,y10_star,y01_star,y11_star"
    assert protocol.descriptor.ranking is None


def test_generic_workbench_does_not_claim_a_ranking_policy() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="workbench/generic"))

    assert protocol.descriptor.ranking is None
    assert protocol.compile().semantic_program.ranking is None


def test_four_state_vector_break_rejects_retired_protocol_and_analysis_key() -> None:
    catalog = builtin_protocol_catalog()
    retired_protocol = "logic/" + "sf" + "xi_screen"
    retired_analysis_key = "include_" + "ve" + "c8"

    with pytest.raises(ConfigError, match="Unknown protocol"):
        catalog.bind(ProtocolBinding(id=retired_protocol))
    with pytest.raises(ConfigError, match="unknown keys"):
        catalog.bind(
            ProtocolBinding(
                id="logic/four_state_vector_screen",
                analysis={retired_analysis_key: True},
            )
        )


def test_logic_four_state_vector_semantics_do_not_reference_the_retired_response_block() -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id="logic/four_state_vector_screen"))
    program = protocol.compile().semantic_program

    compiled_paths = {
        path
        for node in (*program.controls, *program.windows, *program.metrics)
        if node.execution is not None
        for path in node.execution.config_paths
    }

    assert "protocol.inputs.response" not in compiled_paths


def test_logic_four_state_vector_screen_exposes_a_concrete_dual_reporter_adapter() -> None:
    catalog = builtin_protocol_catalog()
    protocol = catalog.bind(ProtocolBinding(id="logic/four_state_vector_screen"))
    transform_config = protocol.effective_plugin_config(plugin_id="transform/four_state_vector")

    assert "response" not in {field.key for field in protocol.descriptor.input_fields}
    assert transform_config["response"] == {
        "logic_channel": "YFP/CFP",
        "intensity_channel": "YFP/OD600",
    }
    with pytest.raises(ConfigError, match=r"unknown keys \['response'\]"):
        catalog.bind(
            ProtocolBinding(
                id="logic/four_state_vector_screen",
                inputs={"response": {"logic_channel": "A/B", "intensity_channel": "A/C"}},
            )
        )


def test_logic_four_state_vector_screen_rejects_fold_change_target_outside_its_compiled_adapter() -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id="logic/four_state_vector_screen",
            inputs={"fold_change": {"target": "A/B", "report_times": [8.0]}},
            analysis={"include_fold_change": True},
        )
    )

    with pytest.raises(ConfigError, match="must match the compiled assay ratio 'YFP/CFP'"):
        protocol.compile()


@pytest.mark.parametrize(
    "protocol_id",
    [
        "plate_reader/dual_reporter_screen",
        "plate_reader/single_reporter_screen",
        "logic/four_state_vector_screen",
    ],
)
def test_plate_reader_fold_change_requires_explicit_report_times(protocol_id: str) -> None:
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id=protocol_id, analysis={"include_fold_change": True}))

    with pytest.raises(ConfigError, match="fold_change.report_times.*explicit non-empty list"):
        protocol.compile()


@pytest.mark.parametrize(
    ("protocol_id", "figure_id", "time_key"),
    [
        ("plate_reader/dual_reporter_screen", "endpoint_by_condition", "time"),
        ("plate_reader/dual_reporter_screen", "intensity_overview", "snap_time"),
        ("plate_reader/single_reporter_screen", "endpoint_by_design", "time"),
        ("plate_reader/single_reporter_screen", "subject_comparison", "snap_time"),
    ],
)
def test_plate_reader_endpoint_plots_require_explicit_time(
    protocol_id: str,
    figure_id: str,
    time_key: str,
) -> None:
    protocol = builtin_protocol_catalog().bind(
        ProtocolBinding(
            id=protocol_id,
            outputs={"plots": {"profile": "none", "include": [figure_id]}},
        )
    )

    with pytest.raises(
        ConfigError,
        match=rf"protocol.outputs.plots.views.{figure_id}.{time_key} must be explicit",
    ):
        protocol.compile()


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
            protocol=ProtocolBinding(id="logic/four_state_vector_screen"),
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
