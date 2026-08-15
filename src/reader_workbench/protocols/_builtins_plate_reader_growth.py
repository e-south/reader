from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from reader_workbench.workbench.input_discovery import (
    DEFAULT_INPUT_EXCLUDE as DEFAULT_EXCLUDE,
)
from reader_workbench.workbench.input_discovery import (
    DEFAULT_WORKBOOK_INCLUDE as DEFAULT_INCLUDE,
)

from .compiler import compile_plate_reader_growth_screen
from .model import (
    ProtocolConfigFieldSpec,
    ProtocolDescriptor,
    ProtocolExecutionPlan,
    ProtocolFigureSpec,
    ProtocolMetricSpec,
    ProtocolPlotProfileSpec,
    ProtocolPluginDefaultsSpec,
    ProtocolResourceSpec,
    ProtocolSemanticProfileSpec,
    binding_value,
)


def build_plate_reader_growth_protocol(
    *,
    dual_reporter_protocol: ProtocolDescriptor,
    field_builder: Callable[..., ProtocolConfigFieldSpec],
) -> ProtocolDescriptor:
    """Build the one-channel growth protocol from shared plate-reader inputs."""

    field = field_builder
    ingest_field = next(item for item in dual_reporter_protocol.input_fields if item.key == "ingest")
    channel_map_field = next(item for item in ingest_field.children if item.key == "channel_map")
    growth_ingest = replace(
        ingest_field,
        children=tuple(
            replace(
                item,
                summary=(
                    "Optional raw workbook label to canonical channel mapping. Leave null for "
                    "map-free kinetic discovery; snapshot and mixed parsing require a mapping."
                ),
                default={"OD600:600": "OD600"},
            )
            if item is channel_map_field
            else item
            for item in ingest_field.children
        ),
    )
    inherited_preprocessing = next(
        item for item in dual_reporter_protocol.analysis_fields if item.key == "preprocessing"
    )
    preprocessing_field = replace(
        inherited_preprocessing,
        children=tuple(
            replace(
                item,
                children=tuple(
                    replace(child, default="none") if child.key == "action" else child for child in item.children
                ),
            )
            if item.key == "overflow"
            else item
            for item in inherited_preprocessing.children
        ),
    )

    return ProtocolDescriptor(
        protocol="plate_reader/growth_screen",
        domain="plate_reader",
        family="screen_analysis",
        summary="One-channel plate-reader growth assay with explicit sample and treatment metadata.",
        tags=("plate_reader", "growth", "screen", "single_channel"),
        resources=(
            ProtocolResourceSpec(
                id="sample_map",
                path="./inputs/metadata.xlsx",
                summary="Well-to-sample metadata for the plate-reader workbook.",
            ),
        ),
        input_fields=(growth_ingest,),
        analysis_fields=(
            field(
                "growth_channel",
                "Measured channel used as the growth readout.",
                kind="string",
                default="OD600",
            ),
            preprocessing_field,
        ),
        factors=dual_reporter_protocol.factors,
        semantic_profiles=(
            ProtocolSemanticProfileSpec(
                id="growth_raw",
                family="growth_panel",
                summary="Growth-screen semantics over one configured measurement channel.",
                primary_metric="Growth",
                primary_readout="configured growth channel",
                tags=("growth", "plate_reader", "panel"),
            ),
        ),
        metrics=(
            ProtocolMetricSpec(
                id="Growth",
                stage="raw",
                summary="Raw configured growth trace.",
                formula="configured_growth_channel",
                profiles=("growth_raw",),
            ),
        ),
        figures=(
            ProtocolFigureSpec(
                id="raw_kinetics",
                kind="qc",
                summary="Raw growth trajectories by sample and treatment.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="endpoint_by_condition",
                kind="summary",
                summary="Explicit-time growth comparison grouped by treatment or condition.",
            ),
            ProtocolFigureSpec(
                id="endpoint_by_design",
                kind="summary",
                summary="Explicit-time growth comparison grouped by sample or design.",
            ),
            ProtocolFigureSpec(
                id="growth_overview",
                kind="kinetics",
                summary="Growth trajectories and an explicitly timed endpoint in one view.",
            ),
            ProtocolFigureSpec(
                id="value_distributions",
                kind="qc",
                summary="Distribution view of the configured growth channel.",
            ),
        ),
        plot_profiles=(
            ProtocolPlotProfileSpec(
                id="screen_overview",
                summary="Explicit-time growth screen review.",
                figures=("raw_kinetics", "endpoint_by_condition", "endpoint_by_design", "growth_overview"),
            ),
            ProtocolPlotProfileSpec(
                id="kinetics_qc",
                summary="Growth trajectories and value distributions without an inferred endpoint.",
                figures=("raw_kinetics", "value_distributions"),
            ),
        ),
        default_plot_profile="kinetics_qc",
        execution=ProtocolExecutionPlan(
            plugin_defaults=(
                ProtocolPluginDefaultsSpec(
                    plugin="ingest/synergy_h1",
                    summary="Generic one-channel plate-reader ingest settings.",
                    with_={
                        "mode": binding_value("ingest.mode", "kinetic_only"),
                        "channel_map": binding_value("ingest.channel_map", {"OD600:600": "OD600"}),
                        "sheet_names": binding_value("ingest.sheet_names", None),
                        "time_round_decimals": binding_value("ingest.time_round_decimals", 12),
                        "time_step_h": binding_value("ingest.time_step_h", None),
                        "time_offset_h": binding_value("ingest.time_offset_h", 0.0),
                        "auto_roots": binding_value("ingest.auto_roots", None),
                        "auto_include": binding_value("ingest.auto_include", list(DEFAULT_INCLUDE)),
                        "auto_exclude": binding_value("ingest.auto_exclude", list(DEFAULT_EXCLUDE)),
                        "auto_pick": binding_value("ingest.auto_pick", "single"),
                        "auto_recursive": binding_value("ingest.auto_recursive", False),
                        "print_summary": binding_value("ingest.print_summary", True),
                    },
                ),
            ),
            compiler=compile_plate_reader_growth_screen,
        ),
    )
