from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from reader.workbench.input_discovery import (
    DEFAULT_INPUT_EXCLUDE as DEFAULT_EXCLUDE,
)
from reader.workbench.input_discovery import (
    DEFAULT_WORKBOOK_INCLUDE as DEFAULT_INCLUDE,
)

from .compiler import compile_plate_reader_single_reporter_screen
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


def build_plate_reader_variant_protocol(
    *,
    dual_reporter_protocol: ProtocolDescriptor,
    field_builder: Callable[..., ProtocolConfigFieldSpec],
) -> ProtocolDescriptor:
    """Derive the single-reporter protocol from shared plate-reader contracts."""

    field = field_builder
    preprocessing_field = next(item for item in dual_reporter_protocol.analysis_fields if item.key == "preprocessing")
    ingest_field = next(item for item in dual_reporter_protocol.input_fields if item.key == "ingest")
    channel_map_field = next(item for item in ingest_field.children if item.key == "channel_map")
    map_free_ingest = replace(
        ingest_field,
        children=tuple(
            replace(
                item,
                summary=(
                    "Optional raw workbook label to canonical channel mapping. Leave null for "
                    "map-free kinetic discovery; snapshot and mixed parsing require a mapping."
                ),
                default=None,
            )
            if item is channel_map_field
            else item
            for item in ingest_field.children
        ),
    )
    input_fields = tuple(
        map_free_ingest if item is ingest_field else item for item in dual_reporter_protocol.input_fields
    )

    return ProtocolDescriptor(
        protocol="plate_reader/single_reporter_screen",
        domain="plate_reader",
        family="screen_analysis",
        summary=(
            "Single-reporter plate-reader protocol with configurable reporter and "
            "normalizer channels plus optional fold-change summaries."
        ),
        tags=("plate_reader", "single_reporter", "screen", "ratio", "fold_change"),
        resources=(
            ProtocolResourceSpec(
                id="sample_map",
                path="./inputs/metadata.xlsx",
                summary="Well-to-sample metadata for the plate-reader workbook.",
            ),
        ),
        input_fields=input_fields,
        analysis_fields=(
            field(
                "reporter_channel",
                "Primary reporter channel to normalize against the configured normalizer.",
                kind="string",
                default="RFP",
            ),
            field(
                "normalizer_channel",
                "Denominator channel used to normalize the reporter signal.",
                kind="string",
                default="OD600",
            ),
            field(
                "temporal_reduction",
                "Optional compiler-owned endpoint or interval reduction used by temporal diagnostic outputs.",
                allow_unknown=True,
                allow_none=True,
                default=None,
            ),
            field(
                "observation_aggregation",
                "Optional within-unit and across-unit aggregation policy for temporal diagnostics.",
                allow_unknown=True,
                allow_none=True,
                default=None,
            ),
            field("include_fold_change", "Build the fold-change comparison table.", kind="bool", default=False),
            preprocessing_field,
        ),
        factors=dual_reporter_protocol.factors,
        semantic_profiles=(
            ProtocolSemanticProfileSpec(
                id="single_reporter_raw",
                family="single_reporter_panel",
                summary="Single-reporter semantics over a configured reporter/normalizer ratio.",
                primary_metric="Reporter_Normalizer",
                primary_readout="reporter / normalizer",
                tags=("single_reporter", "ratio", "panel"),
            ),
            ProtocolSemanticProfileSpec(
                id="single_reporter_fold_change",
                family="single_reporter_panel",
                summary="Single-reporter semantics with compiled fold-change summaries.",
                primary_metric="log2FC",
                primary_readout="reporter / normalizer",
                tags=("single_reporter", "ratio", "panel", "fold_change"),
            ),
        ),
        control_rules=(),
        windows=(),
        metrics=(
            ProtocolMetricSpec(
                id="Normalizer",
                stage="raw",
                summary="Raw configured normalizer trace.",
                formula="configured_normalizer_channel",
                profiles=("single_reporter_raw", "single_reporter_fold_change"),
            ),
            ProtocolMetricSpec(
                id="Reporter",
                stage="raw",
                summary="Raw configured reporter trace.",
                formula="configured_reporter_channel",
                profiles=("single_reporter_raw", "single_reporter_fold_change"),
            ),
            ProtocolMetricSpec(
                id="Reporter_Normalizer",
                stage="support",
                summary="Configured reporter normalized by the configured denominator channel.",
                formula="configured_reporter_channel / configured_normalizer_channel",
                depends_on=("Reporter", "Normalizer"),
                value_space="linear_ratio",
                unit="ratio",
                comparable_group="primary_ratio_linear",
                profiles=("single_reporter_raw", "single_reporter_fold_change"),
            ),
            ProtocolMetricSpec(
                id="FC",
                stage="summary",
                summary="Nearest-time fold-change relative to the configured baseline treatment.",
                formula="Reporter_Normalizer(t*) / baseline(Reporter_Normalizer)",
                depends_on=("Reporter_Normalizer",),
                value_space="fold_change_ratio",
                unit="ratio",
                comparable_group="fold_change_linear",
                profiles=("single_reporter_fold_change",),
            ),
            ProtocolMetricSpec(
                id="log2FC",
                stage="summary",
                summary="Log2 fold-change relative to the configured baseline treatment.",
                formula="log2(FC)",
                depends_on=("FC",),
                value_space="log2_fold_change",
                unit="log2_ratio",
                comparable_group="fold_change_log2",
                profiles=("single_reporter_fold_change",),
            ),
        ),
        effect_signs=(),
        figures=(
            ProtocolFigureSpec(
                id="raw_kinetics",
                kind="qc",
                summary="Raw kinetics over the normalizer, reporter, and derived ratio channels.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="single_reporter_diagnostic",
                kind="qc",
                summary=(
                    "One-row normalizer, reporter, ratio, and condition-reduction diagnostic using an explicit "
                    "endpoint or interval."
                ),
            ),
            ProtocolFigureSpec(
                id="endpoint_by_condition",
                kind="summary",
                summary="Endpoint comparison grouped by treatment or condition.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="endpoint_by_design",
                kind="summary",
                summary="Endpoint comparison grouped by sample or design.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="intensity_overview",
                kind="kinetics",
                summary="Combined time-series and endpoint view of the primary ratio.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="subject_comparison",
                kind="summary",
                summary="Opt-in paired normalizer kinetics and treatment-ordered reporter endpoint.",
            ),
            ProtocolFigureSpec(
                id="value_distributions",
                kind="qc",
                summary="Distribution view of the primary single-reporter ratio.",
            ),
        ),
        plot_profiles=(
            ProtocolPlotProfileSpec(
                id="screen_overview",
                summary="Endpoint screen view; each selected endpoint time must be authored explicitly.",
                figures=("raw_kinetics", "endpoint_by_condition", "endpoint_by_design", "intensity_overview"),
            ),
            ProtocolPlotProfileSpec(
                id="kinetics_qc",
                summary="Kinetics-first QC view with raw traces and distributions.",
                figures=("raw_kinetics", "value_distributions"),
            ),
        ),
        default_plot_profile="kinetics_qc",
        execution=ProtocolExecutionPlan(
            plugin_defaults=(
                ProtocolPluginDefaultsSpec(
                    plugin="ingest/synergy_h1",
                    summary="Generic ingest settings; the compiler derives required channels.",
                    with_={
                        "mode": binding_value("ingest.mode", "kinetic_only"),
                        "channel_map": binding_value("ingest.channel_map", None),
                        "sheet_names": binding_value("ingest.sheet_names", None),
                        "time_round_decimals": binding_value("ingest.time_round_decimals", 12),
                        "time_step_h": binding_value("ingest.time_step_h", None),
                        "auto_roots": binding_value("ingest.auto_roots", None),
                        "auto_include": binding_value("ingest.auto_include", list(DEFAULT_INCLUDE)),
                        "auto_exclude": binding_value("ingest.auto_exclude", list(DEFAULT_EXCLUDE)),
                        "auto_pick": binding_value("ingest.auto_pick", "single"),
                        "auto_recursive": binding_value("ingest.auto_recursive", False),
                        "print_summary": binding_value("ingest.print_summary", True),
                    },
                ),
                ProtocolPluginDefaultsSpec(
                    plugin="transform/fold_change",
                    summary="Generic comparison settings; the compiler supplies the ratio target.",
                    with_={
                        "report_times": binding_value("fold_change.report_times"),
                        "time_tolerance": binding_value("fold_change.time_tolerance", 0.51),
                        "observation_stat": binding_value("fold_change.observation_stat", "median"),
                        "treatment_column": binding_value("fold_change.treatment_column", "treatment"),
                        "group_by": binding_value("fold_change.group_by", ["design_id"]),
                        "use_global_baseline": binding_value("fold_change.use_global_baseline", False),
                        "global_baseline_value": binding_value("fold_change.global_baseline_value", None),
                        "overrides": binding_value("fold_change.overrides", []),
                        "fc_column": binding_value("fold_change.fc_column", "FC"),
                        "log2fc_column": binding_value("fold_change.log2fc_column", "log2FC"),
                    },
                ),
            ),
            compiler=compile_plate_reader_single_reporter_screen,
        ),
    )
