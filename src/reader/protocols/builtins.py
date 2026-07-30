from __future__ import annotations

from functools import cache

from reader.workbench.input_discovery import (
    DEFAULT_INPUT_EXCLUDE as DEFAULT_EXCLUDE,
)
from reader.workbench.input_discovery import (
    DEFAULT_WORKBOOK_INCLUDE as DEFAULT_INCLUDE,
)

from ._builtins_plate_reader_variants import build_plate_reader_variant_protocol
from .compiler import (
    compile_cytometry_flow_panel,
    compile_generic_protocol,
    compile_logic_sfxi_screen,
    compile_logic_sfxi_vec8_collection,
    compile_plate_reader_dual_reporter_screen,
    compile_plate_reader_response_window,
)
from .model import (
    ProtocolArtifactSpec,
    ProtocolCatalog,
    ProtocolConfigFieldSpec,
    ProtocolControlRule,
    ProtocolDescriptor,
    ProtocolExecutionPlan,
    ProtocolFactorSpec,
    ProtocolFigureSpec,
    ProtocolMetricSpec,
    ProtocolPlotProfileSpec,
    ProtocolPluginDefaultsSpec,
    ProtocolRankingSpec,
    ProtocolResourceSpec,
    ProtocolSemanticProfileSpec,
    ProtocolWindowSpec,
    binding_value,
)

_MISSING = object()
_DUAL_REPORTER_CHANNEL_MAP = {
    "OD600:600": "OD600",
    "CFP:433,475": "CFP",
    "YFP:500,530": "YFP",
}


def _field(
    key: str,
    summary: str,
    *,
    kind: str = "mapping",
    required: bool = False,
    allow_none: bool = False,
    choices: tuple[str, ...] = (),
    children: tuple[ProtocolConfigFieldSpec, ...] = (),
    allow_unknown: bool = False,
    default: object = _MISSING,
    example: object = _MISSING,
) -> ProtocolConfigFieldSpec:
    kwargs: dict[str, object] = {}
    if default is not _MISSING:
        kwargs["default"] = default
    if example is not _MISSING:
        kwargs["example"] = example
    return ProtocolConfigFieldSpec(
        key=key,
        summary=summary,
        kind=kind,
        required=required,
        allow_none=allow_none,
        choices=choices,
        children=children,
        allow_unknown=allow_unknown,
        **kwargs,
    )


BUILTIN_PROTOCOLS: tuple[ProtocolDescriptor, ...] = (
    ProtocolDescriptor(
        protocol="workbench/generic",
        domain="generic",
        family="general_workbench",
        summary="Generic explicit protocol binding for experiments that do not yet fit a domain-specific protocol.",
        tags=("generic", "explicit_binding"),
        input_fields=(),
        analysis_fields=(),
        factors=(
            ProtocolFactorSpec(name="sample", role="sample", summary="Primary experimental unit."),
            ProtocolFactorSpec(
                name="replicate",
                role="replicate",
                summary="Source-declared replicate grouping axis.",
                required=False,
            ),
            ProtocolFactorSpec(name="time", role="time", summary="Observation axis when present.", required=False),
        ),
        ranking=ProtocolRankingSpec(
            primary_metric="domain_defined",
            direction="higher_is_better",
            summary="Generic protocol leaves ranking to the domain-specific analysis layer.",
        ),
        execution=ProtocolExecutionPlan(
            compiler=compile_generic_protocol,
        ),
    ),
    ProtocolDescriptor(
        protocol="logic/sfxi_vec8_collection",
        domain="logic",
        family="record_collection",
        summary="Combine SFXI vec8 records from declared Reader experiments with exact revision provenance.",
        tags=("logic", "aggregate", "records"),
        input_fields=(
            _field(
                "record_resources",
                "Ordered resources.by_id keys that identify source dataframe records.",
                kind="string_list",
                default=[],
            ),
        ),
        figures=(
            ProtocolFigureSpec(
                id="vec8_collection_heatmap",
                kind="summary",
                summary="Cross-experiment heatmap over the collected vec8 records.",
                primary=True,
            ),
        ),
        plot_profiles=(
            ProtocolPlotProfileSpec(
                id="collection_overview",
                summary="Primary cross-experiment vec8 heatmap.",
                figures=("vec8_collection_heatmap",),
            ),
        ),
        default_plot_profile="collection_overview",
        artifacts=(ProtocolArtifactSpec(id="vec8_table", summary="CSV table of the collected vec8 rows."),),
        execution=ProtocolExecutionPlan(
            compiler=compile_logic_sfxi_vec8_collection,
        ),
    ),
    ProtocolDescriptor(
        protocol="plate_reader/response_window",
        domain="plate_reader",
        family="event_relative_record_collection",
        summary="Materialize event-relative summaries from aligned records owned by declared Reader experiments.",
        tags=("plate_reader", "aggregate", "event", "window", "records"),
        input_fields=tuple(
            _field(
                key,
                summary,
                kind="string_list",
                default=[],
            )
            for key, summary in (
                ("response_records", "Ordered record resources for the response signal."),
                ("magnitude_records", "Ordered record resources for the magnitude signal."),
                ("trajectory_records", "Ordered record resources for the trajectory signal."),
            )
        ),
        analysis_fields=(
            _field(
                "source",
                "Signal, reference, and state-value bindings.",
                allow_unknown=True,
                default={
                    "response_channel": "response",
                    "magnitude_channel": "magnitude",
                    "growth_channel": "growth",
                    "reference_design_id": "reference",
                    "state_column": "state",
                    "state_values": {"00": "00", "10": "10", "01": "01", "11": "11"},
                    "state_values_case_sensitive": True,
                },
            ),
            _field(
                "event",
                "Declared event and acquisition-segment semantics.",
                allow_unknown=True,
                default={
                    "event_id": "event",
                    "event_kind": "declared_transition",
                    "segment_column": "segment",
                    "pre_segment_index": 0,
                    "post_segment_index": 1,
                    "estimate_method": "segment_gap_midpoint",
                    "declaration": "Transition between declared acquisition segments.",
                },
            ),
            _field(
                "reductions",
                "Event-relative reduction definitions.",
                kind="mapping_list",
                default=[
                    {
                        "id": "primary",
                        "window_start_event_h": 0.0,
                        "window_end_event_h": 1.0,
                        "method": "geometric_time_mean",
                        "response_basis": "post_window",
                        "role": "primary",
                    }
                ],
            ),
            _field(
                "aggregation",
                "Within-experiment observation aggregation and descriptive resampling settings.",
                allow_unknown=True,
                default={
                    "observation_stat": "median",
                    "descriptive_resampling_draws": 100,
                    "descriptive_interval_mass": 0.95,
                    "random_seed": 0,
                },
            ),
            _field(
                "quality",
                "Trace support and value-quality requirements.",
                allow_unknown=True,
                default={
                    "positive_floor": 1e-09,
                    "max_interior_gap_h": 1.0,
                    "min_observations_per_state": 2,
                },
            ),
        ),
        figures=(
            ProtocolFigureSpec(
                id="response_window_summary",
                kind="summary",
                summary="Primary event-relative response and anchored-magnitude components.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="response_window_diagnostic",
                kind="kinetics",
                summary="Focused trajectories and reduced components for one source design.",
            ),
        ),
        plot_profiles=(
            ProtocolPlotProfileSpec(
                id="response_window_overview",
                summary="Primary event-relative summary view.",
                figures=("response_window_summary",),
            ),
        ),
        default_plot_profile="response_window_overview",
        artifacts=(
            ProtocolArtifactSpec(id="designs_table", summary="CSV of design-level response-window summaries."),
            ProtocolArtifactSpec(id="events_table", summary="CSV of source event intervals."),
        ),
        execution=ProtocolExecutionPlan(
            compiler=compile_plate_reader_response_window,
        ),
    ),
    ProtocolDescriptor(
        protocol="plate_reader/dual_reporter_screen",
        domain="plate_reader",
        family="screen_analysis",
        summary=(
            "Dual-reporter plate-reader panel protocol with compiled ratio/fold-change summaries and "
            "optional crosstalk pair selection."
        ),
        tags=("plate_reader", "dual_reporter", "screen", "ratio", "fold_change"),
        resources=(
            ProtocolResourceSpec(
                id="sample_map",
                path="./inputs/metadata.xlsx",
                summary="Well-to-sample metadata for the plate-reader workbook.",
            ),
        ),
        input_fields=(
            _field(
                "ingest",
                "Plate-reader ingest settings and workbook selection.",
                children=(
                    _field(
                        "mode",
                        "Ingest mode for Synergy H1 parsing.",
                        kind="string",
                        choices=("snapshot_only", "kinetic_only", "mixed"),
                    ),
                    _field(
                        "channel_map",
                        "Raw workbook label to canonical channel mapping; required for snapshot or mixed parsing.",
                        kind="mapping",
                        allow_unknown=True,
                        allow_none=True,
                        default=dict(_DUAL_REPORTER_CHANNEL_MAP),
                    ),
                    _field(
                        "sheet_names", "Optional workbook sheet names to parse.", kind="string_list", allow_none=True
                    ),
                    _field(
                        "time_round_decimals",
                        "Rounding precision for parsed time values.",
                        kind="integer",
                        allow_none=True,
                        default=12,
                    ),
                    _field("time_step_h", "Override time-step spacing in hours.", kind="number", allow_none=True),
                    _field(
                        "auto_roots",
                        "Directories to scan for workbook auto-discovery.",
                        kind="string_list",
                        allow_none=True,
                    ),
                    _field(
                        "auto_include",
                        "Filename globs to include during auto-discovery.",
                        kind="string_list",
                        default=list(DEFAULT_INCLUDE),
                    ),
                    _field(
                        "auto_exclude",
                        "Filename globs to exclude during auto-discovery.",
                        kind="string_list",
                        default=list(DEFAULT_EXCLUDE),
                    ),
                    _field(
                        "auto_pick",
                        "Multi-file selection policy for auto-discovery.",
                        kind="string",
                        choices=("single", "latest"),
                        default="single",
                    ),
                    _field(
                        "auto_recursive",
                        "Recurse into child directories when discovering workbooks.",
                        kind="bool",
                        default=False,
                    ),
                    _field("print_summary", "Print an ingest summary to the log.", kind="bool", default=True),
                ),
            ),
            _field(
                "fold_change",
                "Fold-change summary inputs for screen-style comparisons.",
                children=(
                    _field(
                        "report_times",
                        "Explicit report times in hours for fold-change snapshots.",
                        kind="number_list",
                    ),
                    _field("time_tolerance", "Nearest-time tolerance in hours.", kind="number", default=0.51),
                    _field(
                        "observation_stat",
                        "Observation aggregation statistic.",
                        kind="string",
                        choices=("median", "mean"),
                        default="median",
                    ),
                    _field("treatment_column", "Treatment-state column name.", kind="string", default="treatment"),
                    _field(
                        "group_by",
                        "Grouping columns for comparison baselines.",
                        kind="string_list",
                        default=["design_id"],
                    ),
                    _field(
                        "use_global_baseline",
                        "Use one shared baseline instead of per-group baselines.",
                        kind="bool",
                        default=False,
                    ),
                    _field(
                        "global_baseline_value",
                        "Explicit global baseline label when global mode is enabled.",
                        kind="string",
                        allow_none=True,
                    ),
                    _field(
                        "overrides",
                        "Explicit baseline overrides keyed by group columns.",
                        kind="mapping_list",
                        default=[],
                    ),
                    _field("fc_column", "Output fold-change column name.", kind="string", default="FC"),
                    _field("log2fc_column", "Output log2 fold-change column name.", kind="string", default="log2FC"),
                ),
            ),
        ),
        analysis_fields=(
            _field("include_fold_change", "Build the fold-change comparison table.", kind="bool", default=False),
            _field(
                "preprocessing",
                "Pre-ingest cleanup policy for blanks and overflow.",
                children=(
                    _field(
                        "blank",
                        "Blank-correction policy.",
                        children=(
                            _field(
                                "method",
                                "Blank handling strategy.",
                                kind="string",
                                choices=("disregard", "subtract"),
                                default="disregard",
                            ),
                            _field(
                                "capture_blanks",
                                "Emit a blanks side table for downstream QC.",
                                kind="bool",
                                default=True,
                            ),
                        ),
                    ),
                    _field(
                        "overflow",
                        "Overflow handling policy for saturated channels.",
                        children=(
                            _field(
                                "action",
                                "Overflow action.",
                                kind="string",
                                choices=("max", "drop", "nan", "none"),
                                default="max",
                            ),
                            _field("clip_quantile", "Quantile cap used when action=max.", kind="number", default=0.999),
                            _field(
                                "cap_strategy",
                                "How per-channel caps are determined when action=max.",
                                kind="string",
                                choices=("provided", "infer", "quantile"),
                                default="quantile",
                            ),
                            _field(
                                "per_channel_caps",
                                "Explicit per-channel caps when cap_strategy=provided.",
                                kind="mapping",
                                allow_unknown=True,
                                allow_none=True,
                            ),
                            _field(
                                "flag_column",
                                "Column used to mark overflowed wells before capping.",
                                kind="string",
                                default="overflow",
                            ),
                            _field(
                                "treat_inf_as_overflow",
                                "Treat infinite values as overflowed rows.",
                                kind="bool",
                                default=True,
                            ),
                        ),
                    ),
                ),
            ),
            _field(
                "crosstalk_pairs",
                "Pair-selection analysis over fold-change tables.",
                children=(
                    _field("enabled", "Compute crosstalk-safe pair candidates.", kind="bool", default=False),
                    _field("export", "Export the crosstalk pairs table when present.", kind="bool", default=False),
                    _field("value_column", "Value column to score.", kind="string", default="log2FC"),
                    _field("value_scale", "Scale of the value column.", kind="string", default="log2"),
                    _field("time_mode", "Time-selection mode.", kind="string", default="all"),
                    _field("design_column", "Design/grouping column.", kind="string", default="design_id"),
                    _field("treatment_column", "Treatment-state column.", kind="string", default="treatment"),
                    _field("mapping_mode", "Pair mapping mode.", kind="string", default="explicit"),
                    _field("require_self_treatment", "Require a self-treatment pair.", kind="bool", default=True),
                    _field("require_self_is_top1", "Require self-treatment to rank first.", kind="bool", default=True),
                    _field("min_self", "Minimum on-target score.", kind="number", default=1.0),
                    _field("max_cross", "Maximum tolerated cross-score.", kind="number", default=0.5),
                    _field(
                        "min_selectivity_delta",
                        "Minimum on-target minus off-target margin.",
                        kind="number",
                        default=1.0,
                    ),
                    _field(
                        "design_treatment_map",
                        "Explicit design -> self-treatment mapping when mapping_mode=explicit.",
                        kind="mapping",
                        allow_unknown=True,
                    ),
                ),
            ),
        ),
        factors=(
            ProtocolFactorSpec(name="design_id", role="construct", summary="Primary construct or design grouping."),
            ProtocolFactorSpec(name="treatment", role="condition", summary="Primary treatment or assay condition."),
            ProtocolFactorSpec(name="position", role="observation", summary="Observed well or sample position."),
            ProtocolFactorSpec(name="time", role="time", summary="Time after assay start."),
            ProtocolFactorSpec(name="plate_id", role="plate", summary="Plate-local normalization boundary."),
        ),
        semantic_profiles=(
            ProtocolSemanticProfileSpec(
                id="yfp_cfp_raw",
                family="dual_reporter_panel",
                summary="Generic dual-reporter panel semantics over raw/support ratio traces.",
                primary_metric="Ratio",
                primary_readout="YFP / CFP",
                tags=("dual_reporter", "ratio", "panel"),
            ),
            ProtocolSemanticProfileSpec(
                id="yfp_cfp_fold_change",
                family="dual_reporter_panel",
                summary="Dual-reporter panel semantics with compiled fold-change summaries.",
                primary_metric="log2FC",
                primary_readout="YFP / CFP",
                tags=("dual_reporter", "ratio", "panel", "fold_change"),
            ),
            ProtocolSemanticProfileSpec(
                id="yfp_cfp_crosstalk",
                family="dual_reporter_crosstalk",
                summary="Dual-reporter panel semantics with compiled crosstalk pair selection.",
                primary_metric="log2FC",
                primary_readout="YFP / CFP",
                tags=("dual_reporter", "ratio", "crosstalk"),
            ),
        ),
        control_rules=(),
        windows=(),
        metrics=(
            ProtocolMetricSpec(
                id="OD",
                stage="raw",
                summary="Raw OD600 trace.",
                formula="OD600",
                profiles=(
                    "yfp_cfp_raw",
                    "yfp_cfp_fold_change",
                    "yfp_cfp_crosstalk",
                ),
            ),
            ProtocolMetricSpec(
                id="CFP",
                stage="raw",
                summary="Raw CFP trace.",
                formula="CFP",
                profiles=("yfp_cfp_raw", "yfp_cfp_fold_change", "yfp_cfp_crosstalk"),
            ),
            ProtocolMetricSpec(
                id="YFP",
                stage="raw",
                summary="Raw YFP trace.",
                formula="YFP",
                profiles=("yfp_cfp_raw", "yfp_cfp_fold_change", "yfp_cfp_crosstalk"),
            ),
            ProtocolMetricSpec(
                id="YFP_OD",
                stage="support",
                summary="Supporting YFP per biomass proxy.",
                formula="YFP / OD600",
                depends_on=("YFP", "OD"),
                value_space="linear_ratio",
                unit="ratio",
                comparable_group="support_ratio_linear",
                profiles=("yfp_cfp_raw", "yfp_cfp_fold_change", "yfp_cfp_crosstalk"),
            ),
            ProtocolMetricSpec(
                id="CFP_OD",
                stage="support",
                summary="Supporting CFP per biomass proxy.",
                formula="CFP / OD600",
                depends_on=("CFP", "OD"),
                value_space="linear_ratio",
                unit="ratio",
                comparable_group="support_ratio_linear",
                profiles=("yfp_cfp_raw", "yfp_cfp_fold_change", "yfp_cfp_crosstalk"),
            ),
            ProtocolMetricSpec(
                id="Ratio",
                stage="derived",
                summary="Primary within-well assay ratio.",
                formula="YFP / CFP",
                depends_on=("YFP", "CFP"),
                value_space="linear_ratio",
                unit="ratio",
                comparable_group="primary_ratio_linear",
                profiles=("yfp_cfp_raw", "yfp_cfp_fold_change", "yfp_cfp_crosstalk"),
            ),
            ProtocolMetricSpec(
                id="FC",
                stage="summary",
                summary="Nearest-time fold-change relative to the configured baseline treatment.",
                formula="Ratio(t*) / baseline(Ratio)",
                depends_on=("Ratio",),
                value_space="fold_change_ratio",
                unit="ratio",
                comparable_group="fold_change_linear",
                profiles=("yfp_cfp_fold_change", "yfp_cfp_crosstalk"),
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
                profiles=("yfp_cfp_fold_change", "yfp_cfp_crosstalk"),
            ),
        ),
        effect_signs=(),
        figures=(
            ProtocolFigureSpec(
                id="raw_kinetics",
                kind="qc",
                summary="Raw kinetics view over OD600 and reporter channels.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="dual_reporter_triptych",
                kind="kinetics",
                summary="Per-design growth and reporter-ratio kinetics with an explicitly timed endpoint summary.",
            ),
            ProtocolFigureSpec(
                id="endpoint_by_condition",
                kind="summary",
                summary="Endpoint comparison grouped by treatment/condition.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="endpoint_by_design",
                kind="summary",
                summary="Endpoint comparison grouped by construct/design.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="state_summary",
                kind="summary",
                summary="2x2 state summary using alias-mapped treatment states.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="intensity_overview",
                kind="kinetics",
                summary="Combined time-series and endpoint view of the intensity/support channel.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="ratio_overview",
                kind="kinetics",
                summary="Combined time-series and endpoint view of the primary ratio channel.",
            ),
            ProtocolFigureSpec(
                id="value_distributions",
                kind="qc",
                summary="Distribution view of the primary measurement channel.",
            ),
            ProtocolFigureSpec(
                id="ratio_heatmap",
                kind="summary",
                summary="Endpoint heatmap over the primary ratio channel.",
            ),
            ProtocolFigureSpec(
                id="support_heatmap",
                kind="summary",
                summary="Endpoint heatmap over the supporting CFP/OD600 channel.",
            ),
        ),
        plot_profiles=(
            ProtocolPlotProfileSpec(
                id="screen_overview",
                summary="Endpoint screen view; each selected endpoint time must be authored explicitly.",
                figures=("raw_kinetics", "endpoint_by_condition", "endpoint_by_design", "intensity_overview"),
            ),
            ProtocolPlotProfileSpec(
                id="ratio_screen",
                summary="Ratio-centric screen view for dual-reporter comparisons.",
                figures=("raw_kinetics", "state_summary", "ratio_overview"),
            ),
            ProtocolPlotProfileSpec(
                id="kinetics_qc",
                summary="Kinetics-first QC view with raw traces and distributions.",
                figures=("raw_kinetics", "value_distributions"),
            ),
            ProtocolPlotProfileSpec(
                id="heatmap_review",
                summary="Endpoint heatmap review across primary and supporting channels.",
                figures=("ratio_heatmap", "support_heatmap"),
            ),
        ),
        default_plot_profile="kinetics_qc",
        artifacts=(
            ProtocolArtifactSpec(
                id="crosstalk_pairs_table",
                summary="CSV export of crosstalk-safe pair candidates.",
            ),
        ),
        ranking=ProtocolRankingSpec(
            primary_metric="log2FC",
            direction="higher_is_better",
            supporting_metrics=("FC",),
            summary=(
                "When crosstalk selection is enabled, rank self-pairs by compiled fold-change and the configured "
                "selectivity thresholds."
            ),
            profiles=("yfp_cfp_crosstalk",),
        ),
        execution=ProtocolExecutionPlan(
            plugin_defaults=(
                ProtocolPluginDefaultsSpec(
                    plugin="ingest/synergy_h1",
                    summary="Dual-reporter screens parse one explicitly selected Synergy workbook in mixed mode.",
                    with_={
                        "mode": binding_value("ingest.mode", "mixed"),
                        "channels": binding_value("ingest.channels", ["OD600", "CFP", "YFP"]),
                        "channel_map": binding_value("ingest.channel_map", dict(_DUAL_REPORTER_CHANNEL_MAP)),
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
                    summary="Shared fold-change defaults keep plate-reader comparisons and naming consistent.",
                    with_={
                        "target": binding_value("fold_change.target", "YFP/CFP"),
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
            compiler=compile_plate_reader_dual_reporter_screen,
        ),
    ),
    ProtocolDescriptor(
        protocol="logic/sfxi_screen",
        domain="logic",
        family="logic_summary",
        summary="Dual-reporter plate-reader adapter for SFXI vec8 logic and intensity summaries.",
        tags=("logic", "sfxi", "screen", "dual_reporter", "plate_reader"),
        resources=(
            ProtocolResourceSpec(
                id="sample_map",
                path="./inputs/metadata.xlsx",
                summary="Well-to-sample metadata for the logic-screen workbook.",
            ),
        ),
        example_annotations={
            "ordered_state_spaces": {
                "induction_logic": {
                    "column": "treatment",
                    "state_order": ["00", "10", "01", "11"],
                    "values": {
                        "00": "-input-a/-input-b",
                        "10": "+input-a/-input-b",
                        "01": "-input-a/+input-b",
                        "11": "+input-a/+input-b",
                    },
                    "case_sensitive": True,
                }
            }
        },
        input_fields=(
            _field(
                "ingest",
                "Plate-reader ingest settings for the logic screen.",
                children=(
                    _field(
                        "mode",
                        "Ingest mode for Synergy H1 parsing.",
                        kind="string",
                        choices=("snapshot_only", "kinetic_only", "mixed"),
                    ),
                    _field("channels", "Ordered channel names to keep from the workbook.", kind="string_list"),
                    _field(
                        "channel_map",
                        "Raw workbook label to canonical channel mapping; required for snapshot or mixed parsing.",
                        kind="mapping",
                        allow_unknown=True,
                        allow_none=True,
                        default=dict(_DUAL_REPORTER_CHANNEL_MAP),
                    ),
                    _field(
                        "sheet_names", "Optional workbook sheet names to parse.", kind="string_list", allow_none=True
                    ),
                    _field(
                        "time_round_decimals",
                        "Rounding precision for parsed time values.",
                        kind="integer",
                        allow_none=True,
                        default=12,
                    ),
                    _field("time_step_h", "Override time-step spacing in hours.", kind="number", allow_none=True),
                    _field(
                        "auto_roots",
                        "Directories to scan for workbook auto-discovery.",
                        kind="string_list",
                        allow_none=True,
                    ),
                    _field(
                        "auto_include",
                        "Filename globs to include during auto-discovery.",
                        kind="string_list",
                        default=list(DEFAULT_INCLUDE),
                    ),
                    _field(
                        "auto_exclude",
                        "Filename globs to exclude during auto-discovery.",
                        kind="string_list",
                        default=list(DEFAULT_EXCLUDE),
                    ),
                    _field(
                        "auto_pick",
                        "Multi-file selection policy for auto-discovery.",
                        kind="string",
                        choices=("single", "latest"),
                        default="single",
                    ),
                    _field(
                        "auto_recursive",
                        "Recurse into child directories when discovering workbooks.",
                        kind="bool",
                        default=False,
                    ),
                    _field("print_summary", "Print an ingest summary to the log.", kind="bool", default=True),
                ),
            ),
            _field(
                "fold_change",
                "Optional fold-change summary inputs used before vec8 export.",
                children=(
                    _field("target", "Primary fold-change channel.", kind="string", default="YFP/CFP"),
                    _field(
                        "report_times",
                        "Explicit report times in hours for fold-change snapshots.",
                        kind="number_list",
                    ),
                    _field("time_tolerance", "Nearest-time tolerance in hours.", kind="number", default=0.51),
                    _field(
                        "observation_stat",
                        "Observation aggregation statistic.",
                        kind="string",
                        choices=("median", "mean"),
                        default="median",
                    ),
                    _field("treatment_column", "Treatment-state column name.", kind="string", default="treatment"),
                    _field(
                        "group_by",
                        "Grouping columns for comparison baselines.",
                        kind="string_list",
                        default=["design_id"],
                    ),
                    _field(
                        "use_global_baseline",
                        "Use one shared baseline instead of per-group baselines.",
                        kind="bool",
                        default=True,
                    ),
                    _field(
                        "global_baseline_value",
                        "Explicit global baseline label when global mode is enabled.",
                        kind="string",
                        allow_none=True,
                    ),
                    _field(
                        "overrides",
                        "Explicit baseline overrides keyed by group columns.",
                        kind="mapping_list",
                        default=[],
                    ),
                    _field("fc_column", "Output fold-change column name.", kind="string", default="FC"),
                    _field("log2fc_column", "Output log2 fold-change column name.", kind="string", default="log2FC"),
                ),
            ),
            _field(
                "reference",
                "Reference design and aggregation policy for vec8 normalization.",
                children=(
                    _field("design_id", "Reference design id.", kind="string", default="REF"),
                    _field("observation_stat", "Reference observation statistic.", kind="string", default="mean"),
                ),
            ),
            _field("design_by", "Grouping columns for logic designs.", kind="string_list", default=["design_id"]),
            _field("time_column", "Column containing assay time in hours.", kind="string", default="time"),
            _field(
                "time_mode",
                "Time-selection mode for vec8 extraction.",
                kind="string",
                choices=("nearest", "last_before", "first_after", "exact"),
                default="nearest",
            ),
            _field("target_time_h", "Target timepoint for vec8 extraction.", kind="number", allow_none=True),
            _field("time_tolerance_h", "Nearest-time tolerance in hours.", kind="number", default=0.5),
            _field(
                "state_map_ref",
                "Reference to an annotations.ordered_state_spaces entry.",
                kind="string",
                default="induction_logic",
            ),
            _field(
                "promote",
                "Promotion settings for tidy_plus_map conversion.",
                children=(
                    _field("synthesize_batch", "Add a synthetic batch column when missing.", kind="bool", default=True),
                    _field(
                        "drop_where_null_in",
                        "Drop rows with NULL in these columns before promotion.",
                        kind="string_list",
                        default=["treatment", "design_id"],
                    ),
                ),
            ),
            _field(
                "require_all_corners_per_design",
                "Require each design to expose all logic corners before vec8 export.",
                kind="bool",
                default=True,
            ),
            _field(
                "exclude_reference_from_output",
                "Drop the reference design from the final vec8 output.",
                kind="bool",
                default=True,
            ),
            _field(
                "carry_metadata",
                "Metadata columns to carry through vec8 output.",
                kind="string_list",
                default=["sequence", "id"],
            ),
        ),
        analysis_fields=(
            _field("include_fold_change", "Build the fold-change comparison table.", kind="bool", default=False),
            _field("include_vec8", "Build the vec8 summary table.", kind="bool", default=True),
            _field("include_export", "Emit the workbook export when vec8 is present.", kind="bool", default=True),
            _field(
                "logic_symmetry",
                "Logic-symmetry summary settings used when that deliverable is selected.",
                children=(
                    _field("batch_col", "Observation-batch column.", kind="string", default="batch"),
                    _field(
                        "treatment_column",
                        "Optional treatment column override; otherwise the ordered state space owns it.",
                        kind="string",
                        allow_none=True,
                    ),
                    _field(
                        "observation_stat",
                        "Observation aggregation statistic for each state corner.",
                        kind="string",
                        choices=("mean", "median"),
                        default="mean",
                    ),
                    _field(
                        "prep",
                        "Optional time-selection policy before corner aggregation.",
                        children=(
                            _field("enable", "Apply explicit time selection.", kind="bool", default=False),
                            _field(
                                "mode",
                                "Time-selection mode.",
                                kind="string",
                                choices=("first", "last", "median", "exact", "nearest"),
                                default="last",
                            ),
                            _field(
                                "target_time",
                                "Target time for exact or nearest selection.",
                                kind="number",
                                allow_none=True,
                            ),
                            _field("tolerance", "Allowed time distance.", kind="number", default=0.51),
                            _field(
                                "align_corners",
                                "Use one shared time anchor across all four states.",
                                kind="bool",
                                default=False,
                            ),
                            _field(
                                "case_sensitive_treatments",
                                "Optional case-sensitivity override for source state labels.",
                                kind="bool",
                                allow_none=True,
                            ),
                            _field("time_column", "Column containing time values.", kind="string", default="time"),
                        ),
                    ),
                ),
            ),
            _field(
                "sfxi_vec8",
                "SFXI vec8 transform settings.",
                children=(
                    _field(
                        "intensity_log2_offset_delta",
                        "Non-negative log2 offset applied to the intensity channel.",
                        kind="number",
                        default=0.0,
                    ),
                ),
            ),
            _field(
                "preprocessing",
                "Pre-ingest cleanup policy for blanks and overflow.",
                children=(
                    _field(
                        "blank",
                        "Blank-correction policy.",
                        children=(
                            _field(
                                "method",
                                "Blank handling strategy.",
                                kind="string",
                                choices=("disregard", "subtract"),
                                default="disregard",
                            ),
                            _field(
                                "capture_blanks",
                                "Emit a blanks side table for downstream QC.",
                                kind="bool",
                                default=True,
                            ),
                        ),
                    ),
                    _field(
                        "overflow",
                        "Overflow handling policy for saturated channels.",
                        children=(
                            _field(
                                "action",
                                "Overflow action.",
                                kind="string",
                                choices=("max", "drop", "nan", "none"),
                                default="max",
                            ),
                            _field("clip_quantile", "Quantile cap used when action=max.", kind="number", default=0.999),
                            _field(
                                "cap_strategy",
                                "How per-channel caps are determined when action=max.",
                                kind="string",
                                choices=("provided", "infer", "quantile"),
                                default="quantile",
                            ),
                            _field(
                                "per_channel_caps",
                                "Explicit per-channel caps when cap_strategy=provided.",
                                kind="mapping",
                                allow_unknown=True,
                                allow_none=True,
                            ),
                            _field(
                                "flag_column",
                                "Column used to mark overflowed wells before capping.",
                                kind="string",
                                default="overflow",
                            ),
                            _field(
                                "treat_inf_as_overflow",
                                "Treat infinite values as overflowed rows.",
                                kind="bool",
                                default=True,
                            ),
                        ),
                    ),
                ),
            ),
        ),
        factors=(
            ProtocolFactorSpec(name="design_id", role="design", summary="Design grouping for logic comparison."),
            ProtocolFactorSpec(name="time", role="time", summary="Selected summary timepoint."),
            ProtocolFactorSpec(
                name="ordered_state_space",
                role="mapping",
                summary="Ordered 00/10/01/11 treatment-to-corner state space.",
            ),
            ProtocolFactorSpec(
                name="reference_design", role="control", summary="Reference design for intensity normalization."
            ),
        ),
        control_rules=(
            ProtocolControlRule(
                id="logic_corner_map",
                summary="Resolve treatment states to 00/10/01/11 corners through the configured ordered state space.",
                control_selector="state_map_ref",
            ),
        ),
        windows=(
            ProtocolWindowSpec(
                id="summary_timepoint",
                summary="Select a single summary timepoint by nearest/exact/neighbor rules.",
                anchor="analysis_time",
                selector="time_mode",
                params={"default_mode": "nearest"},
            ),
        ),
        metrics=(
            ProtocolMetricSpec(
                id="vec8",
                stage="summary",
                summary="Eight-value SFXI summary vector over logic and intensity channels.",
                formula="v00,v10,v01,v11,y00_star,y10_star,y01_star,y11_star",
            ),
        ),
        figures=(
            ProtocolFigureSpec(
                id="raw_kinetics",
                kind="qc",
                summary="Raw kinetics view over OD600 and reporter channels.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="endpoint_by_condition",
                kind="summary",
                summary="Endpoint comparison grouped by treatment/condition.",
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
                summary="Combined time-series and endpoint view of the intensity channel.",
                primary=True,
            ),
            ProtocolFigureSpec(
                id="logic_symmetry",
                kind="summary",
                summary="Logic symmetry geometry over the configured response channel.",
            ),
            ProtocolFigureSpec(
                id="sfxi_diagnostic",
                kind="summary",
                summary=(
                    "Per-design growth and response trajectories beside the persisted vec8 components "
                    "at the persisted selection time."
                ),
            ),
            ProtocolFigureSpec(
                id="sfxi_vec8_heatmap",
                kind="summary",
                summary="Heatmap over per-design SFXI vec8 logic shape and reference-normalized intensity.",
            ),
        ),
        plot_profiles=(
            ProtocolPlotProfileSpec(
                id="kinetics_qc",
                summary="Time-series QC without selecting a scientific endpoint.",
                figures=("raw_kinetics",),
            ),
            ProtocolPlotProfileSpec(
                id="logic_overview",
                summary="SFXI kinetics and endpoint summaries with explicitly authored plot times.",
                figures=(
                    "raw_kinetics",
                    "endpoint_by_condition",
                    "endpoint_by_design",
                    "intensity_overview",
                ),
            ),
            ProtocolPlotProfileSpec(
                id="logic_geometry",
                summary="Geometry-only logic symmetry review.",
                figures=("logic_symmetry",),
            ),
            ProtocolPlotProfileSpec(
                id="logic_diagnostic",
                summary="Record-driven per-design trajectory and vec8 review.",
                figures=("sfxi_diagnostic",),
            ),
            ProtocolPlotProfileSpec(
                id="logic_full",
                summary="Full logic review with kinetics and symmetry geometry.",
                figures=(
                    "raw_kinetics",
                    "endpoint_by_condition",
                    "endpoint_by_design",
                    "intensity_overview",
                    "logic_symmetry",
                    "sfxi_diagnostic",
                    "sfxi_vec8_heatmap",
                ),
            ),
        ),
        default_plot_profile="kinetics_qc",
        artifacts=(
            ProtocolArtifactSpec(
                id="logic_summary_workbook",
                summary="Workbook export of the SFXI vec8 summary.",
            ),
        ),
        execution=ProtocolExecutionPlan(
            plugin_defaults=(
                ProtocolPluginDefaultsSpec(
                    plugin="ingest/synergy_h1",
                    summary="Logic screens default to CFP/YFP/OD600 Synergy H1 ingest with protocol-bound channel policy.",
                    with_={
                        "mode": binding_value("ingest.mode", "mixed"),
                        "channels": binding_value("ingest.channels", ["OD600", "CFP", "YFP"]),
                        "channel_map": binding_value("ingest.channel_map", dict(_DUAL_REPORTER_CHANNEL_MAP)),
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
                    plugin="validator/to_tidy_plus_map",
                    summary="Promote plate-reader tidy data into the annotated SFXI-compatible table shape.",
                    with_={
                        "synthesize_batch": binding_value("promote.synthesize_batch", True),
                        "drop_where_null_in": binding_value(
                            "promote.drop_where_null_in",
                            ["treatment", "design_id"],
                        ),
                    },
                ),
                ProtocolPluginDefaultsSpec(
                    plugin="transform/fold_change",
                    summary="Shared logic-screen fold-change defaults keep target and baseline policy in one place.",
                    with_={
                        "target": binding_value("fold_change.target", "YFP/CFP"),
                        "report_times": binding_value("fold_change.report_times"),
                        "time_tolerance": binding_value("fold_change.time_tolerance", 0.51),
                        "observation_stat": binding_value("fold_change.observation_stat", "median"),
                        "treatment_column": binding_value("fold_change.treatment_column", "treatment"),
                        "group_by": binding_value("fold_change.group_by", ["design_id"]),
                        "use_global_baseline": binding_value("fold_change.use_global_baseline", True),
                        "global_baseline_value": binding_value("fold_change.global_baseline_value", None),
                        "overrides": binding_value("fold_change.overrides", []),
                        "fc_column": binding_value("fold_change.fc_column", "FC"),
                        "log2fc_column": binding_value("fold_change.log2fc_column", "log2FC"),
                    },
                ),
                ProtocolPluginDefaultsSpec(
                    plugin="transform/sfxi",
                    summary="The dual-reporter adapter binds its concrete channels to the generic SFXI transform.",
                    with_={
                        "response": {
                            "logic_channel": "YFP/CFP",
                            "intensity_channel": "YFP/OD600",
                        },
                        "design_by": binding_value("design_by", ["design_id"]),
                        "time_column": binding_value("time_column", "time"),
                        "time_mode": binding_value("time_mode", "nearest"),
                        "target_time_h": binding_value("target_time_h", None),
                        "time_tolerance_h": binding_value("time_tolerance_h", 0.5),
                        "state_map_ref": binding_value("state_map_ref", "induction_logic"),
                        "reference": {
                            "design_id": binding_value("reference.design_id", "REF"),
                            "observation_stat": binding_value("reference.observation_stat", "mean"),
                        },
                        "require_all_corners_per_design": binding_value("require_all_corners_per_design", True),
                        "exclude_reference_from_output": binding_value("exclude_reference_from_output", True),
                        "carry_metadata": binding_value("carry_metadata", ["sequence", "id"]),
                    },
                ),
            ),
            compiler=compile_logic_sfxi_screen,
        ),
    ),
    ProtocolDescriptor(
        protocol="cytometry/flow_panel",
        domain="cytometry",
        family="panel_analysis",
        summary="Flow-cytometry panel protocol for gated event tables and channel-level summaries.",
        tags=("cytometry", "fcs", "panel"),
        resources=(
            ProtocolResourceSpec(
                id="metadata",
                path="./inputs/metadata.csv",
                summary="Sample metadata joined to imported flow-cytometry events.",
            ),
        ),
        input_fields=(
            _field(
                "ingest",
                "Flow-cytometer ingest and channel naming settings.",
                children=(
                    _field("auto_roots", "Directories to scan for .fcs files.", kind="string_list", allow_none=True),
                    _field(
                        "auto_include", "Filename globs to include.", kind="string_list", default=["*.fcs", "*.FCS"]
                    ),
                    _field(
                        "auto_exclude", "Filename globs to exclude.", kind="string_list", default=list(DEFAULT_EXCLUDE)
                    ),
                    _field(
                        "auto_pick",
                        "Multi-file selection policy.",
                        kind="string",
                        choices=("single", "latest", "merge"),
                        default="merge",
                    ),
                    _field(
                        "auto_recursive",
                        "Recurse into child directories when discovering files.",
                        kind="bool",
                        default=False,
                    ),
                    _field(
                        "channel_name_field", "FCS metadata field used as channel label.", kind="string", default="pns"
                    ),
                    _field(
                        "channel_map",
                        "Optional channel rename mapping.",
                        kind="mapping",
                        allow_unknown=True,
                        allow_none=True,
                    ),
                    _field("drop_channels", "Channels to drop after ingest.", kind="string_list", allow_none=True),
                    _field(
                        "sample_id_from",
                        "How sample ids are derived from filenames.",
                        kind="string",
                        choices=("stem", "name"),
                        default="stem",
                    ),
                    _field("time_value", "Time value applied to snapshot cytometry rows.", kind="number", default=0.0),
                    _field("print_summary", "Print an ingest summary to the log.", kind="bool", default=True),
                ),
            ),
            _field(
                "metadata",
                "Metadata merge requirements for the cytometry panel.",
                children=(
                    _field("key", "Join key between metadata and sample rows.", kind="string", default="sample_id"),
                    _field(
                        "require_columns",
                        "Metadata columns that must exist after merge.",
                        kind="string_list",
                        default=[],
                    ),
                    _field(
                        "require_non_null",
                        "Require merged metadata columns to be non-null.",
                        kind="bool",
                        default=False,
                    ),
                ),
            ),
            _field(
                "gating",
                "Explicit cytometry gating, fluorescence threshold, grouping, and QC policy.",
                required=True,
                children=(
                    _field(
                        "cells_enabled",
                        "Whether to apply the rectangular cells gate.",
                        kind="bool",
                        required=True,
                        example=False,
                    ),
                    _field(
                        "cells_x_channel",
                        "X channel for the cells gate.",
                        kind="string",
                        required=True,
                        example="<cells-x-channel>",
                    ),
                    _field(
                        "cells_y_channel",
                        "Y channel for the cells gate.",
                        kind="string",
                        required=True,
                        example="<cells-y-channel>",
                    ),
                    _field(
                        "cells_x_range",
                        "Closed two-value range for the cells-gate X channel.",
                        kind="number_list",
                        required=True,
                        example=[0.0, 1.0],
                    ),
                    _field(
                        "cells_y_range",
                        "Closed two-value range for the cells-gate Y channel.",
                        kind="number_list",
                        required=True,
                        example=[0.0, 1.0],
                    ),
                    _field(
                        "singlets_enabled",
                        "Whether to apply the singlet-ratio gate after the cells gate.",
                        kind="bool",
                        required=True,
                        example=False,
                    ),
                    _field(
                        "singlet_x_channel",
                        "Denominator channel for the singlet ratio (Y / X).",
                        kind="string",
                        required=True,
                        example="<singlet-denominator-channel>",
                    ),
                    _field(
                        "singlet_y_channel",
                        "Numerator channel for the singlet ratio (Y / X).",
                        kind="string",
                        required=True,
                        example="<singlet-numerator-channel>",
                    ),
                    _field(
                        "singlet_ratio_range",
                        "Closed two-value range for the singlet ratio.",
                        kind="number_list",
                        required=True,
                        example=[0.0, 1.0],
                    ),
                    _field(
                        "fluorescence_channel",
                        "Fluorescence channel summarized after gating.",
                        kind="string",
                        required=True,
                        example="<fluorescence-channel>",
                    ),
                    _field(
                        "threshold_mode",
                        "Positive-event threshold policy.",
                        kind="string",
                        choices=("manual", "from_control_quantile"),
                        required=True,
                        example="manual",
                    ),
                    _field(
                        "threshold_value",
                        "Manual fluorescence threshold; null for control-quantile policy.",
                        kind="number",
                        required=True,
                        allow_none=True,
                        example=0.0,
                    ),
                    _field(
                        "threshold_group_column",
                        "Metadata column containing the threshold control; null for manual policy.",
                        kind="string",
                        required=True,
                        allow_none=True,
                        example=None,
                    ),
                    _field(
                        "threshold_control_value",
                        "Exact control value used for threshold estimation; null for manual policy.",
                        kind="string",
                        required=True,
                        allow_none=True,
                        example=None,
                    ),
                    _field(
                        "threshold_quantile",
                        "Control quantile used as the fluorescence threshold; null for manual policy.",
                        kind="number",
                        required=True,
                        allow_none=True,
                        example=None,
                    ),
                    _field(
                        "group_column",
                        "Metadata column for group summaries; explicit null disables group summaries.",
                        kind="string",
                        required=True,
                        allow_none=True,
                        example=None,
                    ),
                    _field(
                        "minimum_final_events",
                        "Minimum retained singlet events required per sample.",
                        kind="integer",
                        required=True,
                        example=0,
                    ),
                    _field(
                        "minimum_final_percent",
                        "Minimum percentage of input events retained per sample.",
                        kind="number",
                        required=True,
                        example=0.0,
                    ),
                    _field(
                        "maximum_nonpositive_percent",
                        "Maximum allowed percentage of nonpositive fluorescence values per sample.",
                        kind="number",
                        required=True,
                        example=100.0,
                    ),
                    _field(
                        "nonpositive_scope",
                        "Event population used for nonpositive-fluorescence QC.",
                        kind="string",
                        choices=("all_events", "gated_events"),
                        required=True,
                        example="all_events",
                    ),
                ),
            ),
        ),
        analysis_fields=(),
        factors=(
            ProtocolFactorSpec(name="sample", role="sample", summary="Sample/run identifier."),
            ProtocolFactorSpec(name="condition", role="condition", summary="Experimental condition."),
            ProtocolFactorSpec(name="gate", role="gate", summary="Gate or subset definition.", required=False),
        ),
        ranking=ProtocolRankingSpec(
            primary_metric="domain_defined",
            direction="higher_is_better",
            summary="Use domain-defined cytometry comparisons and gating summaries for ranking.",
        ),
        figures=(
            ProtocolFigureSpec(
                id="gating_diagnostic",
                kind="diagnostic",
                summary="Configured cells, singlets, fluorescence, and final-retention diagnostics.",
                primary=True,
            ),
        ),
        plot_profiles=(
            ProtocolPlotProfileSpec(
                id="gating_review",
                summary="Primary record-driven cytometry gating and fluorescence review.",
                figures=("gating_diagnostic",),
            ),
        ),
        default_plot_profile="gating_review",
        artifacts=(
            ProtocolArtifactSpec(
                id="gate_definition_table",
                summary="CSV projection of the resolved gate and threshold policy.",
                default=True,
            ),
            ProtocolArtifactSpec(
                id="sample_stats_table",
                summary="CSV projection of per-sample cytometry statistics.",
                default=True,
            ),
            ProtocolArtifactSpec(
                id="group_stats_table",
                summary="CSV projection of configured group-level cytometry statistics.",
                default=True,
            ),
            ProtocolArtifactSpec(
                id="qc_table",
                summary="CSV projection of per-sample cytometry QC decisions.",
                default=True,
            ),
            ProtocolArtifactSpec(
                id="gated_events_table",
                summary="Optional CSV projection of retained cytometry events.",
            ),
        ),
        execution=ProtocolExecutionPlan(
            plugin_defaults=(
                ProtocolPluginDefaultsSpec(
                    plugin="ingest/flow_cytometer",
                    summary="Flow-panel ingest defaults come from protocol parameters instead of per-experiment plugin blobs.",
                    with_={
                        "auto_roots": binding_value("ingest.auto_roots", None),
                        "auto_include": binding_value("ingest.auto_include", ["*.fcs", "*.FCS"]),
                        "auto_exclude": binding_value("ingest.auto_exclude", list(DEFAULT_EXCLUDE)),
                        "auto_pick": binding_value("ingest.auto_pick", "merge"),
                        "auto_recursive": binding_value("ingest.auto_recursive", False),
                        "channel_name_field": binding_value("ingest.channel_name_field", "pns"),
                        "channel_map": binding_value("ingest.channel_map", None),
                        "drop_channels": binding_value("ingest.drop_channels", None),
                        "sample_id_from": binding_value("ingest.sample_id_from", "stem"),
                        "time_value": binding_value("ingest.time_value", 0.0),
                        "print_summary": binding_value("ingest.print_summary", True),
                    },
                ),
                ProtocolPluginDefaultsSpec(
                    plugin="transform/sample_metadata",
                    summary="Metadata merge requirements are protocol parameters, not raw plugin overrides.",
                    with_={
                        "key": binding_value("metadata.key", "sample_id"),
                        "require_columns": binding_value("metadata.require_columns", []),
                        "require_non_null": binding_value("metadata.require_non_null", False),
                    },
                ),
                ProtocolPluginDefaultsSpec(
                    plugin="transform/cytometry_gating",
                    summary="Cytometry gates and QC are explicit experiment parameters compiled into typed records.",
                    with_={
                        "cells_enabled": binding_value("gating.cells_enabled"),
                        "cells_x_channel": binding_value("gating.cells_x_channel"),
                        "cells_y_channel": binding_value("gating.cells_y_channel"),
                        "cells_x_range": binding_value("gating.cells_x_range"),
                        "cells_y_range": binding_value("gating.cells_y_range"),
                        "singlets_enabled": binding_value("gating.singlets_enabled"),
                        "singlet_x_channel": binding_value("gating.singlet_x_channel"),
                        "singlet_y_channel": binding_value("gating.singlet_y_channel"),
                        "singlet_ratio_range": binding_value("gating.singlet_ratio_range"),
                        "fluorescence_channel": binding_value("gating.fluorescence_channel"),
                        "threshold_mode": binding_value("gating.threshold_mode"),
                        "threshold_value": binding_value("gating.threshold_value"),
                        "threshold_group_column": binding_value("gating.threshold_group_column"),
                        "threshold_control_value": binding_value("gating.threshold_control_value"),
                        "threshold_quantile": binding_value("gating.threshold_quantile"),
                        "group_column": binding_value("gating.group_column"),
                        "minimum_final_events": binding_value("gating.minimum_final_events"),
                        "minimum_final_percent": binding_value("gating.minimum_final_percent"),
                        "maximum_nonpositive_percent": binding_value("gating.maximum_nonpositive_percent"),
                        "nonpositive_scope": binding_value("gating.nonpositive_scope"),
                    },
                ),
            ),
            compiler=compile_cytometry_flow_panel,
        ),
    ),
)

_DUAL_REPORTER_PROTOCOL = next(
    item for item in BUILTIN_PROTOCOLS if item.protocol == "plate_reader/dual_reporter_screen"
)
_PLATE_READER_SINGLE_REPORTER_PROTOCOL = build_plate_reader_variant_protocol(
    dual_reporter_protocol=_DUAL_REPORTER_PROTOCOL,
    field_builder=_field,
)

BUILTIN_PROTOCOLS = (
    next(item for item in BUILTIN_PROTOCOLS if item.protocol == "workbench/generic"),
    _DUAL_REPORTER_PROTOCOL,
    _PLATE_READER_SINGLE_REPORTER_PROTOCOL,
    next(item for item in BUILTIN_PROTOCOLS if item.protocol == "plate_reader/response_window"),
    next(item for item in BUILTIN_PROTOCOLS if item.protocol == "logic/sfxi_screen"),
    next(item for item in BUILTIN_PROTOCOLS if item.protocol == "logic/sfxi_vec8_collection"),
    next(item for item in BUILTIN_PROTOCOLS if item.protocol == "cytometry/flow_panel"),
)


@cache
def builtin_protocol_catalog() -> ProtocolCatalog:
    return ProtocolCatalog(list(BUILTIN_PROTOCOLS))
