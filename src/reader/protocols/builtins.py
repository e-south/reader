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
    compile_plate_reader_dual_reporter_screen,
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
    ProtocolNotebookPolicy,
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
) -> ProtocolConfigFieldSpec:
    kwargs: dict[str, object] = {}
    if default is not _MISSING:
        kwargs["default"] = default
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
            ProtocolFactorSpec(name="replicate", role="replicate", summary="Replicate grouping axis.", required=False),
            ProtocolFactorSpec(name="time", role="time", summary="Observation axis when present.", required=False),
        ),
        ranking=ProtocolRankingSpec(
            primary_metric="domain_defined",
            direction="higher_is_better",
            summary="Generic protocol leaves ranking to the domain-specific analysis layer.",
        ),
        execution=ProtocolExecutionPlan(
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/basic",
                allowed_templates=(
                    "notebook/basic",
                    "notebook/eda",
                    "notebook/dual_reporter_triptych",
                    "notebook/cytometry",
                    "notebook/sfxi_eda",
                ),
                summary="Generic workbench protocol defaults to the minimal record explorer and allows explicit overrides.",
            ),
            compiler=compile_generic_protocol,
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
                        "Report times in hours for fold-change snapshots.",
                        kind="number_list",
                        default=[14.0],
                    ),
                    _field("time_tolerance", "Nearest-time tolerance in hours.", kind="number", default=0.51),
                    _field("agg", "Replicate aggregator.", kind="string", choices=("median", "mean"), default="median"),
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
            _field("include_fold_change", "Build the fold-change comparison table.", kind="bool", default=True),
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
            ProtocolFactorSpec(name="replicate_id", role="replicate", summary="Replicate well identifier."),
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
                summary="Balanced default set for screen-style plate-reader experiments.",
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
        default_plot_profile="screen_overview",
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
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/eda",
                allowed_templates=(
                    "notebook/eda",
                    "notebook/dual_reporter_triptych",
                    "notebook/basic",
                ),
                summary="Dual-reporter plate-reader screens default to the EDA notebook with plot support.",
            ),
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
                        "agg": binding_value("fold_change.agg", "median"),
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
        summary="SFXI logic-screen protocol for mapping inducible corners to vec8 logic/intensity summaries.",
        tags=("logic", "sfxi", "screen"),
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
                        "00": "-inducer/-stress",
                        "10": "+inducer/-stress",
                        "01": "-inducer/+stress",
                        "11": "+inducer/+stress",
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
                        "Report times in hours for fold-change snapshots.",
                        kind="number_list",
                        default=[14.0],
                    ),
                    _field("time_tolerance", "Nearest-time tolerance in hours.", kind="number", default=0.51),
                    _field("agg", "Replicate aggregator.", kind="string", choices=("median", "mean"), default="median"),
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
                "response",
                "Response/intensity channel binding for vec8 summaries.",
                children=(
                    _field("logic_channel", "Channel used for logic fidelity.", kind="string", default="YFP/CFP"),
                    _field(
                        "intensity_channel", "Channel used for intensity scaling.", kind="string", default="YFP/OD600"
                    ),
                ),
            ),
            _field(
                "reference",
                "Reference design and aggregation policy for vec8 normalization.",
                children=(
                    _field("design_id", "Reference design id.", kind="string", default="REF"),
                    _field("stat", "Reference aggregation statistic.", kind="string", default="mean"),
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
            _field("include_fold_change", "Build the fold-change comparison table.", kind="bool", default=True),
            _field("include_vec8", "Build the vec8 summary table.", kind="bool", default=True),
            _field("include_export", "Emit the workbook export when vec8 is present.", kind="bool", default=True),
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
                id="sfxi_vec8_heatmap",
                kind="summary",
                summary="Heatmap over per-design SFXI vec8 logic shape and reference-normalized intensity.",
            ),
        ),
        plot_profiles=(
            ProtocolPlotProfileSpec(
                id="logic_overview",
                summary="Default SFXI overview with kinetics and endpoint summaries.",
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
                id="logic_full",
                summary="Full logic review with kinetics and symmetry geometry.",
                figures=(
                    "raw_kinetics",
                    "endpoint_by_condition",
                    "endpoint_by_design",
                    "intensity_overview",
                    "logic_symmetry",
                    "sfxi_vec8_heatmap",
                ),
            ),
        ),
        default_plot_profile="logic_overview",
        artifacts=(
            ProtocolArtifactSpec(
                id="logic_summary_workbook",
                summary="Workbook export of the SFXI vec8 summary.",
            ),
        ),
        execution=ProtocolExecutionPlan(
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/sfxi_eda",
                allowed_templates=(
                    "notebook/sfxi_eda",
                    "notebook/dual_reporter_triptych",
                    "notebook/eda",
                    "notebook/basic",
                ),
                summary="SFXI logic screens default to the vec8-aware notebook scaffold.",
            ),
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
                        "agg": binding_value("fold_change.agg", "median"),
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
                    summary="Default SFXI vec8 build settings derive from the protocol instead of every step config.",
                    with_={
                        "response": {
                            "logic_channel": binding_value("response.logic_channel", "YFP/CFP"),
                            "intensity_channel": binding_value("response.intensity_channel", "YFP/OD600"),
                        },
                        "design_by": binding_value("design_by", ["design_id"]),
                        "time_column": binding_value("time_column", "time"),
                        "time_mode": binding_value("time_mode", "nearest"),
                        "target_time_h": binding_value("target_time_h", None),
                        "time_tolerance_h": binding_value("time_tolerance_h", 0.5),
                        "state_map_ref": binding_value("state_map_ref", "induction_logic"),
                        "reference": {
                            "design_id": binding_value("reference.design_id", "REF"),
                            "stat": binding_value("reference.stat", "mean"),
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
        execution=ProtocolExecutionPlan(
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/cytometry",
                allowed_templates=("notebook/cytometry", "notebook/basic"),
                summary="Flow-cytometry panels default to the cytometry EDA notebook scaffold.",
            ),
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
    BUILTIN_PROTOCOLS[0],
    _DUAL_REPORTER_PROTOCOL,
    _PLATE_READER_SINGLE_REPORTER_PROTOCOL,
    *BUILTIN_PROTOCOLS[2:],
)


@cache
def builtin_protocol_catalog() -> ProtocolCatalog:
    return ProtocolCatalog(list(BUILTIN_PROTOCOLS))
