from __future__ import annotations

from functools import cache

from reader.domains.plate_reader.analysis._retron_sponge_contract import DEFAULT_PRIMARY_POST_STRESS_HOURS
from reader.plugins.ingest.discovery_policy import DEFAULT_EXCLUDE, DEFAULT_INCLUDE

from .compiler import (
    compile_cytometry_flow_panel,
    compile_generic_protocol,
    compile_logic_sfxi_screen,
    compile_plate_reader_dual_reporter_screen,
    compile_plate_reader_retron_sponge_screen,
    compile_plate_reader_single_reporter_screen,
)
from .model import (
    ProtocolArtifactSpec,
    ProtocolCatalog,
    ProtocolConfigFieldSpec,
    ProtocolControlRule,
    ProtocolDescriptor,
    ProtocolEffectSignSpec,
    ProtocolExecutionPlan,
    ProtocolFactorSpec,
    ProtocolFigureSpec,
    ProtocolMetricSpec,
    ProtocolNotebookPolicy,
    ProtocolPlotProfileSpec,
    ProtocolPluginDefaultsSpec,
    ProtocolRankingSpec,
    ProtocolSemanticProfileOverride,
    ProtocolSemanticProfileSpec,
    ProtocolWindowSpec,
    binding_value,
)

_MISSING = object()


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
        figures=(
            ProtocolFigureSpec(
                id="generic_qc",
                kind="qc",
                summary="Use domain-specific QC views before interpreting downstream summaries.",
                primary=True,
            ),
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
                    "notebook/retron_sponge_aggregate",
                    "notebook/eda",
                    "notebook/microplate",
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
        input_fields=(
            _field(
                "ingest",
                "Plate-reader ingest settings and workbook selection.",
                children=(
                    _field(
                        "mode",
                        "Ingest mode for Synergy H1 parsing.",
                        kind="string",
                        choices=("auto", "snapshot_only", "kinetic_only", "mixed"),
                        default="auto",
                    ),
                    _field(
                        "channel_map",
                        "Optional channel rename mapping.",
                        kind="mapping",
                        allow_unknown=True,
                        allow_none=True,
                    ),
                    _field(
                        "sheet_names", "Optional workbook sheet names to parse.", kind="string_list", allow_none=True
                    ),
                    _field("add_sheet", "Attach source sheet_name to each row.", kind="bool", default=False),
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
                        choices=("single", "latest", "merge"),
                        default="single",
                    ),
                    _field(
                        "auto_recursive",
                        "Recurse into child directories when discovering workbooks.",
                        kind="bool",
                        default=False,
                    ),
                    _field(
                        "add_source_column",
                        "Attach a source file column when merging multiple workbooks.",
                        kind="bool",
                        default=False,
                    ),
                    _field("source_col", "Name of the source file column.", kind="string", default="source_file"),
                    _field("print_summary", "Print an ingest summary to the log.", kind="bool", default=True),
                ),
            ),
            _field(
                "fold_change",
                "Fold-change summary inputs for screen-style comparisons.",
                children=(
                    _field("report_times", "Report times in hours for fold-change snapshots.", kind="number_list"),
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
            _field("strict", "Treat runtime contract mismatches as hard errors.", kind="bool", default=True),
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
                allowed_templates=("notebook/eda", "notebook/microplate", "notebook/basic"),
                summary="Dual-reporter plate-reader screens default to the EDA notebook with plot support.",
            ),
            plugin_defaults=(
                ProtocolPluginDefaultsSpec(
                    plugin="ingest/synergy_h1",
                    summary="Dual-reporter screens default to CFP/YFP/OD600 ingest in auto mode.",
                    with_={
                        "mode": binding_value("ingest.mode", "auto"),
                        "channels": binding_value("ingest.channels", ["OD600", "CFP", "YFP"]),
                        "channel_map": binding_value("ingest.channel_map", None),
                        "sheet_names": binding_value("ingest.sheet_names", None),
                        "add_sheet": binding_value("ingest.add_sheet", False),
                        "time_round_decimals": binding_value("ingest.time_round_decimals", 12),
                        "time_step_h": binding_value("ingest.time_step_h", None),
                        "auto_roots": binding_value("ingest.auto_roots", None),
                        "auto_include": binding_value("ingest.auto_include", list(DEFAULT_INCLUDE)),
                        "auto_exclude": binding_value("ingest.auto_exclude", list(DEFAULT_EXCLUDE)),
                        "auto_pick": binding_value("ingest.auto_pick", "single"),
                        "auto_recursive": binding_value("ingest.auto_recursive", False),
                        "add_source_column": binding_value("ingest.add_source_column", False),
                        "source_col": binding_value("ingest.source_col", "source_file"),
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
        input_fields=(
            _field(
                "ingest",
                "Plate-reader ingest settings for the logic screen.",
                children=(
                    _field(
                        "mode",
                        "Ingest mode for Synergy H1 parsing.",
                        kind="string",
                        choices=("auto", "snapshot_only", "kinetic_only", "mixed"),
                        default="auto",
                    ),
                    _field("channels", "Ordered channel names to keep from the workbook.", kind="string_list"),
                    _field(
                        "channel_map",
                        "Optional channel rename mapping.",
                        kind="mapping",
                        allow_unknown=True,
                        allow_none=True,
                    ),
                    _field(
                        "sheet_names", "Optional workbook sheet names to parse.", kind="string_list", allow_none=True
                    ),
                    _field("add_sheet", "Attach source sheet_name to each row.", kind="bool", default=False),
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
                        choices=("single", "latest", "merge"),
                        default="single",
                    ),
                    _field(
                        "auto_recursive",
                        "Recurse into child directories when discovering workbooks.",
                        kind="bool",
                        default=False,
                    ),
                    _field(
                        "add_source_column",
                        "Attach a source file column when merging multiple workbooks.",
                        kind="bool",
                        default=False,
                    ),
                    _field("source_col", "Name of the source file column.", kind="string", default="source_file"),
                    _field("print_summary", "Print an ingest summary to the log.", kind="bool", default=True),
                ),
            ),
            _field(
                "fold_change",
                "Optional fold-change summary inputs used before vec8 export.",
                children=(
                    _field("target", "Primary fold-change channel.", kind="string", default="YFP/CFP"),
                    _field("report_times", "Report times in hours for fold-change snapshots.", kind="number_list"),
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
                "logic_map_ref", "Reference to annotations.logic_maps entry.", kind="string", default="induction_logic"
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
            _field("strict", "Treat runtime contract mismatches as hard errors.", kind="bool", default=True),
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
            ProtocolFactorSpec(name="logic_map", role="mapping", summary="00/10/01/11 treatment-to-corner map."),
            ProtocolFactorSpec(
                name="reference_design", role="control", summary="Reference design for intensity normalization."
            ),
        ),
        control_rules=(
            ProtocolControlRule(
                id="logic_corner_map",
                summary="Resolve treatment states to 00/10/01/11 corners through the configured logic map.",
                control_selector="logic_map_ref",
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
                formula="v00,v10,v01,v11,y00,y10,y01,y11",
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
                summary="Endpoint comparison grouped by construct/design.",
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
        ),
        plot_profiles=(
            ProtocolPlotProfileSpec(
                id="logic_overview",
                summary="Default SFXI overview with kinetics and endpoint summaries.",
                figures=("raw_kinetics", "endpoint_by_condition", "endpoint_by_design", "intensity_overview"),
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
        ranking=ProtocolRankingSpec(
            primary_metric="vec8",
            direction="higher_is_better",
            summary="Interpret SFXI through the vec8 summary surface and its downstream logic-specific analyses.",
        ),
        execution=ProtocolExecutionPlan(
            notebook=ProtocolNotebookPolicy(
                default_template="notebook/sfxi_eda",
                allowed_templates=("notebook/sfxi_eda", "notebook/eda", "notebook/basic"),
                summary="SFXI logic screens default to the vec8-aware notebook scaffold.",
            ),
            plugin_defaults=(
                ProtocolPluginDefaultsSpec(
                    plugin="ingest/synergy_h1",
                    summary="Logic screens default to CFP/YFP/OD600 Synergy H1 ingest with protocol-bound channel policy.",
                    with_={
                        "mode": binding_value("ingest.mode", "auto"),
                        "channels": binding_value("ingest.channels", ["OD600", "CFP", "YFP"]),
                        "channel_map": binding_value("ingest.channel_map", None),
                        "sheet_names": binding_value("ingest.sheet_names", None),
                        "add_sheet": binding_value("ingest.add_sheet", False),
                        "time_round_decimals": binding_value("ingest.time_round_decimals", 12),
                        "time_step_h": binding_value("ingest.time_step_h", None),
                        "auto_roots": binding_value("ingest.auto_roots", None),
                        "auto_include": binding_value("ingest.auto_include", list(DEFAULT_INCLUDE)),
                        "auto_exclude": binding_value("ingest.auto_exclude", list(DEFAULT_EXCLUDE)),
                        "auto_pick": binding_value("ingest.auto_pick", "single"),
                        "auto_recursive": binding_value("ingest.auto_recursive", False),
                        "add_source_column": binding_value("ingest.add_source_column", False),
                        "source_col": binding_value("ingest.source_col", "source_file"),
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
                        "logic_map_ref": binding_value("logic_map_ref", "induction_logic"),
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
        analysis_fields=(
            _field("strict", "Treat runtime contract mismatches as hard errors.", kind="bool", default=True),
        ),
        factors=(
            ProtocolFactorSpec(name="sample", role="sample", summary="Sample/run identifier."),
            ProtocolFactorSpec(name="condition", role="condition", summary="Experimental condition."),
            ProtocolFactorSpec(name="gate", role="gate", summary="Gate or subset definition.", required=False),
        ),
        figures=(
            ProtocolFigureSpec(
                id="cytometry_qc",
                kind="qc",
                summary="Channel/gating QC precedes downstream comparisons.",
                primary=True,
            ),
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
_DUAL_STRICT_FIELD = next(item for item in _DUAL_REPORTER_PROTOCOL.analysis_fields if item.key == "strict")
_DUAL_PREPROCESSING_FIELD = next(
    item for item in _DUAL_REPORTER_PROTOCOL.analysis_fields if item.key == "preprocessing"
)
_PLATE_READER_MEASUREMENT_FIELD = _field(
    "measurement",
    "Primary matched-control measurement family.",
    kind="string",
    choices=("yfp_cfp", "single_reporter"),
    default="yfp_cfp",
)

_RETRON_SPONGE_FIGURES = (
    ProtocolFigureSpec(
        id="raw_kinetics",
        kind="qc",
        summary="Raw growth and reporter kinetics for early QC before matched-control normalization.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="support_kinetics",
        kind="qc",
        summary="Growth-normalized support ratios that contextualize broad physiology vs reporter-specific effects.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="control_burden_panel",
        kind="qc",
        summary="tetO-only burden panel over the primary readout and growth-rate traces across the full run.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="baseline_shifted_kinetics",
        kind="kinetics",
        summary="Baseline-shifted kinetics that isolate post-stress movement from pre-stress offsets.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="matched_control_kinetics",
        kind="kinetics",
        summary="Per-arm matched-control-normalized kinetics that show deviation from same-sensor tetO controls across the full run.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="induced_effect_kinetics",
        kind="kinetics",
        summary="Per-arm post-stress increment trajectories after matched-control normalization, paired with a compact D_AUC sidecar.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="absolute_effect_kinetics",
        kind="kinetics",
        summary="Per-arm total effect beyond matched tetO that preserves pre-stress preload differences, paired with a compact D_abs_AUC sidecar.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="control_anchored_decomposition",
        kind="summary",
        summary="Per-pair sponge-versus-matched-tetO assay summary with relevant-stress traces, H2O context, and preload/total/increment/burden summaries.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="interaction_summary",
        kind="summary",
        summary="2x2 state summary over the matched-control-normalized endpoint or AUC surface.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="library_heatmaps",
        kind="summary",
        summary="Library-wide heatmaps over absolute effect, post-stress increment, and preload shift.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="stress_modulation_scores",
        kind="summary",
        summary="Stress-modulation score review across on-target sponge/sensor pairs.",
        primary=True,
    ),
    ProtocolFigureSpec(
        id="pareto_ranking",
        kind="summary",
        summary="Pareto-style ranking of absolute on-target effect against burden and leakiness.",
        primary=True,
    ),
)

_RETRON_SPONGE_PLOT_PROFILES = (
    ProtocolPlotProfileSpec(
        id="screen_overview",
        summary="Reader-first default set for matched-control sponge screens from QC through decision and ranking.",
        figures=(
            "raw_kinetics",
            "support_kinetics",
            "control_burden_panel",
            "control_anchored_decomposition",
            "absolute_effect_kinetics",
            "induced_effect_kinetics",
            "library_heatmaps",
            "pareto_ranking",
        ),
    ),
    ProtocolPlotProfileSpec(
        id="kinetics_qc",
        summary="QC-first review over raw, support, and tetO burden traces.",
        figures=("raw_kinetics", "support_kinetics", "control_burden_panel"),
    ),
    ProtocolPlotProfileSpec(
        id="analysis_review",
        summary="Expanded semantic review over compiled sponge metrics, intermediate transforms, and rankings.",
        figures=(
            "baseline_shifted_kinetics",
            "matched_control_kinetics",
            "absolute_effect_kinetics",
            "induced_effect_kinetics",
            "control_anchored_decomposition",
            "interaction_summary",
            "library_heatmaps",
            "stress_modulation_scores",
            "pareto_ranking",
        ),
    ),
)

_PLATE_READER_SINGLE_REPORTER_PROTOCOL = ProtocolDescriptor(
    protocol="plate_reader/single_reporter_screen",
    domain="plate_reader",
    family="screen_analysis",
    summary=(
        "Single-reporter plate-reader panel protocol with configurable reporter/normalizer channels and "
        "compiled fold-change summaries."
    ),
    tags=("plate_reader", "single_reporter", "screen", "ratio", "fold_change"),
    input_fields=_DUAL_REPORTER_PROTOCOL.input_fields,
    analysis_fields=(
        _field(
            "reporter_channel",
            "Primary reporter channel to normalize against the configured normalizer.",
            kind="string",
            default="RFP",
        ),
        _field(
            "normalizer_channel",
            "Denominator channel used to normalize the reporter signal.",
            kind="string",
            default="OD600",
        ),
        _field("include_fold_change", "Build the fold-change comparison table.", kind="bool", default=True),
        _DUAL_STRICT_FIELD,
        _DUAL_PREPROCESSING_FIELD,
    ),
    factors=_DUAL_REPORTER_PROTOCOL.factors,
    semantic_profiles=(
        ProtocolSemanticProfileSpec(
            id="single_reporter_raw",
            family="single_reporter_panel",
            summary="Single-reporter panel semantics over a configured reporter/normalizer ratio.",
            primary_metric="Reporter_Normalizer",
            primary_readout="reporter / normalizer",
            tags=("single_reporter", "ratio", "panel"),
        ),
        ProtocolSemanticProfileSpec(
            id="single_reporter_fold_change",
            family="single_reporter_panel",
            summary="Single-reporter panel semantics with compiled fold-change summaries.",
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
            summary="Raw kinetics view over the configured normalizer, reporter, and reporter ratio channels.",
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
            id="intensity_overview",
            kind="kinetics",
            summary="Combined time-series and endpoint view of the primary single-reporter ratio.",
            primary=True,
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
            summary="Balanced default set for single-reporter plate-reader experiments.",
            figures=("raw_kinetics", "endpoint_by_condition", "endpoint_by_design", "intensity_overview"),
        ),
        ProtocolPlotProfileSpec(
            id="kinetics_qc",
            summary="Kinetics-first QC view with raw traces and distributions.",
            figures=("raw_kinetics", "value_distributions"),
        ),
    ),
    default_plot_profile="screen_overview",
    execution=ProtocolExecutionPlan(
        notebook=ProtocolNotebookPolicy(
            default_template="notebook/eda",
            allowed_templates=("notebook/eda", "notebook/microplate", "notebook/basic"),
            summary="Single-reporter plate-reader screens default to the EDA notebook with plot support.",
        ),
        plugin_defaults=(
            ProtocolPluginDefaultsSpec(
                plugin="ingest/synergy_h1",
                summary=(
                    "Single-reporter screens inherit generic ingest settings here; "
                    "the compiler derives the required reporter/normalizer channels."
                ),
                with_={
                    "mode": binding_value("ingest.mode", "auto"),
                    "channel_map": binding_value("ingest.channel_map", None),
                    "sheet_names": binding_value("ingest.sheet_names", None),
                    "add_sheet": binding_value("ingest.add_sheet", False),
                    "time_round_decimals": binding_value("ingest.time_round_decimals", 12),
                    "time_step_h": binding_value("ingest.time_step_h", None),
                    "auto_roots": binding_value("ingest.auto_roots", None),
                    "auto_include": binding_value("ingest.auto_include", list(DEFAULT_INCLUDE)),
                    "auto_exclude": binding_value("ingest.auto_exclude", list(DEFAULT_EXCLUDE)),
                    "auto_pick": binding_value("ingest.auto_pick", "single"),
                    "auto_recursive": binding_value("ingest.auto_recursive", False),
                    "add_source_column": binding_value("ingest.add_source_column", False),
                    "source_col": binding_value("ingest.source_col", "source_file"),
                    "print_summary": binding_value("ingest.print_summary", True),
                },
            ),
            ProtocolPluginDefaultsSpec(
                plugin="transform/fold_change",
                summary=(
                    "Single-reporter fold-change inherits generic comparison settings here; "
                    "the compiler sets the target to the configured reporter/normalizer ratio."
                ),
                with_={
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
        compiler=compile_plate_reader_single_reporter_screen,
    ),
)

_PLATE_READER_RETRON_SPONGE_PROTOCOL = ProtocolDescriptor(
    protocol="plate_reader/retron_sponge_screen",
    domain="plate_reader",
    family="matched_control_screen",
    summary=(
        "Plate-reader retron sponge screen with explicit matched-control kinetics, burden, leakiness, "
        "and cross-sensor ranking summaries."
    ),
    tags=("plate_reader", "retron", "sponge", "matched_control", "screen", "ratio"),
    input_fields=_DUAL_REPORTER_PROTOCOL.input_fields,
    analysis_fields=(
        _PLATE_READER_MEASUREMENT_FIELD,
        _field(
            "reporter_channel",
            "Reporter channel used when measurement=single_reporter.",
            kind="string",
            default="RFP",
        ),
        _field(
            "growth_channel",
            "Growth / biomass proxy channel used when measurement=single_reporter.",
            kind="string",
            default="OD600",
        ),
        _field("include_fold_change", "Optionally build the fold-change comparison table.", kind="bool", default=False),
        _DUAL_STRICT_FIELD,
        _DUAL_PREPROCESSING_FIELD,
        _field(
            "semantic_metrics",
            "Matched-control sponge-analysis settings.",
            children=(
                _field(
                    "design_column",
                    "Design label column used to derive sensor/sponge identities.",
                    kind="string",
                    default="design_id_alias",
                ),
                _field("state_column", "2x2 state label column.", kind="string", default="treatment_alias"),
                _field(
                    "raw_treatment_column",
                    "Raw treatment column used to recover the actual stress label.",
                    kind="string",
                    default="treatment",
                ),
                _field(
                    "plate_column",
                    "Plate-normalization boundary column. Set to null when workbook sheets are acquisition segments of one plate; set it explicitly when sheets encode distinct biological plates.",
                    kind="string",
                    allow_none=True,
                    default=None,
                ),
                _field("replicate_column", "Replicate-well identifier column.", kind="string", default="position"),
                _field("sensor_column", "Optional explicit sensor column.", kind="string", allow_none=True),
                _field("sponge_column", "Optional explicit sponge column.", kind="string", allow_none=True),
                _field("genotype_column", "Optional explicit genotype-id column.", kind="string", allow_none=True),
                _field(
                    "stress_condition_column",
                    "Optional explicit stress-condition column when raw treatment parsing is not canonical.",
                    kind="string",
                    allow_none=True,
                ),
                _field(
                    "relevant_stress_column",
                    "Optional explicit boolean column marking relevant stress rows.",
                    kind="string",
                    allow_none=True,
                ),
                _field(
                    "expected_sign_column",
                    "Optional explicit sign column (-1/+1) for cross-sensor ranking.",
                    kind="string",
                    allow_none=True,
                ),
                _field(
                    "relevant_sensor_pair_column",
                    "Optional explicit boolean column marking on-target sensor/sponge pairs.",
                    kind="string",
                    allow_none=True,
                ),
                _field(
                    "matched_control_group_column",
                    "Optional explicit grouping column for same-sensor tetO control matching.",
                    kind="string",
                    allow_none=True,
                ),
                _field(
                    "sponge_family_size_column",
                    "Optional explicit sponge-family size/category column.",
                    kind="string",
                    allow_none=True,
                ),
                _field(
                    "design_separator",
                    "Separator used when deriving sensor/sponge from the design label.",
                    kind="string",
                    default="/",
                ),
                _field(
                    "control_name", "Control sponge label used for same-sensor matching.", kind="string", default="tetO"
                ),
                _field(
                    "no_stress_label", "Canonical no-stress label for summary outputs.", kind="string", default="H2O"
                ),
                _field(
                    "stress_time_zero_policy",
                    "How to resolve the stress-addition boundary on the assay clock.",
                    kind="string",
                    choices=("explicit", "largest_gap_midpoint"),
                    default="largest_gap_midpoint",
                ),
                _field(
                    "stress_time_zero_h",
                    "Explicit stress-addition time in hours on the assay clock when policy=explicit.",
                    kind="number",
                    allow_none=True,
                    default=None,
                ),
                _field(
                    "max_post_stress_hours",
                    "Optional cap on the primary post-stress window, measured in hours after stress addition, "
                    "before both AUC and endpoint summaries are computed.",
                    kind="number",
                    allow_none=True,
                    default=DEFAULT_PRIMARY_POST_STRESS_HOURS,
                ),
                _field(
                    "pre_reads", "Number of pre-stress reads used for the baseline window.", kind="integer", default=3
                ),
                _field("endpoint_reads", "Number of reads used in the endpoint window.", kind="integer", default=3),
                _field(
                    "states",
                    "Explicit 2x2 IPTG/stress state labels.",
                    children=(
                        _field(
                            "uninduced_unstressed",
                            "Label for the H2O, -IPTG state.",
                            kind="string",
                            default="-IPTG/-stress",
                        ),
                        _field(
                            "induced_unstressed",
                            "Label for the H2O, +IPTG state.",
                            kind="string",
                            default="+IPTG/-stress",
                        ),
                        _field(
                            "uninduced_stressed",
                            "Label for the relevant-stress, -IPTG state.",
                            kind="string",
                            default="-IPTG/+stress",
                        ),
                        _field(
                            "induced_stressed",
                            "Label for the relevant-stress, +IPTG state.",
                            kind="string",
                            default="+IPTG/+stress",
                        ),
                    ),
                ),
                _field(
                    "plateau",
                    "Primary post-stress window policy.",
                    children=(
                        _field(
                            "mode",
                            "Window selector: full trace after stress, or stop once the matched tetO control plateaus.",
                            kind="string",
                            choices=("full_post_stress", "control_plateau"),
                            default="full_post_stress",
                        ),
                        _field(
                            "slope_tolerance",
                            "Absolute OD slope threshold used for plateau detection.",
                            kind="number",
                            default=0.01,
                        ),
                        _field(
                            "min_intervals",
                            "Minimum number of trailing low-slope intervals before calling plateau.",
                            kind="integer",
                            default=2,
                        ),
                    ),
                ),
                _field(
                    "relevant_stress_map",
                    "Sensor -> relevant stress label mapping.",
                    kind="mapping",
                    allow_unknown=True,
                ),
                _field(
                    "sensor_target_map",
                    "Sensor -> cognate sponge motif list.",
                    kind="mapping",
                    allow_unknown=True,
                ),
                _field(
                    "expected_sign_map",
                    "Optional explicit sign overrides for cross-sensor ranking.",
                    kind="mapping",
                    allow_unknown=True,
                ),
            ),
        ),
    ),
    factors=(
        ProtocolFactorSpec(name="sensor", role="sensor", summary="Reporter promoter / sensor arm."),
        ProtocolFactorSpec(name="sponge", role="construct", summary="Real or tetO sponge arm."),
        ProtocolFactorSpec(name="stress_condition", role="stress", summary="Relevant stress or H2O control."),
        ProtocolFactorSpec(
            name="IPTG",
            role="induction",
            summary="IPTG-driven retron-expression state.",
        ),
        ProtocolFactorSpec(name="replicate_id", role="replicate", summary="Replicate well identifier."),
        ProtocolFactorSpec(name="time", role="time", summary="Time on the assay clock in hours."),
        ProtocolFactorSpec(name="plate_id", role="plate", summary="Plate-local normalization boundary."),
        ProtocolFactorSpec(name="genotype_id", role="construct", summary="Sensor/sponge genotype identifier."),
    ),
    semantic_profiles=(
        ProtocolSemanticProfileSpec(
            id="yfp_cfp",
            family="matched_control_dual_reporter",
            summary="Dual-reporter sponge-screen semantics on the log2(YFP/CFP) axis.",
            primary_metric="O_abs_AUC",
            primary_readout="log2(YFP / CFP)",
            tags=("dual_reporter", "matched_control", "sponge"),
        ),
        ProtocolSemanticProfileSpec(
            id="single_reporter",
            family="matched_control_single_reporter",
            summary="Single-reporter sponge-screen semantics on the log2(configured reporter / configured growth channel) axis.",
            primary_metric="O_abs_AUC",
            primary_readout="log2(configured_reporter_channel / configured_growth_channel)",
            tags=("single_reporter", "matched_control", "sponge"),
        ),
    ),
    control_rules=(
        ProtocolControlRule(
            id="matched_same_sensor_control",
            summary=(
                "Normalize every real sponge well to the same-sensor tetO control on the same plate, "
                "matched by stress state, IPTG state, and timepoint."
            ),
            match_on=("sensor", "plate_id", "stress_condition", "IPTG", "time"),
            control_selector="matched_tetO_group",
            profiles=("yfp_cfp", "single_reporter"),
        ),
    ),
    windows=(
        ProtocolWindowSpec(
            id="pre_stress_last_n",
            summary="Use the last N reads before stress addition as the baseline window.",
            anchor="stress_time_zero",
            selector="last_n_before",
            params={"n": 3},
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolWindowSpec(
            id="primary_post_stress",
            summary="Use the post-stress kinetic window through the configured end-of-window policy.",
            anchor="stress_time_zero",
            selector="configured_post_stress_window",
            params={"policy": "semantic_metrics.plateau"},
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolWindowSpec(
            id="endpoint_last_n",
            summary="Use the last N reads inside the primary post-stress window as the endpoint window after any "
            "configured post-stress time cap is applied.",
            anchor="primary_post_stress",
            selector="last_n_within",
            params={"n": 3},
            profiles=("yfp_cfp", "single_reporter"),
        ),
    ),
    metrics=(
        ProtocolMetricSpec(
            id="OD",
            stage="raw",
            summary="Raw OD600 trace.",
            formula="OD600",
            profile_overrides={
                "single_reporter": ProtocolSemanticProfileOverride(
                    summary="Raw configured growth-proxy trace.",
                    formula="configured_growth_channel",
                )
            },
        ),
        ProtocolMetricSpec(id="CFP", stage="raw", summary="Raw CFP trace.", formula="CFP", profiles=("yfp_cfp",)),
        ProtocolMetricSpec(id="YFP", stage="raw", summary="Raw YFP trace.", formula="YFP", profiles=("yfp_cfp",)),
        ProtocolMetricSpec(
            id="Reporter",
            stage="raw",
            summary="Raw configured reporter trace.",
            formula="configured_reporter_channel",
            profiles=("single_reporter",),
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
            profiles=("yfp_cfp",),
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
            profiles=("yfp_cfp",),
        ),
        ProtocolMetricSpec(
            id="Reporter_OD",
            stage="support",
            summary="Supporting configured reporter per biomass proxy.",
            formula="configured_reporter_channel / configured_growth_channel",
            depends_on=("Reporter", "OD"),
            value_space="linear_ratio",
            unit="ratio",
            comparable_group="support_ratio_linear",
            profiles=("single_reporter",),
        ),
        ProtocolMetricSpec(
            id="R",
            stage="derived",
            summary="Primary within-well log2 ratio.",
            formula="log2(YFP / CFP)",
            depends_on=("YFP", "CFP"),
            value_space="log2_ratio",
            unit="log2_ratio",
            comparable_group="primary_ratio_log2",
            profile_overrides={
                "single_reporter": ProtocolSemanticProfileOverride(
                    summary="Primary within-well single-reporter log2 ratio.",
                    formula="log2(configured_reporter_channel / configured_growth_channel)",
                    depends_on=("Reporter", "OD", "Reporter_OD"),
                    value_space="log2_ratio",
                    unit="log2_ratio",
                    comparable_group="primary_ratio_log2",
                )
            },
        ),
        ProtocolMetricSpec(
            id="R_pre",
            stage="summary",
            summary="Mean of the primary ratio in the pre-stress window.",
            formula="mean(R over pre_stress_last_n)",
            depends_on=("R", "pre_stress_last_n"),
            value_space="log2_ratio",
            unit="log2_ratio",
            comparable_group="primary_ratio_log2",
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="P_pre",
            stage="summary",
            summary="Pre-stress matched-control preload shift between +IPTG and -IPTG states.",
            formula="mean(R_pre - R_pre_tetO,matched)(+IPTG) - mean(R_pre - R_pre_tetO,matched)(-IPTG)",
            depends_on=("R_pre", "matched_same_sensor_control"),
            value_space="delta_log2_ratio",
            unit="log2_ratio_delta",
            comparable_group="response_delta_log2",
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="mu",
            stage="support",
            summary="Approximate growth-rate trace from the slope of log(OD600).",
            formula="d(log(OD600)) / dt",
            depends_on=("OD",),
            profile_overrides={
                "single_reporter": ProtocolSemanticProfileOverride(
                    summary="Approximate growth-rate trace from the slope of log(configured growth channel).",
                    formula="d(log(configured_growth_channel)) / dt",
                )
            },
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="B",
            stage="derived",
            summary="Baseline-shifted reporter ratio relative to the well's own pre-stress state.",
            formula="R(t) - R_pre",
            depends_on=("R", "R_pre"),
            value_space="delta_log2_ratio",
            unit="log2_ratio_delta",
            comparable_group="response_delta_log2",
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="C",
            stage="comparison",
            summary="Matched-control-normalized sponge deviation.",
            formula="B(t) - mean(B matched_same_sensor_control at t)",
            depends_on=("B", "matched_same_sensor_control"),
            value_space="delta_log2_ratio",
            unit="log2_ratio_delta",
            comparable_group="response_delta_log2",
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="C_AUC",
            stage="summary",
            summary="AUC of the matched-control-normalized trace over the primary post-stress window.",
            formula="AUC(C over primary_post_stress)",
            depends_on=("C", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="C_END",
            stage="summary",
            summary="Endpoint mean of the matched-control-normalized trace.",
            formula="mean(C over endpoint_last_n)",
            depends_on=("C", "endpoint_last_n"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="D",
            stage="comparison",
            summary="IPTG-state effect after matched-control normalization.",
            formula="mean(C +IPTG) - mean(C -IPTG)",
            depends_on=("C",),
            value_space="delta_log2_ratio",
            unit="log2_ratio_delta",
            comparable_group="response_delta_log2",
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="D_AUC",
            stage="summary",
            summary="AUC of the IPTG-state effect.",
            formula="AUC(D over primary_post_stress)",
            depends_on=("D", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="D_END",
            stage="summary",
            summary="Endpoint mean of the IPTG-state effect.",
            formula="mean(D over endpoint_last_n)",
            depends_on=("D", "endpoint_last_n"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="D_abs",
            stage="comparison",
            summary="Absolute matched-control IPTG-state effect that retains pre-stress preload differences.",
            formula="mean(R - R_tetO,matched)(+IPTG) - mean(R - R_tetO,matched)(-IPTG)",
            depends_on=("R", "matched_same_sensor_control"),
            value_space="delta_log2_ratio",
            unit="log2_ratio_delta",
            comparable_group="response_delta_log2",
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="D_abs_AUC",
            stage="summary",
            summary="AUC of the absolute matched-control IPTG-state effect.",
            formula="AUC(D_abs over primary_post_stress)",
            depends_on=("D_abs", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="D_abs_END",
            stage="summary",
            summary="Endpoint mean of the absolute matched-control IPTG-state effect.",
            formula="mean(D_abs over endpoint_last_n)",
            depends_on=("D_abs", "endpoint_last_n"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="D_growth",
            stage="burden",
            summary="Construct-specific growth burden after same-sensor tetO subtraction.",
            formula="mean(mu - mu_tetO,matched)(+IPTG) - mean(mu - mu_tetO,matched)(-IPTG)",
            depends_on=("mu", "matched_same_sensor_control"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="D_growth_AUC",
            stage="burden",
            summary="AUC of construct-specific growth burden over the primary window.",
            formula="AUC(D_growth over primary_post_stress)",
            depends_on=("D_growth", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="D_growth_END",
            stage="burden",
            summary="Endpoint mean of construct-specific growth burden.",
            formula="mean(D_growth over endpoint_last_n)",
            depends_on=("D_growth", "endpoint_last_n"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="M",
            stage="comparison",
            summary="Stress modulation of the IPTG-state effect after stress addition.",
            formula="D(relevant_stress) - D(H2O)",
            depends_on=("D",),
            value_space="delta_log2_ratio",
            unit="log2_ratio_delta",
            comparable_group="response_delta_log2",
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="M_AUC",
            stage="summary",
            summary="AUC of stress modulation over the post-stress window.",
            formula="AUC(M over primary_post_stress)",
            depends_on=("M", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="M_END",
            stage="summary",
            summary="Endpoint mean of the stress modulation trace.",
            formula="mean(M over endpoint_last_n)",
            depends_on=("M", "endpoint_last_n"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="O",
            stage="ranking",
            summary="Sign-corrected IPTG-state effect.",
            formula="expected_decoy_sign * D",
            depends_on=("D",),
            value_space="delta_log2_ratio",
            unit="log2_ratio_delta",
            comparable_group="response_delta_log2",
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="O_AUC",
            stage="ranking",
            summary="AUC of the sign-corrected IPTG-state effect.",
            formula="AUC(O over primary_post_stress)",
            depends_on=("O", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="O_abs",
            stage="ranking",
            summary="Sign-corrected absolute matched-control IPTG-state effect.",
            formula="expected_decoy_sign * D_abs",
            depends_on=("D_abs",),
            value_space="delta_log2_ratio",
            unit="log2_ratio_delta",
            comparable_group="response_delta_log2",
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="O_abs_AUC",
            stage="ranking",
            summary="AUC of the sign-corrected absolute matched-control IPTG-state effect.",
            formula="AUC(O_abs over primary_post_stress)",
            depends_on=("O_abs", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="G_sensor",
            stage="summary",
            summary="Native tetO sensor response used for cross-sensor scaling.",
            formula="AUC(mean(B tetO,-IPTG,relevant stress) - mean(B tetO,-IPTG,H2O))",
            depends_on=("B", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="S_AUC",
            stage="ranking",
            summary="Cross-sensor scaled effect size relative to the native sensor response.",
            formula="O_AUC / abs(G_sensor)",
            depends_on=("O_AUC", "G_sensor"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="S_abs_AUC",
            stage="ranking",
            summary="Cross-sensor scaled absolute effect size relative to the native sensor response.",
            formula="O_abs_AUC / abs(G_sensor)",
            depends_on=("O_abs_AUC", "G_sensor"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="L_pre",
            stage="leakiness",
            summary="Pre-stress leakiness relative to the matched control.",
            formula="R_pre(real,-IPTG) - mean(R_pre tetO,-IPTG)",
            depends_on=("R_pre", "matched_same_sensor_control"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="L_post_AUC",
            stage="leakiness",
            summary="Uninduced post-stress leakiness over the primary window.",
            formula="AUC(mean(C -IPTG))",
            depends_on=("C", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="T_ratio_AUC",
            stage="burden",
            summary="tetO ratio burden from the +IPTG versus -IPTG state contrast.",
            formula="AUC(mean(B tetO,+IPTG) - mean(B tetO,-IPTG))",
            depends_on=("B", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="T_growth_AUC",
            stage="burden",
            summary="tetO growth burden from the +IPTG versus -IPTG state contrast.",
            formula="AUC(mean(mu tetO,+IPTG) - mean(mu tetO,-IPTG))",
            depends_on=("mu", "primary_post_stress"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
        ProtocolMetricSpec(
            id="T_finalOD",
            stage="burden",
            summary="Endpoint OD burden for the tetO control.",
            formula="mean(OD tetO,+IPTG,end) - mean(OD tetO,-IPTG,end)",
            depends_on=("OD", "endpoint_last_n"),
            profiles=("yfp_cfp", "single_reporter"),
        ),
    ),
    effect_signs=(
        ProtocolEffectSignSpec(
            target="spyP",
            expected_sign="negative",
            summary="Effective decoys reduce the spyP ratio after sign correction.",
        ),
        ProtocolEffectSignSpec(
            target="sulAp",
            expected_sign="positive",
            summary="Effective LexA decoys increase the sulAp ratio.",
        ),
        ProtocolEffectSignSpec(
            target="soxSp",
            expected_sign="negative",
            summary="Effective SoxR/SoxS decoys reduce the soxSp ratio after sign correction.",
        ),
    ),
    figures=_RETRON_SPONGE_FIGURES,
    plot_profiles=_RETRON_SPONGE_PLOT_PROFILES,
    default_plot_profile="screen_overview",
    artifacts=(
        ProtocolArtifactSpec(
            id="semantic_trace_table",
            summary="CSV export of the matched-control sponge trace table.",
            default=True,
        ),
        ProtocolArtifactSpec(
            id="semantic_summary_table",
            summary="CSV export of the matched-control sponge summary table.",
            default=True,
        ),
    ),
    ranking=ProtocolRankingSpec(
        primary_metric="O_abs_AUC",
        direction="higher_is_better",
        penalties=("T_ratio_AUC", "T_finalOD", "L_pre", "L_post_AUC"),
        supporting_metrics=("S_abs_AUC", "P_pre", "D_AUC", "M_AUC"),
        summary="Rank hits by sign-corrected absolute AUC, then inspect preload, post-stress increment, burden, and leakiness.",
        profiles=("yfp_cfp", "single_reporter"),
    ),
    execution=ProtocolExecutionPlan(
        notebook=ProtocolNotebookPolicy(
            default_template="notebook/retron_sponge",
            allowed_templates=("notebook/retron_sponge", "notebook/eda", "notebook/microplate", "notebook/basic"),
            summary=(
                "Retron sponge screens default to the protocol-specific review notebook and keep the generic "
                "record explorers available as fallbacks."
            ),
        ),
        plugin_defaults=(
            ProtocolPluginDefaultsSpec(
                plugin="ingest/synergy_h1",
                summary=(
                    "Retron sponge screens inherit generic ingest settings here; "
                    "the compiler derives the required measurement-family channels."
                ),
                with_={
                    "mode": binding_value("ingest.mode", "auto"),
                    "channel_map": binding_value("ingest.channel_map", None),
                    "sheet_names": binding_value("ingest.sheet_names", None),
                    "add_sheet": binding_value("ingest.add_sheet", False),
                    "time_round_decimals": binding_value("ingest.time_round_decimals", 12),
                    "time_step_h": binding_value("ingest.time_step_h", None),
                    "auto_roots": binding_value("ingest.auto_roots", None),
                    "auto_include": binding_value("ingest.auto_include", list(DEFAULT_INCLUDE)),
                    "auto_exclude": binding_value("ingest.auto_exclude", list(DEFAULT_EXCLUDE)),
                    "auto_pick": binding_value("ingest.auto_pick", "single"),
                    "auto_recursive": binding_value("ingest.auto_recursive", False),
                    "add_source_column": binding_value("ingest.add_source_column", False),
                    "source_col": binding_value("ingest.source_col", "source_file"),
                    "print_summary": binding_value("ingest.print_summary", True),
                },
            ),
            ProtocolPluginDefaultsSpec(
                plugin="transform/fold_change",
                summary=(
                    "Retron sponge fold-change inherits generic comparison settings here; "
                    "the compiler sets the target to the compiled primary ratio."
                ),
                with_={
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
        compiler=compile_plate_reader_retron_sponge_screen,
    ),
)

BUILTIN_PROTOCOLS = (
    BUILTIN_PROTOCOLS[0],
    _DUAL_REPORTER_PROTOCOL,
    _PLATE_READER_SINGLE_REPORTER_PROTOCOL,
    _PLATE_READER_RETRON_SPONGE_PROTOCOL,
    *BUILTIN_PROTOCOLS[2:],
)


@cache
def builtin_protocol_catalog() -> ProtocolCatalog:
    return ProtocolCatalog(list(BUILTIN_PROTOCOLS))
