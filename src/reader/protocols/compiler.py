from __future__ import annotations

from copy import deepcopy
from typing import Any

from reader.errors import ConfigError
from reader.protocols.model import (
    CompiledProtocolPlan,
    ProtocolSemanticExecution,
    ProtocolSemanticNode,
    ProtocolSemanticProgram,
)
from reader.workbench.decl.model import (
    NotebookTemplateCallDecl,
    PluginStepDecl,
    RecipeSourceDecl,
    RecordInputDecl,
    RecordOutputDecl,
    ResourceInputDecl,
)
from reader.workbench.recipes.registry import resolve_recipe_steps

PLATE_READER_PLOT_OUTPUTS = {
    "raw_kinetics",
    "endpoint_by_condition",
    "endpoint_by_design",
    "state_summary",
    "intensity_overview",
    "ratio_overview",
    "value_distributions",
    "ratio_heatmap",
    "support_heatmap",
    "logic_symmetry",
}
PLATE_READER_EXPORT_OUTPUTS = {"crosstalk_pairs_table"}
LOGIC_EXPORT_OUTPUTS = {"logic_summary_workbook"}


def compile_generic_protocol(protocol: Any):
    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    return CompiledProtocolPlan(
        pipeline=(),
        plots=(),
        exports=(),
        notebooks=(default_notebook_call(template),),
        semantic_program=protocol.descriptor.semantic_program(),
    )


def compile_plate_reader_dual_reporter_screen(protocol: Any):
    analysis = _analysis_options(protocol)
    measurement = _analysis_choice(
        analysis,
        key="measurement",
        default="yfp_cfp",
        allowed={"yfp_cfp", "rfp_od600"},
    )
    include_fold_change = _analysis_bool(analysis, key="include_fold_change", default=True)
    strict = _analysis_bool(analysis, key="strict", default=True)
    preprocessing = _analysis_mapping(analysis, key="preprocessing")
    blank_cfg = _analysis_child(preprocessing, key="blank")
    overflow_cfg = _analysis_child(preprocessing, key="overflow")
    crosstalk_cfg = _analysis_child(analysis, key="crosstalk_pairs")
    include_crosstalk_pairs = _cfg_bool(crosstalk_cfg, key="enabled", default=False)
    include_crosstalk_export = _cfg_bool(crosstalk_cfg, key="export", default=include_crosstalk_pairs)

    if measurement == "rfp_od600" and include_crosstalk_pairs:
        raise ConfigError("plate_reader/dual_reporter_screen does not support crosstalk_pairs with rfp_od600.")

    pipeline = list(_plate_reader_base_steps(measurement=measurement, blank_cfg=blank_cfg, overflow_cfg=overflow_cfg))
    if include_fold_change:
        pipeline.append(_plate_reader_fold_change_step(measurement=measurement))
    if include_crosstalk_pairs:
        if not include_fold_change:
            raise ConfigError(
                "plate_reader/dual_reporter_screen requires include_fold_change when crosstalk_pairs.enabled."
            )
        pipeline.append(_plate_reader_crosstalk_pairs_step(config=crosstalk_cfg))

    selected_plots = protocol.select_plot_outputs(
        allowed=_plate_reader_plot_output_ids(measurement=measurement),
    )
    plots = [
        _plate_reader_plot_output(
            protocol,
            output_id=deliverable_id,
            measurement=measurement,
        )
        for deliverable_id in selected_plots
    ]

    default_exports = ("crosstalk_pairs_table",) if include_crosstalk_export else ()
    selected_exports = protocol.select_export_outputs(
        defaults=default_exports,
        allowed=PLATE_READER_EXPORT_OUTPUTS,
    )
    exports = [_plate_reader_export_output(protocol, output_id=deliverable_id) for deliverable_id in selected_exports]

    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    return CompiledProtocolPlan(
        runtime={"strict": strict},
        pipeline=tuple(pipeline),
        plots=tuple(plots),
        exports=tuple(exports),
        notebooks=(default_notebook_call(template),),
        semantic_program=_plate_reader_semantic_program(
            protocol, measurement=measurement, include_crosstalk_pairs=include_crosstalk_pairs
        ),
    )


def compile_logic_sfxi_screen(protocol: Any):
    analysis = _analysis_options(protocol)
    strict = _analysis_bool(analysis, key="strict", default=True)
    include_fold_change = _analysis_bool(analysis, key="include_fold_change", default=True)
    include_vec8 = _analysis_bool(analysis, key="include_vec8", default=True)
    preprocessing = _analysis_mapping(analysis, key="preprocessing")
    blank_cfg = _analysis_child(preprocessing, key="blank")
    overflow_cfg = _analysis_child(preprocessing, key="overflow")

    pipeline = list(_plate_reader_base_steps(measurement="yfp_cfp", blank_cfg=blank_cfg, overflow_cfg=overflow_cfg))
    if include_fold_change:
        pipeline.append(_plate_reader_fold_change_step(measurement="yfp_cfp"))
    selected_plot_ids = protocol.select_plot_outputs(
        allowed={
            "raw_kinetics",
            "endpoint_by_condition",
            "endpoint_by_design",
            "intensity_overview",
            "logic_symmetry",
        },
    )
    requires_promoted_df = include_vec8 or "logic_symmetry" in selected_plot_ids
    if requires_promoted_df:
        pipeline.append(_sfxi_promote_step())
    if include_vec8:
        pipeline.append(_sfxi_vec8_step())

    plots = [
        _plate_reader_plot_output(protocol, output_id=deliverable_id, measurement="yfp_cfp")
        for deliverable_id in selected_plot_ids
    ]

    default_exports = (
        ("logic_summary_workbook",)
        if include_vec8 and _analysis_bool(analysis, key="include_export", default=include_vec8)
        else ()
    )
    selected_exports = protocol.select_export_outputs(
        defaults=default_exports,
        allowed=LOGIC_EXPORT_OUTPUTS,
    )
    exports = [_logic_export_output(protocol, output_id=deliverable_id) for deliverable_id in selected_exports]

    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    return CompiledProtocolPlan(
        runtime={"strict": strict},
        pipeline=tuple(pipeline),
        plots=tuple(plots),
        exports=tuple(exports),
        notebooks=(default_notebook_call(template),),
        semantic_program=_logic_semantic_program(protocol, include_vec8=include_vec8),
    )


def compile_cytometry_flow_panel(protocol: Any):
    analysis = _analysis_options(protocol)
    strict = _analysis_bool(analysis, key="strict", default=True)
    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    try:
        selected_plots = protocol.select_plot_outputs(allowed=set())
    except ConfigError as exc:
        raise ConfigError(f"cytometry/flow_panel does not currently compile plot outputs. {exc}") from exc
    try:
        selected_exports = protocol.select_export_outputs(defaults=(), allowed=set())
    except ConfigError as exc:
        raise ConfigError(f"cytometry/flow_panel does not currently compile export artifacts. {exc}") from exc
    if selected_plots:
        raise ConfigError("cytometry/flow_panel does not currently compile plot outputs.")
    if selected_exports:
        raise ConfigError("cytometry/flow_panel does not currently compile export artifacts.")
    return CompiledProtocolPlan(
        runtime={"strict": strict},
        pipeline=(
            _step(
                id="ingest_cytometer",
                plugin="ingest/flow_cytometer",
                writes={"df": RecordOutputDecl(record_id="ingest/df")},
            ),
            _step(
                id="merge_metadata",
                plugin="transform/sample_metadata",
                reads={
                    "df": RecordInputDecl(record_id="ingest/df"),
                    "metadata": ResourceInputDecl(resource_id="metadata"),
                },
                writes={"df": RecordOutputDecl(record_id="merged/df")},
            ),
        ),
        plots=(),
        exports=(),
        notebooks=(default_notebook_call(template),),
        semantic_program=_cytometry_semantic_program(protocol),
    )


def _semantic_program(
    protocol: Any,
    *,
    overrides: dict[str, ProtocolSemanticExecution],
    active_profile: str | None = None,
) -> ProtocolSemanticProgram:
    descriptor_program = protocol.descriptor.semantic_program(active_profile=active_profile)

    def _apply(nodes: tuple[ProtocolSemanticNode, ...]) -> tuple[ProtocolSemanticNode, ...]:
        return tuple(
            ProtocolSemanticNode(
                id=node.id,
                kind=node.kind,
                summary=node.summary,
                profiles=node.profiles,
                stage=node.stage,
                formula=node.formula,
                depends_on=node.depends_on,
                anchor=node.anchor,
                selector=node.selector,
                params=node.params,
                match_on=node.match_on,
                control_selector=node.control_selector,
                primary_metric=node.primary_metric,
                direction=node.direction,
                penalties=node.penalties,
                supporting_metrics=node.supporting_metrics,
                execution=overrides.get(node.id, node.execution),
            )
            for node in nodes
        )

    ranking = descriptor_program.ranking
    if ranking is not None:
        ranking = ProtocolSemanticNode(
            id=ranking.id,
            kind=ranking.kind,
            summary=ranking.summary,
            profiles=ranking.profiles,
            stage=ranking.stage,
            formula=ranking.formula,
            depends_on=ranking.depends_on,
            anchor=ranking.anchor,
            selector=ranking.selector,
            params=ranking.params,
            match_on=ranking.match_on,
            control_selector=ranking.control_selector,
            primary_metric=ranking.primary_metric,
            direction=ranking.direction,
            penalties=ranking.penalties,
            supporting_metrics=ranking.supporting_metrics,
            execution=overrides.get(ranking.id, ranking.execution),
        )

    return ProtocolSemanticProgram(
        protocol=descriptor_program.protocol,
        profiles=descriptor_program.profiles,
        active_profile=descriptor_program.active_profile,
        controls=_apply(descriptor_program.controls),
        windows=_apply(descriptor_program.windows),
        metrics=_apply(descriptor_program.metrics),
        ranking=ranking,
    )


def _plate_reader_semantic_program(
    protocol: Any,
    *,
    measurement: str,
    include_crosstalk_pairs: bool,
) -> ProtocolSemanticProgram:
    overrides: dict[str, ProtocolSemanticExecution] = {
        "OD": ProtocolSemanticExecution(
            status="compiled",
            step_ids=("ingest",),
            plugin_ids=("ingest/synergy_h1",),
            record_ids=("ingest/df",),
            note="Raw OD600 values are materialized on the ingest dataframe.",
        ),
    }
    if measurement == "yfp_cfp":
        overrides.update(
            {
                "CFP": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ingest",),
                    plugin_ids=("ingest/synergy_h1",),
                    record_ids=("ingest/df",),
                    note="Raw CFP values are materialized on the ingest dataframe.",
                ),
                "YFP": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ingest",),
                    plugin_ids=("ingest/synergy_h1",),
                    record_ids=("ingest/df",),
                    note="Raw YFP values are materialized on the ingest dataframe.",
                ),
                "CFP_OD": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ratio_cfp_od600",),
                    plugin_ids=("transform/ratio",),
                    record_ids=("ratio_cfp_od600/df",),
                    note="The CFP/OD600 support channel is materialized as a ratio step output.",
                ),
                "YFP_OD": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ratio_yfp_od600",),
                    plugin_ids=("transform/ratio",),
                    record_ids=("ratio_yfp_od600/df",),
                    note="The YFP/OD600 support channel is materialized as a ratio step output.",
                ),
                "R": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ratio_yfp_cfp",),
                    plugin_ids=("transform/ratio",),
                    record_ids=("ratio_yfp_cfp/df",),
                    note="The primary YFP/CFP ratio is materialized as a ratio step output.",
                ),
            }
        )
    else:
        overrides.update(
            {
                "RFP": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ingest",),
                    plugin_ids=("ingest/synergy_h1",),
                    record_ids=("ingest/df",),
                    note="Raw RFP values are materialized on the ingest dataframe.",
                ),
                "R": ProtocolSemanticExecution(
                    status="compiled",
                    step_ids=("ratio_rfp_od600",),
                    plugin_ids=("transform/ratio",),
                    record_ids=("ratio_rfp_od600/df",),
                    note="The primary RFP/OD600 ratio is materialized as a ratio step output.",
                ),
            }
        )
    if include_crosstalk_pairs:
        overrides["ranking"] = ProtocolSemanticExecution(
            status="compiled",
            step_ids=("crosstalk_pairs",),
            plugin_ids=("transform/crosstalk_pairs",),
            record_ids=("crosstalk_pairs/table",),
            config_paths=("protocol.analysis.crosstalk_pairs",),
            note="When crosstalk pair analysis is enabled, the exported pair table is compiled from fold-change output.",
        )
    return _semantic_program(protocol, overrides=overrides, active_profile=measurement)


def _logic_semantic_program(protocol: Any, *, include_vec8: bool) -> ProtocolSemanticProgram:
    overrides: dict[str, ProtocolSemanticExecution] = {}
    if include_vec8:
        vec8_binding = ProtocolSemanticExecution(
            status="compiled",
            step_ids=("sfxi_vec8",),
            plugin_ids=("transform/sfxi",),
            record_ids=("sfxi_vec8/vec8",),
            config_paths=(
                "protocol.inputs.response",
                "protocol.inputs.reference",
                "protocol.inputs.design_by",
                "protocol.inputs.logic_map_ref",
                "protocol.inputs.time_mode",
                "protocol.inputs.target_time_h",
                "protocol.inputs.time_tolerance_h",
            ),
            note="The SFXI vec8 transform materializes the protocol control rule, summary window, metric, and ranking surface.",
        )
        overrides.update(
            {
                "logic_corner_map": vec8_binding,
                "summary_timepoint": vec8_binding,
                "vec8": vec8_binding,
                "ranking": vec8_binding,
            }
        )
    return _semantic_program(protocol, overrides=overrides)


def _cytometry_semantic_program(protocol: Any) -> ProtocolSemanticProgram:
    return _semantic_program(
        protocol,
        overrides={
            "ranking": ProtocolSemanticExecution(
                status="descriptive_only",
                note="Cytometry ranking remains domain-defined until a typed analysis program is introduced.",
            )
        },
    )


def _analysis_options(protocol: Any) -> dict[str, Any]:
    raw = getattr(protocol, "analysis", {}) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"protocol.analysis for {protocol.id!r} must be a mapping")
    return dict(raw)


def _analysis_mapping(raw: dict[str, Any], *, key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"protocol.analysis.{key} must be a mapping")
    return dict(value)


def _analysis_child(raw: dict[str, Any], *, key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"protocol.analysis.{key} must be a mapping")
    return dict(value)


def _analysis_bool(raw: dict[str, Any], *, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    raise ConfigError(f"protocol.analysis.{key} must be true or false")


def _analysis_choice(raw: dict[str, Any], *, key: str, default: str, allowed: set[str]) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ConfigError(f"protocol.analysis.{key} must be one of: {options}")
    return value


def _cfg_bool(raw: dict[str, Any], *, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{key} must be true or false")


def _plate_reader_plot_output_ids(*, measurement: str) -> set[str]:
    if measurement == "yfp_cfp":
        return {
            "raw_kinetics",
            "endpoint_by_condition",
            "endpoint_by_design",
            "state_summary",
            "intensity_overview",
            "ratio_overview",
            "value_distributions",
            "ratio_heatmap",
            "support_heatmap",
        }
    if measurement == "rfp_od600":
        return {
            "raw_kinetics",
            "endpoint_by_condition",
            "endpoint_by_design",
            "intensity_overview",
            "value_distributions",
        }
    raise ConfigError(f"Unsupported plate-reader measurement {measurement!r}")


def _plate_reader_base_steps(
    *,
    measurement: str,
    blank_cfg: dict[str, Any],
    overflow_cfg: dict[str, Any],
) -> tuple[PluginStepDecl, ...]:
    if measurement == "yfp_cfp":
        steps = list(resolve_recipe_steps("plate_reader/synergy_h1")) + list(
            resolve_recipe_steps("plate_reader/dual_reporter_screen_base")
        )
        return tuple(_with_pipeline_overrides(steps, blank_cfg=blank_cfg, overflow_cfg=overflow_cfg))
    if measurement == "rfp_od600":
        return (
            _step(id="ingest", plugin="ingest/synergy_h1"),
            _step(
                id="merge_map",
                plugin="transform/sample_map",
                reads={
                    "df": RecordInputDecl(record_id="ingest/df"),
                    "sample_map": ResourceInputDecl(resource_id="sample_map"),
                },
            ),
            _step(
                id="labels",
                plugin="transform/assay_labels",
                reads={"df": RecordInputDecl(record_id="merge_map/df")},
            ),
            _step(
                id="blank",
                plugin="transform/blank_correction",
                reads={"df": RecordInputDecl(record_id="labels/df")},
                with_=blank_cfg,
            ),
            _step(
                id="overflow",
                plugin="transform/overflow_handling",
                reads={"df": RecordInputDecl(record_id="blank/df")},
                with_=overflow_cfg,
            ),
            _step(
                id="ratio_rfp_od600",
                plugin="transform/ratio",
                reads={"df": RecordInputDecl(record_id="overflow/df")},
                with_={"name": "RFP/OD600", "numerator": "RFP", "denominator": "OD600"},
            ),
        )
    raise ConfigError(f"Unsupported plate-reader measurement {measurement!r}")


def _with_pipeline_overrides(
    steps: list[PluginStepDecl],
    *,
    blank_cfg: dict[str, Any],
    overflow_cfg: dict[str, Any],
) -> list[PluginStepDecl]:
    normalized: list[PluginStepDecl] = []
    for step in steps:
        with_block = dict(step.with_ or {})
        if step.id == "blank" and blank_cfg:
            with_block = _deep_merge(with_block, blank_cfg)
        if step.id == "overflow" and overflow_cfg:
            with_block = _deep_merge(with_block, overflow_cfg)
        normalized.append(
            PluginStepDecl(
                id=step.id,
                plugin=step.plugin,
                reads=dict(step.reads or {}),
                writes=dict(step.writes or {}),
                with_=with_block,
                source_recipe=step.source_recipe,
            )
        )
    return normalized


def _plate_reader_fold_change_step(*, measurement: str) -> PluginStepDecl:
    if measurement == "yfp_cfp":
        return _step(
            id="fold_change__yfp_over_cfp",
            plugin="transform/fold_change",
            reads={"df": RecordInputDecl(record_id="ratio_yfp_cfp/df")},
            writes={"table": RecordOutputDecl(record_id="fold_change__yfp_over_cfp/table")},
        )
    if measurement == "rfp_od600":
        return _step(
            id="fold_change__rfp_od600",
            plugin="transform/fold_change",
            reads={"df": RecordInputDecl(record_id="ratio_rfp_od600/df")},
            with_={"target": "RFP/OD600"},
            writes={"table": RecordOutputDecl(record_id="fold_change__rfp_od600/table")},
        )
    raise ConfigError(f"Unsupported fold-change measurement {measurement!r}")


def _plate_reader_crosstalk_pairs_step(*, config: dict[str, Any]) -> PluginStepDecl:
    defaults = {
        "value_column": "log2FC",
        "value_scale": "log2",
        "target": "YFP/CFP",
        "time_mode": "all",
        "design_column": "design_id",
        "treatment_column": "treatment",
        "mapping_mode": "explicit",
        "require_self_treatment": True,
        "require_self_is_top1": True,
        "min_self": 1.0,
        "max_cross": 0.5,
        "min_selectivity_delta": 1.0,
    }
    with_block = _deep_merge(
        defaults, {key: value for key, value in config.items() if key not in {"enabled", "export"}}
    )
    return _step(
        id="crosstalk_pairs",
        plugin="transform/crosstalk_pairs",
        reads={"table": RecordInputDecl(record_id="fold_change__yfp_over_cfp/table")},
        with_=with_block,
        writes={"table": RecordOutputDecl(record_id="crosstalk_pairs/table")},
    )


def _sfxi_promote_step() -> PluginStepDecl:
    return _step(
        id="promote_to_tidy_plus_map",
        plugin="validator/to_tidy_plus_map",
        reads={"df": RecordInputDecl(record_id="ratio_yfp_od600/df")},
        writes={"df": RecordOutputDecl(record_id="promote_to_tidy_plus_map/df")},
    )


def _sfxi_vec8_step() -> PluginStepDecl:
    return _step(
        id="sfxi_vec8",
        plugin="transform/sfxi",
        reads={"df": RecordInputDecl(record_id="promote_to_tidy_plus_map/df")},
        writes={"vec8": RecordOutputDecl(record_id="sfxi_vec8/vec8")},
    )


def _plate_reader_plot_output(protocol: Any, *, output_id: str, measurement: str) -> PluginStepDecl:
    settings = protocol.plot_view_config(figure_id=output_id)
    plot_reads = _plate_reader_plot_reads(measurement=measurement)
    if output_id == "raw_kinetics":
        defaults = {
            "partition": {"by": "design_id"},
            "hue": "treatment",
            "y": (
                ["OD600", "YFP", "YFP/CFP", "YFP/OD600"] if measurement == "yfp_cfp" else ["OD600", "RFP", "RFP/OD600"]
            ),
            "add_sheet_line": True,
        }
        return _step(
            id="raw_kinetics",
            plugin="plot/time_series",
            reads={"df": plot_reads["df"], "blanks": plot_reads["blanks"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "endpoint_by_condition":
        defaults = {
            "x": "treatment",
            "y": (["OD600", "YFP/OD600"] if measurement == "yfp_cfp" else ["OD600", "RFP/OD600"]),
            "partition": {"by": "design_id"},
            "time": 14.0,
        }
        return _step(
            id="endpoint_by_condition",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "endpoint_by_design":
        defaults = {
            "x": "design_id",
            "y": ("YFP/OD600" if measurement == "yfp_cfp" else "RFP/OD600"),
            "hue": "treatment",
            "time": 14.0,
        }
        return _step(
            id="endpoint_by_design",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "state_summary":
        defaults = {
            "x": "treatment_alias",
            "y": ["OD600", "CFP/OD600", "YFP/OD600", "YFP/CFP"],
            "partition": {"by": "design_id_alias"},
            "time": 14.0,
        }
        return _step(
            id="state_summary",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "intensity_overview":
        if measurement == "yfp_cfp":
            defaults = {
                "partition": {"by": "design_id"},
                "ts_channel": "YFP/OD600",
                "ts_hue": "treatment",
                "ts_add_sheet_line": True,
                "ts_mark_snap_time": True,
                "snap_channel": "YFP/OD600",
                "snap_time": 14.0,
            }
            step_id = "intensity_overview"
        else:
            defaults = {
                "partition": {"by": "design_id"},
                "ts_channel": "RFP/OD600",
                "ts_hue": "treatment",
                "ts_add_sheet_line": True,
                "ts_mark_snap_time": True,
                "snap_channel": "RFP/OD600",
                "snap_time": 14.0,
            }
            step_id = "intensity_overview"
        return _step(
            id=step_id,
            plugin="plot/ts_and_snap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "ratio_overview":
        defaults = {
            "partition": {"by": "design_id_alias"},
            "ts_channel": "OD600",
            "ts_hue": "treatment_alias",
            "ts_add_sheet_line": True,
            "ts_mark_snap_time": True,
            "snap_x": "treatment_alias",
            "snap_channel": "YFP/CFP",
            "snap_time": 14.0,
        }
        return _step(
            id="ratio_overview",
            plugin="plot/ts_and_snap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "value_distributions":
        defaults = {"channels": ["YFP/CFP"], "partition": {"by": "design_id"}}
        if measurement == "rfp_od600":
            defaults["channels"] = ["RFP/OD600"]
        return _step(
            id="value_distributions",
            plugin="plot/distributions",
            reads={"df": plot_reads["df"], "blanks": plot_reads["blanks"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "ratio_heatmap":
        defaults = {
            "channel": "YFP/CFP",
            "time": 14.0,
            "x": "treatment_alias",
            "y": "design_id_alias",
        }
        return _step(
            id="ratio_heatmap",
            plugin="plot/snapshot_heatmap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "support_heatmap":
        defaults = {
            "channel": "CFP/OD600",
            "time": 14.0,
            "x": "treatment_alias",
            "y": "design_id_alias",
        }
        return _step(
            id="support_heatmap",
            plugin="plot/snapshot_heatmap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "logic_symmetry":
        defaults = {"response_channel": "YFP/CFP"}
        return _step(
            id="logic_symmetry",
            plugin="plot/logic_symmetry",
            reads={"df": RecordInputDecl(record_id="promote_to_tidy_plus_map/df")},
            with_=_deep_merge(defaults, settings),
        )
    raise ConfigError(f"Unknown plate-reader plot output {output_id!r}")


def _plate_reader_export_output(protocol: Any, *, output_id: str) -> PluginStepDecl:
    if output_id == "crosstalk_pairs_table":
        defaults = {"path": "crosstalk_pairs.csv"}
        settings = protocol.export_artifact_config(artifact_id=output_id)
        return _step(
            id="crosstalk_pairs_table",
            plugin="export/csv",
            reads={"df": RecordInputDecl(record_id="crosstalk_pairs/table")},
            with_=_deep_merge(defaults, settings),
        )
    raise ConfigError(f"Unknown plate-reader export output {output_id!r}")


def _logic_export_output(protocol: Any, *, output_id: str) -> PluginStepDecl:
    if output_id == "logic_summary_workbook":
        settings = protocol.export_artifact_config(artifact_id=output_id)
        return _step(
            id="logic_summary_workbook",
            plugin="export/xlsx",
            reads={"df": RecordInputDecl(record_id="sfxi_vec8/vec8")},
            with_=_deep_merge({"path": "sfxi/vec8.xlsx", "sheet_name": "vec8"}, settings),
        )
    raise ConfigError(f"Unknown logic export output {output_id!r}")


def _deep_merge(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in (mapping or {}).items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = _deep_merge(merged[key], value)
                continue
            merged[key] = deepcopy(value)
    return merged


def _plate_reader_plot_reads(*, measurement: str) -> dict[str, RecordInputDecl]:
    if measurement == "yfp_cfp":
        return {
            "df": RecordInputDecl(record_id="ratio_yfp_od600/df"),
            "blanks": RecordInputDecl(record_id="blank/blanks"),
        }
    if measurement == "rfp_od600":
        return {
            "df": RecordInputDecl(record_id="ratio_rfp_od600/df"),
            "blanks": RecordInputDecl(record_id="blank/blanks"),
        }
    raise ConfigError(f"Unsupported plate-reader measurement {measurement!r}")


def _step(
    *,
    id: str,
    plugin: str,
    reads: dict[str, Any] | None = None,
    writes: dict[str, Any] | None = None,
    with_: dict[str, Any] | None = None,
    source_recipe: str | None = None,
) -> PluginStepDecl:
    return PluginStepDecl(
        id=id,
        plugin=plugin,
        reads=dict(reads or {}),
        writes=dict(writes or {}),
        with_=dict(with_ or {}),
        source_recipe=(RecipeSourceDecl(recipe=source_recipe) if source_recipe else None),
    )


def default_notebook_call(template: str) -> NotebookTemplateCallDecl:
    return NotebookTemplateCallDecl(id="default", template=template)
