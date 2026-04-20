from __future__ import annotations

from copy import deepcopy
from typing import Any

from reader.errors import ConfigError
from reader.protocols.model import CompiledProtocolPlan
from reader.protocols.semantic_coverage import (
    _cytometry_semantic_program,
    _logic_semantic_program,
    _plate_reader_retron_sponge_semantic_program,
    _plate_reader_semantic_program,
    _plate_reader_single_reporter_semantic_program,
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
    include_fold_change = _analysis_bool(analysis, key="include_fold_change", default=True)
    strict = _analysis_bool(analysis, key="strict", default=True)
    preprocessing = _analysis_mapping(analysis, key="preprocessing")
    blank_cfg = _analysis_mapping(preprocessing, key="blank")
    overflow_cfg = _analysis_mapping(preprocessing, key="overflow")
    crosstalk_cfg = _analysis_mapping(analysis, key="crosstalk_pairs")
    include_crosstalk_pairs = _cfg_bool(crosstalk_cfg, key="enabled", default=False)
    include_crosstalk_export = _cfg_bool(crosstalk_cfg, key="export", default=include_crosstalk_pairs)
    ingest_channels = _configured_ingest_channels(protocol, required=("OD600", "CFP", "YFP"))
    pipeline = list(
        _plate_reader_base_steps(
            measurement="yfp_cfp",
            ingest_channels=ingest_channels,
            blank_cfg=blank_cfg,
            overflow_cfg=overflow_cfg,
        )
    )
    if include_fold_change:
        _configured_fold_change_target(protocol, expected="YFP/CFP")
        pipeline.append(_plate_reader_fold_change_step(measurement="yfp_cfp"))
    if include_crosstalk_pairs:
        if not include_fold_change:
            raise ConfigError(
                "plate_reader/dual_reporter_screen requires include_fold_change when crosstalk_pairs.enabled."
            )
        pipeline.append(_plate_reader_crosstalk_pairs_step(config=crosstalk_cfg))

    selected_plots = protocol.select_plot_outputs(
        allowed=_plate_reader_plot_output_ids(measurement="yfp_cfp"),
    )
    plots = [
        _plate_reader_plot_output(
            protocol,
            output_id=deliverable_id,
            measurement="yfp_cfp",
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
            protocol,
            include_crosstalk_pairs=include_crosstalk_pairs,
            include_fold_change=include_fold_change,
        ),
    )


def compile_plate_reader_single_reporter_screen(protocol: Any):
    analysis = _analysis_options(protocol)
    reporter_channel = _analysis_channel(analysis, key="reporter_channel", default="RFP")
    normalizer_channel = _analysis_channel(analysis, key="normalizer_channel", default="OD600")
    if reporter_channel == normalizer_channel:
        raise ConfigError(
            "plate_reader/single_reporter_screen requires distinct reporter_channel and normalizer_channel."
        )
    include_fold_change = _analysis_bool(analysis, key="include_fold_change", default=True)
    strict = _analysis_bool(analysis, key="strict", default=True)
    preprocessing = _analysis_mapping(analysis, key="preprocessing")
    blank_cfg = _analysis_mapping(preprocessing, key="blank")
    overflow_cfg = _analysis_mapping(preprocessing, key="overflow")
    ingest_channels = _configured_ingest_channels(protocol, required=(normalizer_channel, reporter_channel))
    ratio_label = _single_reporter_ratio_label(
        reporter_channel=reporter_channel,
        normalizer_channel=normalizer_channel,
    )

    pipeline = list(
        _plate_reader_single_reporter_base_steps(
            ingest_channels=ingest_channels,
            reporter_channel=reporter_channel,
            normalizer_channel=normalizer_channel,
            blank_cfg=blank_cfg,
            overflow_cfg=overflow_cfg,
        )
    )
    if include_fold_change:
        _configured_fold_change_target(protocol, expected=ratio_label)
        pipeline.append(
            _plate_reader_single_reporter_fold_change_step(
                reporter_channel=reporter_channel,
                normalizer_channel=normalizer_channel,
            )
        )

    selected_plots = protocol.select_plot_outputs(allowed=_plate_reader_single_reporter_plot_output_ids())
    plots = [
        _plate_reader_single_reporter_plot_output(
            protocol,
            output_id=deliverable_id,
            reporter_channel=reporter_channel,
            normalizer_channel=normalizer_channel,
        )
        for deliverable_id in selected_plots
    ]

    selected_exports = protocol.select_export_outputs(defaults=(), allowed=set())
    if selected_exports:
        raise ConfigError("plate_reader/single_reporter_screen does not currently compile export artifacts.")

    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    return CompiledProtocolPlan(
        runtime={"strict": strict},
        pipeline=tuple(pipeline),
        plots=tuple(plots),
        exports=(),
        notebooks=(default_notebook_call(template),),
        semantic_program=_plate_reader_single_reporter_semantic_program(
            protocol,
            reporter_channel=reporter_channel,
            normalizer_channel=normalizer_channel,
            include_fold_change=include_fold_change,
        ),
    )


def compile_plate_reader_retron_sponge_screen(protocol: Any):
    analysis = _analysis_options(protocol)
    measurement = _analysis_choice(
        analysis,
        key="measurement",
        default="yfp_cfp",
        allowed={"yfp_cfp", "single_reporter"},
    )
    reporter_channel = _analysis_channel(analysis, key="reporter_channel", default="RFP")
    growth_channel = _analysis_channel(analysis, key="growth_channel", default="OD600")
    include_fold_change = _analysis_bool(analysis, key="include_fold_change", default=False)
    strict = _analysis_bool(analysis, key="strict", default=True)
    preprocessing = _analysis_mapping(analysis, key="preprocessing")
    blank_cfg = _analysis_mapping(preprocessing, key="blank")
    overflow_cfg = _analysis_mapping(preprocessing, key="overflow")
    semantic_cfg = _analysis_mapping(analysis, key="semantic_metrics")

    if measurement == "yfp_cfp":
        ingest_channels = _configured_ingest_channels(protocol, required=("OD600", "CFP", "YFP"))
        pipeline = list(
            _plate_reader_base_steps(
                measurement=measurement,
                ingest_channels=ingest_channels,
                blank_cfg=blank_cfg,
                overflow_cfg=overflow_cfg,
                base_recipe="plate_reader/retron_sponge_screen_base",
            )
        )
        semantic_step = _plate_reader_semantic_metrics_step(
            measurement_channel="YFP/CFP",
            record_id="ratio_yfp_cfp/df",
            config=semantic_cfg,
        )
    else:
        ingest_channels = _configured_ingest_channels(protocol, required=(growth_channel, reporter_channel))
        pipeline = list(
            _plate_reader_single_reporter_base_steps(
                ingest_channels=ingest_channels,
                reporter_channel=reporter_channel,
                normalizer_channel=growth_channel,
                blank_cfg=blank_cfg,
                overflow_cfg=overflow_cfg,
                base_recipe="plate_reader/retron_sponge_single_reporter_base",
            )
        )
        semantic_step = _plate_reader_semantic_metrics_step(
            measurement_channel=_single_reporter_ratio_label(
                reporter_channel=reporter_channel,
                normalizer_channel=growth_channel,
            ),
            record_id="ratio_reporter_normalizer/df",
            config=_deep_merge({"growth_channel": growth_channel}, semantic_cfg),
        )
    pipeline.append(semantic_step)
    if include_fold_change:
        if measurement == "yfp_cfp":
            _configured_fold_change_target(protocol, expected="YFP/CFP")
            pipeline.append(_plate_reader_fold_change_step(measurement=measurement))
        else:
            _configured_fold_change_target(
                protocol,
                expected=_single_reporter_ratio_label(
                    reporter_channel=reporter_channel,
                    normalizer_channel=growth_channel,
                ),
            )
            pipeline.append(
                _plate_reader_single_reporter_fold_change_step(
                    reporter_channel=reporter_channel,
                    normalizer_channel=growth_channel,
                )
            )

    if measurement == "yfp_cfp":
        selected_plots = protocol.select_plot_outputs(
            allowed=_plate_reader_retron_plot_output_ids(),
        )
        plots = [
            _plate_reader_retron_plot_output(
                protocol,
                output_id=deliverable_id,
                measurement=measurement,
            )
            for deliverable_id in selected_plots
        ]
    else:
        selected_plots = protocol.select_plot_outputs(
            allowed=_plate_reader_retron_plot_output_ids(),
        )
        plots = [
            _plate_reader_retron_plot_output(
                protocol,
                output_id=deliverable_id,
                measurement=measurement,
                reporter_channel=reporter_channel,
                normalizer_channel=growth_channel,
            )
            for deliverable_id in selected_plots
        ]

    selected_exports = protocol.select_export_outputs(
        defaults=("semantic_summary_table", "semantic_trace_table"),
        allowed=_plate_reader_retron_export_output_ids(),
    )
    exports = [
        _plate_reader_retron_export_output(protocol, output_id=deliverable_id) for deliverable_id in selected_exports
    ]

    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    return CompiledProtocolPlan(
        runtime={"strict": strict},
        pipeline=tuple(pipeline),
        plots=tuple(plots),
        exports=tuple(exports),
        notebooks=(default_notebook_call(template),),
        semantic_program=_plate_reader_retron_sponge_semantic_program(
            protocol,
            measurement=measurement,
            reporter_channel=reporter_channel,
            growth_channel=growth_channel,
        ),
    )


def compile_logic_sfxi_screen(protocol: Any):
    analysis = _analysis_options(protocol)
    strict = _analysis_bool(analysis, key="strict", default=True)
    include_fold_change = _analysis_bool(analysis, key="include_fold_change", default=True)
    include_vec8 = _analysis_bool(analysis, key="include_vec8", default=True)
    preprocessing = _analysis_mapping(analysis, key="preprocessing")
    blank_cfg = _analysis_mapping(preprocessing, key="blank")
    overflow_cfg = _analysis_mapping(preprocessing, key="overflow")

    ingest_channels = _configured_ingest_channels(protocol, required=("OD600", "CFP", "YFP"))
    pipeline = list(
        _plate_reader_base_steps(
            measurement="yfp_cfp",
            ingest_channels=ingest_channels,
            blank_cfg=blank_cfg,
            overflow_cfg=overflow_cfg,
        )
    )
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


def _analysis_channel(raw: dict[str, Any], *, key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"protocol.analysis.{key} must be a non-empty string")
    return value.strip()


def _input_mapping(protocol: Any, *, key: str) -> dict[str, Any]:
    raw = getattr(protocol, "inputs", {}) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"protocol.inputs for {protocol.id!r} must be a mapping")
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"protocol.inputs.{key} for {protocol.id!r} must be a mapping")
    return dict(value)


def _configured_ingest_channels(protocol: Any, *, required: tuple[str, ...]) -> list[str]:
    ingest = _input_mapping(protocol, key="ingest")
    configured = ingest.get("channels")
    if configured in (None, ()):
        return list(required)
    if not isinstance(configured, list) or any(not isinstance(item, str) or not item.strip() for item in configured):
        raise ConfigError(f"protocol.inputs.ingest.channels for {protocol.id!r} must be a list of non-empty strings")
    channels = [item.strip() for item in configured]
    missing = [channel for channel in required if channel not in channels]
    if missing:
        missing_text = ", ".join(missing)
        raise ConfigError(
            f"protocol.inputs.ingest.channels for {protocol.id!r} must include required channel(s): {missing_text}"
        )
    return channels


def _configured_fold_change_target(protocol: Any, *, expected: str) -> str:
    fold_change = _input_mapping(protocol, key="fold_change")
    configured = fold_change.get("target")
    if configured is None:
        return expected
    if not isinstance(configured, str) or not configured.strip():
        raise ConfigError(f"protocol.inputs.fold_change.target for {protocol.id!r} must be a non-empty string")
    target = configured.strip()
    if target != expected:
        raise ConfigError(
            f"protocol.inputs.fold_change.target for {protocol.id!r} must match the compiled assay ratio "
            f"{expected!r}, not {target!r}"
        )
    return target


def _cfg_bool(raw: dict[str, Any], *, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{key} must be true or false")


def _plate_reader_plot_output_ids(*, measurement: str) -> set[str]:
    if measurement != "yfp_cfp":
        raise ConfigError(f"Unsupported plate-reader measurement {measurement!r}")
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


def _plate_reader_single_reporter_plot_output_ids() -> set[str]:
    return {
        "raw_kinetics",
        "endpoint_by_condition",
        "endpoint_by_design",
        "intensity_overview",
        "value_distributions",
    }


def _plate_reader_retron_plot_output_ids() -> set[str]:
    return {
        "raw_kinetics",
        "support_kinetics",
        "control_burden_panel",
        "baseline_shifted_kinetics",
        "matched_control_kinetics",
        "induced_effect_kinetics",
        "absolute_effect_kinetics",
        "control_anchored_decomposition",
        "interaction_summary",
        "library_heatmaps",
        "stress_modulation_scores",
        "pareto_ranking",
    }


def _plate_reader_retron_export_output_ids() -> set[str]:
    return {"semantic_trace_table", "semantic_summary_table"}


def _plate_reader_base_steps(
    *,
    measurement: str,
    ingest_channels: list[str],
    blank_cfg: dict[str, Any],
    overflow_cfg: dict[str, Any],
    base_recipe: str = "plate_reader/dual_reporter_screen_base",
) -> tuple[PluginStepDecl, ...]:
    if measurement != "yfp_cfp":
        raise ConfigError(f"Unsupported plate-reader measurement {measurement!r}")
    steps = list(resolve_recipe_steps("plate_reader/synergy_h1")) + list(resolve_recipe_steps(base_recipe))
    return tuple(
        _with_pipeline_overrides(
            steps,
            ingest_channels=ingest_channels,
            blank_cfg=blank_cfg,
            overflow_cfg=overflow_cfg,
        )
    )


def _plate_reader_single_reporter_base_steps(
    *,
    ingest_channels: list[str],
    reporter_channel: str,
    normalizer_channel: str,
    blank_cfg: dict[str, Any],
    overflow_cfg: dict[str, Any],
    base_recipe: str = "plate_reader/single_reporter_screen_base",
) -> tuple[PluginStepDecl, ...]:
    ratio_label = _single_reporter_ratio_label(
        reporter_channel=reporter_channel,
        normalizer_channel=normalizer_channel,
    )
    steps = list(resolve_recipe_steps("plate_reader/synergy_h1")) + list(
        resolve_recipe_steps(
            base_recipe,
            with_args={
                "reporter_channel": reporter_channel,
                "normalizer_channel": normalizer_channel,
            },
        )
    )
    return tuple(
        _with_pipeline_overrides(
            steps,
            ingest_channels=ingest_channels,
            blank_cfg=blank_cfg,
            overflow_cfg=overflow_cfg,
            ratio_name=ratio_label,
            ratio_numerator=reporter_channel,
            ratio_denominator=normalizer_channel,
        )
    )


def _with_pipeline_overrides(
    steps: list[PluginStepDecl],
    *,
    ingest_channels: list[str] | None = None,
    blank_cfg: dict[str, Any],
    overflow_cfg: dict[str, Any],
    ratio_name: str | None = None,
    ratio_numerator: str | None = None,
    ratio_denominator: str | None = None,
) -> list[PluginStepDecl]:
    normalized: list[PluginStepDecl] = []
    for step in steps:
        with_block = dict(step.with_ or {})
        if step.id == "ingest" and ingest_channels is not None:
            with_block = _deep_merge(with_block, {"channels": ingest_channels})
        if step.id == "blank" and blank_cfg:
            with_block = _deep_merge(with_block, blank_cfg)
        if step.id == "overflow" and overflow_cfg:
            with_block = _deep_merge(with_block, overflow_cfg)
        if step.id == "ratio_reporter_normalizer":
            ratio_overrides = {
                key: value
                for key, value in {
                    "name": ratio_name,
                    "numerator": ratio_numerator,
                    "denominator": ratio_denominator,
                }.items()
                if value is not None
            }
            if ratio_overrides:
                with_block = _deep_merge(with_block, ratio_overrides)
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
    if measurement != "yfp_cfp":
        raise ConfigError(f"Unsupported fold-change measurement {measurement!r}")
    return _step(
        id="fold_change__yfp_over_cfp",
        plugin="transform/fold_change",
        reads={"df": RecordInputDecl(record_id="ratio_yfp_cfp/df")},
        writes={"table": RecordOutputDecl(record_id="fold_change__yfp_over_cfp/table")},
    )


def _plate_reader_single_reporter_fold_change_step(
    *,
    reporter_channel: str,
    normalizer_channel: str,
) -> PluginStepDecl:
    return _step(
        id="fold_change__single_reporter",
        plugin="transform/fold_change",
        reads={"df": RecordInputDecl(record_id="ratio_reporter_normalizer/df")},
        with_={
            "target": _single_reporter_ratio_label(
                reporter_channel=reporter_channel, normalizer_channel=normalizer_channel
            )
        },
        writes={"table": RecordOutputDecl(record_id="fold_change__single_reporter/table")},
    )


def _plate_reader_semantic_metrics_step(
    *,
    measurement_channel: str,
    record_id: str,
    config: dict[str, Any],
) -> PluginStepDecl:
    defaults = {"measurement_channel": measurement_channel}
    return _step(
        id="semantic_metrics",
        plugin="transform/retron_sponge_metrics",
        reads={"df": RecordInputDecl(record_id=record_id)},
        with_=_deep_merge(defaults, config),
        writes={
            "trace": RecordOutputDecl(record_id="semantic_metrics/trace"),
            "summary": RecordOutputDecl(record_id="semantic_metrics/summary"),
        },
    )


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
    if with_block["target"] != "YFP/CFP":
        raise ConfigError(
            "protocol.analysis.crosstalk_pairs.target must remain 'YFP/CFP' for plate_reader/dual_reporter_screen"
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
            "y": ["OD600", "YFP", "YFP/CFP", "YFP/OD600"],
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
            "y": ["OD600", "YFP/OD600"],
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
            "y": "YFP/OD600",
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
        defaults = {
            "partition": {"by": "design_id"},
            "ts_channel": "YFP/OD600",
            "ts_hue": "treatment",
            "ts_add_sheet_line": True,
            "ts_mark_snap_time": True,
            "snap_channel": "YFP/OD600",
            "snap_time": 14.0,
        }
        return _step(
            id="intensity_overview",
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


def _plate_reader_retron_plot_output(
    protocol: Any,
    *,
    output_id: str,
    measurement: str,
    reporter_channel: str | None = None,
    normalizer_channel: str | None = None,
) -> PluginStepDecl:
    if measurement == "single_reporter":
        if reporter_channel is None or normalizer_channel is None:
            raise ConfigError("single-reporter retron plots require reporter_channel and normalizer_channel")
        raw_channels = [normalizer_channel, reporter_channel]
        support_channels = [
            _single_reporter_ratio_label(
                reporter_channel=reporter_channel,
                normalizer_channel=normalizer_channel,
            )
        ]
        raw_ylabel_map = {
            normalizer_channel: normalizer_channel,
            reporter_channel: reporter_channel,
        }
        treatment_label_map = {
            "-IPTG/-stress": "No stress, -IPTG",
            "+IPTG/-stress": "No stress, +IPTG",
            "-IPTG/+stress": "Relevant stress, -IPTG",
            "+IPTG/+stress": "Relevant stress, +IPTG",
        }
        support_ylabel_map = {
            _single_reporter_ratio_label(
                reporter_channel=reporter_channel,
                normalizer_channel=normalizer_channel,
            ): _single_reporter_ratio_label(
                reporter_channel=reporter_channel,
                normalizer_channel=normalizer_channel,
            ),
        }
        control_metric_label_map = {
            "R": f"log2({_single_reporter_ratio_label(reporter_channel=reporter_channel, normalizer_channel=normalizer_channel)})",
            "mu": f"d ln({normalizer_channel}) / dt",
        }
    elif measurement == "yfp_cfp":
        raw_channels = ["OD600", "CFP", "YFP"]
        support_channels = ["YFP/OD600", "CFP/OD600"]
        raw_ylabel_map = {
            "OD600": "OD600",
            "CFP": "CFP",
            "YFP": "YFP",
        }
        treatment_label_map = {
            "-IPTG/-stress": "No stress, -IPTG",
            "+IPTG/-stress": "No stress, +IPTG",
            "-IPTG/+stress": "Relevant stress, -IPTG",
            "+IPTG/+stress": "Relevant stress, +IPTG",
        }
        support_ylabel_map = {
            "YFP/OD600": "YFP/OD600",
            "CFP/OD600": "CFP/OD600",
        }
        control_metric_label_map = {
            "R": "log2(YFP/CFP)",
            "mu": "d ln(OD600) / dt",
        }
    else:
        raise ConfigError(f"Unsupported retron plot measurement {measurement!r}")

    settings = protocol.plot_view_config(figure_id=output_id)
    plot_reads = _plate_reader_retron_plot_reads(
        measurement=measurement,
        reporter_channel=reporter_channel,
        normalizer_channel=normalizer_channel,
    )
    control_name = _plate_reader_retron_control_name(protocol)
    no_stress_label = _plate_reader_retron_no_stress_label(protocol)
    stress_order = _plate_reader_retron_stress_order(protocol)
    state_order = _plate_reader_retron_state_order(protocol)

    if output_id == "raw_kinetics":
        defaults = {
            "partition": {"by": "design_id"},
            "hue": "treatment",
            "xlabel": "Time from stress addition (h)",
            "y": raw_channels,
            "ylabel_map": raw_ylabel_map,
            "hue_label_map": treatment_label_map,
            "add_sheet_line": True,
            "shared_legend": True,
            "show_replicates": False,
            "fig": {
                "figsize": [4.9, 4.9],
                "axis_label_size": 8.2,
                "title_fontsize": 8.2,
                "tick_label_size": 6.4,
                "legend_fontsize": 5.4,
                "legend_marker_size": 5.4,
                "mean_marker_size": 10.0,
                "replicate_marker_size": 12.0,
                "line_width": 1.5,
                "top": 0.84,
                "bottom": 0.12,
                "left": 0.09,
                "right": 0.98,
                "wspace": 0.12,
                "hspace": 0.36,
            },
            "filename": "raw_kinetics",
        }
        return _step(
            id="raw_kinetics",
            plugin="plot/time_series",
            reads={"df": plot_reads["df"], "blanks": plot_reads["blanks"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "support_kinetics":
        defaults = {
            "partition": {"by": "design_id"},
            "hue": "treatment",
            "xlabel": "Time from stress addition (h)",
            "y": support_channels,
            "ylabel_map": support_ylabel_map,
            "hue_label_map": treatment_label_map,
            "add_sheet_line": True,
            "shared_legend": True,
            "show_replicates": False,
            "fig": {
                "figsize": [4.9, 4.9],
                "axis_label_size": 8.1,
                "tick_label_size": 6.4,
                "legend_fontsize": 6.2,
                "mean_marker_size": 14.0,
                "replicate_marker_size": 12.0,
                "line_width": 1.5,
                "top": 0.84,
                "bottom": 0.12,
                "left": 0.09,
                "right": 0.98,
                "wspace": 0.12,
                "hspace": 0.36,
            },
            "filename": "support_kinetics",
        }
        return _step(
            id="support_kinetics",
            plugin="plot/time_series",
            reads={"df": plot_reads["df"], "blanks": plot_reads["blanks"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "control_burden_panel":
        defaults = {
            "metrics": ["R", "mu"],
            "title": "tetO burden check",
            "filename": "control_burden_panel",
            "control_name": control_name,
            "include_control": True,
            "only_control": True,
            "stress_order": stress_order,
            "metric_label_map": control_metric_label_map,
            "fig": {"figsize": [10.2, 3.2]},
        }
        return _step(
            id="control_burden_panel",
            plugin="plot/retron_trace",
            reads={"trace": plot_reads["trace"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "baseline_shifted_kinetics":
        defaults = {
            "metrics": ["B"],
            "title": "Advanced: shift from pre-stress state",
            "filename": "baseline_shifted_kinetics",
            "control_name": control_name,
            "include_control": True,
            "stress_order": stress_order,
        }
        return _step(
            id="baseline_shifted_kinetics",
            plugin="plot/retron_trace",
            reads={"trace": plot_reads["trace"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "matched_control_kinetics":
        defaults = {
            "metrics": ["C"],
            "title": "Advanced: matched-control deviation",
            "filename": "matched_control_kinetics",
            "control_name": control_name,
            "relevant_only": True,
            "stress_order": stress_order,
            "panel_by": "sponge",
        }
        return _step(
            id="matched_control_kinetics",
            plugin="plot/retron_trace",
            reads={"trace": plot_reads["trace"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "induced_effect_kinetics":
        defaults = {
            "metrics": ["D"],
            "title": "Post-stress increment over time",
            "filename": "induced_effect_kinetics",
            "control_name": control_name,
            "relevant_only": True,
            "stress_order": stress_order,
            "panel_by": "sponge",
        }
        return _step(
            id="induced_effect_kinetics",
            plugin="plot/retron_trace",
            reads={"trace": plot_reads["trace"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "absolute_effect_kinetics":
        defaults = {
            "metrics": ["D_abs"],
            "title": "Total effect beyond matched tetO over time",
            "filename": "absolute_effect_kinetics",
            "control_name": control_name,
            "relevant_only": True,
            "stress_order": stress_order,
            "panel_by": "sponge",
        }
        return _step(
            id="absolute_effect_kinetics",
            plugin="plot/retron_trace",
            reads={"trace": plot_reads["trace"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "control_anchored_decomposition":
        defaults = {
            "view": "decomposition",
            "metric": "O_AUC",
            "title": "Reporter-ratio shifts by IPTG state against matched tetO",
            "filename": "control_anchored_decomposition",
            "control_name": control_name,
            "no_stress_label": no_stress_label,
            "relevant_only": True,
        }
        return _step(
            id="control_anchored_decomposition",
            plugin="plot/retron_summary",
            reads={"summary": plot_reads["summary"], "trace": plot_reads["trace"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "interaction_summary":
        defaults = {
            "view": "interaction",
            "metric": "C_AUC",
            "title": "IPTG and stress state summary",
            "filename": "interaction_summary",
            "control_name": control_name,
            "no_stress_label": no_stress_label,
            "relevant_only": True,
            "state_order": state_order,
        }
        return _step(
            id="interaction_summary",
            plugin="plot/retron_summary",
            reads={"summary": plot_reads["summary"], "trace": plot_reads["trace"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "library_heatmaps":
        defaults = {
            "view": "heatmap",
            "title": "Library heatmaps",
            "filename": "library_heatmaps",
            "control_name": control_name,
            "no_stress_label": no_stress_label,
            "relevant_only": True,
        }
        return _step(
            id="library_heatmaps",
            plugin="plot/retron_summary",
            reads={"summary": plot_reads["summary"], "trace": plot_reads["trace"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "stress_modulation_scores":
        defaults = {
            "view": "stress_modulation",
            "metric": "M_AUC",
            "title": "Stress modulation scores",
            "filename": "stress_modulation_scores",
            "control_name": control_name,
            "no_stress_label": no_stress_label,
            "relevant_only": True,
        }
        return _step(
            id="stress_modulation_scores",
            plugin="plot/retron_summary",
            reads={"summary": plot_reads["summary"], "trace": plot_reads["trace"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "pareto_ranking":
        defaults = {
            "view": "pareto",
            "title": "Pareto ranking",
            "filename": "pareto_ranking",
            "control_name": control_name,
            "no_stress_label": no_stress_label,
            "metric": "S_abs_AUC",
            "burden_metric": "D_growth_AUC",
        }
        return _step(
            id="pareto_ranking",
            plugin="plot/retron_summary",
            reads={"summary": plot_reads["summary"], "trace": plot_reads["trace"]},
            with_=_deep_merge(defaults, settings),
        )
    raise ConfigError(f"Unknown retron plot output {output_id!r}")


def _plate_reader_single_reporter_plot_output(
    protocol: Any,
    *,
    output_id: str,
    reporter_channel: str,
    normalizer_channel: str,
) -> PluginStepDecl:
    settings = protocol.plot_view_config(figure_id=output_id)
    plot_reads = _plate_reader_single_reporter_plot_reads()
    ratio_label = _single_reporter_ratio_label(
        reporter_channel=reporter_channel,
        normalizer_channel=normalizer_channel,
    )
    if output_id == "raw_kinetics":
        defaults = {
            "partition": {"by": "design_id"},
            "hue": "treatment",
            "y": [normalizer_channel, reporter_channel, ratio_label],
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
            "y": [normalizer_channel, ratio_label],
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
            "y": ratio_label,
            "hue": "treatment",
            "time": 14.0,
        }
        return _step(
            id="endpoint_by_design",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "intensity_overview":
        defaults = {
            "partition": {"by": "design_id"},
            "ts_channel": ratio_label,
            "ts_hue": "treatment",
            "ts_add_sheet_line": True,
            "ts_mark_snap_time": True,
            "snap_channel": ratio_label,
            "snap_time": 14.0,
        }
        return _step(
            id="intensity_overview",
            plugin="plot/ts_and_snap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "value_distributions":
        defaults = {"channels": [ratio_label], "partition": {"by": "design_id"}}
        return _step(
            id="value_distributions",
            plugin="plot/distributions",
            reads={"df": plot_reads["df"], "blanks": plot_reads["blanks"]},
            with_=_deep_merge(defaults, settings),
        )
    raise ConfigError(f"Unknown single-reporter plot output {output_id!r}")


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


def _plate_reader_retron_export_output(protocol: Any, *, output_id: str) -> PluginStepDecl:
    settings = protocol.export_artifact_config(artifact_id=output_id)
    if output_id == "semantic_trace_table":
        defaults = {"path": "retron/semantic_trace.csv"}
        return _step(
            id="semantic_trace_table",
            plugin="export/csv",
            reads={"df": RecordInputDecl(record_id="semantic_metrics/trace")},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "semantic_summary_table":
        defaults = {"path": "retron/semantic_summary.csv"}
        return _step(
            id="semantic_summary_table",
            plugin="export/csv",
            reads={"df": RecordInputDecl(record_id="semantic_metrics/summary")},
            with_=_deep_merge(defaults, settings),
        )
    raise ConfigError(f"Unknown retron export output {output_id!r}")


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
    if measurement != "yfp_cfp":
        raise ConfigError(f"Unsupported plate-reader measurement {measurement!r}")
    return {
        "df": RecordInputDecl(record_id="ratio_yfp_od600/df"),
        "blanks": RecordInputDecl(record_id="blank/blanks"),
    }


def _plate_reader_single_reporter_plot_reads() -> dict[str, RecordInputDecl]:
    return {
        "df": RecordInputDecl(record_id="ratio_reporter_normalizer/df"),
        "blanks": RecordInputDecl(record_id="blank/blanks"),
    }


def _plate_reader_retron_plot_reads(
    *,
    measurement: str,
    reporter_channel: str | None,
    normalizer_channel: str | None,
) -> dict[str, RecordInputDecl]:
    if measurement == "yfp_cfp":
        base_reads = _plate_reader_plot_reads(measurement="yfp_cfp")
    elif measurement == "single_reporter":
        if reporter_channel is None or normalizer_channel is None:
            raise ConfigError("single-reporter retron plots require reporter_channel and normalizer_channel")
        base_reads = _plate_reader_single_reporter_plot_reads()
    else:
        raise ConfigError(f"Unsupported retron plot measurement {measurement!r}")
    return {
        **base_reads,
        "trace": RecordInputDecl(record_id="semantic_metrics/trace"),
        "summary": RecordInputDecl(record_id="semantic_metrics/summary"),
    }


def _plate_reader_retron_control_name(protocol: Any) -> str:
    semantic_cfg = _analysis_mapping(_analysis_options(protocol), key="semantic_metrics")
    return str(semantic_cfg.get("control_name", "tetO"))


def _plate_reader_retron_no_stress_label(protocol: Any) -> str:
    semantic_cfg = _analysis_mapping(_analysis_options(protocol), key="semantic_metrics")
    return str(semantic_cfg.get("no_stress_label", "H2O"))


def _plate_reader_retron_stress_order(protocol: Any) -> tuple[str, ...]:
    semantic_cfg = _analysis_mapping(_analysis_options(protocol), key="semantic_metrics")
    no_stress_label = _plate_reader_retron_no_stress_label(protocol)
    relevant_stress_map = semantic_cfg.get("relevant_stress_map", {}) or {}
    if not isinstance(relevant_stress_map, dict):
        raise ConfigError("protocol.analysis.semantic_metrics.relevant_stress_map must be a mapping when provided")
    ordered: list[str] = [no_stress_label]
    for value in relevant_stress_map.values():
        label = str(value).strip()
        if label and label not in ordered:
            ordered.append(label)
    return tuple(ordered)


def _plate_reader_retron_state_order(protocol: Any) -> tuple[str, ...]:
    semantic_cfg = _analysis_mapping(_analysis_options(protocol), key="semantic_metrics")
    states = _analysis_mapping(semantic_cfg, key="states")
    defaults = (
        str(states.get("uninduced_unstressed", "-IPTG/-stress")),
        str(states.get("induced_unstressed", "+IPTG/-stress")),
        str(states.get("uninduced_stressed", "-IPTG/+stress")),
        str(states.get("induced_stressed", "+IPTG/+stress")),
    )
    return tuple(defaults)


def _single_reporter_ratio_label(*, reporter_channel: str, normalizer_channel: str) -> str:
    return f"{reporter_channel}/{normalizer_channel}"


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
