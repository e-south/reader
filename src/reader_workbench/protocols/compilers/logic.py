from __future__ import annotations

from typing import Any

from reader_workbench.errors import ConfigError
from reader_workbench.protocols.model import CompiledProtocolPlan
from reader_workbench.protocols.semantic_coverage import _logic_semantic_program
from reader_workbench.workbench.decl.model import (
    PluginStepDecl,
    RecordCollectionInputDecl,
    RecordInputDecl,
    RecordOutputDecl,
)

from .common import (
    _analysis_bool,
    _analysis_mapping,
    _analysis_options,
    _deep_merge,
    _step,
)
from .plate_reader import (
    _configured_fold_change_enabled,
    _configured_fold_change_report_times,
    _configured_fold_change_target,
    _configured_ingest_channels,
    _plate_reader_fold_change_step,
    _plate_reader_plot_output,
)
from .plate_reader_pipeline import compose_dual_reporter_pipeline

LOGIC_EXPORT_OUTPUTS = {"logic_summary_workbook"}
FOUR_STATE_VECTOR_SCREEN_LOGIC_CHANNEL = "YFP/CFP"


def compile_logic_four_state_vector_collection(protocol: Any):
    resource_ids = tuple(protocol.effective_inputs().get("record_resources", ()))
    pipeline = (
        _step(
            id="four_state_vector_collection",
            plugin="transform/four_state_vector_collection",
            reads={"sources": RecordCollectionInputDecl(resource_ids=resource_ids)},
            writes={"vectors": RecordOutputDecl(record_id="four_state_vector_collection/vectors")},
        ),
    )
    selected_plots = protocol.select_plot_outputs(allowed={"four_state_vector_heatmap"})
    plots = tuple(
        _step(
            id="four_state_vector_heatmap",
            plugin="plot/four_state_vector_collection",
            reads={"vectors": RecordInputDecl(record_id="four_state_vector_collection/vectors")},
            with_=protocol.plot_view_config(figure_id="four_state_vector_heatmap"),
        )
        for _ in selected_plots
    )
    selected_exports = protocol.select_export_outputs(defaults=("vector_table",), allowed={"vector_table"})
    exports = tuple(
        _step(
            id="vector_table",
            plugin="export/csv",
            reads={"df": RecordInputDecl(record_id="four_state_vector_collection/vectors")},
            with_=_deep_merge(
                {"path": "four_state_vector_collection.csv"},
                protocol.export_artifact_config(artifact_id="vector_table"),
            ),
        )
        for _ in selected_exports
    )
    return CompiledProtocolPlan(
        pipeline=pipeline,
        plots=plots,
        exports=exports,
        semantic_program=protocol.descriptor.semantic_program(),
    )


def compile_logic_four_state_vector_screen(protocol: Any):
    analysis = _analysis_options(protocol)
    include_fold_change = _configured_fold_change_enabled(protocol, analysis=analysis)
    include_four_state_vector = _analysis_bool(analysis, key="include_four_state_vector", default=True)
    preprocessing = _analysis_mapping(analysis, key="preprocessing")
    blank_cfg = _analysis_mapping(preprocessing, key="blank")
    overflow_cfg = _analysis_mapping(preprocessing, key="overflow")

    ingest_channels = _configured_ingest_channels(protocol, required=("OD600", "CFP", "YFP"))
    pipeline = list(
        compose_dual_reporter_pipeline(
            ingest_channels=ingest_channels,
            blank_config=blank_cfg,
            overflow_config=overflow_cfg,
        )
    )
    if include_fold_change:
        _configured_fold_change_target(protocol, expected=FOUR_STATE_VECTOR_SCREEN_LOGIC_CHANNEL)
        _configured_fold_change_report_times(protocol)
        pipeline.append(_plate_reader_fold_change_step(measurement="yfp_cfp"))
    default_exports = (
        ("logic_summary_workbook",)
        if include_four_state_vector
        and _analysis_bool(analysis, key="include_export", default=include_four_state_vector)
        else ()
    )
    selected_exports = protocol.select_export_outputs(
        defaults=default_exports,
        allowed=LOGIC_EXPORT_OUTPUTS,
    )
    selected_plot_ids = protocol.select_plot_outputs(
        allowed={
            "raw_kinetics",
            "endpoint_by_condition",
            "endpoint_by_design",
            "intensity_overview",
            "logic_symmetry",
            "four_state_vector_diagnostic",
            "four_state_vector_heatmap",
        },
    )
    requires_four_state_vector = (
        include_four_state_vector
        or "four_state_vector_diagnostic" in selected_plot_ids
        or "four_state_vector_heatmap" in selected_plot_ids
        or "logic_summary_workbook" in selected_exports
    )
    requires_logic_symmetry = "logic_symmetry" in selected_plot_ids
    requires_promoted_df = requires_four_state_vector or requires_logic_symmetry
    if requires_promoted_df:
        pipeline.append(_four_state_vector_promote_step())
    if requires_logic_symmetry:
        pipeline.append(_logic_symmetry_step(protocol))
    if requires_four_state_vector:
        pipeline.append(_four_state_vector_step(protocol))

    plots = [_logic_plot_output(protocol, output_id=output_id) for output_id in selected_plot_ids]
    exports = [_logic_export_output(protocol, output_id=output_id) for output_id in selected_exports]

    return CompiledProtocolPlan(
        pipeline=tuple(pipeline),
        plots=tuple(plots),
        exports=tuple(exports),
        semantic_program=_logic_semantic_program(protocol, include_four_state_vector=requires_four_state_vector),
    )


def _four_state_vector_promote_step() -> PluginStepDecl:
    return _step(
        id="promote_to_tidy_plus_map",
        plugin="validator/to_tidy_plus_map",
        reads={"df": RecordInputDecl(record_id="ratio_yfp_od600/df")},
        writes={"df": RecordOutputDecl(record_id="promote_to_tidy_plus_map/df")},
    )


def _four_state_vector_step(protocol: Any) -> PluginStepDecl:
    return _step(
        id="four_state_vector",
        plugin="transform/four_state_vector",
        reads={"df": RecordInputDecl(record_id="promote_to_tidy_plus_map/df")},
        with_={"log2_offset_delta": _four_state_vector_delta(protocol)},
        writes={"vector": RecordOutputDecl(record_id="four_state_vector/vector")},
    )


def _logic_symmetry_step(protocol: Any) -> PluginStepDecl:
    inputs = protocol.effective_inputs()
    settings = _analysis_mapping(protocol.effective_analysis(), key="logic_symmetry")
    return _step(
        id="logic_symmetry_summary",
        plugin="transform/logic_symmetry",
        reads={"df": RecordInputDecl(record_id="promote_to_tidy_plus_map/df")},
        with_=_deep_merge(
            {
                "response_channel": FOUR_STATE_VECTOR_SCREEN_LOGIC_CHANNEL,
                "design_by": inputs.get("design_by", ["design_id"]),
                "state_map_ref": inputs.get("state_map_ref", "induction_logic"),
            },
            settings,
        ),
        writes={"table": RecordOutputDecl(record_id="logic_symmetry/table")},
    )


def _four_state_vector_delta(protocol: Any) -> float:
    settings = _analysis_mapping(_analysis_options(protocol), key="four_state_vector")
    return float(settings.get("intensity_log2_offset_delta", 0.0))


def _four_state_vector_heatmap_defaults(protocol: Any) -> dict[str, Any]:
    return dict(_analysis_mapping(_analysis_options(protocol), key="four_state_vector_heatmap"))


def _logic_plot_output(protocol: Any, *, output_id: str) -> PluginStepDecl:
    settings = protocol.plot_view_config(figure_id=output_id)
    if output_id == "four_state_vector_diagnostic":
        reserved = {"growth_channel", "response_channel", "state_map_ref", "time_column"}
        overridden = sorted(reserved.intersection(settings))
        if overridden:
            raise ConfigError(
                "protocol.outputs.plots.views.four_state_vector_diagnostic cannot override compiler-owned settings: "
                + ", ".join(overridden)
            )
        inputs = protocol.effective_inputs()
        return _step(
            id="four_state_vector_diagnostic",
            plugin="plot/four_state_vector_diagnostic",
            reads={
                "df": RecordInputDecl(record_id="promote_to_tidy_plus_map/df"),
                "vector": RecordInputDecl(record_id="four_state_vector/vector"),
            },
            with_=_deep_merge(
                {
                    "growth_channel": "OD600",
                    "response_channel": FOUR_STATE_VECTOR_SCREEN_LOGIC_CHANNEL,
                    "state_map_ref": inputs.get("state_map_ref", "induction_logic"),
                    "time_column": inputs.get("time_column", "time"),
                },
                settings,
            ),
        )
    if output_id == "four_state_vector_heatmap":
        return _step(
            id="four_state_vector_heatmap",
            plugin="plot/four_state_vector_heatmap",
            reads={"vector": RecordInputDecl(record_id="four_state_vector/vector")},
            with_=_deep_merge(_four_state_vector_heatmap_defaults(protocol), settings),
        )
    if output_id == "logic_symmetry":
        return _step(
            id="logic_symmetry",
            plugin="plot/logic_symmetry",
            reads={"table": RecordInputDecl(record_id="logic_symmetry/table")},
            with_=settings,
        )
    return _plate_reader_plot_output(protocol, output_id=output_id, measurement="yfp_cfp")


def _logic_export_output(protocol: Any, *, output_id: str) -> PluginStepDecl:
    if output_id == "logic_summary_workbook":
        settings = protocol.export_artifact_config(artifact_id=output_id)
        return _step(
            id="logic_summary_workbook",
            plugin="export/xlsx",
            reads={"df": RecordInputDecl(record_id="four_state_vector/vector")},
            with_=_deep_merge({"path": "four_state_vector/vector.xlsx", "sheet_name": "vector"}, settings),
        )
    raise ConfigError(f"Unknown logic export output {output_id!r}")
