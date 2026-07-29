from __future__ import annotations

from copy import deepcopy
from typing import Any

from reader.errors import ConfigError
from reader.protocols.model import CompiledProtocolPlan
from reader.protocols.semantic_coverage import _logic_semantic_program
from reader.workbench.decl.model import PluginStepDecl, RecordInputDecl, RecordOutputDecl, ResourceInputDecl

from .common import (
    _analysis_bool,
    _analysis_mapping,
    _analysis_options,
    _deep_merge,
    _step,
    default_notebook_call,
)
from .plate_reader import (
    _configured_ingest_channels,
    _plate_reader_base_steps,
    _plate_reader_fold_change_step,
    _plate_reader_plot_output,
)

LOGIC_EXPORT_OUTPUTS = {"logic_summary_workbook"}


def compile_logic_sfxi_screen(protocol: Any):
    analysis = _analysis_options(protocol)
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
    default_exports = (
        ("logic_summary_workbook",)
        if include_vec8 and _analysis_bool(analysis, key="include_export", default=include_vec8)
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
            "sfxi_setpoint_scatter",
            "sfxi_triptych_sequence",
            "sfxi_vec8_heatmap",
        },
    )
    requires_vec8 = (
        include_vec8
        or bool({"sfxi_setpoint_scatter", "sfxi_triptych_sequence", "sfxi_vec8_heatmap"} & set(selected_plot_ids))
        or "logic_summary_workbook" in selected_exports
    )
    requires_promoted_df = requires_vec8 or bool({"logic_symmetry", "sfxi_triptych_sequence"} & set(selected_plot_ids))
    if requires_promoted_df:
        pipeline.append(_sfxi_promote_step())
    if requires_vec8:
        pipeline.append(_sfxi_vec8_step(protocol))

    plots = [_logic_plot_output(protocol, output_id=output_id) for output_id in selected_plot_ids]
    exports = [_logic_export_output(protocol, output_id=output_id) for output_id in selected_exports]

    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    return CompiledProtocolPlan(
        pipeline=tuple(pipeline),
        plots=tuple(plots),
        exports=tuple(exports),
        notebooks=(default_notebook_call(template),),
        semantic_program=_logic_semantic_program(protocol, include_vec8=requires_vec8),
    )


def _sfxi_promote_step() -> PluginStepDecl:
    return _step(
        id="promote_to_tidy_plus_map",
        plugin="validator/to_tidy_plus_map",
        reads={"df": RecordInputDecl(record_id="ratio_yfp_od600/df")},
        writes={"df": RecordOutputDecl(record_id="promote_to_tidy_plus_map/df")},
    )


def _sfxi_vec8_step(protocol: Any) -> PluginStepDecl:
    return _step(
        id="sfxi_vec8",
        plugin="transform/sfxi",
        reads={"df": RecordInputDecl(record_id="promote_to_tidy_plus_map/df")},
        with_={"log2_offset_delta": _sfxi_objective_delta(protocol)},
        writes={"vec8": RecordOutputDecl(record_id="sfxi_vec8/vec8")},
    )


def _sfxi_objective_delta(protocol: Any) -> float:
    objective = _analysis_mapping(_analysis_options(protocol), key="sfxi_objective")
    return float(objective.get("intensity_log2_offset_delta", 0.0))


def _sfxi_setpoint_scatter_defaults(protocol: Any) -> dict[str, Any]:
    objective = _analysis_mapping(_analysis_options(protocol), key="sfxi_objective")
    scaling = _analysis_mapping(objective, key="scaling")
    exponents = _analysis_mapping(objective, key="exponents")
    return {
        "setpoints": deepcopy(objective.get("setpoints", {"and": [0.0, 0.0, 0.0, 1.0]})),
        "scaling_percentile": int(scaling.get("percentile", 95)),
        "scaling_min_n": int(scaling.get("min_n", 5)),
        "scaling_eps": float(scaling.get("eps", 1.0e-8)),
        "logic_exponent_beta": float(exponents.get("logic_exponent_beta", 1.0)),
        "intensity_exponent_gamma": float(exponents.get("intensity_exponent_gamma", 1.0)),
        "intensity_log2_offset_delta": _sfxi_objective_delta(protocol),
    }


def _sfxi_triptych_sequence_defaults(protocol: Any) -> dict[str, Any]:
    configured = deepcopy(_analysis_mapping(_analysis_options(protocol), key="sfxi_triptych_sequence"))
    duplicated = sorted(
        {"state_map_ref", "treatment_col", "treatment_column", "treatment_map", "treatments"} & set(configured)
    )
    if duplicated:
        raise ConfigError(
            f"protocol.analysis.sfxi_triptych_sequence must not duplicate SFXI treatment identity; remove: {duplicated}"
        )
    sfxi_cfg = protocol.effective_plugin_config(plugin_id="transform/sfxi")
    state_map_ref = sfxi_cfg.get("state_map_ref")
    if not isinstance(state_map_ref, str) or not state_map_ref.strip():
        raise ConfigError("logic/sfxi_screen requires protocol.inputs.state_map_ref to be a non-empty string.")
    configured["state_map_ref"] = state_map_ref.strip()
    return configured


def _sfxi_vec8_heatmap_defaults(protocol: Any) -> dict[str, Any]:
    return deepcopy(_analysis_mapping(_analysis_options(protocol), key="sfxi_vec8_heatmap"))


def _logic_plot_output(protocol: Any, *, output_id: str) -> PluginStepDecl:
    settings = protocol.plot_view_config(figure_id=output_id)
    if output_id == "sfxi_vec8_heatmap":
        return _step(
            id="sfxi_vec8_heatmap",
            plugin="plot/sfxi_vec8_heatmap",
            reads={"vec8": RecordInputDecl(record_id="sfxi_vec8/vec8")},
            with_=_deep_merge(_sfxi_vec8_heatmap_defaults(protocol), settings),
        )
    if output_id == "logic_symmetry":
        return _step(
            id="logic_symmetry",
            plugin="plot/logic_symmetry",
            reads={"df": RecordInputDecl(record_id="promote_to_tidy_plus_map/df")},
            with_=_deep_merge({"response_channel": "YFP/CFP"}, settings),
        )
    if output_id == "sfxi_setpoint_scatter":
        return _step(
            id="sfxi_setpoint_scatter",
            plugin="plot/sfxi_setpoint_scatter",
            reads={"vec8": RecordInputDecl(record_id="sfxi_vec8/vec8")},
            with_=_deep_merge(_sfxi_setpoint_scatter_defaults(protocol), settings),
        )
    if output_id == "sfxi_triptych_sequence":
        duplicated = sorted(
            {"state_map_ref", "treatment_col", "treatment_column", "treatment_map", "treatments"} & set(settings)
        )
        if duplicated:
            raise ConfigError(
                "protocol.outputs.plots.views.sfxi_triptych_sequence must not duplicate SFXI treatment identity; "
                f"remove: {duplicated}"
            )
        triptych_cfg = _deep_merge(_sfxi_triptych_sequence_defaults(protocol), settings)
        candidate_bindings_resource = triptych_cfg.pop("candidate_bindings_resource", None)
        if not isinstance(candidate_bindings_resource, str) or not candidate_bindings_resource.strip():
            raise ConfigError(
                "logic/sfxi_screen sfxi_triptych_sequence requires analysis.sfxi_triptych_sequence."
                "candidate_bindings_resource."
            )
        return _step(
            id="sfxi_triptych_sequence",
            plugin="plot/sfxi_triptych_sequence",
            reads={
                "vec8": RecordInputDecl(record_id="sfxi_vec8/vec8"),
                "assay": RecordInputDecl(record_id="promote_to_tidy_plus_map/df"),
                "candidate_bindings": ResourceInputDecl(resource_id=candidate_bindings_resource.strip()),
            },
            with_=triptych_cfg,
        )
    return _plate_reader_plot_output(protocol, output_id=output_id, measurement="yfp_cfp")


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
