from __future__ import annotations

import math
from typing import Any

from reader.errors import ConfigError
from reader.protocols.model import CompiledProtocolPlan
from reader.protocols.semantic_coverage import (
    _plate_reader_semantic_program,
    _plate_reader_single_reporter_semantic_program,
)
from reader.workbench.decl.model import (
    PluginStepDecl,
    RecordCollectionInputDecl,
    RecordInputDecl,
    RecordOutputDecl,
)

from .common import (
    _analysis_bool,
    _analysis_channel,
    _analysis_mapping,
    _analysis_options,
    _config_bool,
    _deep_merge,
    _input_mapping,
    _step,
)
from .plate_reader_pipeline import compose_dual_reporter_pipeline, compose_single_reporter_pipeline

PLATE_READER_EXPORT_OUTPUTS = {"crosstalk_pairs_table"}


def compile_plate_reader_response_window(protocol: Any):
    input_names = ("response_records", "magnitude_records", "trajectory_records")
    effective_inputs = protocol.effective_inputs()
    analysis = protocol.effective_analysis()
    resource_ids = {name: tuple(effective_inputs.get(name, ())) for name in input_names}
    writes = {
        name: RecordOutputDecl(record_id=f"response_window/{name}")
        for name in ("wells", "designs", "bootstrap_draws", "traces", "events")
    }
    pipeline = (
        _step(
            id="response_window",
            plugin="transform/response_window",
            reads={name: RecordCollectionInputDecl(resource_ids=resource_ids[name]) for name in input_names},
            writes=writes,
            with_=analysis,
        ),
    )
    primary_reduction = _primary_reduction_config(analysis)
    selected_plots = protocol.select_plot_outputs(allowed={"response_window_summary", "response_window_diagnostic"})
    plots = tuple(
        _response_window_plot_output(
            protocol,
            output_id=output_id,
            primary_reduction=primary_reduction,
        )
        for output_id in selected_plots
    )
    selected_exports = protocol.select_export_outputs(
        defaults=("designs_table", "events_table"),
        allowed={"designs_table", "events_table"},
    )
    export_records = {
        "designs_table": ("response_window/designs", "response_window_designs.csv"),
        "events_table": ("response_window/events", "response_window_events.csv"),
    }
    exports = tuple(
        _step(
            id=artifact_id,
            plugin="export/csv",
            reads={"df": RecordInputDecl(record_id=export_records[artifact_id][0])},
            with_=_deep_merge(
                {"path": export_records[artifact_id][1]},
                protocol.export_artifact_config(artifact_id=artifact_id),
            ),
        )
        for artifact_id in selected_exports
    )
    return CompiledProtocolPlan(
        pipeline=pipeline,
        plots=plots,
        exports=exports,
        semantic_program=protocol.descriptor.semantic_program(),
    )


def _response_window_plot_output(
    protocol: Any,
    *,
    output_id: str,
    primary_reduction: dict[str, Any],
) -> PluginStepDecl:
    settings = protocol.plot_view_config(figure_id=output_id)
    compiler_owned = {"primary_reduction_id", "pre_window_duration_h"}
    overridden = sorted(compiler_owned.intersection(settings))
    if overridden:
        raise ConfigError(
            f"protocol.outputs.plots.views.{output_id} cannot override compiler-owned fields {overridden}; "
            "mark exactly one protocol.analysis.reductions entry as primary instead"
        )
    primary_reduction_id = str(primary_reduction["id"])
    if output_id == "response_window_summary":
        return _step(
            id=output_id,
            plugin="plot/response_window_summary",
            reads={"designs": RecordInputDecl(record_id="response_window/designs")},
            with_=_deep_merge({"primary_reduction_id": primary_reduction_id}, settings),
        )
    if output_id == "response_window_diagnostic":
        for key in ("source_experiment_id", "design_id"):
            value = settings.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"protocol.outputs.plots.views.{output_id}.{key} must be a non-empty string")
        return _step(
            id=output_id,
            plugin="plot/response_window_diagnostic",
            reads={
                "designs": RecordInputDecl(record_id="response_window/designs"),
                "traces": RecordInputDecl(record_id="response_window/traces"),
            },
            with_=_deep_merge(
                {
                    "primary_reduction_id": primary_reduction_id,
                    "pre_window_duration_h": primary_reduction.get("pre_window_duration_h"),
                },
                settings,
            ),
        )
    raise ConfigError(f"Unsupported response-window plot output {output_id!r}")


def _primary_reduction_config(analysis: dict[str, Any]) -> dict[str, Any]:
    reductions = analysis.get("reductions", ())
    primary = [item for item in reductions if isinstance(item, dict) and item.get("role") == "primary"]
    if len(primary) != 1 or not isinstance(primary[0].get("id"), str) or not primary[0]["id"].strip():
        raise ConfigError("protocol.analysis.reductions must declare exactly one primary reduction id")
    result = dict(primary[0])
    result["id"] = result["id"].strip()
    return result


def compile_plate_reader_dual_reporter_screen(protocol: Any):
    analysis = _analysis_options(protocol)
    include_fold_change = _configured_fold_change_enabled(protocol, analysis=analysis)
    preprocessing = _analysis_mapping(analysis, key="preprocessing")
    blank_cfg = _analysis_mapping(preprocessing, key="blank")
    overflow_cfg = _analysis_mapping(preprocessing, key="overflow")
    crosstalk_cfg = _analysis_mapping(analysis, key="crosstalk_pairs")
    include_crosstalk_pairs = _config_bool(crosstalk_cfg, key="enabled", default=False)
    include_crosstalk_export = _config_bool(crosstalk_cfg, key="export", default=include_crosstalk_pairs)
    ingest_channels = _configured_ingest_channels(protocol, required=("OD600", "CFP", "YFP"))
    pipeline = list(
        compose_dual_reporter_pipeline(
            ingest_channels=ingest_channels,
            blank_config=blank_cfg,
            overflow_config=overflow_cfg,
        )
    )
    if include_fold_change:
        _configured_fold_change_target(protocol, expected="YFP/CFP")
        _configured_fold_change_report_times(protocol)
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

    return CompiledProtocolPlan(
        pipeline=tuple(pipeline),
        plots=tuple(plots),
        exports=tuple(exports),
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
    include_fold_change = _configured_fold_change_enabled(protocol, analysis=analysis)
    preprocessing = _analysis_mapping(analysis, key="preprocessing")
    blank_cfg = _analysis_mapping(preprocessing, key="blank")
    overflow_cfg = _analysis_mapping(preprocessing, key="overflow")
    ingest_channels = _configured_ingest_channels(protocol, required=(normalizer_channel, reporter_channel))
    ratio_label = _single_reporter_ratio_label(
        reporter_channel=reporter_channel,
        normalizer_channel=normalizer_channel,
    )

    pipeline = list(
        compose_single_reporter_pipeline(
            ingest_channels=ingest_channels,
            reporter_channel=reporter_channel,
            normalizer_channel=normalizer_channel,
            blank_config=blank_cfg,
            overflow_config=overflow_cfg,
        )
    )
    if include_fold_change:
        _configured_fold_change_target(protocol, expected=ratio_label)
        _configured_fold_change_report_times(protocol)
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

    return CompiledProtocolPlan(
        pipeline=tuple(pipeline),
        plots=tuple(plots),
        exports=(),
        semantic_program=_plate_reader_single_reporter_semantic_program(
            protocol,
            reporter_channel=reporter_channel,
            normalizer_channel=normalizer_channel,
            include_fold_change=include_fold_change,
        ),
    )


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


def _configured_fold_change_enabled(protocol: Any, *, analysis: dict[str, Any]) -> bool:
    if "include_fold_change" in analysis:
        return _analysis_bool(analysis, key="include_fold_change", default=False)
    fold_change = _input_mapping(protocol, key="fold_change")
    return "report_times" in fold_change


def _configured_fold_change_report_times(protocol: Any) -> tuple[float, ...]:
    fold_change = _input_mapping(protocol, key="fold_change")
    configured = fold_change.get("report_times")
    where = f"protocol.inputs.fold_change.report_times for {protocol.id!r}"
    if not isinstance(configured, list) or not configured:
        raise ConfigError(f"{where} must be an explicit non-empty list of finite hours")
    report_times: list[float] = []
    for value in configured:
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ConfigError(f"{where} must contain only finite numbers")
        report_times.append(float(value))
    return tuple(report_times)


def _require_plot_time(*, settings: dict[str, Any], output_id: str, key: str) -> float:
    where = f"protocol.outputs.plots.views.{output_id}.{key}"
    if key not in settings:
        raise ConfigError(f"{where} must be explicit")
    value = settings[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ConfigError(f"{where} must be a finite number")
    return float(value)


def _plate_reader_plot_output_ids(*, measurement: str) -> set[str]:
    if measurement != "yfp_cfp":
        raise ConfigError(f"Unsupported plate-reader measurement {measurement!r}")
    return {
        "dual_reporter_triptych",
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
        "subject_comparison",
        "value_distributions",
    }


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


def _plate_reader_plot_output(protocol: Any, *, output_id: str, measurement: str) -> PluginStepDecl:
    settings = protocol.plot_view_config(figure_id=output_id)
    plot_reads = _plate_reader_plot_reads(measurement=measurement)
    if output_id == "dual_reporter_triptych":
        if "snapshot_time_h" not in settings:
            raise ConfigError("protocol.outputs.plots.views.dual_reporter_triptych.snapshot_time_h must be explicit")
        snapshot_time_h = settings["snapshot_time_h"]
        if (
            isinstance(snapshot_time_h, bool)
            or not isinstance(snapshot_time_h, int | float)
            or not math.isfinite(float(snapshot_time_h))
        ):
            raise ConfigError(
                "protocol.outputs.plots.views.dual_reporter_triptych.snapshot_time_h must be a finite number"
            )
        defaults = {
            "design_column": "design_id",
            "treatment_column": "treatment",
            "time_column": "time",
            "growth_channel": "OD600",
            "ratio_channel": "YFP/CFP",
            "snapshot_channel": "YFP/CFP",
            "snapshot_time_mode": "nearest",
            "snapshot_time_tolerance_h": 0.51,
        }
        return _step(
            id="dual_reporter_triptych",
            plugin="plot/dual_reporter_triptych",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
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
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "endpoint_by_condition":
        _require_plot_time(settings=settings, output_id=output_id, key="time")
        defaults = {
            "x": "treatment",
            "y": ["OD600", "YFP/OD600"],
            "partition": {"by": "design_id"},
        }
        return _step(
            id="endpoint_by_condition",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "endpoint_by_design":
        _require_plot_time(settings=settings, output_id=output_id, key="time")
        defaults = {
            "x": "design_id",
            "y": "YFP/OD600",
            "hue": "treatment",
        }
        return _step(
            id="endpoint_by_design",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "state_summary":
        _require_plot_time(settings=settings, output_id=output_id, key="time")
        defaults = {
            "x": "treatment_alias",
            "y": ["OD600", "CFP/OD600", "YFP/OD600", "YFP/CFP"],
            "partition": {"by": "design_id_alias"},
        }
        return _step(
            id="state_summary",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "intensity_overview":
        _require_plot_time(settings=settings, output_id=output_id, key="snap_time")
        defaults = {
            "partition": {"by": "design_id"},
            "ts_channel": "YFP/OD600",
            "ts_hue": "treatment",
            "ts_add_sheet_line": True,
            "ts_mark_snap_time": True,
            "snap_channel": "YFP/OD600",
        }
        return _step(
            id="intensity_overview",
            plugin="plot/ts_and_snap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "ratio_overview":
        _require_plot_time(settings=settings, output_id=output_id, key="snap_time")
        defaults = {
            "partition": {"by": "design_id_alias"},
            "ts_channel": "OD600",
            "ts_hue": "treatment_alias",
            "ts_add_sheet_line": True,
            "ts_mark_snap_time": True,
            "snap_x": "treatment_alias",
            "snap_channel": "YFP/CFP",
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
        _require_plot_time(settings=settings, output_id=output_id, key="time")
        defaults = {
            "channel": "YFP/CFP",
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
        _require_plot_time(settings=settings, output_id=output_id, key="time")
        defaults = {
            "channel": "CFP/OD600",
            "x": "treatment_alias",
            "y": "design_id_alias",
        }
        return _step(
            id="support_heatmap",
            plugin="plot/snapshot_heatmap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    raise ConfigError(f"Unknown plate-reader plot output {output_id!r}")


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
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "endpoint_by_condition":
        _require_plot_time(settings=settings, output_id=output_id, key="time")
        defaults = {
            "x": "treatment",
            "y": [normalizer_channel, ratio_label],
            "partition": {"by": "design_id"},
        }
        return _step(
            id="endpoint_by_condition",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "endpoint_by_design":
        _require_plot_time(settings=settings, output_id=output_id, key="time")
        defaults = {
            "x": "design_id",
            "y": ratio_label,
            "hue": "treatment",
        }
        return _step(
            id="endpoint_by_design",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "intensity_overview":
        _require_plot_time(settings=settings, output_id=output_id, key="snap_time")
        defaults = {
            "partition": {"by": "design_id"},
            "ts_channel": ratio_label,
            "ts_hue": "treatment",
            "ts_add_sheet_line": True,
            "ts_mark_snap_time": True,
            "snap_channel": ratio_label,
        }
        return _step(
            id="intensity_overview",
            plugin="plot/ts_and_snap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if output_id == "subject_comparison":
        _require_plot_time(settings=settings, output_id=output_id, key="snap_time")
        defaults = {
            "ts_channel": normalizer_channel,
            "ts_hue": "treatment",
            "ts_style": "design_id",
            "snap_x": "design_id",
            "snap_channel": ratio_label,
            "snap_hue": "treatment",
            "square_panels": True,
            "title": "Subject comparison",
        }
        return _step(
            id="subject_comparison",
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


def _single_reporter_ratio_label(*, reporter_channel: str, normalizer_channel: str) -> str:
    return f"{reporter_channel}/{normalizer_channel}"
