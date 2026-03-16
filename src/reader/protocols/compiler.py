from __future__ import annotations

from copy import deepcopy
from typing import Any

from reader.errors import ConfigError
from reader.protocols.model import CompiledProtocolPlan
from reader.workbench.decl.model import (
    NotebookTemplateCallDecl,
    PluginStepDecl,
    RecipeSourceDecl,
    RecordInputDecl,
    RecordOutputDecl,
    ResourceInputDecl,
)
from reader.workbench.recipes.registry import resolve_recipe_steps

PLATE_READER_PLOT_DELIVERABLES = {
    "time_series",
    "snapshot_by_channel",
    "snapshot_by_design",
    "snapshot_state",
    "ts_and_snap_intensity",
    "ts_and_snap_ratio",
    "distributions",
    "snapshot_heatmap_yfp_cfp",
    "snapshot_heatmap_cfp_od600",
    "logic_symmetry_yfp_cfp",
}
PLATE_READER_EXPORT_DELIVERABLES = {"crosstalk_pairs_csv"}
LOGIC_EXPORT_DELIVERABLES = {"vec8_xlsx"}


def compile_generic_protocol(protocol: Any):
    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    return CompiledProtocolPlan(
        pipeline=(),
        plots=(),
        exports=(),
        notebooks=(default_notebook_call(template),),
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

    plot_profile = _deliverable_profile(
        protocol, surface="plots", default=("rfp_full" if measurement == "rfp_od600" else "yfp_full")
    )
    defaults = _plate_reader_plot_profile_defaults(profile=plot_profile, measurement=measurement)
    selected_plots = protocol.select_deliverables(
        surface="plots",
        defaults=defaults,
        allowed=PLATE_READER_PLOT_DELIVERABLES,
    )
    plots = [
        _plate_reader_plot_deliverable(
            protocol,
            deliverable_id=deliverable_id,
            measurement=measurement,
        )
        for deliverable_id in selected_plots
    ]

    default_exports = ("crosstalk_pairs_csv",) if include_crosstalk_export else ()
    selected_exports = protocol.select_deliverables(
        surface="exports",
        defaults=default_exports,
        allowed=PLATE_READER_EXPORT_DELIVERABLES,
    )
    exports = [
        _plate_reader_export_deliverable(protocol, deliverable_id=deliverable_id) for deliverable_id in selected_exports
    ]

    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    return CompiledProtocolPlan(
        runtime={"strict": strict},
        pipeline=tuple(pipeline),
        plots=tuple(plots),
        exports=tuple(exports),
        notebooks=(default_notebook_call(template),),
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
    if include_vec8:
        pipeline.append(_sfxi_promote_step())
        pipeline.append(_sfxi_vec8_step())

    plot_profile = _deliverable_profile(protocol, surface="plots", default="yfp_full")
    defaults = _logic_plot_profile_defaults(plot_profile)
    plots = [
        _plate_reader_plot_deliverable(protocol, deliverable_id=deliverable_id, measurement="yfp_cfp")
        for deliverable_id in protocol.select_deliverables(
            surface="plots",
            defaults=defaults,
            allowed=PLATE_READER_PLOT_DELIVERABLES,
        )
    ]

    default_exports = (
        ("vec8_xlsx",) if include_vec8 and _analysis_bool(analysis, key="include_export", default=include_vec8) else ()
    )
    selected_exports = protocol.select_deliverables(
        surface="exports",
        defaults=default_exports,
        allowed=LOGIC_EXPORT_DELIVERABLES,
    )
    exports = [
        _logic_export_deliverable(protocol, deliverable_id=deliverable_id) for deliverable_id in selected_exports
    ]

    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
    return CompiledProtocolPlan(
        runtime={"strict": strict},
        pipeline=tuple(pipeline),
        plots=tuple(plots),
        exports=tuple(exports),
        notebooks=(default_notebook_call(template),),
    )


def compile_cytometry_flow_panel(protocol: Any):
    analysis = _analysis_options(protocol)
    strict = _analysis_bool(analysis, key="strict", default=True)
    template = protocol.resolve_notebook_template(configured_template=protocol.configured_notebook_template())
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


def _deliverable_profile(protocol: Any, *, surface: str, default: str) -> str:
    block = (
        getattr(protocol, "deliverables", {}).get(surface, {})
        if isinstance(getattr(protocol, "deliverables", {}), dict)
        else {}
    )
    if not isinstance(block, dict):
        raise ConfigError(f"protocol.deliverables.{surface} for {protocol.id!r} must be a mapping")
    value = block.get("profile", default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"protocol.deliverables.{surface}.profile for {protocol.id!r} must be a non-empty string")
    return value


def _plate_reader_plot_profile_defaults(profile: str, *, measurement: str) -> tuple[str, ...]:
    profiles: dict[str, tuple[str, ...]] = {
        "yfp_full": ("time_series", "snapshot_by_channel", "snapshot_by_design", "ts_and_snap_intensity"),
        "rfp_full": ("time_series", "snapshot_by_channel", "snapshot_by_design", "ts_and_snap_intensity"),
        "yfp_time_series": ("time_series", "distributions"),
        "dual_reporter_core": ("time_series", "snapshot_state", "ts_and_snap_ratio"),
        "none": (),
    }
    if profile not in profiles:
        options = ", ".join(sorted(profiles))
        raise ConfigError(f"protocol.deliverables.plots.profile must be one of: {options}")
    defaults = profiles[profile]
    if measurement == "rfp_od600":
        defaults = tuple("snapshot_state" if item == "snapshot_state" else item for item in defaults)
    return defaults


def _logic_plot_profile_defaults(profile: str) -> tuple[str, ...]:
    profiles: dict[str, tuple[str, ...]] = {
        "yfp_full": ("time_series", "snapshot_by_channel", "snapshot_by_design", "ts_and_snap_intensity"),
        "none": (),
    }
    if profile not in profiles:
        options = ", ".join(sorted(profiles))
        raise ConfigError(f"protocol.deliverables.plots.profile must be one of: {options}")
    return profiles[profile]


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


def _plate_reader_plot_deliverable(protocol: Any, *, deliverable_id: str, measurement: str) -> PluginStepDecl:
    settings = protocol.deliverable_settings(surface="plots", deliverable_id=deliverable_id)
    plot_reads = _plate_reader_plot_reads(measurement=measurement)
    if deliverable_id == "time_series":
        defaults = {
            "partition": {"by": "design_id"},
            "hue": "treatment",
            "y": (
                ["OD600", "YFP", "YFP/CFP", "YFP/OD600"] if measurement == "yfp_cfp" else ["OD600", "RFP", "RFP/OD600"]
            ),
            "add_sheet_line": True,
        }
        return _step(
            id="time_series",
            plugin="plot/time_series",
            reads={"df": plot_reads["df"], "blanks": plot_reads["blanks"]},
            with_=_deep_merge(defaults, settings),
        )
    if deliverable_id == "snapshot_by_channel":
        defaults = {
            "x": "treatment",
            "y": (["OD600", "YFP/OD600"] if measurement == "yfp_cfp" else ["OD600", "RFP/OD600"]),
            "partition": {"by": "design_id"},
            "time": 14.0,
        }
        return _step(
            id="snapshot_by_channel",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if deliverable_id == "snapshot_by_design":
        defaults = {
            "x": "design_id",
            "y": ("YFP/OD600" if measurement == "yfp_cfp" else "RFP/OD600"),
            "hue": "treatment",
            "time": 14.0,
        }
        return _step(
            id="snapshot_by_design",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if deliverable_id == "snapshot_state":
        defaults = {
            "x": "treatment_alias",
            "y": ["OD600", "CFP/OD600", "YFP/OD600", "YFP/CFP"],
            "partition": {"by": "design_id_alias"},
            "time": 14.0,
        }
        return _step(
            id="snapshot_state",
            plugin="plot/snapshot_barplot",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if deliverable_id == "ts_and_snap_intensity":
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
            step_id = "ts_and_snap_intensity"
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
            step_id = "ts_and_snap_intensity"
        return _step(
            id=step_id,
            plugin="plot/ts_and_snap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if deliverable_id == "ts_and_snap_ratio":
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
            id="ts_and_snap_ratio",
            plugin="plot/ts_and_snap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if deliverable_id == "distributions":
        defaults = {"channels": ["YFP/CFP"], "partition": {"by": "design_id"}}
        return _step(
            id="distributions",
            plugin="plot/distributions",
            reads={"df": plot_reads["df"], "blanks": plot_reads["blanks"]},
            with_=_deep_merge(defaults, settings),
        )
    if deliverable_id == "snapshot_heatmap_yfp_cfp":
        defaults = {
            "channel": "YFP/CFP",
            "time": 14.0,
            "x": "treatment_alias",
            "y": "design_id_alias",
        }
        return _step(
            id="snapshot_heatmap_yfp_cfp",
            plugin="plot/snapshot_heatmap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if deliverable_id == "snapshot_heatmap_cfp_od600":
        defaults = {
            "channel": "CFP/OD600",
            "time": 14.0,
            "x": "treatment_alias",
            "y": "design_id_alias",
        }
        return _step(
            id="snapshot_heatmap_cfp_od600",
            plugin="plot/snapshot_heatmap",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    if deliverable_id == "logic_symmetry_yfp_cfp":
        defaults = {"response_channel": "YFP/CFP"}
        return _step(
            id="logic_symmetry_yfp_cfp",
            plugin="plot/logic_symmetry",
            reads={"df": plot_reads["df"]},
            with_=_deep_merge(defaults, settings),
        )
    raise ConfigError(f"Unknown plate-reader plot deliverable {deliverable_id!r}")


def _plate_reader_export_deliverable(protocol: Any, *, deliverable_id: str) -> PluginStepDecl:
    if deliverable_id == "crosstalk_pairs_csv":
        defaults = {"path": "crosstalk_pairs.csv"}
        settings = protocol.deliverable_settings(surface="exports", deliverable_id=deliverable_id)
        return _step(
            id="crosstalk_pairs_csv",
            plugin="export/csv",
            reads={"df": RecordInputDecl(record_id="crosstalk_pairs/table")},
            with_=_deep_merge(defaults, settings),
        )
    raise ConfigError(f"Unknown plate-reader export deliverable {deliverable_id!r}")


def _logic_export_deliverable(protocol: Any, *, deliverable_id: str) -> PluginStepDecl:
    if deliverable_id == "vec8_xlsx":
        settings = protocol.deliverable_settings(surface="exports", deliverable_id=deliverable_id)
        return _step(
            id="vec8_xlsx",
            plugin="export/xlsx",
            reads={"df": RecordInputDecl(record_id="sfxi_vec8/vec8")},
            with_=_deep_merge({"path": "sfxi/vec8.xlsx", "sheet_name": "vec8"}, settings),
        )
    raise ConfigError(f"Unknown logic export deliverable {deliverable_id!r}")


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
