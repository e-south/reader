from __future__ import annotations

from pathlib import Path
from typing import Any

from reader.errors import ConfigError
from reader.workbench.decl import (
    FileInputDecl,
    PluginStepDecl,
    RecordCollectionInputDecl,
    RecordInputDecl,
    RecordOutputDecl,
    ResourceInputDecl,
    WorkbenchDecl,
)
from reader.workbench.experiment import ResourceCatalog
from reader.workbench.graph.nodes import PluginStep, Workbench, ensure_unique_workbench_ids
from reader.workbench.graph.refs import (
    FileRef,
    InputRef,
    OutputRef,
    RecordCollectionRef,
    RecordRef,
    ResourceRef,
    SourceRecordRef,
)
from reader.workbench.ontology import WorkbenchPluginStepKind, get_workbench_surface_semantics
from reader.workbench.paths import resolve_path_within_root


def resolve_workbench(decl: WorkbenchDecl) -> Workbench:
    pipeline = tuple(_resolve_plugin_steps(decl, kind="pipeline", specs=decl.pipeline.steps))
    plots = tuple(_resolve_plugin_steps(decl, kind="plot", specs=decl.plots.specs))
    exports = tuple(_resolve_plugin_steps(decl, kind="export", specs=decl.exports.specs))
    ensure_unique_workbench_ids(pipeline, plots, exports)
    return Workbench(pipeline=pipeline, plots=plots, exports=exports)


def materialize_workbench(decl: WorkbenchDecl) -> dict[str, list[dict[str, Any]]]:
    workbench = resolve_workbench(decl)
    return {
        "pipeline": [item.to_dict() for item in workbench.pipeline],
        "plots": [item.to_dict() for item in workbench.plots],
        "exports": [item.to_dict() for item in workbench.exports],
    }


def select_workbench_specs(
    specs: list[PluginStep],
    *,
    only: list[str],
    exclude: list[str],
    kind_label: str,
) -> list[PluginStep]:
    ids = [item.id for item in specs]
    available = set(ids)
    if only:
        only_ids = set(only)
        missing = sorted(only_ids - available)
        if missing:
            raise ConfigError(f"Unknown {kind_label} id(s): {missing}.")
        selected = [item for item in specs if item.id in only_ids]
    else:
        selected = list(specs)
    if exclude:
        exclude_ids = set(exclude)
        missing = sorted(exclude_ids - available)
        if missing:
            raise ConfigError(f"Unknown {kind_label} id(s): {missing}.")
        selected = [item for item in selected if item.id not in exclude_ids]
    return selected


def normalize_input_binding(
    binding: Any,
    *,
    root: Path,
    resources: ResourceCatalog,
    section: str,
    step_id: str,
    key: str,
) -> InputRef:
    if isinstance(binding, RecordInputDecl):
        return RecordRef(record_id=binding.record_id)
    if isinstance(binding, FileInputDecl):
        try:
            path = resolve_path_within_root(binding.path, root=root)
        except ValueError as err:
            raise ConfigError(
                f"{section} {step_id}: reads '{key}' must stay under the experiment root after resolving symlinks."
            ) from err
        return FileRef(path=path)
    if isinstance(binding, ResourceInputDecl):
        try:
            resource = resources.require_file(binding.resource_id)
        except ValueError as err:
            raise ConfigError(
                f"{section} {step_id}: reads '{key}' references unknown resource '{binding.resource_id}'."
            ) from err
        return ResourceRef(resource_id=binding.resource_id, path=resource.path)
    if isinstance(binding, RecordCollectionInputDecl):
        records: list[SourceRecordRef] = []
        for resource_id in binding.resource_ids:
            try:
                resource = resources.require_record(resource_id)
            except ValueError as err:
                raise ConfigError(
                    f"{section} {step_id}: reads '{key}' references unknown record resource '{resource_id}'."
                ) from err
            records.append(
                SourceRecordRef(
                    resource_id=resource_id,
                    experiment_id=resource.experiment_id,
                    record_id=resource.record_id,
                    experiment_root=resource.experiment_root,
                    outputs_dir=resource.outputs_dir,
                )
            )
        return RecordCollectionRef(records=tuple(records))
    raise ConfigError(f"{section} {step_id}: reads '{key}' uses unsupported binding type {type(binding).__name__}")


def normalize_output_binding(binding: RecordOutputDecl) -> OutputRef:
    return OutputRef(record_id=binding.record_id)


def _resolve_plugin_steps(
    decl: WorkbenchDecl,
    *,
    kind: WorkbenchPluginStepKind,
    specs: tuple[PluginStepDecl, ...],
) -> list[PluginStep]:
    semantics = get_workbench_surface_semantics(kind)
    root = decl.experiment.root
    resources = decl.experiment_semantics.resources
    seen: set[str] = set()
    resolved: list[PluginStep] = []
    for step in specs or ():
        step_id = step.id
        if not step_id or not isinstance(step_id, str):
            raise ConfigError(f"Every {semantics.section} spec must include an id.")
        if step_id in seen:
            raise ConfigError(f"{semantics.section} contains duplicate spec id(s): {step_id}")
        seen.add(step_id)
        plugin = step.plugin
        if not plugin or not isinstance(plugin, str):
            raise ConfigError(f"{semantics.section} {step_id}: plugin must be a non-empty string")
        if "/" not in plugin:
            raise ConfigError(f"{semantics.section} {step_id}: plugin must be 'category/key'")
        category = plugin.split("/", 1)[0]
        expected_category = semantics.expected_plugin_category
        if kind == "pipeline" and category in {"plot", "export"}:
            raise ConfigError(f"pipeline {step_id}: plot/export plugins are not allowed in pipeline.")
        if expected_category is not None and category != expected_category:
            raise ConfigError(f"{semantics.section} {step_id}: plugin must be {expected_category}/*")
        resolved.append(
            PluginStep(
                kind=kind,
                id=step_id,
                plugin=plugin,
                reads={
                    key: normalize_input_binding(
                        value,
                        root=root,
                        resources=resources,
                        section=semantics.section,
                        step_id=step_id,
                        key=key,
                    )
                    for key, value in (step.reads or {}).items()
                },
                with_=dict(step.with_ or {}),
                writes={key: normalize_output_binding(value) for key, value in (step.writes or {}).items()},
                source_recipe=step.source_recipe,
            )
        )
    return resolved
