from __future__ import annotations

from pathlib import Path
from typing import Any

from reader.core.errors import ConfigError
from reader.workbench.config import (
    InputBindingSpec,
    OutputBindingSpec,
    PluginStepSpec,
    ReaderSpec,
    RecipeCallSpec,
    ResourceSpec,
    SpecDefaults,
)
from reader.workbench.experiment import (
    AssayCollections,
    AssayCollectionSpec,
    AssayLabels,
    AssayLabelSpec,
    AssayOrders,
    AssayOrderSpec,
    AssaySemantics,
    ExperimentSemantics,
    LogicMaps,
    LogicMapSpec,
    OutputLayout,
    ResourceCatalog,
    ResourceEntry,
)

from .model import (
    ExperimentDecl,
    FileInputDecl,
    NotebookDecl,
    NotebookTemplateCallDecl,
    PipelineDecl,
    PluginStepDecl,
    RecipeCallDecl,
    RecipeSourceDecl,
    RecordInputDecl,
    RecordOutputDecl,
    ResourceInputDecl,
    SpecDefaultsDecl,
    SurfaceDecl,
    WorkbenchDecl,
    ensure_decl_step_override_shape,
)


def load_workbench_decl(path: Path) -> WorkbenchDecl:
    spec = ReaderSpec.load(path)
    return build_workbench_decl(spec, source_path=path)


def build_workbench_decl(spec: ReaderSpec, *, source_path: Path) -> WorkbenchDecl:
    root = source_path.parent.resolve()
    experiment = ExperimentDecl(
        id=spec.experiment.id,
        title=spec.experiment.title or spec.experiment.id,
        root=root,
    )
    layout = OutputLayout(
        outputs_dir=_resolve_outputs_dir(spec.paths.outputs, root=root),
        plots_subdir=_validate_output_subdir(spec.paths.plots, key="plots"),
        exports_subdir=_validate_output_subdir(spec.paths.exports, key="exports"),
        notebooks_subdir=_validate_output_subdir(spec.paths.notebooks, key="notebooks"),
    )
    resources = _bind_resources(spec.resources.by_id or {}, root=root)
    experiment_semantics = ExperimentSemantics(
        assay=AssaySemantics(
            labels=AssayLabels(
                by_id={
                    key: AssayLabelSpec(
                        source=value.source,
                        values=dict(value.values or {}),
                        output=value.output,
                    )
                    for key, value in (spec.assay.labels or {}).items()
                }
            ),
            orders=AssayOrders(
                by_id={
                    key: AssayOrderSpec(column=value.column, values=list(value.values or []))
                    for key, value in (spec.assay.orders or {}).items()
                }
            ),
            collections=AssayCollections(
                by_id={
                    key: AssayCollectionSpec(
                        column=value.column,
                        items={item_key: list(item_values) for item_key, item_values in (value.items or {}).items()},
                    )
                    for key, value in (spec.assay.collections or {}).items()
                }
            ),
            logic_maps=LogicMaps(
                by_id={
                    key: LogicMapSpec(
                        column=value.column,
                        corners=dict(value.corners),
                        case_sensitive=bool(value.case_sensitive),
                    )
                    for key, value in (spec.assay.logic_maps or {}).items()
                }
            ),
        ),
        resources=resources,
        layout=layout,
    )
    pipeline = PipelineDecl(
        recipes=tuple(_bind_recipe_call(item) for item in (spec.pipeline.recipes or [])),
        runtime=dict(spec.pipeline.runtime or {}),
        overrides=_bind_overrides(spec.pipeline.overrides or {}, root=root, resources=resources),
        steps=tuple(_bind_step(item, root=root, resources=resources) for item in (spec.pipeline.steps or [])),
    )
    plots = SurfaceDecl(
        recipes=tuple(_bind_recipe_call(item) for item in (spec.plots.recipes or [])),
        defaults=_bind_defaults(spec.plots.defaults, root=root, resources=resources),
        overrides=_bind_overrides(spec.plots.overrides or {}, root=root, resources=resources),
        specs=tuple(_bind_step(item, root=root, resources=resources) for item in (spec.plots.specs or [])),
    )
    exports = SurfaceDecl(
        recipes=tuple(_bind_recipe_call(item) for item in (spec.exports.recipes or [])),
        defaults=_bind_defaults(spec.exports.defaults, root=root, resources=resources),
        overrides=_bind_overrides(spec.exports.overrides or {}, root=root, resources=resources),
        specs=tuple(_bind_step(item, root=root, resources=resources) for item in (spec.exports.specs or [])),
    )
    notebooks = NotebookDecl(
        specs=tuple(
            NotebookTemplateCallDecl(id=item.id, template=item.template) for item in (spec.notebooks.specs or [])
        )
    )
    for section, overrides in (
        ("pipeline", pipeline.overrides),
        ("plots", plots.overrides),
        ("exports", exports.overrides),
    ):
        if not isinstance(overrides, dict):
            raise ConfigError(f"{section}.overrides must be a mapping")
        for step_id, override in overrides.items():
            ensure_decl_step_override_shape(override=override, where=f"{section}.overrides.{step_id}")
    return WorkbenchDecl(
        experiment=experiment,
        experiment_semantics=experiment_semantics,
        plotting_palette=spec.plotting.palette if spec.plotting else None,
        pipeline=pipeline,
        plots=plots,
        exports=exports,
        notebooks=notebooks,
    )


def _resolve_outputs_dir(raw: str, *, root: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError("paths.outputs must be a non-empty string path")
    path = Path(raw).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _validate_output_subdir(raw: str, *, key: str) -> str:
    if raw is None:
        raise ConfigError(f"paths.{key} must be a string subdirectory (use '.' to flatten).")
    if not isinstance(raw, str):
        raise ConfigError(f"paths.{key} must be a string subdirectory")
    subdir = Path(raw)
    if subdir.is_absolute():
        raise ConfigError(f"paths.{key} must be relative to paths.outputs, not absolute.")
    normalized = Path(".") / subdir
    if any(part == ".." for part in normalized.parts):
        raise ConfigError(f"paths.{key} must stay under paths.outputs and may not escape via '..'.")
    return raw


def _bind_resources(resources: dict[str, ResourceSpec], *, root: Path) -> ResourceCatalog:
    bound: dict[str, ResourceEntry] = {}
    for resource_id, resource in resources.items():
        path = Path(resource.path).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        bound[resource_id] = ResourceEntry(kind=resource.kind, path=path)
    return ResourceCatalog(by_id=bound)


def _bind_defaults(defaults: SpecDefaults, *, root: Path, resources: ResourceCatalog) -> SpecDefaultsDecl:
    return SpecDefaultsDecl(
        reads={
            key: _bind_input(value, root=root, resources=resources) for key, value in (defaults.reads or {}).items()
        },
        with_=dict(defaults.with_ or {}),
    )


def _bind_recipe_call(item: str | RecipeCallSpec) -> RecipeCallDecl:
    if isinstance(item, str):
        return RecipeCallDecl(recipe=item)
    return RecipeCallDecl(recipe=item.recipe, with_=dict(item.with_ or {}))


def _bind_overrides(
    raw_overrides: dict[str, Any],
    *,
    root: Path,
    resources: ResourceCatalog,
) -> dict[str, Any]:
    bound: dict[str, Any] = {}
    for step_id, override in raw_overrides.items():
        if not isinstance(override, dict):
            raise ConfigError(f"override for {step_id!r} must be a mapping")
        payload = dict(override)
        if "reads" in payload:
            reads = payload["reads"]
            if not isinstance(reads, dict):
                raise ConfigError(f"override for {step_id!r}.reads must be a mapping")
            payload["reads"] = {
                key: _bind_input(InputBindingSpec.model_validate(value), root=root, resources=resources)
                for key, value in reads.items()
            }
        if "writes" in payload:
            writes = payload["writes"]
            if not isinstance(writes, dict):
                raise ConfigError(f"override for {step_id!r}.writes must be a mapping")
            payload["writes"] = {
                key: _bind_output(OutputBindingSpec.model_validate(value)) for key, value in writes.items()
            }
        if "with" in payload and not isinstance(payload["with"], dict):
            raise ConfigError(f"override for {step_id!r}.with must be a mapping")
        bound[step_id] = payload
    return bound


def _bind_step(
    item: PluginStepSpec,
    *,
    root: Path,
    resources: ResourceCatalog,
    source_recipe: RecipeSourceDecl | None = None,
) -> PluginStepDecl:
    return PluginStepDecl(
        id=item.id,
        plugin=item.plugin,
        reads={key: _bind_input(value, root=root, resources=resources) for key, value in (item.reads or {}).items()},
        writes={key: _bind_output(value) for key, value in (item.writes or {}).items()},
        with_=dict(item.with_ or {}),
        source_recipe=source_recipe,
    )


def bind_recipe_steps(
    specs: list[PluginStepSpec],
    *,
    root: Path,
    resources: ResourceCatalog,
    recipe: str,
    with_args: dict[str, Any] | None = None,
) -> list[PluginStepDecl]:
    source = RecipeSourceDecl(recipe=recipe, with_=dict(with_args or {}))
    return [_bind_step(item, root=root, resources=resources, source_recipe=source) for item in specs]


def _bind_input(binding: InputBindingSpec, *, root: Path, resources: ResourceCatalog):
    if binding.record is not None:
        return RecordInputDecl(record_id=binding.record)
    if binding.file is not None:
        return FileInputDecl(path=binding.file)
    if binding.resource is not None:
        resource_id = binding.resource
        if resources.get(resource_id) is None:
            raise ConfigError(f"Unknown resource {resource_id!r}. Declare it under resources or use a file binding.")
        return ResourceInputDecl(resource_id=resource_id)
    raise ConfigError("input binding must declare record, file, or resource")


def _bind_output(binding: OutputBindingSpec) -> RecordOutputDecl:
    return RecordOutputDecl(record_id=binding.record)
