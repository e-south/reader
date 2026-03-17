from __future__ import annotations

from pathlib import Path

from reader.errors import ConfigError
from reader.protocols.model import ProtocolBinding, ProtocolCatalog
from reader.workbench.config import ProtocolBindingSpec, ReaderSpec, ResourceSpec
from reader.workbench.experiment import (
    AnnotationCollections,
    AnnotationCollectionSpec,
    AnnotationLabels,
    AnnotationLabelSpec,
    AnnotationOrders,
    AnnotationOrderSpec,
    AnnotationSemantics,
    ExperimentSemantics,
    LogicMaps,
    LogicMapSpec,
    OutputLayout,
    ResourceCatalog,
    ResourceEntry,
)

from .model import ExperimentDecl, NotebookDecl, PipelineDecl, SurfaceDecl, WorkbenchDecl


def load_workbench_decl(path: Path, *, protocols: ProtocolCatalog) -> WorkbenchDecl:
    spec = ReaderSpec.load(path)
    return build_workbench_decl(spec, source_path=path, protocols=protocols)


def build_workbench_decl(
    spec: ReaderSpec,
    *,
    source_path: Path,
    protocols: ProtocolCatalog,
) -> WorkbenchDecl:
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
    protocol_binding = _bind_protocol(spec.protocol)
    bound_protocol = protocols.bind(protocol_binding)
    compiled = bound_protocol.compile()

    experiment_semantics = ExperimentSemantics(
        protocol=protocol_binding,
        annotations=AnnotationSemantics(
            labels=AnnotationLabels(
                by_id={
                    key: AnnotationLabelSpec(
                        source=value.source,
                        values=dict(value.values or {}),
                        output=value.output,
                    )
                    for key, value in (spec.annotations.labels or {}).items()
                }
            ),
            orders=AnnotationOrders(
                by_id={
                    key: AnnotationOrderSpec(column=value.column, values=list(value.values or []))
                    for key, value in (spec.annotations.orders or {}).items()
                }
            ),
            collections=AnnotationCollections(
                by_id={
                    key: AnnotationCollectionSpec(
                        column=value.column,
                        items={item_key: list(item_values) for item_key, item_values in (value.items or {}).items()},
                    )
                    for key, value in (spec.annotations.collections or {}).items()
                }
            ),
            logic_maps=LogicMaps(
                by_id={
                    key: LogicMapSpec(
                        column=value.column,
                        corners=dict(value.corners),
                        case_sensitive=bool(value.case_sensitive),
                    )
                    for key, value in (spec.annotations.logic_maps or {}).items()
                }
            ),
        ),
        resources=resources,
        layout=layout,
        protocol_program=compiled.semantic_program,
    )

    return WorkbenchDecl(
        experiment=experiment,
        experiment_semantics=experiment_semantics,
        plotting_palette=spec.plotting.palette if spec.plotting else None,
        pipeline=PipelineDecl(runtime=dict(compiled.runtime or {}), steps=tuple(compiled.pipeline)),
        plots=SurfaceDecl(specs=tuple(compiled.plots)),
        exports=SurfaceDecl(specs=tuple(compiled.exports)),
        notebooks=NotebookDecl(specs=tuple(compiled.notebooks)),
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


def _bind_protocol(spec: ProtocolBindingSpec) -> ProtocolBinding:
    return ProtocolBinding(
        id=spec.id,
        inputs=dict(spec.inputs or {}),
        analysis=dict(spec.analysis or {}),
        outputs=spec.outputs.model_dump(exclude_none=True),
    )
