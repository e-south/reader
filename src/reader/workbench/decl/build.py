from __future__ import annotations

from pathlib import Path

from reader.errors import ConfigError
from reader.protocols.model import ProtocolBinding, ProtocolCatalog
from reader.workbench.config import (
    FileResourceSpec,
    ProtocolBindingSpec,
    ReaderSpec,
    RecordResourceSpec,
    ResourceSpec,
    reader_spec_digest,
)
from reader.workbench.experiment import (
    AnnotationCollections,
    AnnotationCollectionSpec,
    AnnotationLabels,
    AnnotationLabelSpec,
    AnnotationOrders,
    AnnotationOrderSpec,
    AnnotationSemantics,
    ExperimentEvidence,
    ExperimentSemantics,
    FileResourceEntry,
    OrderedStateSpaces,
    OrderedStateSpaceSpec,
    OutputLayout,
    RecordResourceEntry,
    ResourceCatalog,
)
from reader.workbench.experiments import ExperimentCatalog
from reader.workbench.paths import resolve_path_within_root

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
        lifecycle=spec.experiment.lifecycle,
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
            ordered_state_spaces=OrderedStateSpaces(
                by_id={
                    key: OrderedStateSpaceSpec(
                        column=value.column,
                        state_order=tuple(value.state_order),
                        source_values=dict(value.values),
                        case_sensitive=bool(value.case_sensitive),
                    )
                    for key, value in (spec.annotations.ordered_state_spaces or {}).items()
                }
            ),
        ),
        resources=resources,
        layout=layout,
        protocol_program=compiled.semantic_program,
        evidence=(
            ExperimentEvidence(
                data_class=spec.evidence.data_class,
                data_class_reason=spec.evidence.data_class_reason,
                replicate_kind=spec.evidence.replicate_kind,
                replicate_identity_field=spec.evidence.replicate_identity_field,
            )
            if spec.evidence is not None
            else None
        ),
    )

    return WorkbenchDecl(
        experiment=experiment,
        experiment_semantics=experiment_semantics,
        plotting_palette=spec.plotting.palette if spec.plotting else None,
        pipeline=PipelineDecl(runtime=dict(compiled.runtime or {}), steps=tuple(compiled.pipeline)),
        plots=SurfaceDecl(specs=tuple(compiled.plots)),
        exports=SurfaceDecl(specs=tuple(compiled.exports)),
        notebooks=NotebookDecl(specs=tuple(compiled.notebooks)),
        config_digest=reader_spec_digest(spec),
    )


def _resolve_outputs_dir(raw: str, *, root: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError("paths.outputs must be a non-empty string path")
    try:
        return resolve_path_within_root(raw, root=root)
    except ValueError as err:
        raise ConfigError("paths.outputs must stay under the experiment root after resolving symlinks.") from err


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
    bound: dict[str, FileResourceEntry | RecordResourceEntry] = {}
    experiment_catalog: ExperimentCatalog | None = None
    for resource_id, resource in resources.items():
        if isinstance(resource, FileResourceSpec):
            try:
                path = resolve_path_within_root(resource.path, root=root)
            except ValueError as err:
                raise ConfigError(
                    f"resources.{resource_id}.path must stay under the experiment root after resolving symlinks."
                ) from err
            bound[resource_id] = FileResourceEntry(kind="file", path=path)
            continue
        if isinstance(resource, RecordResourceSpec):
            experiment_catalog = experiment_catalog or ExperimentCatalog.from_experiment_root(root)
            location = experiment_catalog.resolve(resource.experiment)
            bound[resource_id] = RecordResourceEntry(
                kind="record",
                experiment_id=location.id,
                record_id=resource.record,
                experiment_root=location.root,
                outputs_dir=location.outputs_dir,
            )
            continue
        raise ConfigError(f"resources.{resource_id} has unsupported kind {resource.kind!r}")
    return ResourceCatalog(by_id=bound)


def _bind_protocol(spec: ProtocolBindingSpec) -> ProtocolBinding:
    return ProtocolBinding(
        id=spec.id,
        inputs=dict(spec.inputs or {}),
        analysis=dict(spec.analysis or {}),
        outputs=spec.outputs.model_dump(exclude_none=True),
    )
