from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from reader_workbench.errors import RecordError
from reader_workbench.runtime import ReaderRuntime
from reader_workbench.workbench.config import ReaderSpec
from reader_workbench.workbench.decl import WorkbenchDecl
from reader_workbench.workbench.engine.setup import slice_pipeline_steps
from reader_workbench.workbench.engine.validation import validation_summary
from reader_workbench.workbench.experiment import FileResourceEntry, RecordResourceEntry, ResourceEntry
from reader_workbench.workbench.graph import resolve_workbench

from .common import (
    count_visible_files,
    format_relative_path,
    preview_output_files,
    resolve_output_subdir,
    summarize_outputs_dir,
    visible_relative_files,
)
from .readiness import experiment_readiness_payload
from .results import record_entries_payload
from .runtime import (
    compiled_workbench_payload,
    implementation_plan_payload,
    pipeline_step_payload,
    record_producer_map,
)
from .semantics import semantic_program_payload

EXPERIMENT_INSPECT_SECTIONS = (
    "identity",
    "authoring",
    "semantics",
    "plan",
    "compiled",
    "inputs",
    "generated",
    "readiness",
)


def _resource_entry_payload(resource_id: str, entry: ResourceEntry, *, base: Path) -> dict[str, str]:
    if isinstance(entry, FileResourceEntry):
        return {
            "id": resource_id,
            "kind": entry.kind,
            "path": format_relative_path(entry.path, base=base),
        }
    if isinstance(entry, RecordResourceEntry):
        return {
            "id": resource_id,
            "kind": entry.kind,
            "experiment": entry.experiment_id,
            "record": entry.record_id,
        }
    raise TypeError(f"Unsupported experiment resource entry: {type(entry).__name__}")


def experiment_authoring_payload(
    *, inputs: dict[str, Any], analysis: dict[str, Any], outputs: dict[str, Any]
) -> dict[str, object]:
    return {
        "inputs": deepcopy(inputs),
        "analysis": deepcopy(analysis),
        "outputs": deepcopy(outputs),
    }


def experiment_config_authoring_payload(*, document: dict[str, Any]) -> dict[str, object]:
    return deepcopy(document)


def experiment_implementation_payload(
    *,
    plan: dict[str, object] | None = None,
    compiled: dict[str, object] | None = None,
    inputs: dict[str, object] | None = None,
    generated: dict[str, object] | None = None,
    readiness: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    if plan is not None:
        payload["plan"] = deepcopy(plan)
    if compiled is not None:
        payload["compiled"] = deepcopy(compiled)
    if inputs is not None:
        payload["inputs"] = deepcopy(inputs)
    if generated is not None:
        payload["generated"] = deepcopy(generated)
    if readiness is not None:
        payload["readiness"] = deepcopy(readiness)
    return payload


def experiment_surface_payload(
    *,
    experiment: dict[str, object],
    authoring: dict[str, object],
    semantic_program,
    implementation: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment": deepcopy(experiment),
        "authoring": deepcopy(authoring),
        "semantics": {
            "program": semantic_program_payload(semantic_program, include_execution=False),
        },
        "implementation": deepcopy(implementation),
    }


def experiment_inspect_section_payload(
    payload: dict[str, object],
    *,
    section: str,
) -> dict[str, object]:
    """Project an inspect document through a stable semantic section name."""

    if section not in EXPERIMENT_INSPECT_SECTIONS:
        raise ValueError(f"Unknown experiment inspect section {section!r}")
    projected = {"experiment": deepcopy(payload["experiment"])}
    if section == "identity":
        return projected
    if section in {"authoring", "semantics"}:
        projected[section] = deepcopy(payload[section])
        return projected
    implementation = payload["implementation"]
    if not isinstance(implementation, dict) or section not in implementation:
        raise ValueError(f"Experiment inspect payload does not contain section {section!r}")
    projected[section] = deepcopy(implementation[section])
    return projected


def experiment_identity_payload(
    *,
    job_path: Path,
    decl: WorkbenchDecl,
    protocol_id: str | None = None,
) -> dict[str, object]:
    evidence = decl.experiment_semantics.evidence
    return {
        "id": decl.experiment.id,
        "title": decl.experiment.title,
        "lifecycle": decl.experiment.lifecycle,
        "protocol": protocol_id or decl.experiment_semantics.protocol.id,
        "config": str(job_path),
        "root": str(decl.experiment.root),
        "evidence": evidence.to_payload() if evidence is not None else None,
    }


def bound_experiment_surface_context(
    *,
    job_path: Path,
    spec: ReaderSpec,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
) -> tuple[object, dict[str, object], dict[str, object]]:
    bound_protocol = runtime.bind_protocol(decl.experiment_semantics.protocol)
    return (
        bound_protocol,
        {
            **experiment_identity_payload(job_path=job_path, decl=decl, protocol_id=bound_protocol.id),
            "plot_profile": spec.protocol.outputs.plots.profile or bound_protocol.default_plot_profile,
        },
        experiment_authoring_payload(
            inputs=spec.protocol.inputs,
            analysis=spec.protocol.analysis,
            outputs=spec.protocol.outputs.model_dump(exclude_none=True),
        ),
    )


def experiment_explain_payload(
    *,
    job_path: Path,
    spec: ReaderSpec,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
) -> dict[str, object]:
    bound_protocol, experiment_payload, authoring_payload = bound_experiment_surface_context(
        job_path=job_path,
        spec=spec,
        decl=decl,
        runtime=runtime,
    )
    workbench = resolve_workbench(decl)
    pipeline_steps = list(workbench.pipeline)
    plot_steps = list(workbench.plots)
    export_steps = list(workbench.exports)
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    semantic_program = decl.experiment_semantics.protocol_program
    compiled_payload = compiled_workbench_payload(
        bound_protocol=bound_protocol,
        pipeline_steps=pipeline_steps,
        plot_steps=plot_steps,
        export_steps=export_steps,
        runtime=runtime,
        record_producers=record_producers,
    )
    compiled_payload["semantic_program"] = semantic_program_payload(semantic_program)
    return experiment_surface_payload(
        experiment=experiment_payload,
        authoring=authoring_payload,
        semantic_program=semantic_program,
        implementation=experiment_implementation_payload(
            plan=implementation_plan_payload(
                bound_protocol=bound_protocol,
                decl=decl,
                pipeline_steps=pipeline_steps,
                plot_steps=plot_steps,
                export_steps=export_steps,
            ),
            compiled=compiled_payload,
        ),
    )


def experiment_run_dry_run_payload(
    *,
    job_path: Path,
    spec: ReaderSpec,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
    resume_from: str | None,
    until: str | None,
    only: str | None = None,
) -> dict[str, object]:
    validation_summary(decl, check_files=False, exp_root=decl.experiment.root, runtime=runtime)
    workbench = resolve_workbench(decl)
    pipeline_steps = slice_pipeline_steps(
        list(workbench.pipeline),
        resume_from=only or resume_from,
        until=only or until,
    )
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    payload = experiment_explain_payload(job_path=job_path, spec=spec, decl=decl, runtime=runtime)
    payload["dry_run"] = True
    payload["slice"] = {
        "from": resume_from,
        "until": until,
        "only": only,
    }
    payload["implementation"]["plan"]["pipeline_flow"] = [step.id for step in pipeline_steps]
    payload["implementation"]["plan"]["plots"] = []
    payload["implementation"]["plan"]["exports"] = []
    payload["implementation"]["compiled"]["pipeline"] = [
        pipeline_step_payload(step, runtime=runtime, record_producers=record_producers) for step in pipeline_steps
    ]
    payload["implementation"]["compiled"]["plots"] = []
    payload["implementation"]["compiled"]["exports"] = []
    return payload


def experiment_steps_payload(
    *,
    job_path: Path,
    spec: ReaderSpec,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
) -> dict[str, object]:
    bound_protocol, experiment_payload, authoring_payload = bound_experiment_surface_context(
        job_path=job_path,
        spec=spec,
        decl=decl,
        runtime=runtime,
    )
    workbench = resolve_workbench(decl)
    pipeline_steps = list(workbench.pipeline)
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    plan_payload = implementation_plan_payload(
        bound_protocol=bound_protocol,
        decl=decl,
        pipeline_steps=pipeline_steps,
        plot_steps=[],
        export_steps=[],
    )
    plan_payload["pipeline_count"] = len(pipeline_steps)
    semantic_program = decl.experiment_semantics.protocol_program
    compiled_payload = {
        "pipeline": [
            pipeline_step_payload(step, runtime=runtime, record_producers=record_producers) for step in pipeline_steps
        ],
        "plots": [],
        "exports": [],
    }
    compiled_payload["semantic_program"] = semantic_program_payload(semantic_program)
    return experiment_surface_payload(
        experiment=experiment_payload,
        authoring=authoring_payload,
        semantic_program=semantic_program,
        implementation=experiment_implementation_payload(
            plan=plan_payload,
            compiled=compiled_payload,
        ),
    )


def experiment_config_json_payload(
    *,
    job_path: Path,
    spec: ReaderSpec,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
) -> dict[str, object]:
    bound_protocol, experiment_payload, _ = bound_experiment_surface_context(
        job_path=job_path,
        spec=spec,
        decl=decl,
        runtime=runtime,
    )
    workbench = resolve_workbench(decl)
    pipeline_steps = list(workbench.pipeline)
    plot_steps = list(workbench.plots)
    export_steps = list(workbench.exports)
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    semantic_program = decl.experiment_semantics.protocol_program
    compiled_payload = compiled_workbench_payload(
        bound_protocol=bound_protocol,
        pipeline_steps=pipeline_steps,
        plot_steps=plot_steps,
        export_steps=export_steps,
        runtime=runtime,
        record_producers=record_producers,
    )
    compiled_payload["semantic_program"] = semantic_program_payload(semantic_program)
    return experiment_surface_payload(
        experiment=experiment_payload,
        authoring=experiment_config_authoring_payload(document=spec.model_dump(by_alias=True)),
        semantic_program=semantic_program,
        implementation=experiment_implementation_payload(
            plan=implementation_plan_payload(
                bound_protocol=bound_protocol,
                decl=decl,
                pipeline_steps=pipeline_steps,
                plot_steps=plot_steps,
                export_steps=export_steps,
            ),
            compiled=compiled_payload,
        ),
    )


def experiment_inspect_payload(
    *,
    job_path: Path,
    spec: ReaderSpec,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
) -> dict[str, object]:
    bound_protocol, experiment_payload, authoring_payload = bound_experiment_surface_context(
        job_path=job_path,
        spec=spec,
        decl=decl,
        runtime=runtime,
    )
    workbench = resolve_workbench(decl)
    exp_root = decl.experiment.root
    inputs_dir = exp_root / "inputs"
    outputs_dir = decl.experiment_semantics.layout.outputs_dir
    output_counts = summarize_outputs_dir(
        outputs_dir,
        plots_subdir=decl.experiment_semantics.layout.plots_subdir,
        exports_subdir=decl.experiment_semantics.layout.exports_subdir,
        notebooks_subdir=decl.experiment_semantics.layout.notebooks_subdir,
    )
    plots_dir = resolve_output_subdir(outputs_dir, decl.experiment_semantics.layout.plots_subdir)
    exports_dir = resolve_output_subdir(outputs_dir, decl.experiment_semantics.layout.exports_subdir)
    notebooks_dir = resolve_output_subdir(outputs_dir, decl.experiment_semantics.layout.notebooks_subdir)
    artifacts_dir = outputs_dir / "artifacts"
    store = runtime.record_store(
        outputs_dir,
        plots_subdir=decl.experiment_semantics.layout.plots_subdir,
        exports_subdir=decl.experiment_semantics.layout.exports_subdir,
        create=False,
    )
    input_files = visible_relative_files(inputs_dir, base=exp_root, limit=8)
    input_file_count = count_visible_files(inputs_dir)
    resource_rows = [
        _resource_entry_payload(resource_id, entry, base=exp_root)
        for resource_id, entry in sorted(decl.experiment_semantics.resources.by_id.items())
    ]
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    records_payload: list[dict[str, object]] = []
    records_error: str | None = None
    if store.catalog_exists():
        try:
            records_payload = record_entries_payload(
                store=store,
                outputs_dir=outputs_dir,
                runtime=runtime,
            )
        except RecordError as exc:
            records_error = str(exc)
    pipeline_steps = list(workbench.pipeline)
    plot_steps = list(workbench.plots)
    export_steps = list(workbench.exports)
    semantic_program = decl.experiment_semantics.protocol_program
    compiled_payload = compiled_workbench_payload(
        bound_protocol=bound_protocol,
        pipeline_steps=pipeline_steps,
        plot_steps=plot_steps,
        export_steps=export_steps,
        runtime=runtime,
        record_producers=record_producers,
    )
    compiled_payload["semantic_program"] = semantic_program_payload(semantic_program)
    readiness_payload = experiment_readiness_payload(job_path=job_path, decl=decl, runtime=runtime)
    generated_payload: dict[str, object] = {
        "counts": output_counts,
        "examples": {
            "records": preview_output_files(artifacts_dir, base=exp_root),
            "plots": preview_output_files(plots_dir, base=exp_root),
            "exports": preview_output_files(exports_dir, base=exp_root),
            "notebooks": preview_output_files(notebooks_dir, base=exp_root),
        },
        "records": records_payload,
    }
    if records_error is not None:
        generated_payload["records_error"] = records_error
    return experiment_surface_payload(
        experiment=experiment_payload,
        authoring=authoring_payload,
        semantic_program=semantic_program,
        implementation=experiment_implementation_payload(
            plan=implementation_plan_payload(
                bound_protocol=bound_protocol,
                decl=decl,
                pipeline_steps=pipeline_steps,
                plot_steps=plot_steps,
                export_steps=export_steps,
            ),
            compiled=compiled_payload,
            inputs={
                "counts": {
                    "files": input_file_count,
                    "resources": len(resource_rows),
                },
                "files": input_files,
                "resources": resource_rows,
            },
            generated=generated_payload,
            readiness=readiness_payload,
        ),
    )
