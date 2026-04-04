from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from reader.runtime import ReaderRuntime
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl
from reader.workbench.engine.setup import slice_pipeline_steps
from reader.workbench.graph import resolve_workbench

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
            "program": semantic_program_payload(semantic_program) if semantic_program is not None else None,
        },
        "implementation": deepcopy(implementation),
    }


def experiment_identity_payload(
    *,
    job_path: Path,
    decl: WorkbenchDecl,
    protocol_id: str | None = None,
) -> dict[str, object]:
    return {
        "id": decl.experiment.id,
        "title": decl.experiment.title,
        "lifecycle": decl.experiment.lifecycle,
        "protocol": protocol_id or decl.experiment_semantics.protocol.id,
        "config": str(job_path),
        "root": str(decl.experiment.root),
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
            "notebook_template": spec.protocol.outputs.notebook.template or bound_protocol.default_notebook_template,
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
    notebook_steps = list(workbench.notebooks)
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    return experiment_surface_payload(
        experiment=experiment_payload,
        authoring=authoring_payload,
        semantic_program=decl.experiment_semantics.protocol_program,
        implementation=experiment_implementation_payload(
            plan=implementation_plan_payload(
                bound_protocol=bound_protocol,
                decl=decl,
                pipeline_steps=pipeline_steps,
                plot_steps=plot_steps,
                export_steps=export_steps,
                notebook_steps=notebook_steps,
            ),
            compiled=compiled_workbench_payload(
                bound_protocol=bound_protocol,
                pipeline_steps=pipeline_steps,
                plot_steps=plot_steps,
                export_steps=export_steps,
                notebook_steps=notebook_steps,
                runtime=runtime,
                record_producers=record_producers,
            ),
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
        notebook_steps=[],
    )
    plan_payload["pipeline_count"] = len(pipeline_steps)
    return experiment_surface_payload(
        experiment=experiment_payload,
        authoring=authoring_payload,
        semantic_program=decl.experiment_semantics.protocol_program,
        implementation=experiment_implementation_payload(
            plan=plan_payload,
            compiled={
                "pipeline": [
                    pipeline_step_payload(step, runtime=runtime, record_producers=record_producers)
                    for step in pipeline_steps
                ],
                "plots": [],
                "exports": [],
                "notebooks": [],
            },
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
    notebook_steps = list(workbench.notebooks)
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    return experiment_surface_payload(
        experiment=experiment_payload,
        authoring=experiment_config_authoring_payload(document=spec.model_dump(by_alias=True)),
        semantic_program=decl.experiment_semantics.protocol_program,
        implementation=experiment_implementation_payload(
            plan=implementation_plan_payload(
                bound_protocol=bound_protocol,
                decl=decl,
                pipeline_steps=pipeline_steps,
                plot_steps=plot_steps,
                export_steps=export_steps,
                notebook_steps=notebook_steps,
            ),
            compiled=compiled_workbench_payload(
                bound_protocol=bound_protocol,
                pipeline_steps=pipeline_steps,
                plot_steps=plot_steps,
                export_steps=export_steps,
                notebook_steps=notebook_steps,
                runtime=runtime,
                record_producers=record_producers,
            ),
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
        (resource_id, format_relative_path(entry.path, base=exp_root))
        for resource_id, entry in sorted(decl.experiment_semantics.resources.by_id.items())
    ]
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    records_payload = (
        record_entries_payload(store=store, outputs_dir=outputs_dir, base=exp_root) if store.catalog_exists() else []
    )
    pipeline_steps = list(workbench.pipeline)
    plot_steps = list(workbench.plots)
    export_steps = list(workbench.exports)
    notebook_steps = list(workbench.notebooks)
    return experiment_surface_payload(
        experiment=experiment_payload,
        authoring=authoring_payload,
        semantic_program=decl.experiment_semantics.protocol_program,
        implementation=experiment_implementation_payload(
            plan=implementation_plan_payload(
                bound_protocol=bound_protocol,
                decl=decl,
                pipeline_steps=pipeline_steps,
                plot_steps=plot_steps,
                export_steps=export_steps,
                notebook_steps=notebook_steps,
            ),
            compiled=compiled_workbench_payload(
                bound_protocol=bound_protocol,
                pipeline_steps=pipeline_steps,
                plot_steps=plot_steps,
                export_steps=export_steps,
                notebook_steps=notebook_steps,
                runtime=runtime,
                record_producers=record_producers,
            ),
            inputs={
                "counts": {
                    "files": input_file_count,
                    "resources": len(resource_rows),
                },
                "files": input_files,
                "resources": [
                    {
                        "id": resource_id,
                        "path": path_text,
                    }
                    for resource_id, path_text in resource_rows
                ],
            },
            generated={
                "counts": output_counts,
                "examples": {
                    "records": preview_output_files(artifacts_dir, base=exp_root),
                    "plots": preview_output_files(plots_dir, base=exp_root),
                    "exports": preview_output_files(exports_dir, base=exp_root),
                    "notebooks": preview_output_files(notebooks_dir, base=exp_root),
                },
                "records": records_payload,
            },
            readiness=experiment_readiness_payload(job_path=job_path, decl=decl, runtime=runtime),
        ),
    )
