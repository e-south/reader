from __future__ import annotations

from rich.console import Console

from reader.plotting.mpl import ensure_mpl_cache_dir
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.commands import reader_command
from reader.workbench.decl import WorkbenchDecl
from reader.workbench.graph import ensure_unique_workbench_ids, resolve_workbench
from reader.workbench.inspection.reports import workflow_explain_renderables

from ._shared import collect_categories


def build_next_steps(
    decl: WorkbenchDecl,
    *,
    job_label: str | None = None,
    runtime: ReaderRuntime | None = None,
    include_plot: bool = True,
    include_export: bool = True,
) -> list[tuple[str, str]]:
    runtime = runtime or builtin_runtime()
    label = (job_label or "").strip()

    def _cmd(base: str, tail: str = "") -> str:
        return reader_command(base, label, tail)

    steps: list[tuple[str, str]] = []
    workbench = resolve_workbench(decl)
    plot_specs = list(workbench.plots)
    export_specs = list(workbench.exports)
    steps.append((_cmd("records"), "Review generated records"))
    if plot_specs and include_plot:
        steps.append((_cmd("plot"), "Save plot files to outputs/plots"))
    if export_specs and include_export:
        steps.append((_cmd("export"), "Write export files to outputs/exports"))
    steps.append((_cmd("notebook"), "Open the canonical experiment notebook"))
    return steps


def explain(
    decl: WorkbenchDecl,
    *,
    console: Console,
    registry=None,
    plot_specs=None,
    export_specs=None,
    runtime: ReaderRuntime | None = None,
) -> None:
    runtime = runtime or builtin_runtime()
    bound_protocol = runtime.bind_protocol(decl.experiment_semantics.protocol)
    workbench = resolve_workbench(decl)
    pipeline_steps = list(workbench.pipeline)
    plot_specs = list(plot_specs) if plot_specs is not None else list(workbench.plots)
    export_specs = list(export_specs) if export_specs is not None else list(workbench.exports)
    ensure_unique_workbench_ids(pipeline_steps, plot_specs, export_specs)
    categories = collect_categories(list(workbench.plugin_steps()))
    if "plot" in categories:
        ensure_mpl_cache_dir()
    registry = registry or (runtime.plugins if categories else None)
    if (pipeline_steps or plot_specs or export_specs) and registry is None:
        raise RuntimeError("plugin-backed workflow explanation requires a plugin registry")
    for renderable in workflow_explain_renderables(
        bound_protocol=bound_protocol,
        decl=decl,
        pipeline_steps=pipeline_steps,
        plot_specs=plot_specs,
        export_specs=export_specs,
        registry=registry,
    ):
        console.print(renderable)
