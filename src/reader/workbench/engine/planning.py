from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from reader.plotting.mpl import ensure_mpl_cache_dir
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.decl import WorkbenchDecl
from reader.workbench.graph import (
    OutputRef,
    ensure_unique_workbench_ids,
    input_ref_display,
    output_ref_display,
    resolve_workbench,
)
from reader.workbench.templates import require_notebook_template_for_protocol, resolve_notebook_template_descriptor

from ._shared import collect_categories


def _plan_table(steps: list[Any], registry: Any, *, title: str) -> Table:
    table = Table(
        title=title,
        title_justify="left",
        title_style="bold cyan",
        header_style="bold",
        box=box.ROUNDED,
        expand=True,
        show_lines=False,
    )
    table.add_column("#", justify="right", style="muted")
    table.add_column("Step ID", style="accent")
    table.add_column("Plugin")
    table.add_column("Type")
    table.add_column("Inputs")
    table.add_column("Outputs")
    for index, step in enumerate(steps, 1):
        descriptor = registry.resolve_descriptor(step.plugin)
        plugin_cls = descriptor.cls
        input_lines: list[str] = []
        for name, port in plugin_cls.input_ports().items():
            suffix = ", optional" if port.optional else ""
            bound_ref = (step.reads or {}).get(name)
            if bound_ref is not None:
                input_lines.append(f"{name} <- {input_ref_display(bound_ref)} ({port.render()}{suffix})")
            else:
                input_lines.append(f"{name} ({port.render()}{suffix})")

        output_lines: list[str] = []
        for out_name, port in plugin_cls.output_ports().items():
            if port.kind == "dataframe":
                label_ref = (step.writes or {}).get(out_name, OutputRef(record_id=f"{step.id}/{out_name}"))
                output_lines.append(f"{output_ref_display(label_ref)} ({port.render()})")
                continue
            if title.lower().startswith("plot"):
                output_lines.append(f"{out_name} ({port.render()} → outputs/plots/)")
                continue
            if title.lower().startswith("export"):
                output_lines.append(f"{out_name} ({port.render()} → outputs/exports/)")
                continue
            output_lines.append(f"{out_name} ({port.render()})")

        table.add_row(
            str(index),
            step.id,
            step.plugin,
            f"{descriptor.domain}/{descriptor.family}",
            "\n".join(input_lines) if input_lines else "—",
            "\n".join(output_lines) if output_lines else "—",
        )
    return table


def _notebook_table(steps: list[Any], *, title: str) -> Table:
    table = Table(
        title=title,
        title_justify="left",
        title_style="bold cyan",
        header_style="bold",
        box=box.ROUNDED,
        expand=True,
        show_lines=False,
    )
    table.add_column("#", justify="right", style="muted")
    table.add_column("Spec ID", style="accent")
    table.add_column("Template")
    table.add_column("Type")
    table.add_column("Config")
    for index, step in enumerate(steps, 1):
        descriptor = resolve_notebook_template_descriptor(step.template)
        table.add_row(
            str(index),
            step.id,
            step.template,
            descriptor.family,
            "—",
        )
    return table


def build_next_steps(
    decl: WorkbenchDecl,
    *,
    job_label: str | None = None,
    runtime: ReaderRuntime | None = None,
) -> list[tuple[str, str]]:
    runtime = runtime or builtin_runtime()
    label = (job_label or "").strip()

    def _cmd(base: str, tail: str = "") -> str:
        return f"{base} {label}{tail}" if label else f"{base}{tail}"

    steps: list[tuple[str, str]] = []
    workbench = resolve_workbench(decl)
    plot_specs = list(workbench.plots)
    export_specs = list(workbench.exports)
    notebook_specs = list(workbench.notebooks)
    bound_protocol = runtime.bind_protocol(decl.experiment_semantics.protocol)
    notebook_template = bound_protocol.resolve_notebook_template(
        configured_template=(notebook_specs[0].template if notebook_specs else None)
    )
    require_notebook_template_for_protocol(notebook_template, protocol=bound_protocol)
    steps.append((_cmd("reader records"), "Review generated workbench records (QC)"))
    if plot_specs:
        steps.append((_cmd("reader plot"), "Save plot files to outputs/plots"))
    if export_specs:
        steps.append((_cmd("reader export"), "Write export files to outputs/exports"))
    steps.append((_cmd("reader notebook"), f"Open a notebook (template {notebook_template})"))
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
    notebook_specs = list(workbench.notebooks)
    ensure_unique_workbench_ids(pipeline_steps, plot_specs, export_specs, notebook_specs)
    categories = collect_categories(list(workbench.plugin_steps()))
    if "plot" in categories:
        ensure_mpl_cache_dir()
    registry = registry or (runtime.plugins if categories else None)
    summary = Table(box=box.ROUNDED, expand=True, show_header=False)
    summary.add_column("Field", style="accent", no_wrap=True)
    summary.add_column("Value")
    summary.add_row("Protocol", bound_protocol.id)
    summary.add_row("Input sections", ", ".join(sorted(bound_protocol.inputs)) if bound_protocol.inputs else "—")
    summary.add_row("Analysis knobs", ", ".join(sorted(bound_protocol.analysis)) if bound_protocol.analysis else "—")
    summary.add_row(
        "Pipeline flow",
        " -> ".join(step.id for step in pipeline_steps) if pipeline_steps else "—",
    )
    summary.add_row("Plots", ", ".join(step.id for step in plot_specs) if plot_specs else "—")
    summary.add_row("Exports", ", ".join(step.id for step in export_specs) if export_specs else "—")
    summary.add_row("Notebooks", ", ".join(step.template for step in notebook_specs) if notebook_specs else "—")
    resources = tuple(decl.experiment_semantics.resources.by_id.keys())
    if resources:
        summary.add_row("Resources", ", ".join(resources))
    console.print(Panel(summary, border_style="cyan", box=box.ROUNDED, title="Protocol plan"))
    if pipeline_steps:
        if registry is None:
            raise RuntimeError("pipeline explanation requires a plugin registry")
        pipeline = _plan_table(pipeline_steps, registry, title="Pipeline")
        console.print(
            Panel(
                pipeline,
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(pipeline_steps)} steps", style="dim"),
            )
        )
    if plot_specs:
        if registry is None:
            raise RuntimeError("plot explanation requires a plugin registry")
        plots_table = _plan_table(plot_specs, registry, title="Plots")
        console.print(
            Panel(
                plots_table,
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(plot_specs)} specs", style="dim"),
            )
        )
    if export_specs:
        if registry is None:
            raise RuntimeError("export explanation requires a plugin registry")
        exports_table = _plan_table(export_specs, registry, title="Exports")
        console.print(
            Panel(
                exports_table,
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(export_specs)} specs", style="dim"),
            )
        )
    if notebook_specs:
        notebooks_table = _notebook_table(notebook_specs, title="Notebooks")
        console.print(
            Panel(
                notebooks_table,
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(notebook_specs)} specs", style="dim"),
            )
        )
