from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from reader.core.config import ReaderSpec
from reader.core.mpl import ensure_mpl_cache_dir
from reader.core.notebooks import normalize_notebook_preset, resolve_notebook_template_descriptor
from reader.core.registry import load_entry_points
from reader.core.workbench import ensure_unique_workbench_ids, resolve_workbench

from ._shared import collect_categories, has_cytometry_step


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
        descriptor = registry.resolve_descriptor(step.uses)
        plugin_cls = descriptor.cls
        input_lines: list[str] = []
        for raw_name, contract in plugin_cls.input_contracts().items():
            optional = raw_name.endswith("?")
            name = raw_name[:-1] if optional else raw_name
            contract_label = contract
            target = (step.reads or {}).get(name)
            if contract == "none" and isinstance(target, str) and target.startswith("file:"):
                contract_label = "file"
            suffix = ", optional" if optional else ""
            input_lines.append(f"{name} ({contract_label}{suffix})")

        output_lines: list[str] = []
        output_contracts = plugin_cls.output_contracts()
        contract_surfaces = plugin_cls.output_contract_surfaces()
        for out_name, contract in output_contracts.items():
            if contract == "none" and out_name == "files":
                if title.lower().startswith("plot"):
                    output_lines.append("files → outputs/plots/")
                    continue
                if title.lower().startswith("export"):
                    output_lines.append("files → outputs/exports/")
                    continue
            label = (step.writes or {}).get(out_name, f"{step.id}/{out_name}") if hasattr(step, "writes") else out_name
            surface = contract_surfaces.get(out_name)
            contract_label = surface.render() if surface is not None else contract
            output_lines.append(f"{label} ({contract_label})")

        table.add_row(
            str(index),
            step.id,
            step.uses,
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
        descriptor = resolve_notebook_template_descriptor(step.uses)
        config_lines = [f"{key}={value!r}" for key, value in sorted((step.with_ or {}).items())]
        table.add_row(
            str(index),
            step.id,
            step.uses,
            descriptor.family,
            "\n".join(config_lines) if config_lines else "—",
        )
    return table


def build_next_steps(spec: ReaderSpec, *, job_label: str | None = None) -> list[tuple[str, str]]:
    label = (job_label or "").strip()

    def _cmd(base: str, tail: str = "") -> str:
        return f"{base} {label}{tail}" if label else f"{base}{tail}"

    steps: list[tuple[str, str]] = []
    workbench = resolve_workbench(spec)
    plot_specs = list(workbench.plots)
    export_specs = list(workbench.exports)
    notebook_specs = list(workbench.notebooks)
    notebook_uses = notebook_specs[0].uses if notebook_specs else None
    if not notebook_uses:
        if plot_specs:
            notebook_uses = "notebook/eda"
        elif has_cytometry_step(spec):
            notebook_uses = "notebook/cytometry"
        else:
            notebook_uses = "notebook/basic"
    notebook_uses = normalize_notebook_preset(notebook_uses)
    steps.append((_cmd("reader records"), "Review generated workbench records (QC)"))
    if plot_specs:
        steps.append((_cmd("reader plot"), "Save plot files to outputs/plots"))
    if export_specs:
        steps.append((_cmd("reader export"), "Write export files to outputs/exports"))
    steps.append((_cmd("reader notebook"), f"Open a notebook (template {notebook_uses})"))
    return steps


def explain(
    spec: ReaderSpec,
    *,
    console: Console,
    registry=None,
    plot_specs=None,
    export_specs=None,
) -> None:
    workbench = resolve_workbench(spec)
    pipeline_steps = list(workbench.pipeline)
    plot_specs = list(plot_specs) if plot_specs is not None else list(workbench.plots)
    export_specs = list(export_specs) if export_specs is not None else list(workbench.exports)
    notebook_specs = list(workbench.notebooks)
    ensure_unique_workbench_ids(pipeline_steps, plot_specs, export_specs, notebook_specs)
    categories = collect_categories(list(workbench.plugin_specs()))
    if "plot" in categories:
        ensure_mpl_cache_dir()
    registry = registry or (load_entry_points(categories=categories) if categories else None)
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
