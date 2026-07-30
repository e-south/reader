from __future__ import annotations

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from reader_workbench.workbench.graph import OutputRef, input_ref_display, output_ref_display

from .common import flatten_binding_rows
from .results import record_payload_detail_text
from .runtime import render_read_binding
from .semantics import semantic_program_table


def experiment_inspect_renderables(
    *,
    payload: dict[str, object],
    semantic_program,
) -> list[Panel]:
    experiment = dict(payload.get("experiment") or {})
    authoring = dict(payload.get("authoring") or {})
    implementation = dict(payload.get("implementation") or {})
    compiled = dict(implementation.get("compiled") or {})
    inputs = dict(implementation.get("inputs") or {})
    generated = dict(implementation.get("generated") or {})
    readiness = dict(implementation.get("readiness") or {})
    generated_counts = dict(generated.get("counts") or {})
    generated_examples = dict(generated.get("examples") or {})
    input_counts = dict(inputs.get("counts") or {})
    input_files = [str(item) for item in (inputs.get("files") or [])]
    resources = [dict(item) for item in (inputs.get("resources") or []) if isinstance(item, dict)]
    records_payload = [dict(item) for item in (generated.get("records") or []) if isinstance(item, dict)]
    authoring_rows: list[tuple[str, str, str]] = []
    for section in ("inputs", "analysis", "outputs"):
        values = authoring.get(section)
        if isinstance(values, dict):
            for path, value in flatten_binding_rows(values):
                authoring_rows.append((section, path, value))

    renderables: list[Panel] = []

    overview = Table(box=box.ROUNDED, expand=True, show_header=False)
    overview.add_column("Field", style="accent", no_wrap=True)
    overview.add_column("Value")
    overview.add_row("Experiment", str(experiment.get("id") or "—"))
    if str(experiment.get("lifecycle") or "active") != "active":
        overview.add_row("Lifecycle", str(experiment.get("lifecycle") or "—"))
    overview.add_row("Protocol", str(experiment.get("protocol") or "—"))
    overview.add_row("Config", str(experiment.get("config") or "—"))
    overview.add_row("Root", str(experiment.get("root") or "—"))
    overview.add_row("Inputs", f"{int(input_counts.get('files') or len(input_files))} file(s) under inputs/")
    overview.add_row(
        "Generated",
        (
            f"{int(generated_counts.get('records') or 0)} records, "
            f"{int(generated_counts.get('plots') or 0)} plots, "
            f"{int(generated_counts.get('exports') or 0)} exports, "
            f"{int(generated_counts.get('notebooks') or 0)} notebooks"
        ),
    )
    overview.add_row("Plot profile", str(experiment.get("plot_profile") or "—"))
    renderables.append(Panel(overview, title="Experiment overview", border_style="accent", box=box.ROUNDED))

    if readiness:
        readiness_table = _table("Readiness")
        readiness_table.add_column("field", style="accent", width=18)
        readiness_table.add_column("value", overflow="fold")
        preflight = dict(readiness.get("preflight") or {})
        capabilities = dict(readiness.get("capabilities") or {})
        next_steps = [dict(item) for item in (readiness.get("next_steps") or []) if isinstance(item, dict)]
        readiness_table.add_row("State", str(readiness.get("summary") or readiness.get("state") or "—"))
        readiness_table.add_row(
            "Preflight",
            (
                f"{preflight.get('status', '—')} • "
                f"files={preflight.get('files', '—')} • "
                f"deps={preflight.get('dependencies', '—')}"
            ),
        )
        readiness_table.add_row(
            "Capabilities",
            ", ".join(f"{key}={'yes' if bool(value) else 'no'}" for key, value in capabilities.items()) or "—",
        )
        errors = [str(item) for item in (readiness.get("errors") or [])]
        readiness_table.add_row("Issues", errors[0] if errors else "—")
        if len(errors) > 1:
            readiness_table.add_row("More issues", f"{len(errors) - 1} more")
        if next_steps:
            readiness_table.add_row(
                "Next",
                "\n".join(f"{item.get('command')} — {item.get('description')}" for item in next_steps[:4]),
            )
        renderables.append(Panel(readiness_table, border_style="accent", box=box.ROUNDED))

    authoring_table = _table("Config values")
    authoring_table.add_column("section", style="accent", width=10)
    authoring_table.add_column("path", overflow="fold")
    authoring_table.add_column("value", overflow="fold")
    if authoring_rows:
        for section, path, value in authoring_rows:
            authoring_table.add_row(section, path, value)
    else:
        authoring_table.add_row("—", "—", "No explicit bindings; protocol defaults only.")
    renderables.append(Panel(authoring_table, border_style="accent", box=box.ROUNDED))

    renderables.append(
        Panel(
            semantic_program_table(semantic_program, include_execution=False),
            border_style="accent",
            box=box.ROUNDED,
        )
    )
    renderables.append(
        Panel(
            semantic_program_table(
                semantic_program,
                title="Compiled Semantic Execution",
            ),
            border_style="accent",
            box=box.ROUNDED,
        )
    )

    filesystem = _table("Inputs + resources")
    filesystem.add_column("kind", style="accent", width=10)
    filesystem.add_column("entry")
    filesystem.add_column("details")
    if input_files:
        for relpath in input_files:
            filesystem.add_row("input", relpath, "detected under inputs/")
        remaining_inputs = int(input_counts.get("files") or len(input_files)) - len(input_files)
        if remaining_inputs > 0:
            filesystem.add_row("input", "…", f"{remaining_inputs} more file(s)")
    else:
        filesystem.add_row("input", "—", "No visible files under inputs/")
    if resources:
        for resource in resources:
            filesystem.add_row("resource", str(resource.get("id") or "—"), str(resource.get("path") or "—"))
    renderables.append(Panel(filesystem, border_style="accent", box=box.ROUNDED))

    generated_table = _table("Generated outputs")
    generated_table.add_column("kind", style="accent", width=10)
    generated_table.add_column("count", justify="right", width=7)
    generated_table.add_column("examples", overflow="fold")
    generated_table.add_row(
        "records", str(generated_counts.get("records") or 0), str(generated_examples.get("records") or "—")
    )
    generated_table.add_row(
        "plots", str(generated_counts.get("plots") or 0), str(generated_examples.get("plots") or "—")
    )
    generated_table.add_row(
        "exports", str(generated_counts.get("exports") or 0), str(generated_examples.get("exports") or "—")
    )
    generated_table.add_row(
        "notebooks",
        str(generated_counts.get("notebooks") or 0),
        str(generated_examples.get("notebooks") or "—"),
    )
    renderables.append(Panel(generated_table, border_style="accent", box=box.ROUNDED))

    records_table = _table("Records")
    records_table.add_column("record", style="accent", overflow="fold")
    records_table.add_column("kind", width=18)
    records_table.add_column("producer", overflow="fold")
    records_table.add_column("detail", overflow="fold")
    if records_payload:
        for record in records_payload:
            records_table.add_row(
                str(record.get("record_id") or "—"),
                str(record.get("kind") or "—"),
                str(record.get("producer_label") or "—"),
                record_payload_detail_text(record),
            )
    else:
        records_table.add_row("—", "—", "—", "No records found under outputs/manifests/records.json.")
    renderables.append(Panel(records_table, border_style="accent", box=box.ROUNDED))

    pipeline_rows = [dict(item) for item in (compiled.get("pipeline") or []) if isinstance(item, dict)]
    pipeline_table = _table("Pipeline chain")
    pipeline_table.add_column("#", justify="right", style="muted")
    pipeline_table.add_column("stage", style="accent")
    pipeline_table.add_column("id", overflow="fold")
    pipeline_table.add_column("plugin", overflow="fold")
    pipeline_table.add_column("from", overflow="fold")
    pipeline_table.add_column("writes", overflow="fold")
    for idx, step in enumerate(pipeline_rows, 1):
        from_refs = ", ".join(render_read_binding(item) for item in (step.get("reads") or [])) or "—"
        writes = (
            ", ".join(
                (
                    f"{item['label']} -> {item['display']}"
                    if isinstance(item, dict) and item.get("kind") == "dataframe"
                    else str(item.get("label") if isinstance(item, dict) else item)
                )
                for item in (step.get("writes") or [])
            )
            or "—"
        )
        pipeline_table.add_row(
            str(idx),
            str(step.get("stage") or "—"),
            str(step.get("id") or "—"),
            str(step.get("plugin") or "—"),
            from_refs,
            writes,
        )
    renderables.append(
        Panel(
            pipeline_table,
            border_style="accent",
            box=box.ROUNDED,
            subtitle=f"[muted]{len(pipeline_rows)} step(s)[/muted]",
        )
    )

    renderables.append(_surface_specs_panel(title="Plot outputs", rows=(compiled.get("plots") or [])))
    renderables.append(_surface_specs_panel(title="Exports", rows=(compiled.get("exports") or [])))

    return renderables


def workflow_explain_renderables(
    *,
    bound_protocol,
    decl,
    pipeline_steps,
    plot_specs,
    export_specs,
    registry,
) -> list[Panel]:
    renderables: list[Panel] = []
    summary = Table(box=box.ROUNDED, expand=True, show_header=False)
    summary.add_column("Field", style="accent", no_wrap=True)
    summary.add_column("Value")
    summary.add_row("Protocol", bound_protocol.id)
    summary.add_row("Input sections", ", ".join(sorted(bound_protocol.inputs)) if bound_protocol.inputs else "—")
    summary.add_row("Analysis knobs", ", ".join(sorted(bound_protocol.analysis)) if bound_protocol.analysis else "—")
    summary.add_row("Pipeline flow", " -> ".join(step.id for step in pipeline_steps) if pipeline_steps else "—")
    summary.add_row("Plots", ", ".join(step.id for step in plot_specs) if plot_specs else "—")
    summary.add_row("Exports", ", ".join(step.id for step in export_specs) if export_specs else "—")
    resources = tuple(decl.experiment_semantics.resources.by_id.keys())
    if resources:
        summary.add_row("Resources", ", ".join(resources))
    renderables.append(Panel(summary, border_style="cyan", box=box.ROUNDED, title="Plan summary"))

    semantic_program = decl.experiment_semantics.protocol_program
    renderables.append(
        Panel(
            semantic_program_table(semantic_program, include_execution=False),
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    renderables.append(
        Panel(
            semantic_program_table(semantic_program, title="Compiled Semantic Execution"),
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    if pipeline_steps:
        renderables.append(
            Panel(
                _workflow_plan_table(pipeline_steps, registry, title="Pipeline"),
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(pipeline_steps)} steps", style="dim"),
            )
        )
    if plot_specs:
        renderables.append(
            Panel(
                _workflow_plan_table(plot_specs, registry, title="Plots"),
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(plot_specs)} specs", style="dim"),
            )
        )
    if export_specs:
        renderables.append(
            Panel(
                _workflow_plan_table(export_specs, registry, title="Exports"),
                border_style="cyan",
                box=box.ROUNDED,
                subtitle=Text(f"{len(export_specs)} specs", style="dim"),
            )
        )
    return renderables


def _workflow_plan_table(steps, registry, *, title: str) -> Table:
    table = _table(title)
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


def _surface_specs_panel(*, title: str, rows) -> Panel:
    table = _table(title)
    table.add_column("#", justify="right", style="muted")
    table.add_column("id", style="accent", overflow="fold")
    table.add_column("summary", overflow="fold")
    table.add_column("from", overflow="fold")
    payload_rows = [dict(item) for item in rows if isinstance(item, dict)]
    if payload_rows:
        for idx, item in enumerate(payload_rows, 1):
            from_refs = ", ".join(render_read_binding(read) for read in (item.get("reads") or [])) or "—"
            table.add_row(str(idx), str(item.get("id") or "—"), str(item.get("summary") or "—"), from_refs)
    else:
        empty = "No plot outputs selected." if title == "Plot outputs" else "No exports selected."
        table.add_row("—", "—", empty, "—")
    return Panel(table, border_style="accent", box=box.ROUNDED)


def _table(title: str) -> Table:
    return Table(
        title=title,
        title_justify="left",
        title_style="bold cyan",
        header_style="bold",
        box=box.ROUNDED,
        expand=True,
        show_lines=False,
    )
