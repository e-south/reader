from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich import box
from rich.panel import Panel
from rich.table import Table

from reader.errors import ConfigError
from reader.runtime import builtin_runtime
from reader.workbench.inspection.protocols import (
    protocol_artifacts_table,
    protocol_descriptor_payload,
    protocol_example_config,
    protocol_example_document,
    protocol_pipeline_table,
    protocol_plot_outputs_table,
    protocol_plot_profiles_table,
    protocol_surface_impl_table,
    protocol_surface_rows,
    protocol_surface_table,
)
from reader.workbench.inspection.runtime import binding_display, export_output_summaries, plot_output_summaries
from reader.workbench.inspection.semantics import (
    semantic_program_table,
)

from . import shared
from .helpers import default_protocol_plan
from .shared import app, emit_json, normalize_output_format, table


@app.command(help="List built-in protocols or describe one.")
def protocols(
    name: str | None = typer.Argument(
        None,
        metavar="[NAME]",
        help="Optional protocol id to describe (e.g., plate_reader/dual_reporter_screen).",
    ),
    domain: str | None = typer.Option(None, "--domain", metavar="NAME", help="Filter protocols by semantic domain."),
    family: str | None = typer.Option(None, "--family", metavar="NAME", help="Filter protocols by semantic family."),
    example_config: bool = typer.Option(
        False,
        "--example-config",
        help="Print a starter YAML outline for the named protocol.",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
):
    runtime = builtin_runtime()
    try:
        fmt = normalize_output_format(format)
        if example_config and not name:
            raise typer.BadParameter("--example-config requires a protocol name.")
        if name:
            descriptor = runtime.protocols.resolve(name)
            if fmt == "json":
                emit_json(protocol_descriptor_payload(descriptor, runtime=runtime))
                return
            bound_protocol, compiled_plan = default_protocol_plan(descriptor=descriptor, runtime=runtime)
            semantic_program = compiled_plan.semantic_program or descriptor.semantic_program()
            summary = table(f"Protocol: {descriptor.protocol}")
            summary.add_column("Section", style="accent")
            summary.add_column("Details")
            summary.add_row("Domain", descriptor.domain)
            summary.add_row("Family", descriptor.family)
            summary.add_row("Summary", descriptor.summary)
            if descriptor.tags:
                summary.add_row("Tags", ", ".join(descriptor.tags))
            if descriptor.semantic_profiles:
                summary.add_row("Profiles", ", ".join(profile.id for profile in descriptor.semantic_profiles))
            if descriptor.factors:
                summary.add_row("Factors", ", ".join(f"{item.name} ({item.role})" for item in descriptor.factors))
            if descriptor.windows:
                summary.add_row("Windows", ", ".join(item.id for item in descriptor.windows))
            if descriptor.metrics:
                summary.add_row("Metrics", ", ".join(item.id for item in descriptor.metrics))
            if descriptor.ranking is not None:
                summary.add_row("Primary ranking", descriptor.ranking.primary_metric)
            summary.add_row(
                "Semantic nodes",
                str(
                    len(semantic_program.controls)
                    + len(semantic_program.windows)
                    + len(semantic_program.metrics)
                    + (1 if semantic_program.ranking is not None else 0)
                ),
            )
            summary.add_row("Default notebook", descriptor.execution.notebook.default_template)
            summary.add_row("Allowed notebooks", ", ".join(descriptor.execution.notebook.allowed_templates))
            if descriptor.default_plot_profile is not None:
                summary.add_row("Default plot profile", descriptor.default_plot_profile)
            shared.console.print(Panel(summary, border_style="accent", box=box.ROUNDED))

            input_rows = protocol_surface_rows(descriptor.input_fields)
            if input_rows:
                shared.console.print(
                    Panel(protocol_surface_table("Inputs Surface", input_rows), border_style="accent", box=box.ROUNDED)
                )
            analysis_rows = protocol_surface_rows(descriptor.analysis_fields)
            if analysis_rows:
                shared.console.print(
                    Panel(
                        protocol_surface_table("Analysis Surface", analysis_rows),
                        border_style="accent",
                        box=box.ROUNDED,
                    )
                )
            if (
                semantic_program.controls
                or semantic_program.windows
                or semantic_program.metrics
                or semantic_program.ranking
            ):
                shared.console.print(
                    Panel(semantic_program_table(semantic_program), border_style="accent", box=box.ROUNDED)
                )
            if descriptor.plot_profiles:
                shared.console.print(
                    Panel(protocol_plot_profiles_table(descriptor), border_style="accent", box=box.ROUNDED)
                )
            if descriptor.figures:
                shared.console.print(
                    Panel(protocol_plot_outputs_table(descriptor), border_style="accent", box=box.ROUNDED)
                )
            if descriptor.artifacts:
                shared.console.print(
                    Panel(protocol_artifacts_table(descriptor), border_style="accent", box=box.ROUNDED)
                )
            if compiled_plan.pipeline:
                shared.console.print(
                    Panel(protocol_pipeline_table(compiled_plan.pipeline), border_style="accent", box=box.ROUNDED)
                )
            if compiled_plan.plots:
                shared.console.print(
                    Panel(
                        protocol_surface_impl_table(
                            "Plot Implementations",
                            compiled_plan.plots,
                            plot_output_summaries(bound_protocol),
                            binding_display=binding_display,
                        ),
                        border_style="accent",
                        box=box.ROUNDED,
                    )
                )
            if compiled_plan.exports:
                shared.console.print(
                    Panel(
                        protocol_surface_impl_table(
                            "Export Implementations",
                            compiled_plan.exports,
                            export_output_summaries(bound_protocol),
                            binding_display=binding_display,
                        ),
                        border_style="accent",
                        box=box.ROUNDED,
                    )
                )
            if example_config:
                shared.console.print(
                    Panel(
                        protocol_example_config(descriptor),
                        title="Starter YAML",
                        border_style="accent",
                        box=box.ROUNDED,
                    )
                )
            return

        if fmt == "json":
            emit_json(
                {
                    "protocols": [
                        {
                            "protocol": descriptor.protocol,
                            "domain": descriptor.domain,
                            "family": descriptor.family,
                            "summary": descriptor.summary,
                            "tags": list(descriptor.tags),
                        }
                        for descriptor in runtime.protocols.all()
                        if (not domain or descriptor.domain == domain) and (not family or descriptor.family == family)
                    ]
                }
            )
            return

        listing = table("Protocols")
        listing.add_column("Name", style="accent", min_width=24)
        listing.add_column("Domain")
        listing.add_column("Family")
        listing.add_column("Description")
        for descriptor in runtime.protocols.all():
            if domain and descriptor.domain != domain:
                continue
            if family and descriptor.family != family:
                continue
            listing.add_row(descriptor.protocol, descriptor.domain, descriptor.family, descriptor.summary)
        shared.console.print(Panel(listing, border_style="accent", box=box.ROUNDED))
    except ConfigError as err:
        raise typer.BadParameter(str(err)) from err


@app.command(help="Scaffold a new experiment directory from a protocol starter config.")
def init(
    target: str = typer.Argument(..., metavar="DIR", help="Experiment directory to create."),
    protocol: str = typer.Option(
        ...,
        "--protocol",
        "-p",
        metavar="ID",
        help="Protocol id to bind, for example: plate_reader/dual_reporter_screen.",
    ),
    title: str | None = typer.Option(None, "--title", metavar="TEXT", help="Optional human-readable experiment title."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite config.yaml when the target directory already contains one.",
    ),
):
    runtime = builtin_runtime()
    try:
        descriptor = runtime.protocols.resolve(protocol)
    except ConfigError as err:
        raise typer.BadParameter(str(err)) from err
    target_dir = Path(target).expanduser()
    if target_dir.suffix:
        raise typer.BadParameter("init expects a directory path, not a file path")
    target_dir = target_dir.resolve()
    config_path = target_dir / "config.yaml"
    if config_path.exists() and not force:
        shared.abort(f"{config_path} already exists. Pass --force to overwrite it.")
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "inputs").mkdir(exist_ok=True)
    (target_dir / "notebooks").mkdir(exist_ok=True)
    example_document = protocol_example_document(descriptor)
    experiment_block = dict(example_document["experiment"])
    experiment_block["id"] = target_dir.name
    if title:
        experiment_block["title"] = title
    example_document["experiment"] = experiment_block
    config_path.write_text(yaml.safe_dump(example_document, sort_keys=False), encoding="utf-8")

    summary = Table(box=box.ROUNDED, expand=True, show_header=False)
    summary.add_column("Field", style="accent", no_wrap=True)
    summary.add_column("Value")
    summary.add_row("Protocol", descriptor.protocol)
    summary.add_row("Experiment", experiment_block["id"])
    summary.add_row("Config", str(config_path))
    summary.add_row("Inputs dir", str(target_dir / "inputs"))
    summary.add_row("Notebook dir", str(target_dir / "notebooks"))
    shared.console.print(Panel(summary, title="Experiment scaffolded", border_style="accent", box=box.ROUNDED))
    shared.console.print(
        Panel(
            "\n".join(
                [
                    f"reader inspect {config_path}",
                    f"reader validate {config_path} --no-files",
                    f"reader protocols {descriptor.protocol}",
                ]
            ),
            title="Next steps",
            border_style="accent",
            box=box.ROUNDED,
        )
    )
