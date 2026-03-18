from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.panel import Panel

from reader.errors import ConfigError
from reader.runtime import builtin_runtime
from reader.workbench.graph import resolve_workbench
from reader.workbench.notebooks import write_experiment_notebook
from reader.workbench.templates import (
    builtin_notebook_template_catalog,
    compatible_notebook_templates,
    require_notebook_template_for_protocol,
)

from ..spec_overrides import select_surface_specs
from . import shared
from .helpers import (
    bind_decl_protocol,
    default_notebook_name,
    infer_job_path,
    load_job_models,
    next_available_path,
    spec_to_dict,
    template_requirements_satisfied,
)
from .shared import (
    NOTEBOOK_MODE_OPTION,
    NOTEBOOK_PLOT_EXCLUDE_OPTION,
    NOTEBOOK_PLOT_ONLY_OPTION,
    app,
    table,
)


def render_marimo_help(target: Path, *, mode: str, has_fcs: bool) -> None:
    sync_cmd = "uv sync --locked --group notebooks"
    if has_fcs:
        sync_cmd = f"{sync_cmd} --group cytometry"
    marimo_cmd = f"{shared.sys.executable} -m marimo {mode} {target}"
    uvx_cmd = f"uvx marimo {mode} --sandbox {target}"
    shared.console.print(
        Panel.fit(
            "Could not launch marimo automatically.\n\n"
            "Try:\n"
            f"  1) {sync_cmd}\n"
            "     (Note: uv sync removes undeclared packages; include extra groups you use)\n"
            f"  2) {marimo_cmd}\n"
            f"  3) {uvx_cmd}\n\n"
            f"Notebook: [path]{target}[/path]",
            border_style="warn",
            box=box.ROUNDED,
        )
    )


def _launch_marimo(mode: str, target: Path, *, has_fcs: bool) -> None:
    cmd = [shared.sys.executable, "-m", "marimo", mode, str(target)]
    try:
        result = shared.subprocess.run(cmd, check=False)
    except FileNotFoundError:
        render_marimo_help(target, mode=mode, has_fcs=has_fcs)
        raise typer.Exit(code=1) from None
    if result.returncode != 0:
        render_marimo_help(target, mode=mode, has_fcs=has_fcs)
        raise typer.Exit(code=1)


def _scaffold_notebook(
    *,
    job: str | None,
    name: str | None,
    template_name: str | None,
    list_templates: bool,
    overwrite: bool,
    new: bool,
    refresh: bool,
    mode: str,
    plot_only: list[str] | None,
    plot_exclude: list[str] | None,
    scan_records: bool,
) -> None:
    try:
        if list_templates:
            runtime = builtin_runtime()
            bound_protocol = None
            title = "Notebook templates"
            descriptors = builtin_notebook_template_catalog().all()
            if job is not None:
                job_path = infer_job_path(job)
                _, decl = load_job_models(job_path)
                bound_protocol = bind_decl_protocol(decl=decl, runtime=runtime)
                title = f"Notebook templates: {bound_protocol.id}"
                descriptors = compatible_notebook_templates(protocol=bound_protocol)
            listing = table(title)
            listing.add_column("Name", style="accent")
            listing.add_column("Domain")
            listing.add_column("Family")
            if bound_protocol is not None:
                listing.add_column("Default", justify="center")
            listing.add_column("Description")
            for descriptor in descriptors:
                row = [descriptor.template, descriptor.domain, descriptor.family]
                if bound_protocol is not None:
                    row.append("yes" if descriptor.template == bound_protocol.default_notebook_template else "")
                row.append(descriptor.summary)
                listing.add_row(*row)
            shared.console.print(Panel(listing, border_style="accent", box=box.ROUNDED))
            return
        if overwrite and new:
            raise typer.BadParameter("--overwrite cannot be combined with --new.")
        if refresh:
            overwrite = True
        mode_value = (mode or "").strip().lower()
        if mode_value not in {"edit", "run", "none"}:
            raise typer.BadParameter("--mode must be one of: edit, run, none.")
        job_path = infer_job_path(job)
        exp_dir = job_path.parent
        _, decl = load_job_models(job_path)
        runtime = builtin_runtime()
        bound_protocol = bind_decl_protocol(decl=decl, runtime=runtime)
        workbench = resolve_workbench(decl)
        plot_specs = list(workbench.plots)
        notebook_specs = list(workbench.notebooks)
        configured_notebook = notebook_specs[0] if notebook_specs and not template_name else None
        selected_template = bound_protocol.resolve_notebook_template(
            explicit_template=template_name,
            configured_template=(configured_notebook.template if configured_notebook is not None else None),
        )
        descriptor = require_notebook_template_for_protocol(selected_template, protocol=bound_protocol)
        if (plot_only or plot_exclude) and not descriptor.capabilities.supports_plot_filters:
            raise typer.BadParameter(
                f"--only/--exclude are not supported with template {descriptor.template}. "
                "Choose a template that declares plot-filter capability."
            )
        layout = decl.experiment_semantics.layout
        outputs_dir = layout.outputs_dir
        if not template_requirements_satisfied(selected_template, decl, outputs_dir, runtime=runtime):
            raise typer.BadParameter(
                f"Template {descriptor.template} does not satisfy its declared requirements for this experiment. "
                "Check pipeline assets or existing dataframe records."
            )
        notebooks_cfg = layout.notebooks_subdir
        nb_dir = outputs_dir if notebooks_cfg in ("", ".", "./") else outputs_dir / str(notebooks_cfg)
        target_name = name or default_notebook_name()
        target = nb_dir / target_name
        if new:
            target = next_available_path(target)
        elif overwrite and target.exists():
            confirm = typer.confirm(f"Notebook already exists at {target}. Overwrite?", default=False)
            if not confirm:
                overwrite = False
        has_fcs = any(path.suffix.lower() == ".fcs" for path in exp_dir.rglob("*.fcs"))
        existed = target.exists()
        plot_specs_payload = None
        if descriptor.capabilities.inject_plot_specs:
            selected = select_surface_specs(
                plot_specs, only=plot_only or [], exclude=plot_exclude or [], kind="plot spec"
            )
            plot_specs_payload = [spec_to_dict(spec) for spec in selected]
        target, created = write_experiment_notebook(
            target,
            template=selected_template,
            overwrite=overwrite,
            plot_specs=plot_specs_payload,
            allow_record_scan=scan_records,
        )
        if created:
            if existed and overwrite:
                status = f"✓ Notebook overwritten: [path]{target}[/path]\n[muted]template[/muted]: {selected_template}"
            else:
                status = f"✓ Notebook created: [path]{target}[/path]\n[muted]template[/muted]: {selected_template}"
            border_style = "ok"
        else:
            action = "opening existing" if mode_value != "none" else "using existing"
            status = f"Notebook already exists: [path]{target}[/path] {action}."
            border_style = "warn"
        shared.console.print(Panel.fit(status, border_style=border_style, box=box.ROUNDED))
        if mode_value == "none":
            shared.console.print(str(target))
            return
        launch_cmd = f"{shared.sys.executable} -m marimo {mode_value} {target}"
        shared.console.print(Panel.fit(f"Launching: {launch_cmd}", border_style="accent", box=box.ROUNDED))
        _launch_marimo(mode_value, target, has_fcs=has_fcs)
    except ConfigError as err:
        raise typer.BadParameter(str(err)) from err


@app.command(help="Scaffold an interactive marimo notebook and open it.")
def notebook(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help="Experiment config path, directory, or index from 'uv run reader ls'.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="Notebook filename (created under outputs/notebooks). Defaults to EDA_YYYYMMDD.py.",
    ),
    template_name: str | None = typer.Option(
        None,
        "--template",
        help="Notebook template (defaults to the protocol default or protocol.outputs.notebook.template).",
    ),
    list_templates: bool = typer.Option(False, "--list-templates", help="List notebook templates and exit."),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "--force",
        help="Overwrite today's notebook if it already exists (asks for confirmation).",
    ),
    new: bool = typer.Option(
        False,
        "--new",
        help="Create an additional notebook by appending a numeric suffix if needed.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Regenerate the notebook even if it exists (same as --overwrite).",
    ),
    scan_records: bool = typer.Option(
        False,
        "--scan-records",
        help="Allow notebook templates to scan outputs/artifacts when records.json is missing.",
    ),
    mode: str = NOTEBOOK_MODE_OPTION,
    only: list[str] = NOTEBOOK_PLOT_ONLY_OPTION,
    exclude: list[str] = NOTEBOOK_PLOT_EXCLUDE_OPTION,
):
    _scaffold_notebook(
        job=job,
        name=name,
        template_name=template_name,
        list_templates=list_templates,
        overwrite=overwrite,
        new=new,
        refresh=refresh,
        scan_records=scan_records,
        mode=mode,
        plot_only=only,
        plot_exclude=exclude,
    )
