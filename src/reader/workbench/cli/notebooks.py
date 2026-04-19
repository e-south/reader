from __future__ import annotations

import os
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel

from reader.errors import ConfigError, RecordError

from . import shared
from ._lazy import load as _load
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


def render_marimo_routes(*, target: Path, url: str, runtime_root: Path) -> None:
    check_cmd = f"uv run marimo check {target}"
    shared.console.print(
        Panel.fit(
            "Review routes:\n"
            f"  Static check: {check_cmd}\n"
            f"  Browser review: {url}\n"
            "  Chrome MCP: open the URL in a fresh isolated page.\n\n"
            f"Managed runtime root: [path]{runtime_root}[/path]",
            border_style="accent",
            box=box.ROUNDED,
        )
    )


def _launch_marimo(
    mode: str,
    target: Path,
    *,
    has_fcs: bool,
    headless: bool = False,
    port: int | None = None,
) -> None:
    launch = _load("reader.workbench.notebooks.launch")
    plan = launch.plan_marimo_launch(
        mode=mode,
        target=target,
        headless=headless,
        preferred_port=port,
        base_env=os.environ.copy(),
    )
    if plan.terminated_sessions:
        shared.console.print(
            Panel.fit(
                f"Pruned {len(plan.terminated_sessions)} existing reader-managed Marimo session(s) "
                "for this experiment before launch.",
                border_style="warn",
                box=box.ROUNDED,
            )
        )
    if plan.reused_session is not None:
        shared.console.print(
            Panel.fit(
                f"Notebook already running: [path]{target}[/path]\n[muted]url[/muted]: {plan.url}",
                border_style="ok",
                box=box.ROUNDED,
            )
        )
        render_marimo_routes(target=target, url=plan.url, runtime_root=plan.runtime_paths.root)
        if not headless:
            launch.open_url(plan.url)
        return
    shared.console.print(
        Panel.fit(
            f"Launching: {' '.join(plan.cmd)}\n[muted]url[/muted]: {plan.url}",
            border_style="accent",
            box=box.ROUNDED,
        )
    )
    render_marimo_routes(target=target, url=plan.url, runtime_root=plan.runtime_paths.root)
    try:
        proc = shared.subprocess.Popen(plan.cmd, env=plan.env)
    except FileNotFoundError:
        render_marimo_help(target, mode=mode, has_fcs=has_fcs)
        raise typer.Exit(code=1) from None
    launch.register_managed_session(
        registry_path=plan.runtime_paths.registry_path,
        pid=proc.pid,
        port=plan.port,
        host=plan.host,
        mode=mode,
        target=target,
    )
    try:
        returncode = proc.wait()
    finally:
        launch.unregister_managed_session(registry_path=plan.runtime_paths.registry_path, pid=proc.pid)
    if returncode != 0:
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
    headless: bool,
    port: int | None,
) -> None:
    try:
        templates = _load("reader.workbench.templates")
        if list_templates:
            runtime = _load("reader.runtime").builtin_runtime()
            bound_protocol = None
            title = "Notebook templates"
            descriptors = templates.builtin_notebook_template_catalog().all()
            selected_template = None
            if job is not None:
                job_path = infer_job_path(job)
                _, decl = load_job_models(job_path)
                bound_protocol = bind_decl_protocol(decl=decl, runtime=runtime)
                workbench = _load("reader.workbench.graph").resolve_workbench(decl)
                configured_notebook = workbench.notebooks[0] if workbench.notebooks else None
                selected_template = bound_protocol.resolve_notebook_template(
                    explicit_template=template_name,
                    configured_template=(configured_notebook.template if configured_notebook is not None else None),
                )
                title = f"Notebook templates: {bound_protocol.id}"
                descriptors = templates.compatible_notebook_templates(protocol=bound_protocol)
            listing = table(title)
            listing.add_column("Name", style="accent")
            listing.add_column("Domain")
            listing.add_column("Family")
            if bound_protocol is not None:
                listing.add_column("Scaffold", justify="center")
            listing.add_column("Description")
            for descriptor in descriptors:
                row = [descriptor.template, descriptor.domain, descriptor.family]
                if bound_protocol is not None:
                    row.append("yes" if descriptor.template == selected_template else "")
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
        runtime = _load("reader.runtime").builtin_runtime()
        bound_protocol = bind_decl_protocol(decl=decl, runtime=runtime)
        workbench = _load("reader.workbench.graph").resolve_workbench(decl)
        plot_specs = list(workbench.plots)
        notebook_specs = list(workbench.notebooks)
        configured_notebook = notebook_specs[0] if notebook_specs and not template_name else None
        selected_template = bound_protocol.resolve_notebook_template(
            explicit_template=template_name,
            configured_template=(configured_notebook.template if configured_notebook is not None else None),
        )
        descriptor = templates.require_notebook_template_for_protocol(selected_template, protocol=bound_protocol)
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
            selected = _load("reader.workbench.spec_overrides").select_surface_specs(
                plot_specs, only=plot_only or [], exclude=plot_exclude or [], kind="plot spec"
            )
            plot_specs_payload = [spec_to_dict(spec) for spec in selected]
        target, created = _load("reader.workbench.notebooks").write_experiment_notebook(
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
        _launch_marimo(mode_value, target, has_fcs=has_fcs, headless=headless, port=port)
    except (ConfigError, RecordError) as err:
        raise typer.BadParameter(str(err)) from err


@app.command(help="Scaffold an interactive marimo notebook and open it.")
def notebook(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help=shared.JOB_ARG_HELP_SHORT,
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
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Launch without opening a browser. Reader prints a loopback URL suitable for Chrome MCP review.",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help="Preferred loopback port. Defaults to a reader-managed clean port starting at 2718.",
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
        headless=headless,
        port=port,
        mode=mode,
        plot_only=only,
        plot_exclude=exclude,
    )
