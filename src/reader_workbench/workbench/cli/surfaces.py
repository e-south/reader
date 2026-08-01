from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.panel import Panel

from reader_workbench.errors import ReaderError

from . import _records_view, _surface_execution, shared
from ._lazy import load as _load
from .helpers import (
    ensure_active_lifecycle,
    find_nearest_experiments_dir,
    find_year_jobs,
    infer_job_path,
    load_job_models,
    require_dataframe_records,
)
from .shared import (
    EXPORT_EXCLUDE_OPTION,
    EXPORT_INPUT_OPTION,
    EXPORT_ONLY_OPTION,
    EXPORT_SET_OPTION,
    PLOT_EXCLUDE_OPTION,
    PLOT_INPUT_OPTION,
    PLOT_ONLY_OPTION,
    PLOT_SET_OPTION,
    abort,
    app,
    emit_json,
    handle_reader_error,
    normalize_output_format,
    table,
)


def _spec_overrides():
    return _surface_execution.spec_overrides()


def _validate_list_mode_flags(
    *,
    list_only: bool,
    dry_run: bool,
    inputs: list[str] | None,
    sets: list[str] | None,
) -> None:
    _surface_execution.validate_list_mode_flags(
        list_only=list_only,
        dry_run=dry_run,
        inputs=inputs,
        sets=sets,
    )


def _apply_surface_overrides(
    selected,
    *,
    inputs: list[str] | None,
    sets: list[str] | None,
    experiment_root: Path,
    resources,
):
    return _surface_execution.apply_surface_overrides(
        selected,
        inputs=inputs,
        sets=sets,
        experiment_root=experiment_root,
        resources=resources,
    )


def _validate_plot_job_for_execution(
    job_path: Path,
    *,
    only: list[str] | None,
    exclude: list[str] | None,
    dry_run: bool,
    inputs: list[str] | None,
    sets: list[str] | None,
) -> None:
    _surface_execution.validate_plot_job_for_execution(
        job_path,
        only=only,
        exclude=exclude,
        dry_run=dry_run,
        inputs=inputs,
        sets=sets,
        ensure_active_lifecycle_fn=ensure_active_lifecycle,
        require_dataframe_records_fn=require_dataframe_records,
    )


def _render_surface_specs_table(
    *, title_text: str, selected, runtime, record_producers, summaries: dict[str, str]
) -> None:
    _surface_execution.render_surface_specs_table(
        title_text=title_text,
        selected=selected,
        runtime=runtime,
        record_producers=record_producers,
        summaries=summaries,
    )


def _surface_next_steps(*, job_hint: str | None, output_dir: Path, include_plot: bool, include_export: bool) -> None:
    _surface_execution.surface_next_steps(
        job_hint=job_hint,
        output_dir=output_dir,
        include_plot=include_plot,
        include_export=include_export,
    )


def _run_plot_job(
    job_path: Path,
    *,
    job_hint: str | None,
    only: list[str] | None,
    exclude: list[str] | None,
    list_only: bool,
    format: str,
    dry_run: bool,
    log_level: str,
    inputs: list[str] | None,
    sets: list[str] | None,
) -> None:
    _surface_execution.run_plot_job(
        job_path,
        job_hint=job_hint,
        only=only,
        exclude=exclude,
        list_only=list_only,
        format=format,
        dry_run=dry_run,
        log_level=log_level,
        inputs=inputs,
        sets=sets,
        ensure_active_lifecycle_fn=ensure_active_lifecycle,
        require_dataframe_records_fn=require_dataframe_records,
    )


def _run_export_job(
    job_path: Path,
    *,
    job_hint: str | None,
    only: list[str] | None,
    exclude: list[str] | None,
    list_only: bool,
    format: str,
    dry_run: bool,
    log_level: str,
    inputs: list[str] | None,
    sets: list[str] | None,
) -> None:
    _surface_execution.run_export_job(
        job_path,
        job_hint=job_hint,
        only=only,
        exclude=exclude,
        list_only=list_only,
        format=format,
        dry_run=dry_run,
        log_level=log_level,
        inputs=inputs,
        sets=sets,
        ensure_active_lifecycle_fn=ensure_active_lifecycle,
        require_dataframe_records_fn=require_dataframe_records,
    )


@app.command(help="List plot specs or save plot files from existing dataframe records.")
def plot(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help=shared.JOB_ARG_HELP_SHORT,
    ),
    year: str | None = typer.Option(
        None, "--year", metavar="YYYY", help="Run plots for all experiments under experiments/YYYY."
    ),
    root: str | None = typer.Option(
        None,
        "--root",
        metavar="DIR",
        help="Override experiments root when using --year (default: nearest experiments/).",
    ),
    only: list[str] = PLOT_ONLY_OPTION,
    exclude: list[str] = PLOT_EXCLUDE_OPTION,
    list_only: bool = typer.Option(False, "--list", help="List plot specs for this config and exit."),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format for --list: table | json (default: table)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan only: validate and print the plot plan without executing."
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        metavar="LEVEL",
        help="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL (default: INFO).",
    ),
    inputs: list[str] = PLOT_INPUT_OPTION,
    sets: list[str] = PLOT_SET_OPTION,
):
    if root and not year:
        raise typer.BadParameter("--root is only valid with --year")
    _validate_list_mode_flags(list_only=list_only, dry_run=dry_run, inputs=inputs, sets=sets)
    fmt = normalize_output_format(format)
    if year:
        if fmt == "json":
            raise typer.BadParameter("--format json is only supported for single-experiment plot listings")
        if job is not None:
            raise typer.BadParameter("--year cannot be combined with CONFIG|DIR|INDEX")
        root_path = find_nearest_experiments_dir(Path.cwd()) if root is None else Path(root).resolve()
        jobs = find_year_jobs(year, root_path)
        if not list_only:
            failures: list[tuple[Path, str]] = []
            for job_path in jobs:
                try:
                    _validate_plot_job_for_execution(
                        job_path,
                        only=only,
                        exclude=exclude,
                        dry_run=dry_run,
                        inputs=inputs,
                        sets=sets,
                    )
                except (ReaderError, typer.BadParameter) as exc:
                    failures.append((job_path, str(exc)))
            if failures:
                lines = [f"{len(failures)} experiment(s) are not runnable for year {year}:"]
                lines += [f"- {path.parent.name}: {msg}" for path, msg in failures]
                abort("\n".join(lines))
        shared.console.print(
            Panel.fit(
                f"Plotting {len(jobs)} experiment(s) for {year} under [path]{root_path}[/path].",
                border_style="accent",
                box=box.ROUNDED,
            )
        )
        failures: list[tuple[Path, str]] = []
        total = len(jobs)
        for index, job_path in enumerate(jobs, 1):
            exp_name = job_path.parent.name
            cmd_line = " ".join(
                _spec_overrides().build_surface_command(
                    "reader plot",
                    job_path,
                    only=only,
                    exclude=exclude,
                    list_only=list_only,
                    dry_run=dry_run,
                    log_level=log_level,
                    inputs=inputs,
                    sets=sets,
                )
            )
            shared.console.print(f"[accent]{index}/{total}[/accent] {exp_name}")
            shared.console.print(f"[muted]{cmd_line}[/muted]")
            try:
                _run_plot_job(
                    job_path,
                    job_hint=str(job_path),
                    only=only,
                    exclude=exclude,
                    list_only=list_only,
                    format=fmt,
                    dry_run=dry_run,
                    log_level=log_level,
                    inputs=inputs,
                    sets=sets,
                )
            except (ReaderError, typer.BadParameter) as exc:
                failures.append((job_path, str(exc)))
                shared.console.print(
                    Panel.fit(f"[error]✗ {exp_name}: {exc}[/error]", border_style="error", box=box.ROUNDED)
                )
        if failures:
            lines = [f"{len(failures)} experiment(s) failed while plotting year {year}:"]
            lines += [f"- {path.parent.name}: {msg}" for path, msg in failures]
            abort("\n".join(lines))
        return

    try:
        job_path = infer_job_path(job)
        _run_plot_job(
            job_path,
            job_hint=job,
            only=only,
            exclude=exclude,
            list_only=list_only,
            format=fmt,
            dry_run=dry_run,
            log_level=log_level,
            inputs=inputs,
            sets=sets,
        )
    except ReaderError as err:
        handle_reader_error(err)


@app.command(help="List export specs or write export files from existing dataframe records.")
def export(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help=shared.JOB_ARG_HELP_SHORT,
    ),
    only: list[str] = EXPORT_ONLY_OPTION,
    exclude: list[str] = EXPORT_EXCLUDE_OPTION,
    list_only: bool = typer.Option(False, "--list", help="List export specs for this config and exit."),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format for --list: table | json (default: table)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan only: validate and print the export plan without executing."
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        metavar="LEVEL",
        help="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL (default: INFO).",
    ),
    inputs: list[str] = EXPORT_INPUT_OPTION,
    sets: list[str] = EXPORT_SET_OPTION,
):
    try:
        _validate_list_mode_flags(list_only=list_only, dry_run=dry_run, inputs=inputs, sets=sets)
        job_path = infer_job_path(job)
        _run_export_job(
            job_path,
            job_hint=job,
            only=only,
            exclude=exclude,
            list_only=list_only,
            format=format,
            dry_run=dry_run,
            log_level=log_level,
            inputs=inputs,
            sets=sets,
        )
    except ReaderError as err:
        handle_reader_error(err)


@app.command(help="List current records from outputs/manifests/records.json.")
def records(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help=shared.JOB_ARG_HELP_WITH_DEFAULT,
    ),
    all: bool = typer.Option(
        False,
        "--all",
        help="Include prior and retired record identities with revision counts.",
    ),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format: table | json (default: table)."
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        metavar="N",
        help="Maximum records per JSON page (default: 25; maximum: 100).",
    ),
    continuation: str | None = typer.Option(
        None,
        "--continuation",
        metavar="TOKEN",
        help="Opaque continuation token from a previous JSON page.",
    ),
):
    try:
        job_path = infer_job_path(job)
        _records_view.render_records(
            job_path=job_path,
            all_revisions=all,
            format=format,
            limit=limit,
            continuation=continuation,
        )
    except ReaderError as err:
        handle_reader_error(err)


@app.command(help="List pipeline steps and bindings for a config.")
def steps(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help=shared.JOB_ARG_HELP_WITH_DEFAULT,
    ),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format: table | json (default: table)."
    ),
):
    try:
        job_path = infer_job_path(job)
        spec, decl = load_job_models(job_path)
    except ReaderError as err:
        handle_reader_error(err)
    runtime = _load("reader_workbench.runtime").builtin_runtime()
    fmt = normalize_output_format(format)
    workbench = _load("reader_workbench.workbench.graph").resolve_workbench(decl)
    pipeline = list(workbench.pipeline)
    inspection_experiments = _load("reader_workbench.workbench.inspection.experiments")
    inspection_runtime = _load("reader_workbench.workbench.inspection.runtime")
    payload = inspection_experiments.experiment_steps_payload(job_path=job_path, spec=spec, decl=decl, runtime=runtime)
    if fmt == "json":
        emit_json(payload)
        return
    listing = table("Steps")
    listing.add_column("#", justify="right", style="muted")
    listing.add_column("stage", style="accent")
    listing.add_column("id", style="accent", overflow="fold")
    listing.add_column("plugin", overflow="fold")
    listing.add_column("from", overflow="fold")
    listing.add_column("writes", overflow="fold")
    for index, item in enumerate(payload["implementation"]["compiled"]["pipeline"], 1):
        from_refs = ", ".join(inspection_runtime.render_read_binding(entry) for entry in item["reads"]) or "—"
        writes = (
            ", ".join(
                (f"{entry['label']} -> {entry['display']}" if entry.get("kind") == "dataframe" else str(entry["label"]))
                for entry in item["writes"]
            )
            or "—"
        )
        listing.add_row(str(index), str(item["stage"]), str(item["id"]), str(item["plugin"]), from_refs, writes)
    shared.console.print(
        Panel(listing, border_style="accent", box=box.ROUNDED, subtitle=f"[muted]{len(pipeline)} total[/muted]")
    )


@app.command(help="List plugins by category, domain, and family.")
def plugins(
    category: str | None = typer.Option(
        None, "--category", metavar="NAME", help="Filter by category: ingest | transform | plot | export | validator"
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        metavar="NAME",
        help="Filter by domain, for example: plate_reader | cytometry | logic | generic",
    ),
    family: str | None = typer.Option(
        None,
        "--family",
        metavar="NAME",
        help="Filter by family, for example: time_series | metadata_merge | workbook_ingest",
    ),
    protocol: str | None = typer.Option(
        None, "--protocol", metavar="ID", help="Limit to plugins used by the named protocol's default plan."
    ),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format: table | json (default: table)."
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        metavar="N",
        help="Maximum plugins per JSON page (default: 25; maximum: 100).",
    ),
    continuation: str | None = typer.Option(
        None,
        "--continuation",
        metavar="TOKEN",
        help="Opaque continuation token from a previous JSON page.",
    ),
):
    protocol = protocol if isinstance(protocol, str) else None
    fmt = normalize_output_format(format)
    limit, continuation = shared.normalize_paging_options(limit, continuation)
    shared.require_json_paging(format=fmt, limit=limit, continuation=continuation)
    try:
        runtime = _load("reader_workbench.runtime").builtin_runtime()
        descriptors = runtime.plugins.catalog().filter(category=category, domain=domain, family=family)
        if protocol:
            bound_protocol = runtime.bind_protocol(_load("reader_workbench.protocols").ProtocolBinding(id=protocol))
            plan = bound_protocol.compile()
            allowed_plugins = {step.plugin for step in (*plan.pipeline, *plan.plots, *plan.exports)}
            descriptors = [descriptor for descriptor in descriptors if descriptor.plugin in allowed_plugins]
    except ReaderError as err:
        handle_reader_error(err)
    if fmt == "json":
        payload = _load("reader_workbench.workbench.inspection.catalogs").plugin_registry_payload(
            descriptors=descriptors,
            category=category,
            domain=domain,
            family=family,
            protocol=protocol,
        )
        page = shared.page_json_collection(
            payload["plugins"],
            key=lambda item: str(item["plugin"]),
            surface="plugins",
            selection=payload["selection"],
            limit=limit,
            continuation=continuation,
        )
        payload["plugins"] = list(page.items)
        emit_json(payload, truncated=page.truncated, continuation=page.continuation)
        return
    listing = table("Plugins")
    listing.add_column("category", style="accent")
    listing.add_column("domain")
    listing.add_column("family")
    listing.add_column("key")
    listing.add_column("summary", overflow="fold")
    listing.add_column("class", style="muted", overflow="fold")
    for descriptor in descriptors:
        listing.add_row(
            descriptor.category,
            descriptor.domain,
            descriptor.family,
            descriptor.key,
            descriptor.summary,
            f"{descriptor.cls.__module__}.{descriptor.cls.__name__}",
        )
    shared.console.print(
        Panel(
            listing,
            border_style="accent",
            box=box.ROUNDED,
            subtitle=(
                f"[muted]{len(descriptors)} plugin(s) discovered"
                f"{f' • protocol: {protocol}' if protocol else ''}[/muted]"
            ),
        )
    )
