from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.panel import Panel

from reader.errors import ReaderError
from reader.workbench.commands import reader_command

from . import shared
from ._lazy import load as _load
from .helpers import (
    append_journal,
    bind_decl_protocol,
    find_nearest_experiments_dir,
    find_year_jobs,
    format_job_arg,
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
    return _load("reader.workbench.spec_overrides")


def _render_surface_specs_table(
    *, title_text: str, selected, runtime, record_producers, summaries: dict[str, str]
) -> None:
    inspection_runtime = _load("reader.workbench.inspection.runtime")
    listing = table(title_text)
    listing.add_column("#", justify="right", style="muted")
    listing.add_column("id", style="accent", overflow="fold")
    listing.add_column("summary", overflow="fold")
    listing.add_column("from", overflow="fold")
    listing.add_column("plugin", overflow="fold")
    for index, spec in enumerate(selected, 1):
        spec_payload = inspection_runtime.spec_step_payload(
            spec, summary=summaries.get(spec.id, "—"), runtime=runtime, record_producers=record_producers
        )
        from_refs = ", ".join(inspection_runtime.render_read_binding(item) for item in spec_payload["reads"]) or "—"
        listing.add_row(str(index), spec.id, summaries.get(spec.id, "—"), from_refs, spec.plugin)
    shared.console.print(
        Panel(listing, border_style="accent", box=box.ROUNDED, subtitle=f"[muted]{len(selected)} total[/muted]")
    )


def _surface_next_steps(*, job_hint: str | None, output_dir: Path, include_plot: bool, include_export: bool) -> None:
    def _cmd(base: str, tail: str = "") -> str:
        return reader_command(base, job_hint, tail)

    lines = [f"Artifacts saved in [path]{output_dir}[/path]", "", "Next steps:"]
    if include_plot:
        lines.append(f"  {_cmd('plot')}")
    if include_export:
        lines.append(f"  {_cmd('export')}")
    lines.append(f"  {_cmd('notebook')}")
    shared.console.print(Panel.fit("\n".join(lines), border_style="green", box=box.ROUNDED))


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
    _, decl = load_job_models(job_path)
    runtime = _load("reader.runtime").builtin_runtime()
    workbench = _load("reader.workbench.graph").resolve_workbench(decl)
    bound_protocol = bind_decl_protocol(decl=decl, runtime=runtime)
    inspection_catalogs = _load("reader.workbench.inspection.catalogs")
    inspection_runtime = _load("reader.workbench.inspection.runtime")
    fmt = normalize_output_format(format)
    if not list_only:
        if fmt == "json":
            raise typer.BadParameter("--format json is only supported with --list")
        if not dry_run:
            require_dataframe_records(decl, job_path, runtime=runtime)
    plot_specs = list(workbench.plots)
    record_producers = inspection_runtime.record_producer_map(workbench.plugin_steps(), runtime=runtime)
    if not plot_specs:
        if list_only:
            if fmt == "json":
                emit_json(
                    inspection_catalogs.workbench_surface_specs_payload(
                        job_path=job_path,
                        decl=decl,
                        runtime=runtime,
                        bound_protocol=bound_protocol,
                        selected=[],
                        kind="plot",
                        only=only or [],
                        exclude=exclude or [],
                    )
                )
                return
            shared.console.print(
                Panel.fit("No plot specs configured in this experiment.", border_style="warn", box=box.ROUNDED)
            )
            return
        raise typer.BadParameter("No plot specs configured in this experiment. Add plots to the config.")
    selected = _spec_overrides().select_surface_specs(
        plot_specs, only=only or [], exclude=exclude or [], kind="plot spec"
    )
    if list_only:
        if fmt == "json":
            emit_json(
                inspection_catalogs.workbench_surface_specs_payload(
                    job_path=job_path,
                    decl=decl,
                    runtime=runtime,
                    bound_protocol=bound_protocol,
                    selected=selected,
                    kind="plot",
                    only=only or [],
                    exclude=exclude or [],
                )
            )
            return
        _render_surface_specs_table(
            title_text="Plots",
            selected=selected,
            runtime=runtime,
            record_producers=record_producers,
            summaries=inspection_runtime.plot_output_summaries(bound_protocol),
        )
        return
    experiment_root = decl.experiment.root
    resources = decl.experiment_semantics.resources
    spec_overrides = _spec_overrides()
    input_overrides = spec_overrides.parse_input_overrides(inputs or [], root=experiment_root, resources=resources)
    set_overrides = spec_overrides.parse_set_overrides(sets or [])
    selected = spec_overrides.apply_step_overrides(
        selected,
        input_overrides=input_overrides,
        set_overrides=set_overrides,
        root=experiment_root,
        resources=resources,
    )
    append_journal(
        job_path,
        " ".join(
            spec_overrides.build_surface_command(
                "reader plot",
                job_path,
                only=only,
                exclude=exclude,
                list_only=False,
                dry_run=dry_run,
                log_level=log_level,
                inputs=inputs,
                sets=sets,
            )
        ),
    )
    _load("reader.workbench.engine").run_spec(
        decl,
        dry_run=dry_run,
        log_level=log_level,
        console=shared.console,
        include_pipeline=False,
        include_plots=True,
        include_exports=False,
        plot_specs=selected,
        runtime=runtime,
    )
    if not dry_run:
        outputs_dir = decl.experiment_semantics.layout.outputs_dir
        plots_cfg = decl.experiment_semantics.layout.plots_subdir
        plots_dir = outputs_dir if plots_cfg in ("", ".", "./") else outputs_dir / str(plots_cfg)
        _surface_next_steps(
            job_hint=format_job_arg(job_hint),
            output_dir=plots_dir,
            include_plot=False,
            include_export=bool(workbench.exports),
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
    _, decl = load_job_models(job_path)
    fmt = normalize_output_format(format)
    workbench = _load("reader.workbench.graph").resolve_workbench(decl)
    runtime = _load("reader.runtime").builtin_runtime()
    inspection_catalogs = _load("reader.workbench.inspection.catalogs")
    inspection_runtime = _load("reader.workbench.inspection.runtime")
    record_producers = inspection_runtime.record_producer_map(workbench.plugin_steps(), runtime=runtime)
    bound_protocol = bind_decl_protocol(decl=decl, runtime=runtime)
    export_specs = list(workbench.exports)
    if not export_specs:
        if list_only:
            if fmt == "json":
                emit_json(
                    inspection_catalogs.workbench_surface_specs_payload(
                        job_path=job_path,
                        decl=decl,
                        runtime=runtime,
                        bound_protocol=bound_protocol,
                        selected=[],
                        kind="export",
                        only=only or [],
                        exclude=exclude or [],
                    )
                )
                return
            shared.console.print(
                Panel.fit("No export specs configured in this experiment.", border_style="warn", box=box.ROUNDED)
            )
            return
        raise typer.BadParameter("No export specs configured in this experiment. Add exports to the config.")
    selected = _spec_overrides().select_surface_specs(
        export_specs, only=only or [], exclude=exclude or [], kind="export spec"
    )
    if list_only:
        if fmt == "json":
            emit_json(
                inspection_catalogs.workbench_surface_specs_payload(
                    job_path=job_path,
                    decl=decl,
                    runtime=runtime,
                    bound_protocol=bound_protocol,
                    selected=selected,
                    kind="export",
                    only=only or [],
                    exclude=exclude or [],
                )
            )
            return
        _render_surface_specs_table(
            title_text="Exports",
            selected=selected,
            runtime=runtime,
            record_producers=record_producers,
            summaries=inspection_runtime.export_output_summaries(bound_protocol),
        )
        return
    if fmt == "json":
        raise typer.BadParameter("--format json is only supported with --list")
    if not dry_run:
        require_dataframe_records(decl, job_path, runtime=runtime)
    experiment_root = decl.experiment.root
    resources = decl.experiment_semantics.resources
    spec_overrides = _spec_overrides()
    input_overrides = spec_overrides.parse_input_overrides(inputs or [], root=experiment_root, resources=resources)
    set_overrides = spec_overrides.parse_set_overrides(sets or [])
    selected = spec_overrides.apply_step_overrides(
        selected,
        input_overrides=input_overrides,
        set_overrides=set_overrides,
        root=experiment_root,
        resources=resources,
    )
    append_journal(
        job_path,
        " ".join(
            spec_overrides.build_surface_command(
                "reader export",
                job_path,
                only=only,
                exclude=exclude,
                list_only=False,
                dry_run=dry_run,
                log_level=log_level,
                inputs=inputs,
                sets=sets,
            )
        ),
    )
    _load("reader.workbench.engine").run_spec(
        decl,
        dry_run=dry_run,
        log_level=log_level,
        console=shared.console,
        include_pipeline=False,
        include_plots=False,
        include_exports=True,
        export_specs=selected,
        runtime=runtime,
    )
    if not dry_run:
        outputs_dir = decl.experiment_semantics.layout.outputs_dir
        exports_cfg = decl.experiment_semantics.layout.exports_subdir
        exports_dir = outputs_dir if exports_cfg in ("", ".", "./") else outputs_dir / str(exports_cfg)
        _surface_next_steps(
            job_hint=format_job_arg(job_hint),
            output_dir=exports_dir,
            include_plot=bool(workbench.plots),
            include_export=False,
        )


@app.command(help="Save plot files from plot specs using existing dataframe records.")
def plot(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help="Experiment config path, directory, or index from 'uv run reader ls'.",
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
    fmt = normalize_output_format(format)
    if year:
        if fmt == "json":
            raise typer.BadParameter("--format json is only supported for single-experiment plot listings")
        if job is not None:
            raise typer.BadParameter("--year cannot be combined with CONFIG|DIR|INDEX")
        root_path = find_nearest_experiments_dir(Path.cwd()) if root is None else Path(root).resolve()
        jobs = find_year_jobs(year, root_path)
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


@app.command(help="Run export specs using existing dataframe records.")
def export(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help="Experiment config path, directory, or index from 'uv run reader ls'.",
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


@app.command(help="List emitted workbench records from outputs/manifests/records.json.")
def records(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'uv run reader ls' (defaults to nearest ./config.yaml)",
    ),
    all: bool = typer.Option(False, "--all", help="Show revision history counts instead of latest entries."),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format: table | json (default: table)."
    ),
):
    try:
        job_path = infer_job_path(job)
        _, decl = load_job_models(job_path)
        outputs_dir = decl.experiment_semantics.layout.outputs_dir
        store = (
            _load("reader.runtime")
            .builtin_runtime()
            .record_store(
                outputs_dir,
                plots_subdir=decl.experiment_semantics.layout.plots_subdir,
                exports_subdir=decl.experiment_semantics.layout.exports_subdir,
                create=False,
            )
        )
        if not store.catalog_exists():
            abort(f"No outputs/manifests/records.json found. Run '{reader_command('run')}' first to produce records.")
    except ReaderError as err:
        handle_reader_error(err)
    fmt = normalize_output_format(format)
    if fmt == "json":
        try:
            emit_json(
                _load("reader.workbench.inspection.results").record_catalog_payload(
                    experiment=_load("reader.workbench.inspection.experiments").experiment_identity_payload(
                        job_path=job_path, decl=decl
                    ),
                    store=store,
                    outputs_dir=outputs_dir,
                    base=decl.experiment.root,
                    include_history=all,
                )
            )
        except ReaderError as err:
            handle_reader_error(err)
        return

    try:
        latest_records = store.iter_latest_records()
    except ReaderError as err:
        handle_reader_error(err)

    if all:
        if not latest_records:
            shared.console.print(
                Panel.fit(
                    f"No record history listed in outputs/manifests/records.json. Run '{reader_command('run')}' first.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        try:
            revision_counts = store.revision_counts(record.record_id for record in latest_records)
        except ReaderError as err:
            handle_reader_error(err)
        listing = table("Records • history")
        listing.add_column("Record")
        listing.add_column("Kind", style="accent")
        listing.add_column("Producer")
        listing.add_column("Revisions", justify="right")
        for record in latest_records:
            listing.add_row(
                record.record_id,
                record.kind,
                f"{record.producer.kind}:{record.producer.id}",
                str(revision_counts[record.record_id]),
            )
    else:
        if not latest_records:
            shared.console.print(
                Panel.fit(
                    f"No records listed in outputs/manifests/records.json. Run '{reader_command('run')}' first.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        listing = table("Records • latest")
        listing.add_column("Record")
        listing.add_column("Kind", style="accent")
        listing.add_column("Producer")
        listing.add_column("Details", style="path")
        for record in latest_records:
            detail = (
                f"{record.contract_id} • {record.path}"
                if record.kind == "dataframe_artifact"
                else ", ".join(str(path) for path in record.files)
            )
            listing.add_row(record.record_id, record.kind, f"{record.producer.kind}:{record.producer.id}", detail)
    shared.console.print(Panel(listing, border_style="accent", box=box.ROUNDED))


@app.command(help="List pipeline steps and bindings for a config.")
def steps(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'uv run reader ls' (defaults to nearest ./config.yaml)",
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
    runtime = _load("reader.runtime").builtin_runtime()
    fmt = normalize_output_format(format)
    workbench = _load("reader.workbench.graph").resolve_workbench(decl)
    pipeline = list(workbench.pipeline)
    inspection_experiments = _load("reader.workbench.inspection.experiments")
    inspection_runtime = _load("reader.workbench.inspection.runtime")
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


@app.command(help="List plugins by workbench ontology: category, domain, and family.")
def plugins(
    category: str | None = typer.Option(
        None, "--category", metavar="NAME", help="Filter by category: ingest | transform | plot | export | validator"
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        metavar="NAME",
        help="Filter by semantic domain, for example: plate_reader | cytometry | logic | generic",
    ),
    family: str | None = typer.Option(
        None,
        "--family",
        metavar="NAME",
        help="Filter by semantic family, for example: time_series | metadata_merge | workbook_ingest",
    ),
    protocol: str | None = typer.Option(
        None, "--protocol", metavar="ID", help="Limit to plugins used by the named protocol's default compiled plan."
    ),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format: table | json (default: table)."
    ),
):
    protocol = protocol if isinstance(protocol, str) else None
    fmt = normalize_output_format(format)
    try:
        runtime = _load("reader.runtime").builtin_runtime()
        descriptors = runtime.plugins.catalog().filter(category=category, domain=domain, family=family)
        if protocol:
            bound_protocol = runtime.bind_protocol(_load("reader.protocols").ProtocolBinding(id=protocol))
            plan = bound_protocol.compile()
            allowed_plugins = {step.plugin for step in (*plan.pipeline, *plan.plots, *plan.exports)}
            descriptors = [descriptor for descriptor in descriptors if descriptor.plugin in allowed_plugins]
    except ReaderError as err:
        handle_reader_error(err)
    if fmt == "json":
        emit_json(
            _load("reader.workbench.inspection.catalogs").plugin_registry_payload(
                descriptors=descriptors,
                category=category,
                domain=domain,
                family=family,
                protocol=protocol,
            )
        )
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
