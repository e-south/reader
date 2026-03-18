from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich import box
from rich.panel import Panel
from rich.table import Table

from reader.errors import ReaderError
from reader.runtime import builtin_runtime
from reader.workbench.config import ReaderSpec
from reader.workbench.engine import explain as explain_job
from reader.workbench.engine import run_job
from reader.workbench.engine import validate as validate_job
from reader.workbench.engine import validation_summary as validate_summary_job
from reader.workbench.graph import materialize_workbench
from reader.workbench.inspection import (
    experiment_config_json_payload,
    experiment_explain_payload,
    experiment_identity_payload,
    experiment_inspect_payload,
    experiment_run_dry_run_payload,
    inventory_summary_payload,
    inventory_surface_payload,
    validation_surface_payload,
)
from reader.workbench.inspection.common import preview_output_files, resolve_output_subdir, summarize_outputs_dir
from reader.workbench.inspection.reports import experiment_inspect_renderables
from reader.workbench.inspection.runtime import generated_summary, selected_plan_payload, selected_plan_summary

from . import shared
from .helpers import (
    append_journal,
    find_jobs,
    find_nearest_experiments_dir,
    format_job_arg,
    infer_job_path,
    load_job_models,
    resolve_pipeline_step_id,
)
from .shared import (
    app,
    checkmark,
    emit_json,
    handle_reader_error,
    normalize_flag,
    normalize_output_format,
    normalize_status_filter,
    table,
)


@app.command(help="List experiments under a root (default: ./experiments).")
def ls(
    root: str = typer.Option(
        "./experiments",
        "--root",
        metavar="DIR",
        help="Directory to search recursively for experiment directories.",
    ),
    include_scaffolds: bool = typer.Option(
        False,
        "--all",
        help="Include scaffold/template directories alongside runnable experiments.",
    ),
    protocol: str | None = typer.Option(
        None, "--protocol", metavar="ID", help="Only show experiments bound to the given protocol id."
    ),
    status: str | None = typer.Option(
        None,
        "--status",
        metavar="STATE",
        help="Only show experiments with status: ok | config_error.",
    ),
    details: bool = typer.Option(
        False,
        "--details",
        help="Show protocol id, selected-plan summary, and generated output counts for each experiment.",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
):
    include_scaffolds = normalize_flag(include_scaffolds)
    details = normalize_flag(details)
    fmt = normalize_output_format(format)
    protocol_filter = protocol.strip() if isinstance(protocol, str) and protocol.strip() else None
    status_filter = normalize_status_filter(status)

    root_path = (
        find_nearest_experiments_dir(Path.cwd()) if str(root).strip() == "./experiments" else Path(root).resolve()
    )
    jobs = find_jobs(root_path, include_scaffolds=include_scaffolds)
    if not jobs:
        if fmt == "json":
            emit_json(
                inventory_surface_payload(
                    root=root_path,
                    include_scaffolds=include_scaffolds,
                    details=details,
                    protocol=protocol_filter,
                    status=status_filter,
                    experiments=[],
                )
            )
            return
        shared.console.print(
            Panel.fit(f"No experiments found under [path]{root_path}[/path].", border_style="warn", box=box.ROUNDED)
        )
        return

    entries: list[dict[str, object]] = []
    runtime = builtin_runtime() if details else None
    for idx, config_path in enumerate(jobs, 1):
        entry: dict[str, object] = {
            "index": idx,
            "name": config_path.parent.name,
            "config": str(config_path),
            "root": str(config_path.parent),
            "protocol": None,
            "generated": {"records": 0, "plots": 0, "exports": 0, "notebooks": 0},
            "selected": None,
            "has_outputs": False,
            "status": "ok",
            "error": None,
        }
        try:
            if details:
                spec, decl = load_job_models(config_path, runtime=runtime)
                entry["selected"] = selected_plan_payload(spec=spec, decl=decl, runtime=runtime)
            else:
                spec = ReaderSpec.load(config_path)
            entry["protocol"] = spec.protocol.id
            outputs_dir = (config_path.parent / spec.paths.outputs).resolve()
            counts = summarize_outputs_dir(
                outputs_dir,
                plots_subdir=spec.paths.plots,
                exports_subdir=spec.paths.exports,
                notebooks_subdir=spec.paths.notebooks,
            )
            entry["generated"] = counts
            entry["has_outputs"] = any(counts.values())
            if details:
                entry["generated_examples"] = {
                    "records": preview_output_files(outputs_dir / "artifacts", base=config_path.parent),
                    "plots": preview_output_files(
                        resolve_output_subdir(outputs_dir, spec.paths.plots), base=config_path.parent
                    ),
                    "exports": preview_output_files(
                        resolve_output_subdir(outputs_dir, spec.paths.exports), base=config_path.parent
                    ),
                    "notebooks": preview_output_files(
                        resolve_output_subdir(outputs_dir, spec.paths.notebooks),
                        base=config_path.parent,
                    ),
                }
        except ReaderError as err:
            entry["status"] = "config_error"
            entry["error"] = str(err)
        if protocol_filter and entry["protocol"] != protocol_filter:
            continue
        if status_filter and entry["status"] != status_filter:
            continue
        entries.append(entry)

    if not entries:
        if fmt == "json":
            emit_json(
                inventory_surface_payload(
                    root=root_path,
                    include_scaffolds=include_scaffolds,
                    details=details,
                    protocol=protocol_filter,
                    status=status_filter,
                    experiments=[],
                )
            )
            return
        filters = []
        if protocol_filter:
            filters.append(f"protocol={protocol_filter}")
        if status_filter:
            filters.append(f"status={status_filter}")
        suffix = f" for filters ({', '.join(filters)})" if filters else ""
        shared.console.print(
            Panel.fit(
                f"No experiments found under [path]{root_path}[/path]{suffix}.",
                border_style="warn",
                box=box.ROUNDED,
            )
        )
        return

    if fmt == "json":
        emit_json(
            inventory_surface_payload(
                root=root_path,
                include_scaffolds=include_scaffolds,
                details=details,
                protocol=protocol_filter,
                status=status_filter,
                experiments=entries,
            )
        )
        return

    inventory_summary = inventory_summary_payload(entries)
    listing = table("Experiments")
    listing.add_column("#", justify="right", style="muted")
    name_values = [str(entry["name"]) for entry in entries]
    max_name = max((len(name) for name in name_values), default=12)
    max_width = int((shared.console.width or 80) * (0.35 if details else 0.6))
    name_width = max(12, min(max_name + 2, max_width))
    listing.add_column("Name", style="accent", max_width=name_width, overflow="ellipsis")
    if details:
        listing.add_column("Protocol", max_width=28, overflow="ellipsis")
        listing.add_column("Status", width=12)
        listing.add_column("Selected", overflow="fold")
        listing.add_column("Generated", overflow="fold")
        listing.add_column("Issue", overflow="fold")
    else:
        listing.add_column("Outputs", justify="center", width=7)

    for entry in entries:
        status_value = str(entry["status"])
        generated = dict(entry["generated"])
        if details:
            listing.add_row(
                str(entry["index"]),
                str(entry["name"]),
                str(entry["protocol"] or "—"),
                status_value,
                selected_plan_summary(entry.get("selected") if isinstance(entry, dict) else None),
                generated_summary(generated),
                str(entry["error"] or "—"),
            )
        else:
            outputs_cell = "[error]ERR[/error]" if status_value != "ok" else checkmark(bool(entry["has_outputs"]))
            listing.add_row(str(entry["index"]), str(entry["name"]), outputs_cell)
    shared.console.print(
        Panel(
            listing,
            border_style="accent",
            box=box.ROUNDED,
            subtitle=(
                f"[muted]root: [path]{root_path}[/path] — {len(entries)} shown"
                + (f" (protocol={protocol_filter})" if protocol_filter else "")
                + (f" (status={status_filter})" if status_filter else "")
                + "[/muted]"
            ),
        )
    )
    if details:
        summary = Table(box=box.ROUNDED, expand=True, show_header=False)
        summary.add_column("Field", style="accent", no_wrap=True)
        summary.add_column("Value")
        summary.add_row("Shown", str(len(entries)))
        summary.add_row(
            "Outputs",
            (
                f"{inventory_summary['outputs']['with_outputs']} with outputs"
                f" • {inventory_summary['outputs']['without_outputs']} without outputs"
            ),
        )
        status_bits = [f"{key}={value}" for key, value in dict(inventory_summary["by_status"]).items()]
        summary.add_row("Status", ", ".join(status_bits) if status_bits else "—")
        protocol_bits = [f"{key}={value}" for key, value in dict(inventory_summary["by_protocol"]).items()]
        summary.add_row("Protocols", ", ".join(protocol_bits) if protocol_bits else "—")
        shared.console.print(Panel(summary, border_style="accent", box=box.ROUNDED, title="Inventory summary"))


@app.command(help="Inspect one experiment: inputs, pipeline chain, plots, artifacts, and generated outputs.")
def inspect(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
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
    fmt = normalize_output_format(format)
    runtime = builtin_runtime()
    payload = experiment_inspect_payload(job_path=job_path, spec=spec, decl=decl, runtime=runtime)
    if fmt == "json":
        emit_json(payload)
        return
    for renderable in experiment_inspect_renderables(
        payload=payload, semantic_program=decl.experiment_semantics.protocol_program
    ):
        shared.console.print(renderable)


@app.command(help="Show planned steps and contracts (no execution).")
def explain(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format: table | json (default: table)."
    ),
):
    try:
        job_path = infer_job_path(job)
        append_journal(job_path, f"reader explain {job_path}")
        spec, decl = load_job_models(job_path)
        runtime = builtin_runtime()
        fmt = normalize_output_format(format)
        if fmt == "json":
            emit_json(experiment_explain_payload(job_path=job_path, spec=spec, decl=decl, runtime=runtime))
            return
        explain_job(decl, console=shared.console, runtime=runtime)
    except ReaderError as err:
        handle_reader_error(err)


@app.command(help="Validate config, plugin params, reads wiring, and input files.")
def validate(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    no_files: bool = typer.Option(False, "--no-files", help="Skip file existence checks (config-only validation)."),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format: table | json (default: table)."
    ),
):
    try:
        job_path = infer_job_path(job)
        append_journal(job_path, f"reader validate {job_path}")
        _, decl = load_job_models(job_path)
        runtime = builtin_runtime()
        fmt = normalize_output_format(format)
        if fmt == "json":
            summary = validate_summary_job(
                decl, check_files=not no_files, exp_root=decl.experiment.root, runtime=runtime
            )
            emit_json(
                validation_surface_payload(
                    experiment=experiment_identity_payload(job_path=job_path, decl=decl),
                    check_files=not no_files,
                    summary=summary,
                )
            )
            if summary["status"] != "ok":
                raise typer.Exit(code=1)
            return
        summary = validate_job(
            decl,
            console=shared.console,
            check_files=not no_files,
            exp_root=decl.experiment.root,
            runtime=runtime,
        )
        if summary["status"] != "ok":
            raise typer.Exit(code=1)
    except ReaderError as err:
        handle_reader_error(err)


@app.command(help="Print the authoring config plus compiled runtime plan.")
def config(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    format: str = typer.Option("yaml", "--format", metavar="FMT", help="Output format: yaml | json (default: yaml)."),
):
    try:
        job_path = infer_job_path(job)
        spec, decl = load_job_models(job_path)
    except ReaderError as err:
        handle_reader_error(err)
    fmt = str(format).strip().lower()
    if fmt == "json":
        runtime = builtin_runtime()
        emit_json(experiment_config_json_payload(job_path=job_path, spec=spec, decl=decl, runtime=runtime))
        return
    if fmt == "yaml":
        payload = spec.model_dump(by_alias=True)
        materialized = materialize_workbench(decl)
        payload["compiled"] = {
            "pipeline": materialized["pipeline"],
            "plots": materialized["plots"],
            "exports": materialized["exports"],
            "notebooks": materialized["notebooks"],
        }
        typer.echo(yaml.safe_dump(payload, sort_keys=False))
        return
    raise typer.BadParameter("format must be 'yaml' or 'json'")


@app.command(help="Run pipeline to generate dataframe records.")
def run(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    from_step: str | None = typer.Option(
        None,
        "--from",
        metavar="STEP_ID",
        help="Start from this pipeline step (inclusive). Use an exact id as declared in the config.",
    ),
    until: str | None = typer.Option(
        None,
        "--until",
        metavar="STEP_ID",
        help="Stop after this step (inclusive). Use an exact id as declared in the config.",
    ),
    only: str | None = typer.Option(None, "--only", metavar="STEP_ID", help="Run exactly one pipeline step by id."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan only: validate and print the plan without executing steps."
    ),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format for --dry-run: table | json (default: table)."
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        metavar="LEVEL",
        help="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL (default: INFO).",
    ),
    compact: bool = typer.Option(False, "--compact", help="Use concise progress output instead of per-step logs."),
):
    job_path = infer_job_path(job)
    parts = ["reader run", str(job_path)]
    if only and (from_step or until):
        raise typer.BadParameter("--only cannot be combined with --from/--until")

    try:
        spec, decl = load_job_models(job_path)
    except ReaderError as err:
        handle_reader_error(err)
    fmt = normalize_output_format(format)
    if fmt == "json" and not dry_run:
        raise typer.BadParameter("--format json is only supported with --dry-run")

    if only:
        resolve_pipeline_step_id(decl, only)
        parts += ["--only", only]
        if dry_run:
            parts += ["--dry-run"]
        if fmt == "json":
            parts += ["--format", "json"]
        if log_level and log_level != "INFO":
            parts += ["--log-level", log_level]
        if compact:
            parts += ["--compact"]
        append_journal(job_path, " ".join(parts))
        try:
            runtime = builtin_runtime()
            if dry_run and fmt == "json":
                emit_json(
                    experiment_run_dry_run_payload(
                        job_path=job_path,
                        spec=spec,
                        decl=decl,
                        runtime=runtime,
                        resume_from=None,
                        until=None,
                        only=only,
                    )
                )
                return
            run_job(
                job_path,
                resume_from=only,
                until=only,
                dry_run=dry_run,
                log_level=log_level,
                verbose=not compact,
                console=shared.console,
                include_pipeline=True,
                include_plots=False,
                include_exports=False,
                runtime=runtime,
            )
        except ReaderError as err:
            handle_reader_error(err)
        return

    if from_step:
        resolve_pipeline_step_id(decl, from_step)
        parts += ["--from", from_step]
    if until:
        resolve_pipeline_step_id(decl, until)
        parts += ["--until", until]
    if dry_run:
        parts += ["--dry-run"]
    if fmt == "json":
        parts += ["--format", "json"]
    if log_level and log_level != "INFO":
        parts += ["--log-level", log_level]
    if compact:
        parts += ["--compact"]
    append_journal(job_path, " ".join(parts))
    try:
        runtime = builtin_runtime()
        if dry_run and fmt == "json":
            emit_json(
                experiment_run_dry_run_payload(
                    job_path=job_path,
                    spec=spec,
                    decl=decl,
                    runtime=runtime,
                    resume_from=from_step,
                    until=until,
                )
            )
            return
        run_job(
            job_path,
            resume_from=from_step,
            until=until,
            dry_run=dry_run,
            log_level=log_level,
            verbose=not compact,
            console=shared.console,
            include_pipeline=True,
            include_plots=False,
            include_exports=False,
            job_label=format_job_arg(job),
            show_next_steps=True,
            runtime=runtime,
        )
    except ReaderError as err:
        handle_reader_error(err)
