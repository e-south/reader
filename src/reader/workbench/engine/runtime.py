from __future__ import annotations

import os
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from reader.errors import ConfigError
from reader.plotting.mpl import ensure_mpl_cache_dir
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.commands import reader_command
from reader.workbench.decl import WorkbenchDecl, load_workbench_decl
from reader.workbench.graph import ensure_unique_workbench_ids, resolve_workbench

from ._shared import collect_categories
from .execution import run_steps
from .planning import build_next_steps, explain
from .setup import build_run_context, configure_logger, resolve_palette_book, slice_pipeline_steps


def run_spec(
    decl: WorkbenchDecl,
    *,
    resume_from: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    verbose: bool = True,
    console: Console | None = None,
    include_pipeline: bool = True,
    include_plots: bool = True,
    include_exports: bool = True,
    plot_specs=None,
    export_specs=None,
    job_label: str | None = None,
    show_next_steps: bool = False,
    runtime: ReaderRuntime | None = None,
) -> None:
    os.environ.setdefault("ARROW_LOG_LEVEL", "FATAL")
    runtime = runtime or builtin_runtime()
    layout = decl.experiment_semantics.layout
    out_dir = layout.outputs_dir.resolve()
    if out_dir.exists() and not out_dir.is_dir():
        raise ConfigError(f"paths.outputs points to a file: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    console = console or Console()
    logger = configure_logger(out_dir=out_dir, log_level=log_level, verbose=verbose, console=console)

    workbench = resolve_workbench(decl)
    pipeline_steps = list(workbench.pipeline)
    plot_steps = list(plot_specs) if plot_specs is not None else list(workbench.plots)
    export_steps = list(export_specs) if export_specs is not None else list(workbench.exports)
    ensure_unique_workbench_ids(pipeline_steps, plot_steps, export_steps, workbench.notebooks)

    if not include_pipeline:
        if resume_from or until:
            raise ConfigError(
                f"--from/--until require pipeline execution; use {reader_command('run')} for sliced runs."
            )
        pipeline_steps = []
    else:
        pipeline_steps = slice_pipeline_steps(pipeline_steps, resume_from=resume_from, until=until)

    if not include_plots:
        plot_steps = []
    if not include_exports:
        export_steps = []

    all_steps = pipeline_steps + plot_steps + export_steps
    if plot_steps:
        ensure_mpl_cache_dir()

    plots_cfg = layout.plots_subdir
    exports_cfg = layout.exports_subdir
    store = runtime.record_store(
        out_dir,
        plots_subdir=(plots_cfg if plots_cfg not in ("", ".", "./") else None),
        exports_subdir=(exports_cfg if exports_cfg not in ("", ".", "./") else None),
    )
    ctx = build_run_context(
        decl=decl,
        runtime=runtime,
        out_dir=out_dir,
        store=store,
        logger=logger,
        palette_book=resolve_palette_book(decl=decl, steps=all_steps, dry_run=dry_run),
    )
    categories = collect_categories(list(workbench.plugin_steps()))
    registry = runtime.plugins if categories else None

    if dry_run:
        console.print(Panel.fit("DRY RUN — printing plan", border_style="yellow", box=box.ROUNDED))
        explain(
            decl,
            console=console,
            registry=registry,
            plot_specs=plot_steps,
            export_specs=export_steps,
            runtime=runtime,
        )
        return

    ctx.logger.info(
        "run • pipeline=%d step(s)%s • plots=%d spec(s) • exports=%d spec(s)",
        len(pipeline_steps),
        " (skipped)" if not include_pipeline else "",
        len(plot_steps),
        len(export_steps),
    )
    if not verbose:
        console.print(
            f"[accent]run[/accent] • pipeline={len(pipeline_steps)} step(s){' (skipped)' if not include_pipeline else ''} "
            f"• plots={len(plot_steps)} spec(s) • exports={len(export_steps)} spec(s)"
        )

    if pipeline_steps and registry is None:
        raise ConfigError("pipeline execution requires plugin workbench steps")
    if plot_steps and registry is None:
        raise ConfigError("plot execution requires plugin workbench specs")
    if export_steps and registry is None:
        raise ConfigError("export execution requires plugin workbench specs")

    run_steps(
        items=pipeline_steps,
        phase="pipeline",
        verbose=verbose,
        console=console,
        ctx=ctx,
        store=store,
        registry=registry,
    )
    if plot_steps:
        run_steps(
            items=plot_steps,
            phase="plots",
            verbose=verbose,
            console=console,
            ctx=ctx,
            store=store,
            registry=registry,
        )
    if export_steps:
        run_steps(
            items=export_steps,
            phase="exports",
            verbose=verbose,
            console=console,
            ctx=ctx,
            store=store,
            registry=registry,
        )

    if show_next_steps:
        next_steps = build_next_steps(decl, job_label=job_label, runtime=runtime)
        table = Table(show_header=True, header_style="bold", box=box.ROUNDED, expand=False)
        table.add_column("Command", style="accent")
        table.add_column("What it does")
        for command, description in next_steps:
            table.add_row(command, description)
        console.print(
            Panel(
                table,
                border_style="green",
                box=box.ROUNDED,
                title=f"Records generated in [path]{out_dir}[/path]",
                title_align="left",
            )
        )
        return
    console.print(Panel.fit(f"✓ Done — outputs in {str(out_dir)}", border_style="green", box=box.ROUNDED))


def run_job(
    spec_path: Path,
    *,
    resume_from: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    verbose: bool = True,
    console: Console | None = None,
    include_pipeline: bool = True,
    include_plots: bool = True,
    include_exports: bool = True,
    job_label: str | None = None,
    show_next_steps: bool = False,
    runtime: ReaderRuntime | None = None,
) -> None:
    runtime = runtime or builtin_runtime()
    decl = load_workbench_decl(spec_path, protocols=runtime.protocols)
    run_spec(
        decl,
        resume_from=resume_from,
        until=until,
        dry_run=dry_run,
        log_level=log_level,
        verbose=verbose,
        console=console,
        include_pipeline=include_pipeline,
        include_plots=include_plots,
        include_exports=include_exports,
        job_label=job_label,
        show_next_steps=show_next_steps,
        runtime=runtime,
    )
