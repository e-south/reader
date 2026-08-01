from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from reader_workbench.errors import ConfigError, ExecutionError, InvocationFinalizationError
from reader_workbench.plotting.mpl import ensure_mpl_cache_dir
from reader_workbench.runtime import ReaderRuntime, builtin_runtime
from reader_workbench.workbench.commands import reader_command
from reader_workbench.workbench.decl import WorkbenchDecl, load_workbench_decl
from reader_workbench.workbench.graph import ensure_unique_workbench_ids, resolve_workbench
from reader_workbench.workbench.paths import resolve_confined_sink_root
from reader_workbench.workbench.records import current_build_identity
from reader_workbench.workbench.records.locking import provenance_lock_scope

from ._shared import collect_categories
from .execution import run_steps
from .invocations import (
    ExecutionResult,
    InvocationLedger,
    ProducedRecordRevision,
    SelectedSteps,
    capture_revision_snapshot,
    declared_input_projection,
    produced_record_revisions,
)
from .planning import build_next_steps, explain
from .setup import build_run_context, configure_logger, normalize_log_level, resolve_palette_book, slice_pipeline_steps
from .validation import (
    _assert_no_source_record_output_collisions,
    _planned_output_record_ids,
    _source_record_preflight_issues,
    validation_summary,
)


def run_spec(
    decl: WorkbenchDecl,
    *,
    resume_from: str | None = None,
    until: str | None = None,
    reset_records: bool = False,
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
) -> ExecutionResult:
    os.environ.setdefault("ARROW_LOG_LEVEL", "FATAL")
    runtime = runtime or builtin_runtime()
    layout = decl.experiment_semantics.layout
    try:
        out_dir = resolve_confined_sink_root(layout.outputs_dir, root=decl.experiment.root, label="outputs")
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if out_dir.exists() and not out_dir.is_dir():
        raise ConfigError(f"paths.outputs points to a file: {out_dir}")

    console = console or Console()
    normalize_log_level(log_level)

    workbench = resolve_workbench(decl)
    pipeline_steps = list(workbench.pipeline)
    plot_steps = list(plot_specs) if plot_specs is not None else list(workbench.plots)
    export_steps = list(export_specs) if export_specs is not None else list(workbench.exports)
    ensure_unique_workbench_ids(pipeline_steps, plot_steps, export_steps)

    if not include_pipeline:
        if resume_from or until:
            raise ConfigError(
                f"--from/--until require pipeline execution; use {reader_command('run')} for sliced runs."
            )
        pipeline_steps = []
    else:
        pipeline_steps = slice_pipeline_steps(pipeline_steps, resume_from=resume_from, until=until)

    if reset_records:
        if dry_run:
            raise ConfigError("reset_records cannot be used for a dry run")
        if not include_pipeline or pipeline_steps != list(workbench.pipeline):
            raise ConfigError("reset_records requires a complete pipeline run without --from, --until, or --only")

    if not include_plots:
        plot_steps = []
    if not include_exports:
        export_steps = []

    all_steps = pipeline_steps + plot_steps + export_steps
    categories = collect_categories(list(workbench.plugin_steps()))
    registry = runtime.plugins if categories else None
    selected_steps = SelectedSteps(
        pipeline=tuple(step.id for step in pipeline_steps),
        plots=tuple(step.id for step in plot_steps),
        exports=tuple(step.id for step in export_steps),
    )
    requested_operations = [
        operation
        for operation, enabled in (
            ("run", include_pipeline),
            ("plot", include_plots),
            ("export", include_exports),
        )
        if enabled
    ]
    operation = requested_operations[0] if len(requested_operations) == 1 else "mixed"

    if dry_run:
        validation_summary(
            decl,
            check_files=False,
            exp_root=decl.experiment.root,
            runtime=runtime,
            plot_specs_override=plot_steps,
            export_specs_override=export_steps,
        )
        console.print(Panel.fit("DRY RUN — printing plan", border_style="yellow", box=box.ROUNDED))
        explain(
            decl,
            console=console,
            registry=registry,
            plot_specs=plot_steps,
            export_specs=export_steps,
            runtime=runtime,
        )
        return ExecutionResult(
            invocation_id=None,
            provenance_epoch_id=None,
            operation=operation,
            status="planned",
            dry_run=True,
            selected_steps=selected_steps,
            produced_record_revisions=(),
            ledger_path=None,
        )

    if registry is not None:
        planned_record_ids = _planned_output_record_ids(
            pipeline_items=pipeline_steps,
            plot_items=plot_steps,
            export_items=export_steps,
            registry=registry,
        )
        _assert_no_source_record_output_collisions(
            items=pipeline_steps,
            planned_record_ids=planned_record_ids,
            experiment_root=decl.experiment.root,
        )
        _, source_record_issues = _source_record_preflight_issues(
            items=pipeline_steps,
            registry=registry,
            contracts=runtime.contracts,
        )
        if source_record_issues:
            raise ConfigError("Run preflight failed before output mutation: " + source_record_issues[0])

    plots_cfg = layout.plots_subdir
    exports_cfg = layout.exports_subdir
    store = runtime.record_store(
        out_dir,
        plots_subdir=(plots_cfg if plots_cfg not in ("", ".", "./") else None),
        exports_subdir=(exports_cfg if exports_cfg not in ("", ".", "./") else None),
        experiment_root=decl.experiment.root,
        create=False,
    )
    preserved_output_paths = (layout.subdir_path("notebooks"),)
    if reset_records:
        store.validate_generated_epoch_reset(preserved_paths=preserved_output_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    if reset_records:
        store.manifests_dir.mkdir(parents=True, exist_ok=True)
    else:
        store.ensure_layout()
    logger = configure_logger(out_dir=out_dir, log_level=log_level, verbose=verbose, console=console)
    if plot_steps:
        ensure_mpl_cache_dir()
    with provenance_lock_scope(
        store.provenance_lock,
        acquire_error=ExecutionError("Could not acquire the experiment writer lease"),
        release_error=ExecutionError(
            "Execution completed, but Reader could not release the experiment writer lease. "
            "Do not retry blindly; run reader verify before continuing."
        ),
        release_note="Reader also could not release the experiment writer lease",
    ):
        return _run_spec_locked(
            decl=decl,
            runtime=runtime,
            out_dir=out_dir,
            store=store,
            logger=logger,
            all_steps=all_steps,
            pipeline_steps=pipeline_steps,
            plot_steps=plot_steps,
            export_steps=export_steps,
            registry=registry,
            selected_steps=selected_steps,
            operation=operation,
            reset_records=reset_records,
            verbose=verbose,
            console=console,
            include_pipeline=include_pipeline,
            show_next_steps=show_next_steps,
            job_label=job_label,
        )


def _run_spec_locked(
    *,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
    out_dir: Path,
    store: Any,
    logger: Any,
    all_steps: list[Any],
    pipeline_steps: list[Any],
    plot_steps: list[Any],
    export_steps: list[Any],
    registry: Any,
    selected_steps: SelectedSteps,
    operation: str,
    reset_records: bool,
    verbose: bool,
    console: Console,
    include_pipeline: bool,
    show_next_steps: bool,
    job_label: str | None,
) -> ExecutionResult:
    """Execute one mutating plan while the caller holds the experiment writer lease."""

    provenance_epoch_id = (
        store.reset_generated_epoch(preserved_paths=(decl.experiment_semantics.layout.subdir_path("notebooks"),))
        if reset_records
        else store.provenance_epoch_id()
    )
    store.bind_provenance_epoch(provenance_epoch_id)
    ledger = InvocationLedger(
        experiment_root=decl.experiment.root,
        outputs_dir=out_dir,
        provenance_epoch_id=provenance_epoch_id,
        epoch_guard=store.assert_provenance_epoch,
        writer_lock=store.provenance_lock,
    )
    ctx = build_run_context(
        decl=decl,
        runtime=runtime,
        out_dir=out_dir,
        store=store,
        logger=logger,
        palette_book=resolve_palette_book(decl=decl, steps=all_steps, dry_run=False),
    )
    steps_by_phase = {
        "pipeline": pipeline_steps,
        "plots": plot_steps,
        "exports": export_steps,
    }
    attempt = ledger.append_attempt(
        config_digest=ctx.config_digest,
        build_identity=current_build_identity(),
        operation=operation,
        selected_step_ids=selected_steps.to_dict(),
        declared_inputs=declared_input_projection(
            steps_by_phase=steps_by_phase,
            experiment_root=decl.experiment.root,
        ),
    )
    before_revisions: dict[str, dict] = {}

    try:
        before_revisions = capture_revision_snapshot(store)
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

        revision_payloads = produced_record_revisions(
            before=before_revisions,
            after=capture_revision_snapshot(store),
        )
    except BaseException as failure:
        try:
            revisions = produced_record_revisions(
                before=before_revisions,
                after=capture_revision_snapshot(store),
            )
        except BaseException:
            revisions = []
        try:
            ledger.append_result(
                attempt,
                exit_status=130 if isinstance(failure, KeyboardInterrupt) else 1,
                produced_record_revisions=revisions,
                failure=failure,
            )
        except BaseException as ledger_failure:
            failure.add_note(
                f"Reader also could not persist the failed invocation result ({type(ledger_failure).__name__})."
            )
        raise

    try:
        ledger.append_result(
            attempt,
            exit_status=0,
            produced_record_revisions=revision_payloads,
        )
    except BaseException as failure:
        raise InvocationFinalizationError(
            "Execution records were committed, but Reader could not confirm the invocation result. "
            "Keep the committed evidence and run reader verify before handoff.",
            invocation_id=attempt.invocation_id,
            produced_record_revisions=tuple(dict(item) for item in revision_payloads),
        ) from failure

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
    else:
        console.print(Panel.fit(f"✓ Done — outputs in {str(out_dir)}", border_style="green", box=box.ROUNDED))
    return ExecutionResult(
        invocation_id=attempt.invocation_id,
        provenance_epoch_id=provenance_epoch_id,
        operation=operation,
        status="succeeded",
        dry_run=False,
        selected_steps=selected_steps,
        produced_record_revisions=tuple(
            ProducedRecordRevision.from_payload(revision) for revision in revision_payloads
        ),
        ledger_path=ledger.path,
    )


def run_job(
    spec_path: Path,
    *,
    resume_from: str | None = None,
    until: str | None = None,
    reset_records: bool = False,
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
) -> ExecutionResult:
    runtime = runtime or builtin_runtime()
    decl = load_workbench_decl(spec_path, protocols=runtime.protocols)
    return run_spec(
        decl,
        resume_from=resume_from,
        until=until,
        reset_records=reset_records,
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
