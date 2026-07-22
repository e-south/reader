from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel

from reader.workbench.commands import reader_command

from . import shared
from ._lazy import load as _load
from .helpers import bind_decl_protocol, format_job_arg, load_job_models
from .shared import emit_json, normalize_output_format, table


def spec_overrides():
    return _load("reader.workbench.spec_overrides")


def validate_list_mode_flags(
    *,
    list_only: bool,
    dry_run: bool,
    inputs: list[str] | None,
    sets: list[str] | None,
) -> None:
    if not list_only:
        return
    if dry_run:
        raise typer.BadParameter("--dry-run cannot be combined with --list")
    if inputs:
        raise typer.BadParameter("--input cannot be combined with --list")
    if sets:
        raise typer.BadParameter("--set cannot be combined with --list")


def apply_surface_overrides(
    selected,
    *,
    inputs: list[str] | None,
    sets: list[str] | None,
    experiment_root: Path,
    resources,
):
    overrides = spec_overrides()
    input_overrides = overrides.parse_input_overrides(inputs or [], root=experiment_root, resources=resources)
    set_overrides = overrides.parse_set_overrides(sets or [])
    selected = overrides.apply_step_overrides(
        selected,
        input_overrides=input_overrides,
        set_overrides=set_overrides,
        root=experiment_root,
        resources=resources,
    )
    return overrides, selected


def _raise_dependency_preflight_errors(*, selected, runtime, bound_protocol, exp_root: Path, label: str) -> None:
    errors: list[str] = []
    for step in selected:
        plugin_cls = runtime.plugins.resolve(step.plugin)
        cfg = plugin_cls.ConfigModel.model_validate(
            bound_protocol.effective_plugin_config(plugin_id=step.plugin, step_with=(step.with_ or {}))
        )
        for issue in plugin_cls.preflight_readiness(exp_dir=exp_root, cfg=cfg, reads=(step.reads or {})):
            if issue.kind == "dependency":
                errors.append(f"{label}:{step.id} • {issue.message}")
    if errors:
        raise typer.BadParameter("\n".join(errors))


def validate_plot_job_for_execution(
    job_path: Path,
    *,
    only: list[str] | None,
    exclude: list[str] | None,
    dry_run: bool,
    inputs: list[str] | None,
    sets: list[str] | None,
    ensure_active_lifecycle_fn: Callable,
    require_dataframe_records_fn: Callable,
) -> None:
    runtime = _load("reader.runtime").builtin_runtime()
    _, decl = load_job_models(job_path, runtime=runtime)
    if not dry_run:
        ensure_active_lifecycle_fn(decl, job_path, command_name="plot")
    workbench = _load("reader.workbench.graph").resolve_workbench(decl)
    plot_specs = list(workbench.plots)
    if not plot_specs:
        raise typer.BadParameter("No plots configured in this experiment. Add plots to the config.")
    selected = spec_overrides().select_surface_specs(
        plot_specs, only=only or [], exclude=exclude or [], kind="plot spec"
    )
    if not selected:
        raise typer.BadParameter("No plots selected. Adjust --only/--exclude or use --list to inspect valid ids.")
    _, selected = apply_surface_overrides(
        selected,
        inputs=inputs,
        sets=sets,
        experiment_root=decl.experiment.root,
        resources=decl.experiment_semantics.resources,
    )
    if dry_run:
        _raise_dependency_preflight_errors(
            selected=selected,
            runtime=runtime,
            bound_protocol=bind_decl_protocol(decl=decl, runtime=runtime),
            exp_root=decl.experiment.root,
            label="plot",
        )
    if not dry_run:
        require_dataframe_records_fn(decl, job_path, runtime=runtime)


def render_surface_specs_table(
    *,
    title_text: str,
    selected,
    runtime,
    record_producers,
    summaries: dict[str, str],
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


def surface_next_steps(
    *,
    job_hint: str | None,
    output_dir: Path,
    include_plot: bool,
    include_export: bool,
) -> None:
    def _cmd(base: str, tail: str = "") -> str:
        return reader_command(base, job_hint, tail)

    lines = [f"Files saved in [path]{output_dir}[/path]", "", "Next steps:"]
    if include_plot:
        lines.append(f"  {_cmd('plot')}")
    if include_export:
        lines.append(f"  {_cmd('export')}")
    lines.append(f"  {_cmd('notebook')}")
    shared.console.print(Panel.fit("\n".join(lines), border_style="green", box=box.ROUNDED))


def run_plot_job(
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
    ensure_active_lifecycle_fn: Callable,
    require_dataframe_records_fn: Callable,
    append_journal_fn: Callable,
) -> None:
    _, decl = load_job_models(job_path)
    if not list_only and not dry_run:
        ensure_active_lifecycle_fn(decl, job_path, command_name="plot")
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
            require_dataframe_records_fn(decl, job_path, runtime=runtime)
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
                Panel.fit("No plots configured in this experiment.", border_style="warn", box=box.ROUNDED)
            )
            return
        raise typer.BadParameter("No plots configured in this experiment. Add plots to the config.")
    selected = spec_overrides().select_surface_specs(
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
        render_surface_specs_table(
            title_text="Plots",
            selected=selected,
            runtime=runtime,
            record_producers=record_producers,
            summaries=inspection_runtime.plot_output_summaries(bound_protocol),
        )
        return
    if not selected:
        raise typer.BadParameter("No plots selected. Adjust --only/--exclude or use --list to inspect valid ids.")
    experiment_root = decl.experiment.root
    resources = decl.experiment_semantics.resources
    overrides, selected = apply_surface_overrides(
        selected,
        inputs=inputs,
        sets=sets,
        experiment_root=experiment_root,
        resources=resources,
    )
    if dry_run:
        _raise_dependency_preflight_errors(
            selected=selected,
            runtime=runtime,
            bound_protocol=bound_protocol,
            exp_root=experiment_root,
            label="plot",
        )
    if not dry_run:
        append_journal_fn(
            job_path,
            " ".join(
                overrides.build_surface_command(
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
        surface_next_steps(
            job_hint=format_job_arg(job_hint),
            output_dir=plots_dir,
            include_plot=False,
            include_export=bool(workbench.exports),
        )


def run_export_job(
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
    ensure_active_lifecycle_fn: Callable,
    require_dataframe_records_fn: Callable,
    append_journal_fn: Callable,
) -> None:
    _, decl = load_job_models(job_path)
    if not list_only and not dry_run:
        ensure_active_lifecycle_fn(decl, job_path, command_name="export")
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
                Panel.fit("No exports configured in this experiment.", border_style="warn", box=box.ROUNDED)
            )
            return
        raise typer.BadParameter("No exports configured in this experiment. Add exports to the config.")
    selected = spec_overrides().select_surface_specs(
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
        render_surface_specs_table(
            title_text="Exports",
            selected=selected,
            runtime=runtime,
            record_producers=record_producers,
            summaries=inspection_runtime.export_output_summaries(bound_protocol),
        )
        return
    if not selected:
        raise typer.BadParameter("No exports selected. Adjust --only/--exclude or use --list to inspect valid ids.")
    if fmt == "json":
        raise typer.BadParameter("--format json is only supported with --list")
    if not dry_run:
        require_dataframe_records_fn(decl, job_path, runtime=runtime)
    experiment_root = decl.experiment.root
    resources = decl.experiment_semantics.resources
    overrides, selected = apply_surface_overrides(
        selected,
        inputs=inputs,
        sets=sets,
        experiment_root=experiment_root,
        resources=resources,
    )
    if not dry_run:
        append_journal_fn(
            job_path,
            " ".join(
                overrides.build_surface_command(
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
        surface_next_steps(
            job_hint=format_job_arg(job_hint),
            output_dir=exports_dir,
            include_plot=bool(workbench.plots),
            include_export=False,
        )
