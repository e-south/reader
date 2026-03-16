"""
--------------------------------------------------------------------------------
<reader project>
src/reader/workbench/cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import typer
import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.traceback import install as rich_tracebacks

from reader.errors import ConfigError, ReaderError, RecordError
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl, build_workbench_decl
from reader.workbench.engine import explain as explain_job
from reader.workbench.engine import run_job, run_spec
from reader.workbench.engine import validate as validate_job
from reader.workbench.engine._shared import pipeline_has_plugin
from reader.workbench.experiment import ResourceCatalog
from reader.workbench.experiments import discover_experiment_configs
from reader.workbench.graph import (
    FileRef,
    InputRef,
    OutputRef,
    RecordRef,
    ResourceRef,
    input_ref_to_dict,
    materialize_workbench,
    output_ref_to_dict,
    resolve_workbench,
    select_workbench_specs,
)
from reader.workbench.notebooks import write_experiment_notebook
from reader.workbench.templates import (
    builtin_notebook_template_catalog,
    compatible_notebook_templates,
    require_notebook_template_for_protocol,
    resolve_notebook_template_descriptor,
)

THEME = Theme(
    {
        "title": "bold cyan",
        "accent": "cyan",
        "ok": "bold green",
        "warn": "bold yellow",
        "error": "bold red",
        "muted": "dim",
        "path": "magenta",
    }
)

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help=(
        "reader — experiment pipeline runner.\n\n"
        "Run pipelines to generate dataframe records, then render plots, exports, or notebooks. "
        "Start with 'reader demo' or 'reader ls'."
    ),
)
console = Console(theme=THEME)
rich_tracebacks(show_locals=False)

PLOT_ONLY_OPTION = typer.Option(None, "--only", help="Run only the specified plot id (repeatable).")
PLOT_EXCLUDE_OPTION = typer.Option(None, "--exclude", help="Exclude the specified plot id (repeatable).")
PLOT_INPUT_OPTION = typer.Option(
    None,
    "--input",
    metavar="KEY=VALUE",
    help="Override reads bindings for selected plot specs (repeatable).",
)
PLOT_SET_OPTION = typer.Option(
    None,
    "--set",
    metavar="PATH=VALUE",
    help="Patch spec fields for selected plots (reads.*, with.*, writes.*). Repeatable.",
)
EXPORT_ONLY_OPTION = typer.Option(None, "--only", help="Run only the specified export id (repeatable).")
EXPORT_EXCLUDE_OPTION = typer.Option(None, "--exclude", help="Exclude the specified export id (repeatable).")
EXPORT_INPUT_OPTION = typer.Option(
    None,
    "--input",
    metavar="KEY=VALUE",
    help="Override reads bindings for selected export specs (repeatable).",
)
EXPORT_SET_OPTION = typer.Option(
    None,
    "--set",
    metavar="PATH=VALUE",
    help="Patch spec fields for selected exports (reads.*, with.*, writes.*). Repeatable.",
)
NOTEBOOK_MODE_OPTION = typer.Option(
    "edit",
    "--mode",
    help="Launch mode: edit | run | none (default: edit).",
)
NOTEBOOK_PLOT_ONLY_OPTION = typer.Option(
    None,
    "--only",
    help="Filter plot ids when using a notebook template that supports plot selection (repeatable).",
)
NOTEBOOK_PLOT_EXCLUDE_OPTION = typer.Option(
    None,
    "--exclude",
    help="Exclude plot ids when using a notebook template that supports plot selection (repeatable).",
)


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    """Show help when no command is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _checkmark(cond: bool) -> str:
    return "[ok]✓[/ok]" if cond else "[muted]—[/muted]"


def _table(title: str) -> Table:
    return Table(
        title=f"[title]{title}[/title]",
        title_justify="left",
        header_style="bold",
        box=box.ROUNDED,
        expand=True,
        show_lines=False,
        show_edge=True,
    )


def _abort(msg: str, *, code: int = 1) -> None:
    console.print(Panel.fit(f"[error]✗ {msg}[/error]", border_style="error", box=box.ROUNDED))
    raise typer.Exit(code=code)


def _handle_reader_error(err: ReaderError) -> None:
    _abort(str(err))


def _render_marimo_help(target: Path, *, mode: str, has_fcs: bool) -> None:
    sync_cmd = "uv sync --locked --group notebooks"
    if has_fcs:
        sync_cmd = f"{sync_cmd} --group cytometry"
    marimo_cmd = f"{sys.executable} -m marimo {mode} {target}"
    uvx_cmd = f"uvx marimo {mode} --sandbox {target}"
    console.print(
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
    cmd = [sys.executable, "-m", "marimo", mode, str(target)]
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        _render_marimo_help(target, mode=mode, has_fcs=has_fcs)
        raise typer.Exit(code=1) from None
    if result.returncode != 0:
        _render_marimo_help(target, mode=mode, has_fcs=has_fcs)
        raise typer.Exit(code=1)


@app.command(help="List built-in protocols or describe one.")
def protocols(
    name: str | None = typer.Argument(
        None,
        metavar="[NAME]",
        help="Optional protocol id to describe (e.g., plate_reader/dual_reporter_screen).",
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        metavar="NAME",
        help="Filter protocols by semantic domain.",
    ),
    family: str | None = typer.Option(
        None,
        "--family",
        metavar="NAME",
        help="Filter protocols by semantic family.",
    ),
):
    runtime = builtin_runtime()
    try:
        if name:
            descriptor = runtime.protocols.resolve(name)
            table = _table(f"Protocol: {descriptor.protocol}")
            table.add_column("Section", style="accent")
            table.add_column("Details")
            table.add_row("Domain", descriptor.domain)
            table.add_row("Family", descriptor.family)
            table.add_row("Summary", descriptor.summary)
            if descriptor.tags:
                table.add_row("Tags", ", ".join(descriptor.tags))
            if descriptor.factors:
                table.add_row("Factors", ", ".join(f"{item.name} ({item.role})" for item in descriptor.factors))
            if descriptor.windows:
                table.add_row("Windows", ", ".join(item.id for item in descriptor.windows))
            if descriptor.metrics:
                table.add_row("Metrics", ", ".join(item.id for item in descriptor.metrics))
            if descriptor.ranking is not None:
                table.add_row("Primary ranking", descriptor.ranking.primary_metric)
            table.add_row("Default notebook", descriptor.execution.notebook.default_template)
            table.add_row("Allowed notebooks", ", ".join(descriptor.execution.notebook.allowed_templates))
            if descriptor.deliverables:
                plot_ids = [item.id for item in descriptor.deliverables if item.surface == "plots"]
                export_ids = [item.id for item in descriptor.deliverables if item.surface == "exports"]
                if plot_ids:
                    table.add_row("Plot deliverables", ", ".join(plot_ids))
                if export_ids:
                    table.add_row("Export deliverables", ", ".join(export_ids))
            if descriptor.execution.plugin_defaults:
                table.add_row(
                    "Plugin defaults", ", ".join(item.plugin for item in descriptor.execution.plugin_defaults)
                )
            console.print(Panel(table, border_style="accent", box=box.ROUNDED))
            return

        table = _table("Protocols")
        table.add_column("Name", style="accent", min_width=24)
        table.add_column("Domain")
        table.add_column("Family")
        table.add_column("Description")
        for descriptor in runtime.protocols.all():
            if domain and descriptor.domain != domain:
                continue
            if family and descriptor.family != family:
                continue
            table.add_row(descriptor.protocol, descriptor.domain, descriptor.family, descriptor.summary)
        console.print(Panel(table, border_style="accent", box=box.ROUNDED))
    except ConfigError as e:
        raise typer.BadParameter(str(e)) from e


def _load_job_models(job_path: Path) -> tuple[ReaderSpec, WorkbenchDecl]:
    runtime = builtin_runtime()
    spec = ReaderSpec.load(job_path)
    return spec, build_workbench_decl(spec, source_path=job_path, protocols=runtime.protocols)


def _has_sfxi_step(decl: WorkbenchDecl, *, runtime: ReaderRuntime) -> bool:
    return pipeline_has_plugin(decl, runtime=runtime, tag="sfxi")


def _dataframe_record_contracts(
    outputs_dir: Path,
    *,
    runtime: ReaderRuntime,
    exact: str | None = None,
    prefix: str | None = None,
) -> list[str]:
    contract_catalog = runtime.contracts
    store = runtime.record_store(outputs_dir, create=False)
    if not store.catalog_exists():
        return []
    try:
        records = store.iter_latest_records(kind="dataframe_artifact")
    except RecordError:
        return []
    matches: list[str] = []
    for record in records:
        contract = record.contract_id
        if exact and contract_catalog.satisfies(actual=contract, expected=exact):
            matches.append(contract)
            continue
        if prefix and contract.startswith(prefix):
            matches.append(contract)
    return matches


def _template_requirements_satisfied(
    template_name: str,
    decl: WorkbenchDecl,
    outputs_dir: Path,
    *,
    runtime: ReaderRuntime,
) -> bool:
    descriptor = resolve_notebook_template_descriptor(template_name)
    requirements = descriptor.capabilities.requires_any
    if not requirements:
        return True
    record_contracts: list[str] | None = None
    for requirement in requirements:
        if requirement.plugin and pipeline_has_plugin(decl, runtime=runtime, plugin=requirement.plugin):
            return True
        if requirement.domain and pipeline_has_plugin(decl, runtime=runtime, domain=requirement.domain):
            return True
        if requirement.tag and pipeline_has_plugin(decl, runtime=runtime, tag=requirement.tag):
            return True
        if requirement.record_contract or requirement.record_contract_prefix:
            if record_contracts is None:
                record_contracts = _dataframe_record_contracts(
                    outputs_dir,
                    runtime=runtime,
                    exact=requirement.record_contract,
                    prefix=requirement.record_contract_prefix,
                )
            elif requirement.record_contract or requirement.record_contract_prefix:
                exact = requirement.record_contract
                prefix = requirement.record_contract_prefix
                record_contracts.extend(
                    _dataframe_record_contracts(outputs_dir, runtime=runtime, exact=exact, prefix=prefix)
                )
            if record_contracts:
                return True
    return False


def _bind_decl_protocol(*, decl: WorkbenchDecl, runtime: ReaderRuntime):
    return runtime.bind_protocol(decl.experiment_semantics.protocol)


def _default_notebook_name() -> str:
    return f"EDA_{datetime.now().strftime('%Y%m%d')}.py"


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


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
                job_path = _infer_job_path(job)
                _, decl = _load_job_models(job_path)
                bound_protocol = _bind_decl_protocol(decl=decl, runtime=runtime)
                title = f"Notebook templates: {bound_protocol.id}"
                descriptors = compatible_notebook_templates(protocol=bound_protocol)
            table = _table(title)
            table.add_column("Name", style="accent")
            table.add_column("Domain")
            table.add_column("Family")
            if bound_protocol is not None:
                table.add_column("Default", justify="center")
            table.add_column("Description")
            for descriptor in descriptors:
                row = [descriptor.template, descriptor.domain, descriptor.family]
                if bound_protocol is not None:
                    row.append("yes" if descriptor.template == bound_protocol.default_notebook_template else "")
                row.append(descriptor.summary)
                table.add_row(*row)
            console.print(Panel(table, border_style="accent", box=box.ROUNDED))
            return
        if overwrite and new:
            raise typer.BadParameter("--overwrite cannot be combined with --new.")
        if refresh:
            overwrite = True
        mode_value = (mode or "").strip().lower()
        if mode_value not in {"edit", "run", "none"}:
            raise typer.BadParameter("--mode must be one of: edit, run, none.")
        job_path = _infer_job_path(job)
        exp_dir = job_path.parent
        _, decl = _load_job_models(job_path)
        runtime = builtin_runtime()
        bound_protocol = _bind_decl_protocol(decl=decl, runtime=runtime)
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
        if not _template_requirements_satisfied(selected_template, decl, outputs_dir, runtime=runtime):
            raise typer.BadParameter(
                f"Template {descriptor.template} does not satisfy its declared requirements for this experiment. "
                "Check pipeline assets or existing dataframe records."
            )
        notebooks_cfg = layout.notebooks_subdir
        nb_dir = outputs_dir if notebooks_cfg in ("", ".", "./") else outputs_dir / str(notebooks_cfg)
        target_name = name or _default_notebook_name()
        target = nb_dir / target_name
        if new:
            target = _next_available_path(target)
        elif overwrite and target.exists():
            confirm = typer.confirm(
                f"Notebook already exists at {target}. Overwrite?",
                default=False,
            )
            if not confirm:
                overwrite = False
        has_fcs = any(p.suffix.lower() == ".fcs" for p in exp_dir.rglob("*.fcs"))
        existed = target.exists()
        plot_specs_payload = None
        if descriptor.capabilities.inject_plot_specs:
            selected = _select_steps(plot_specs, only=plot_only or [], exclude=plot_exclude or [], kind="plot spec")
            plot_specs_payload = [_spec_to_dict(s) for s in selected]
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
        console.print(
            Panel.fit(
                status,
                border_style=border_style,
                box=box.ROUNDED,
            )
        )
        if mode_value == "none":
            console.print(str(target))
            return
        launch_cmd = f"{sys.executable} -m marimo {mode_value} {target}"
        console.print(Panel.fit(f"Launching: {launch_cmd}", border_style="accent", box=box.ROUNDED))
        _launch_marimo(mode_value, target, has_fcs=has_fcs)
    except ConfigError as e:
        raise typer.BadParameter(str(e)) from e


@app.command(help="Scaffold an interactive marimo notebook and open it.")
def notebook(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help="Experiment config path, directory, or index from 'reader ls'.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="Notebook filename (created under outputs/notebooks). Defaults to EDA_YYYYMMDD.py.",
    ),
    template_name: str | None = typer.Option(
        None,
        "--template",
        help="Notebook template (defaults to the protocol default or protocol.deliverables.notebook.template).",
    ),
    list_templates: bool = typer.Option(
        False,
        "--list-templates",
        help="List notebook templates and exit.",
    ),
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


def _find_nearest_experiments_dir(start: Path) -> Path:
    """
    Walk up from 'start' to find the closest 'experiments/' directory.
    Falls back to ./experiments under the current working directory.
    """
    for base in [start] + list(start.parents):
        cand = base / "experiments"
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    return (start / "experiments").resolve()


def _infer_job_path(job: str | None) -> Path:
    """
    Resolve CONFIG argument with explicit, assertive rules:
      • If CONFIG exists and is a directory => <dir>/config.yaml (if present) else error
      • If CONFIG exists and is a file     => use it as-is
      • If CONFIG is a pure integer string => treat as 1-based index into nearest experiments/
      • If CONFIG is omitted               => search for nearest 'config.yaml' upward from CWD
    No silent fallbacks.
    """
    if job:
        s = str(job).strip()
        p = Path(s)
        if p.exists():
            if p.is_dir():
                candidate = p / "config.yaml"
                if candidate.exists():
                    return candidate.resolve()
                raise typer.BadParameter(
                    f"CONFIG directory {p!s} has no 'config.yaml'. "
                    "Pass a file path, an experiment directory that contains config.yaml, or a numeric index (see 'reader ls')."
                )
            # existing file
            return p.resolve()

        # Numeric index: resolve against the nearest experiments/ root (same order as `reader ls`)
        if s.isdigit():
            idx = int(s)
            root_path = _find_nearest_experiments_dir(Path.cwd())
            jobs = _find_jobs(root_path)
            if not jobs:
                raise typer.BadParameter(f"No experiments found under {root_path}. Use 'reader ls' first.")
            if idx < 1 or idx > len(jobs):
                raise typer.BadParameter(
                    f"Experiment index out of range: {idx} (valid: 1..{len(jobs)} under {root_path}). "
                    "Use 'reader ls' to see the index numbers."
                )
            return jobs[idx - 1]

        # Not an existing path, not a numeric index → explicit error
        raise typer.BadParameter(
            f"CONFIG not found: {job!r}. Pass a path to a config.yaml, an experiment directory, "
            "or a numeric experiment index from 'reader ls'."
        )

    cwd = Path.cwd()
    # current dir first
    candidate = cwd / "config.yaml"
    if candidate.exists():
        return candidate.resolve()
    # then walk up
    for base in cwd.parents:
        c = base / "config.yaml"
        if c.exists():
            return c.resolve()
    raise typer.BadParameter(
        "Missing CONFIG and no 'config.yaml' found in the current or parent directories. "
        "Run inside an experiment dir or pass a path to the config (or the experiment dir). "
        "Tip: use 'reader ls' to list experiments and pass its index."
    )


def _format_job_arg(job: str | None) -> str | None:
    if job is None:
        return None
    value = str(job).strip()
    return value or None


def _find_jobs(root: Path, *, include_scaffolds: bool = False) -> list[Path]:
    return discover_experiment_configs(root, include_scaffolds=include_scaffolds)


def _find_year_jobs(year: str, root: Path) -> list[Path]:
    year_str = str(year).strip()
    if not year_str:
        raise typer.BadParameter("--year cannot be empty")
    if not year_str.isdigit() or len(year_str) != 4:
        raise typer.BadParameter("--year expects a 4-digit year (e.g., 2025).")
    if not root.exists() or not root.is_dir():
        raise typer.BadParameter(f"Experiments root not found: {root}")
    year_dir = root / year_str
    if not year_dir.exists() or not year_dir.is_dir():
        raise typer.BadParameter(f"No experiments directory for year {year_str} under {root}.")
    jobs = _find_jobs(year_dir)
    if not jobs:
        raise typer.BadParameter(f"No experiments found under {year_dir}.")
    return jobs


def _require_dataframe_records(decl: WorkbenchDecl, job_path: Path, *, runtime: ReaderRuntime) -> None:
    layout = decl.experiment_semantics.layout
    outputs_dir = layout.outputs_dir
    store = runtime.record_store(
        outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        create=False,
    )
    if not store.catalog_exists():
        raise RecordError(
            f"No outputs/manifests/records.json found. Run 'reader run {job_path}' first to generate dataframe records."
        )
    try:
        records = store.iter_latest_records(kind="dataframe_artifact")
    except RecordError as exc:
        raise RecordError(
            f"Could not read record catalog at {store.records_path}. Run 'reader run {job_path}' first."
        ) from exc
    if not records:
        raise RecordError(
            f"No dataframe records listed in outputs/manifests/records.json. Run 'reader run {job_path}' first."
        )


def _append_journal(job_path: Path, command_line: str) -> None:
    exp_dir = job_path.parent
    # Prefer JOURNAL.md if it exists; otherwise create it (uppercase by default)
    journal = exp_dir / (
        "JOURNAL.md" if (exp_dir / "JOURNAL.md").exists() or not (exp_dir / "journal.md").exists() else "journal.md"
    )
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = "" if journal.exists() else "# Experiment Journal\n\n"
    entry = f"### {ts}\n\n```\n{command_line}\n```\n\n"
    journal.write_text(
        header + (journal.read_text(encoding="utf-8") if journal.exists() else "") + entry, encoding="utf-8"
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
):
    # If user didn't override --root, auto-detect nearest experiments/ so this
    # works from anywhere inside the repository.
    if str(root).strip() == "./experiments":
        root_path = _find_nearest_experiments_dir(Path.cwd())
    else:
        root_path = Path(root).resolve()
    jobs = _find_jobs(root_path, include_scaffolds=include_scaffolds)
    if not jobs:
        console.print(
            Panel.fit(
                f"No experiments found under [path]{root_path}[/path].",
                border_style="warn",
                box=box.ROUNDED,
            )
        )
        return
    t = _table("Experiments")
    t.add_column("#", justify="right", style="muted")
    name_values = [p.parent.name for p in jobs]
    max_name = max((len(n) for n in name_values), default=12)
    max_width = int((console.width or 80) * 0.6)
    name_width = max(12, min(max_name + 2, max_width))
    t.add_column("Name", style="accent", max_width=name_width, overflow="fold")
    t.add_column("Outputs", justify="center", width=7)
    for i, p in enumerate(jobs, 1):
        name = p.parent.name
        outputs_ok = False
        try:
            _, decl = _load_job_models(p)
            man = decl.experiment_semantics.layout.outputs_dir / "manifests" / "records.json"
            outputs_ok = man.exists()
        except ReaderError:
            outputs_ok = False
        t.add_row(str(i), name, _checkmark(outputs_ok))
    console.print(
        Panel(
            t,
            border_style="accent",
            box=box.ROUNDED,
            subtitle=f"[muted]root: [path]{root_path}[/path] — {len(jobs)} found[/muted]",
        )
    )


@app.command(help="Show planned steps and contracts (no execution).")
def explain(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
):
    try:
        job_path = _infer_job_path(job)
        _append_journal(job_path, f"reader explain {job_path}")
        _, decl = _load_job_models(job_path)
        explain_job(decl, console=console, runtime=builtin_runtime())
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(help="Validate config, plugin params, reads wiring, and input files.")
def validate(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    no_files: bool = typer.Option(
        False,
        "--no-files",
        help="Skip file existence checks (config-only validation).",
    ),
):
    try:
        job_path = _infer_job_path(job)
        _append_journal(job_path, f"reader validate {job_path}")
        _, decl = _load_job_models(job_path)
        validate_job(
            decl,
            console=console,
            check_files=not no_files,
            exp_root=decl.experiment.root,
            runtime=builtin_runtime(),
        )
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(help="Print the expanded config (recipes + overrides applied).")
def config(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        metavar="FMT",
        help="Output format: yaml | json (default: yaml).",
    ),
):
    try:
        job_path = _infer_job_path(job)
        spec, decl = _load_job_models(job_path)
    except ReaderError as e:
        _handle_reader_error(e)
    fmt = str(format).strip().lower()
    payload = spec.model_dump(by_alias=True)
    materialized = materialize_workbench(decl)
    payload.setdefault("pipeline", {})
    payload["pipeline"]["steps"] = materialized["pipeline"]
    payload.setdefault("plots", {})
    payload.setdefault("exports", {})
    payload.setdefault("notebooks", {})
    payload["plots"]["specs"] = materialized["plots"]
    payload["exports"]["specs"] = materialized["exports"]
    payload["notebooks"]["specs"] = materialized["notebooks"]
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if fmt == "yaml":
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
    only: str | None = typer.Option(
        None,
        "--only",
        metavar="STEP_ID",
        help="Run exactly one pipeline step by id.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan only: validate and print the plan without executing steps.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        metavar="LEVEL",
        help="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL (default: INFO).",
    ),
    compact: bool = typer.Option(
        False,
        "--compact",
        help="Use concise progress output instead of per-step logs.",
    ),
):
    job_path = _infer_job_path(job)
    parts = ["reader run", str(job_path)]

    if only and (from_step or until):
        raise typer.BadParameter("--only cannot be combined with --from/--until")

    try:
        _, decl = _load_job_models(job_path)
    except ReaderError as e:
        _handle_reader_error(e)

    if only:
        _resolve_pipeline_step_id(decl, only)
        parts += ["--only", only]
        if dry_run:
            parts += ["--dry-run"]
        if log_level and log_level != "INFO":
            parts += ["--log-level", log_level]
        if compact:
            parts += ["--compact"]
        _append_journal(job_path, " ".join(parts))
        try:
            runtime = builtin_runtime()
            run_job(
                job_path,
                resume_from=only,
                until=only,
                dry_run=dry_run,
                log_level=log_level,
                verbose=not compact,
                console=console,
                include_pipeline=True,
                include_plots=False,
                include_exports=False,
                runtime=runtime,
            )
        except ReaderError as e:
            _handle_reader_error(e)
        return

    if from_step:
        _resolve_pipeline_step_id(decl, from_step)
        parts += ["--from", from_step]
    if until:
        _resolve_pipeline_step_id(decl, until)
        parts += ["--until", until]
    if dry_run:
        parts += ["--dry-run"]
    if log_level and log_level != "INFO":
        parts += ["--log-level", log_level]
    if compact:
        parts += ["--compact"]
    _append_journal(job_path, " ".join(parts))
    try:
        runtime = builtin_runtime()
        run_job(
            job_path,
            resume_from=from_step,
            until=until,
            dry_run=dry_run,
            log_level=log_level,
            verbose=not compact,
            console=console,
            include_pipeline=True,
            include_plots=False,
            include_exports=False,
            job_label=_format_job_arg(job),
            show_next_steps=True,
            runtime=runtime,
        )
    except ReaderError as e:
        _handle_reader_error(e)


def _build_plot_command(
    job_path: Path,
    *,
    only: list[str] | None,
    exclude: list[str] | None,
    list_only: bool,
    dry_run: bool,
    log_level: str,
    inputs: list[str] | None,
    sets: list[str] | None,
) -> list[str]:
    parts = ["reader plot", str(job_path)]
    if list_only:
        parts += ["--list"]
    if only:
        for v in only:
            parts += ["--only", v]
    if exclude:
        for v in exclude:
            parts += ["--exclude", v]
    if dry_run:
        parts += ["--dry-run"]
    if log_level and log_level != "INFO":
        parts += ["--log-level", log_level]
    for raw in inputs or []:
        parts += ["--input", raw]
    for raw in sets or []:
        parts += ["--set", raw]
    return parts


def _run_plot_job(
    job_path: Path,
    *,
    job_hint: str | None,
    only: list[str] | None,
    exclude: list[str] | None,
    list_only: bool,
    dry_run: bool,
    log_level: str,
    inputs: list[str] | None,
    sets: list[str] | None,
) -> None:
    _, decl = _load_job_models(job_path)
    runtime = builtin_runtime()
    workbench = resolve_workbench(decl)
    if not list_only:
        _require_dataframe_records(decl, job_path, runtime=runtime)
    plot_specs = list(workbench.plots)
    if not plot_specs:
        if list_only:
            console.print(
                Panel.fit(
                    "No plot specs configured in this experiment.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        raise typer.BadParameter("No plot specs configured in this experiment. Add plots to the config.")
    selected = _select_steps(plot_specs, only=only or [], exclude=exclude or [], kind="plot spec")
    if list_only:
        t = _table("Plots")
        t.add_column("#", justify="right", style="muted")
        t.add_column("id", style="accent")
        t.add_column("plugin")
        for i, s in enumerate(selected, 1):
            t.add_row(str(i), s.id, s.plugin)
        console.print(
            Panel(
                t,
                border_style="accent",
                box=box.ROUNDED,
                subtitle=f"[muted]{len(selected)} total[/muted]",
            )
        )
        return
    experiment_root = decl.experiment.root
    resources = decl.experiment_semantics.resources
    input_overrides = _parse_input_overrides(inputs or [], root=experiment_root, resources=resources)
    set_overrides: list[tuple[str, object]] = []
    for raw in sets or []:
        if "=" not in raw:
            raise typer.BadParameter("--set expects PATH=VALUE")
        path, value_raw = raw.split("=", 1)
        path = path.strip()
        if not path:
            raise typer.BadParameter("--set path cannot be empty")
        value = yaml.safe_load(value_raw)
        set_overrides.append((path, value))
    selected = _apply_step_overrides(
        selected,
        input_overrides=input_overrides,
        set_overrides=set_overrides,
        root=experiment_root,
        resources=resources,
    )
    parts = _build_plot_command(
        job_path,
        only=only,
        exclude=exclude,
        list_only=False,
        dry_run=dry_run,
        log_level=log_level,
        inputs=inputs,
        sets=sets,
    )
    _append_journal(job_path, " ".join(parts))
    run_spec(
        decl,
        dry_run=dry_run,
        log_level=log_level,
        console=console,
        include_pipeline=False,
        include_plots=True,
        include_exports=False,
        plot_specs=selected,
        runtime=runtime,
    )
    if not dry_run:
        job_hint = _format_job_arg(job_hint)
        outputs_dir = decl.experiment_semantics.layout.outputs_dir
        plots_cfg = decl.experiment_semantics.layout.plots_subdir
        plots_dir = outputs_dir if plots_cfg in ("", ".", "./") else outputs_dir / str(plots_cfg)

        def _cmd(base: str, tail: str = "") -> str:
            return f"{base} {job_hint}{tail}" if job_hint else f"{base}{tail}"

        lines = [f"Plots saved in [path]{plots_dir}[/path]", "", "Next steps:"]
        lines.append(f"  {_cmd('reader notebook')}")
        if workbench.exports:
            lines.append(f"  {_cmd('reader export')}")
        console.print(Panel.fit("\n".join(lines), border_style="green", box=box.ROUNDED))


@app.command(help="Save plot files from plot specs using existing dataframe records.")
def plot(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help="Experiment config path, directory, or index from 'reader ls'.",
    ),
    year: str | None = typer.Option(
        None,
        "--year",
        metavar="YYYY",
        help="Run plots for all experiments under experiments/YYYY.",
    ),
    root: str | None = typer.Option(
        None,
        "--root",
        metavar="DIR",
        help="Override experiments root when using --year (default: nearest experiments/).",
    ),
    only: list[str] = PLOT_ONLY_OPTION,
    exclude: list[str] = PLOT_EXCLUDE_OPTION,
    list_only: bool = typer.Option(
        False,
        "--list",
        help="List plot specs for this config and exit.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan only: validate and print the plot plan without executing.",
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
    if year:
        if job is not None:
            raise typer.BadParameter("--year cannot be combined with CONFIG|DIR|INDEX")
        root_path = _find_nearest_experiments_dir(Path.cwd()) if root is None else Path(root).resolve()
        jobs = _find_year_jobs(year, root_path)
        console.print(
            Panel.fit(
                f"Plotting {len(jobs)} experiment(s) for {year} under [path]{root_path}[/path].",
                border_style="accent",
                box=box.ROUNDED,
            )
        )
        failures: list[tuple[Path, str]] = []
        total = len(jobs)
        for idx, job_path in enumerate(jobs, 1):
            exp_name = job_path.parent.name
            cmd_line = " ".join(
                _build_plot_command(
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
            console.print(f"[accent]{idx}/{total}[/accent] {exp_name}")
            console.print(f"[muted]{cmd_line}[/muted]")
            try:
                _run_plot_job(
                    job_path,
                    job_hint=str(job_path),
                    only=only,
                    exclude=exclude,
                    list_only=list_only,
                    dry_run=dry_run,
                    log_level=log_level,
                    inputs=inputs,
                    sets=sets,
                )
            except (ReaderError, typer.BadParameter) as exc:
                failures.append((job_path, str(exc)))
                console.print(
                    Panel.fit(
                        f"[error]✗ {exp_name}: {exc}[/error]",
                        border_style="error",
                        box=box.ROUNDED,
                    )
                )
        if failures:
            lines = [f"{len(failures)} experiment(s) failed while plotting year {year}:"]
            lines += [f"- {path.parent.name}: {msg}" for path, msg in failures]
            _abort("\n".join(lines))
        return

    try:
        job_path = _infer_job_path(job)
        _run_plot_job(
            job_path,
            job_hint=job,
            only=only,
            exclude=exclude,
            list_only=list_only,
            dry_run=dry_run,
            log_level=log_level,
            inputs=inputs,
            sets=sets,
        )
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(help="Run export specs using existing dataframe records.")
def export(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help="Experiment config path, directory, or index from 'reader ls'.",
    ),
    only: list[str] = EXPORT_ONLY_OPTION,
    exclude: list[str] = EXPORT_EXCLUDE_OPTION,
    list_only: bool = typer.Option(
        False,
        "--list",
        help="List export specs for this config and exit.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan only: validate and print the export plan without executing.",
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
        job_path = _infer_job_path(job)
        _, decl = _load_job_models(job_path)
    except ReaderError as e:
        _handle_reader_error(e)
    workbench = resolve_workbench(decl)
    export_specs = list(workbench.exports)
    if not export_specs:
        if list_only:
            console.print(
                Panel.fit(
                    "No export specs configured in this experiment.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        raise typer.BadParameter("No export specs configured in this experiment. Add exports to the config.")
    selected = _select_steps(export_specs, only=only or [], exclude=exclude or [], kind="export spec")
    if list_only:
        t = _table("Exports")
        t.add_column("#", justify="right", style="muted")
        t.add_column("id", style="accent")
        t.add_column("plugin")
        for i, s in enumerate(selected, 1):
            t.add_row(str(i), s.id, s.plugin)
        console.print(
            Panel(
                t,
                border_style="accent",
                box=box.ROUNDED,
                subtitle=f"[muted]{len(selected)} total[/muted]",
            )
        )
        return
    try:
        runtime = builtin_runtime()
        _require_dataframe_records(decl, job_path, runtime=runtime)
        experiment_root = decl.experiment.root
        resources = decl.experiment_semantics.resources
        input_overrides = _parse_input_overrides(inputs or [], root=experiment_root, resources=resources)
        set_overrides: list[tuple[str, object]] = []
        for raw in sets or []:
            if "=" not in raw:
                raise typer.BadParameter("--set expects PATH=VALUE")
            path, value_raw = raw.split("=", 1)
            path = path.strip()
            if not path:
                raise typer.BadParameter("--set path cannot be empty")
            value = yaml.safe_load(value_raw)
            set_overrides.append((path, value))
        selected = _apply_step_overrides(
            selected,
            input_overrides=input_overrides,
            set_overrides=set_overrides,
            root=experiment_root,
            resources=resources,
        )
        parts = ["reader export", str(job_path)]
        if only:
            for v in only:
                parts += ["--only", v]
        if exclude:
            for v in exclude:
                parts += ["--exclude", v]
        if dry_run:
            parts += ["--dry-run"]
        if log_level and log_level != "INFO":
            parts += ["--log-level", log_level]
        for raw in inputs or []:
            parts += ["--input", raw]
        for raw in sets or []:
            parts += ["--set", raw]
        _append_journal(job_path, " ".join(parts))
        run_spec(
            decl,
            dry_run=dry_run,
            log_level=log_level,
            console=console,
            include_pipeline=False,
            include_plots=False,
            include_exports=True,
            export_specs=selected,
            runtime=runtime,
        )
        if not dry_run:
            job_hint = _format_job_arg(job)
            outputs_dir = decl.experiment_semantics.layout.outputs_dir
            exports_cfg = decl.experiment_semantics.layout.exports_subdir
            exports_dir = outputs_dir if exports_cfg in ("", ".", "./") else outputs_dir / str(exports_cfg)

            def _cmd(base: str, tail: str = "") -> str:
                return f"{base} {job_hint}{tail}" if job_hint else f"{base}{tail}"

            lines = [f"Exports saved in [path]{exports_dir}[/path]", "", "Next steps:"]
            if workbench.plots:
                lines.append(f"  {_cmd('reader plot')}")
            lines.append(f"  {_cmd('reader notebook')}")
            console.print(Panel.fit("\n".join(lines), border_style="green", box=box.ROUNDED))
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(help="List emitted workbench records from outputs/manifests/records.json.")
def records(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    all: bool = typer.Option(False, "--all", help="Show revision history counts instead of latest entries."),
):
    try:
        _, decl = _load_job_models(_infer_job_path(job))
        outputs_dir = decl.experiment_semantics.layout.outputs_dir
        store = builtin_runtime().record_store(
            outputs_dir,
            plots_subdir=decl.experiment_semantics.layout.plots_subdir,
            exports_subdir=decl.experiment_semantics.layout.exports_subdir,
            create=False,
        )
        if not store.catalog_exists():
            _abort("No outputs/manifests/records.json found. Run 'reader run' first to produce records.")
    except ReaderError as e:
        _handle_reader_error(e)

    try:
        latest_records = store.iter_latest_records()
    except ReaderError as e:
        _handle_reader_error(e)

    if all:
        if not latest_records:
            console.print(
                Panel.fit(
                    "No record history listed in outputs/manifests/records.json. Run 'reader run' first.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        try:
            revision_counts = {
                record.record_id: len(store.record_history(record.record_id)) for record in latest_records
            }
        except ReaderError as e:
            _handle_reader_error(e)
        t = _table("Records • history")
        t.add_column("Record")
        t.add_column("Kind", style="accent")
        t.add_column("Producer")
        t.add_column("Revisions", justify="right")
        for record in latest_records:
            t.add_row(
                record.record_id,
                record.kind,
                f"{record.producer.kind}:{record.producer.id}",
                str(revision_counts[record.record_id]),
            )
    else:
        if not latest_records:
            console.print(
                Panel.fit(
                    "No records listed in outputs/manifests/records.json. Run 'reader run' first.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        t = _table("Records • latest")
        t.add_column("Record")
        t.add_column("Kind", style="accent")
        t.add_column("Producer")
        t.add_column("Details", style="path")
        for record in latest_records:
            if record.kind == "dataframe_artifact":
                detail = f"{record.contract_id} • {record.path}"
            else:
                detail = ", ".join(str(path) for path in record.files)
            t.add_row(record.record_id, record.kind, f"{record.producer.kind}:{record.producer.id}", detail)
    console.print(Panel(t, border_style="accent", box=box.ROUNDED))


@app.command(help="List step ids and plugins for a config.")
def steps(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
):
    try:
        _, decl = _load_job_models(_infer_job_path(job))
    except ReaderError as e:
        _handle_reader_error(e)
    pipeline = list(resolve_workbench(decl).pipeline)
    t = _table("Steps")
    t.add_column("#", justify="right", style="muted")
    t.add_column("id", style="accent")
    t.add_column("plugin")
    for i, s in enumerate(pipeline, 1):
        t.add_row(str(i), s.id, s.plugin)
    console.print(
        Panel(
            t,
            border_style="accent",
            box=box.ROUNDED,
            subtitle=f"[muted]{len(pipeline)} total[/muted]",
        )
    )


@app.command(help="List plugins by workbench ontology: category, domain, and family.")
def plugins(
    category: str | None = typer.Option(
        None,
        "--category",
        metavar="NAME",
        help="Filter by category: ingest | transform | plot | export | validator",
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
):
    try:
        reg = builtin_runtime().plugins
    except ReaderError as e:
        _handle_reader_error(e)
    descriptors = reg.catalog().filter(category=category, domain=domain, family=family)
    t = _table("Plugins")
    t.add_column("category", style="accent")
    t.add_column("domain")
    t.add_column("family")
    t.add_column("key")
    t.add_column("summary", overflow="fold")
    t.add_column("class", style="muted", overflow="fold")
    for descriptor in descriptors:
        t.add_row(
            descriptor.category,
            descriptor.domain,
            descriptor.family,
            descriptor.key,
            descriptor.summary,
            f"{descriptor.cls.__module__}.{descriptor.cls.__name__}",
        )
    console.print(
        Panel(
            t,
            border_style="accent",
            box=box.ROUNDED,
            subtitle=f"[muted]{len(descriptors)} plugin(s) discovered[/muted]",
        )
    )


# --------------------------- helpers ---------------------------


def _resolve_pipeline_step_id(decl: WorkbenchDecl, which: str) -> str:
    which_str = str(which).strip()
    pipeline = list(resolve_workbench(decl).pipeline)
    if any(s.id == which_str for s in pipeline):
        return which_str
    options = ", ".join(s.id for s in pipeline[:12])
    raise typer.BadParameter(
        f"Unknown pipeline step id '{which_str}'. Tip: use 'reader steps' to list ids "
        f"(first few: {options}{' …' if len(pipeline) > 12 else ''})."
    )


def _spec_to_dict(spec_obj) -> dict:
    if hasattr(spec_obj, "to_dict"):
        return spec_obj.to_dict()
    return {
        "id": spec_obj.id,
        "plugin": spec_obj.plugin,
        "reads": {key: input_ref_to_dict(value) for key, value in (spec_obj.reads or {}).items()},
        "with": dict(spec_obj.with_ or {}),
        "writes": {key: output_ref_to_dict(value) for key, value in (spec_obj.writes or {}).items()},
    }


def _select_steps(steps, *, only: list[str], exclude: list[str], kind: str):
    try:
        return select_workbench_specs(steps, only=only, exclude=exclude, kind_label=kind)
    except ConfigError as err:
        raise typer.BadParameter(f"{err} Use --list to see valid ids.") from err


def _parse_input_overrides(
    raw_inputs: list[str],
    *,
    root: Path,
    resources: ResourceCatalog,
) -> dict[str, InputRef]:
    overrides: dict[str, InputRef] = {}
    for raw in raw_inputs:
        if "=" not in raw:
            raise typer.BadParameter("--input expects KEY=VALUE")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise typer.BadParameter("--input key cannot be empty")
        if not value:
            raise typer.BadParameter("--input value cannot be empty")
        overrides[key] = _coerce_cli_input_ref(yaml.safe_load(value), root=root, resources=resources)
    return overrides


def _coerce_cli_input_ref(value, *, root: Path, resources: ResourceCatalog) -> InputRef:
    if isinstance(value, (RecordRef, FileRef, ResourceRef)):
        return value
    if isinstance(value, dict):
        record = value.get("record")
        file_path = value.get("file")
        resource_id = value.get("resource")
        populated = [item for item in (record, file_path, resource_id) if item is not None]
        if len(populated) != 1:
            raise typer.BadParameter("reads.* must declare exactly one of record, file, or resource")
        if isinstance(record, str) and record.strip():
            return RecordRef(record_id=record.strip())
        if isinstance(file_path, str) and file_path.strip():
            path = Path(file_path.strip()).expanduser()
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()
            return FileRef(path=path)
        if isinstance(resource_id, str) and resource_id.strip():
            return _resolve_cli_resource_ref(resource_id.strip(), root=root, resources=resources)
        raise typer.BadParameter("reads.* binding values must be non-empty strings")
    if isinstance(value, str) and value.strip():
        return RecordRef(record_id=value.strip())
    raise typer.BadParameter("reads.* expects a YAML/JSON mapping like {record: ...}, {file: ...}, or {resource: ...}")


def _resolve_cli_resource_ref(resource_id: str, *, root: Path, resources: ResourceCatalog) -> ResourceRef:
    if not resource_id:
        raise typer.BadParameter("resource bindings require a non-empty resource id")
    try:
        resource = resources.require_file(resource_id)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err
    return ResourceRef(resource_id=resource_id, path=resource.path.resolve())


def _coerce_cli_output_ref(value) -> OutputRef:
    if isinstance(value, OutputRef):
        return value
    if isinstance(value, dict):
        record = value.get("record")
        if isinstance(record, str) and record.strip():
            return OutputRef(record_id=record.strip())
        raise typer.BadParameter("writes.* must declare {record: ...}")
    if isinstance(value, str) and value.strip():
        return OutputRef(record_id=value.strip())
    raise typer.BadParameter("writes.* expects a record id or a {record: ...} mapping")


def _set_nested(mapping: dict, keys: list[str], value) -> None:
    cur = mapping
    for k in keys[:-1]:
        if k not in cur:
            cur[k] = {}
        if not isinstance(cur[k], dict):
            raise typer.BadParameter(f"--set path invalid (non-mapping at '{k}')")
        cur = cur[k]
    cur[keys[-1]] = value


def _apply_step_overrides(
    steps,
    *,
    input_overrides: dict[str, InputRef],
    set_overrides: list[tuple[str, object]],
    root: Path,
    resources: ResourceCatalog,
):
    updated = []
    for step in steps:
        if hasattr(step, "model_copy"):
            s = step.model_copy(deep=True)
            reads = dict(s.reads or {})
            with_block = dict(s.with_ or {})
            writes = dict(s.writes or {})
        else:
            reads = dict(step.reads or {})
            with_block = dict(step.with_ or {})
            writes = dict(step.writes or {})
        if input_overrides:
            reads.update(input_overrides)
        for path, value in set_overrides:
            parts = [p for p in path.split(".") if p]
            if not parts:
                raise typer.BadParameter("--set path cannot be empty")
            section = parts[0]
            if section not in {"reads", "with", "writes"}:
                raise typer.BadParameter("--set path must start with reads., with., or writes.")
            if section in {"reads", "writes"}:
                if len(parts) != 2:
                    raise typer.BadParameter(f"--set {section} expects a single key (e.g., {section}.foo=bar)")
                target = reads if section == "reads" else writes
                target[parts[1]] = (
                    _coerce_cli_input_ref(value, root=root, resources=resources)
                    if section == "reads"
                    else _coerce_cli_output_ref(value)
                )
            else:
                if len(parts) < 2:
                    raise typer.BadParameter("--set with.* requires a key (e.g., with.foo=bar)")
                _set_nested(with_block, parts[1:], value)
        if hasattr(step, "model_copy"):
            s.reads = reads
            s.with_ = with_block
            s.writes = writes
            updated.append(s)
        else:
            payload = {
                "id": step.id,
                "plugin": step.plugin,
                "reads": reads,
                "with_": with_block,
                "writes": writes,
                "source_recipe": getattr(step, "source_recipe", None),
            }
            if hasattr(step, "kind"):
                payload["kind"] = step.kind
            updated.append(step.__class__(**payload))
    return updated


@app.command(help="Show a quick guided walkthrough.")
def demo():
    steps = [
        ("1", "Find experiments", "reader ls"),
        ("2", "List protocols", "reader protocols"),
        ("3", "Explain plan", "reader explain 1"),
        ("4", "Validate config + inputs", "reader validate 1"),
        ("5", "Run pipeline (records)", "reader run 1"),
        ("6", "See records", "reader records 1"),
        ("7", "List plot specs", "reader plot 1 --list"),
        ("8", "Save plots", "reader plot 1"),
        ("9", "Run exports", "reader export 1"),
        ("10", "Notebook (marimo)", "reader notebook 1"),
    ]
    t = _table("Reader Demo")
    t.add_column("#", justify="right", style="muted")
    t.add_column("Goal", style="accent")
    t.add_column("Command", style="path")
    for row in steps:
        t.add_row(*row)
    console.print(
        Panel(
            t,
            border_style="accent",
            box=box.ROUNDED,
            subtitle="[muted]Tip: replace the index with a path or experiment directory[/muted]",
        )
    )
