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

import reader.workbench._cli_bootstrap  # noqa: F401
from reader.errors import ConfigError, ReaderError, RecordError
from reader.protocols import ProtocolBinding
from reader.protocols.model import ProtocolBindingValueRef
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl, build_workbench_decl
from reader.workbench.engine import explain as explain_job
from reader.workbench.engine import run_job, run_spec
from reader.workbench.engine import validate as validate_job
from reader.workbench.engine import validation_summary as validate_summary_job
from reader.workbench.engine._shared import pipeline_has_plugin
from reader.workbench.experiments import discover_experiment_configs
from reader.workbench.graph import (
    input_ref_to_dict,
    materialize_workbench,
    output_ref_to_dict,
    resolve_workbench,
)
from reader.workbench.inspection import (
    experiment_config_json_payload,
    experiment_explain_payload,
    experiment_identity_payload,
    experiment_inspect_payload,
    experiment_run_dry_run_payload,
    experiment_steps_payload,
    inventory_summary_payload,
    inventory_surface_payload,
    plugin_registry_payload,
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
    record_catalog_payload,
    semantic_program_table,
    validation_surface_payload,
    workbench_surface_specs_payload,
)
from reader.workbench.inspection.common import (
    preview_output_files,
    resolve_output_subdir,
    summarize_outputs_dir,
)
from reader.workbench.inspection.reports import experiment_inspect_renderables
from reader.workbench.inspection.runtime import (
    binding_display,
    export_output_summaries,
    generated_summary,
    plot_output_summaries,
    record_producer_map,
    render_read_binding,
    selected_plan_payload,
    selected_plan_summary,
    spec_step_payload,
)
from reader.workbench.notebooks import write_experiment_notebook
from reader.workbench.spec_overrides import (
    apply_step_overrides,
    build_surface_command,
    parse_input_overrides,
    parse_set_overrides,
    select_surface_specs,
)
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
        "reader — experimental workbench.\n\n"
        "Discover assays and experiments, inspect compiled workflow plans, validate authoring YAML, "
        "run pipelines, and materialize plots, exports, or notebooks. "
        "Start with 'reader demo', 'reader ls', or 'reader protocols'."
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


def _normalize_output_format(
    value: str | None,
    *,
    default: str = "table",
    allowed: tuple[str, ...] = ("table", "json"),
) -> str:
    if not isinstance(value, str):
        return default
    fmt = value.strip().lower() or default
    if fmt not in allowed:
        raise typer.BadParameter(f"format must be one of: {', '.join(allowed)}")
    return fmt


def _normalize_flag(value: bool | object, *, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _normalize_status_filter(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    allowed = {"ok", "config_error"}
    if normalized not in allowed:
        raise typer.BadParameter(f"status must be one of: {', '.join(sorted(allowed))}")
    return normalized


def _json_friendly(value):
    if isinstance(value, ProtocolBindingValueRef):
        payload = {"binding_value": value.key}
        if value.has_default:
            payload["default"] = _json_friendly(value.default)
        return payload
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_friendly(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_friendly(item) for item in value]
    if isinstance(value, list):
        return [_json_friendly(item) for item in value]
    return value


def _emit_json(payload: object) -> None:
    typer.echo(json.dumps(_json_friendly(payload), indent=2, sort_keys=True))


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
        fmt = _normalize_output_format(format)
        if example_config and not name:
            raise typer.BadParameter("--example-config requires a protocol name.")
        if name:
            descriptor = runtime.protocols.resolve(name)
            if fmt == "json":
                _emit_json(protocol_descriptor_payload(descriptor, runtime=runtime))
                return
            bound_protocol, compiled_plan = _default_protocol_plan(descriptor=descriptor, runtime=runtime)
            semantic_program = compiled_plan.semantic_program or descriptor.semantic_program()
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
            table.add_row(
                "Semantic nodes",
                str(
                    len(semantic_program.controls)
                    + len(semantic_program.windows)
                    + len(semantic_program.metrics)
                    + (1 if semantic_program.ranking is not None else 0)
                ),
            )
            table.add_row("Default notebook", descriptor.execution.notebook.default_template)
            table.add_row("Allowed notebooks", ", ".join(descriptor.execution.notebook.allowed_templates))
            if descriptor.default_plot_profile is not None:
                table.add_row("Default plot profile", descriptor.default_plot_profile)
            console.print(Panel(table, border_style="accent", box=box.ROUNDED))
            input_rows = protocol_surface_rows(descriptor.input_fields)
            if input_rows:
                console.print(
                    Panel(protocol_surface_table("Inputs Surface", input_rows), border_style="accent", box=box.ROUNDED)
                )
            analysis_rows = protocol_surface_rows(descriptor.analysis_fields)
            if analysis_rows:
                console.print(
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
                or semantic_program.ranking is not None
            ):
                console.print(Panel(semantic_program_table(semantic_program), border_style="accent", box=box.ROUNDED))
            if descriptor.plot_profiles:
                console.print(Panel(protocol_plot_profiles_table(descriptor), border_style="accent", box=box.ROUNDED))
            if descriptor.figures:
                console.print(Panel(protocol_plot_outputs_table(descriptor), border_style="accent", box=box.ROUNDED))
            if descriptor.artifacts:
                console.print(Panel(protocol_artifacts_table(descriptor), border_style="accent", box=box.ROUNDED))
            if compiled_plan.pipeline:
                console.print(
                    Panel(protocol_pipeline_table(compiled_plan.pipeline), border_style="accent", box=box.ROUNDED)
                )
            if compiled_plan.plots:
                console.print(
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
                console.print(
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
                console.print(
                    Panel(
                        protocol_example_config(descriptor),
                        title="Starter YAML",
                        border_style="accent",
                        box=box.ROUNDED,
                    )
                )
            return

        if fmt == "json":
            _emit_json(
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
    title: str | None = typer.Option(
        None,
        "--title",
        metavar="TEXT",
        help="Optional human-readable experiment title.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite config.yaml when the target directory already contains one.",
    ),
):
    runtime = builtin_runtime()
    try:
        descriptor = runtime.protocols.resolve(protocol)
    except ConfigError as e:
        raise typer.BadParameter(str(e)) from e
    target_dir = Path(target).expanduser()
    if target_dir.suffix:
        raise typer.BadParameter("init expects a directory path, not a file path")
    target_dir = target_dir.resolve()
    config_path = target_dir / "config.yaml"
    if config_path.exists() and not force:
        _abort(f"{config_path} already exists. Pass --force to overwrite it.")
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
    console.print(Panel(summary, title="Experiment scaffolded", border_style="accent", box=box.ROUNDED))
    console.print(
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


def _load_job_models(job_path: Path, *, runtime: ReaderRuntime | None = None) -> tuple[ReaderSpec, WorkbenchDecl]:
    runtime = runtime or builtin_runtime()
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


def _default_protocol_plan(*, descriptor, runtime: ReaderRuntime):
    bound_protocol = runtime.bind_protocol(ProtocolBinding(id=descriptor.protocol))
    return bound_protocol, bound_protocol.compile()


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
            selected = select_surface_specs(
                plot_specs, only=plot_only or [], exclude=plot_exclude or [], kind="plot spec"
            )
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
        help="Notebook template (defaults to the protocol default or protocol.outputs.notebook.template).",
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
    protocol: str | None = typer.Option(
        None,
        "--protocol",
        metavar="ID",
        help="Only show experiments bound to the given protocol id.",
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
    # If user didn't override --root, auto-detect nearest experiments/ so this
    # works from anywhere inside the repository.
    include_scaffolds = _normalize_flag(include_scaffolds)
    details = _normalize_flag(details)
    fmt = _normalize_output_format(format)
    protocol_filter = protocol.strip() if isinstance(protocol, str) and protocol.strip() else None
    status_filter = _normalize_status_filter(status)

    if str(root).strip() == "./experiments":
        root_path = _find_nearest_experiments_dir(Path.cwd())
    else:
        root_path = Path(root).resolve()
    jobs = _find_jobs(root_path, include_scaffolds=include_scaffolds)
    if not jobs:
        if fmt == "json":
            _emit_json(
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
        console.print(
            Panel.fit(
                f"No experiments found under [path]{root_path}[/path].",
                border_style="warn",
                box=box.ROUNDED,
            )
        )
        return
    entries: list[dict[str, object]] = []
    runtime = builtin_runtime() if details else None
    for i, p in enumerate(jobs, 1):
        name = p.parent.name
        entry: dict[str, object] = {
            "index": i,
            "name": name,
            "config": str(p),
            "root": str(p.parent),
            "protocol": None,
            "generated": {"records": 0, "plots": 0, "exports": 0, "notebooks": 0},
            "selected": None,
            "has_outputs": False,
            "status": "ok",
            "error": None,
        }
        try:
            if details:
                spec, decl = _load_job_models(p, runtime=runtime)
                entry["selected"] = selected_plan_payload(spec=spec, decl=decl, runtime=runtime)
            else:
                spec = ReaderSpec.load(p)
            entry["protocol"] = spec.protocol.id
            outputs_dir = (p.parent / spec.paths.outputs).resolve()
            output_counts = summarize_outputs_dir(
                outputs_dir,
                plots_subdir=spec.paths.plots,
                exports_subdir=spec.paths.exports,
                notebooks_subdir=spec.paths.notebooks,
            )
            entry["generated"] = output_counts
            entry["has_outputs"] = any(output_counts.values())
            if details:
                entry["generated_examples"] = {
                    "records": preview_output_files(outputs_dir / "artifacts", base=p.parent),
                    "plots": preview_output_files(resolve_output_subdir(outputs_dir, spec.paths.plots), base=p.parent),
                    "exports": preview_output_files(
                        resolve_output_subdir(outputs_dir, spec.paths.exports),
                        base=p.parent,
                    ),
                    "notebooks": preview_output_files(
                        resolve_output_subdir(outputs_dir, spec.paths.notebooks),
                        base=p.parent,
                    ),
                }
        except ReaderError as err:
            entry["status"] = "config_error"
            entry["error"] = str(err)
        protocol_value = entry["protocol"]
        if protocol_filter and protocol_value != protocol_filter:
            continue
        if status_filter and entry["status"] != status_filter:
            continue
        entries.append(entry)

    if not entries:
        if fmt == "json":
            _emit_json(
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
        console.print(
            Panel.fit(
                f"No experiments found under [path]{root_path}[/path]{suffix}.",
                border_style="warn",
                box=box.ROUNDED,
            )
        )
        return

    if fmt == "json":
        _emit_json(
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
    t = _table("Experiments")
    t.add_column("#", justify="right", style="muted")
    name_values = [str(entry["name"]) for entry in entries]
    max_name = max((len(n) for n in name_values), default=12)
    max_width = int((console.width or 80) * (0.35 if details else 0.6))
    name_width = max(12, min(max_name + 2, max_width))
    t.add_column("Name", style="accent", max_width=name_width, overflow="ellipsis")
    if details:
        t.add_column("Protocol", max_width=28, overflow="ellipsis")
        t.add_column("Status", width=12)
        t.add_column("Selected", overflow="fold")
        t.add_column("Generated", overflow="fold")
        t.add_column("Issue", overflow="fold")
    else:
        t.add_column("Outputs", justify="center", width=7)
    for entry in entries:
        status = str(entry["status"])
        generated = dict(entry["generated"])
        if details:
            t.add_row(
                str(entry["index"]),
                str(entry["name"]),
                str(entry["protocol"] or "—"),
                status,
                selected_plan_summary(entry.get("selected") if isinstance(entry, dict) else None),
                generated_summary(generated),
                str(entry["error"] or "—"),
            )
        else:
            outputs_cell = "[error]ERR[/error]" if status != "ok" else _checkmark(bool(entry["has_outputs"]))
            t.add_row(str(entry["index"]), str(entry["name"]), outputs_cell)
    console.print(
        Panel(
            t,
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
        console.print(Panel(summary, border_style="accent", box=box.ROUNDED, title="Inventory summary"))


@app.command(help="Inspect one experiment: inputs, pipeline chain, plots, artifacts, and generated outputs.")
def inspect(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
):
    try:
        job_path = _infer_job_path(job)
        spec, decl = _load_job_models(job_path)
    except ReaderError as e:
        _handle_reader_error(e)
    fmt = _normalize_output_format(format)
    runtime = builtin_runtime()
    payload = experiment_inspect_payload(job_path=job_path, spec=spec, decl=decl, runtime=runtime)

    if fmt == "json":
        _emit_json(payload)
        return

    for renderable in experiment_inspect_renderables(
        payload=payload, semantic_program=decl.experiment_semantics.protocol_program
    ):
        console.print(renderable)


@app.command(help="Show planned steps and contracts (no execution).")
def explain(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
):
    try:
        job_path = _infer_job_path(job)
        _append_journal(job_path, f"reader explain {job_path}")
        spec, decl = _load_job_models(job_path)
        runtime = builtin_runtime()
        fmt = _normalize_output_format(format)
        if fmt == "json":
            _emit_json(experiment_explain_payload(job_path=job_path, spec=spec, decl=decl, runtime=runtime))
            return
        explain_job(decl, console=console, runtime=runtime)
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
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
):
    try:
        job_path = _infer_job_path(job)
        _append_journal(job_path, f"reader validate {job_path}")
        _, decl = _load_job_models(job_path)
        runtime = builtin_runtime()
        fmt = _normalize_output_format(format)
        if fmt == "json":
            summary = validate_summary_job(
                decl,
                check_files=not no_files,
                exp_root=decl.experiment.root,
                runtime=runtime,
            )
            _emit_json(
                validation_surface_payload(
                    experiment=experiment_identity_payload(job_path=job_path, decl=decl),
                    check_files=not no_files,
                    summary=summary,
                )
            )
            return
        validate_job(
            decl,
            console=console,
            check_files=not no_files,
            exp_root=decl.experiment.root,
            runtime=runtime,
        )
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(help="Print the authoring config plus compiled runtime plan.")
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
    if fmt == "json":
        runtime = builtin_runtime()
        _emit_json(experiment_config_json_payload(job_path=job_path, spec=spec, decl=decl, runtime=runtime))
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
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format for --dry-run: table | json (default: table).",
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
        spec, decl = _load_job_models(job_path)
    except ReaderError as e:
        _handle_reader_error(e)
    fmt = _normalize_output_format(format)
    if fmt == "json" and not dry_run:
        raise typer.BadParameter("--format json is only supported with --dry-run")

    if only:
        _resolve_pipeline_step_id(decl, only)
        parts += ["--only", only]
        if dry_run:
            parts += ["--dry-run"]
        if fmt == "json":
            parts += ["--format", "json"]
        if log_level and log_level != "INFO":
            parts += ["--log-level", log_level]
        if compact:
            parts += ["--compact"]
        _append_journal(job_path, " ".join(parts))
        try:
            runtime = builtin_runtime()
            if dry_run and fmt == "json":
                _emit_json(
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
    if fmt == "json":
        parts += ["--format", "json"]
    if log_level and log_level != "INFO":
        parts += ["--log-level", log_level]
    if compact:
        parts += ["--compact"]
    _append_journal(job_path, " ".join(parts))
    try:
        runtime = builtin_runtime()
        if dry_run and fmt == "json":
            _emit_json(
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
    _, decl = _load_job_models(job_path)
    runtime = builtin_runtime()
    workbench = resolve_workbench(decl)
    bound_protocol = _bind_decl_protocol(decl=decl, runtime=runtime)
    fmt = _normalize_output_format(format)
    if not list_only:
        if fmt == "json":
            raise typer.BadParameter("--format json is only supported with --list")
        _require_dataframe_records(decl, job_path, runtime=runtime)
    plot_specs = list(workbench.plots)
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    if not plot_specs:
        if list_only:
            if fmt == "json":
                _emit_json(
                    workbench_surface_specs_payload(
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
            console.print(
                Panel.fit(
                    "No plot specs configured in this experiment.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        raise typer.BadParameter("No plot specs configured in this experiment. Add plots to the config.")
    selected = select_surface_specs(plot_specs, only=only or [], exclude=exclude or [], kind="plot spec")
    if list_only:
        if fmt == "json":
            _emit_json(
                workbench_surface_specs_payload(
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
        t = _table("Plots")
        t.add_column("#", justify="right", style="muted")
        t.add_column("id", style="accent", overflow="fold")
        t.add_column("summary", overflow="fold")
        t.add_column("from", overflow="fold")
        t.add_column("plugin", overflow="fold")
        plot_summaries = plot_output_summaries(bound_protocol)
        for i, s in enumerate(selected, 1):
            spec_payload = spec_step_payload(
                s,
                summary=plot_summaries.get(s.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            from_refs = ", ".join(render_read_binding(item) for item in spec_payload["reads"]) or "—"
            t.add_row(str(i), s.id, plot_summaries.get(s.id, "—"), from_refs, s.plugin)
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
    input_overrides = parse_input_overrides(inputs or [], root=experiment_root, resources=resources)
    set_overrides = parse_set_overrides(sets or [])
    selected = apply_step_overrides(
        selected,
        input_overrides=input_overrides,
        set_overrides=set_overrides,
        root=experiment_root,
        resources=resources,
    )
    parts = build_surface_command(
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
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format for --list: table | json (default: table).",
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
    fmt = _normalize_output_format(format)
    if year:
        if fmt == "json":
            raise typer.BadParameter("--format json is only supported for single-experiment plot listings")
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
                build_surface_command(
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
            console.print(f"[accent]{idx}/{total}[/accent] {exp_name}")
            console.print(f"[muted]{cmd_line}[/muted]")
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
            format=fmt,
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
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format for --list: table | json (default: table).",
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
    fmt = _normalize_output_format(format)
    workbench = resolve_workbench(decl)
    runtime = builtin_runtime()
    record_producers = record_producer_map(workbench.plugin_steps(), runtime=runtime)
    bound_protocol = _bind_decl_protocol(decl=decl, runtime=runtime)
    export_specs = list(workbench.exports)
    if not export_specs:
        if list_only:
            if fmt == "json":
                _emit_json(
                    workbench_surface_specs_payload(
                        job_path=job_path,
                        decl=decl,
                        runtime=builtin_runtime(),
                        bound_protocol=bound_protocol,
                        selected=[],
                        kind="export",
                        only=only or [],
                        exclude=exclude or [],
                    )
                )
                return
            console.print(
                Panel.fit(
                    "No export specs configured in this experiment.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        raise typer.BadParameter("No export specs configured in this experiment. Add exports to the config.")
    selected = select_surface_specs(export_specs, only=only or [], exclude=exclude or [], kind="export spec")
    if list_only:
        if fmt == "json":
            _emit_json(
                workbench_surface_specs_payload(
                    job_path=job_path,
                    decl=decl,
                    runtime=builtin_runtime(),
                    bound_protocol=bound_protocol,
                    selected=selected,
                    kind="export",
                    only=only or [],
                    exclude=exclude or [],
                )
            )
            return
        t = _table("Exports")
        t.add_column("#", justify="right", style="muted")
        t.add_column("id", style="accent", overflow="fold")
        t.add_column("summary", overflow="fold")
        t.add_column("from", overflow="fold")
        t.add_column("plugin", overflow="fold")
        export_summaries = export_output_summaries(bound_protocol)
        for i, s in enumerate(selected, 1):
            spec_payload = spec_step_payload(
                s,
                summary=export_summaries.get(s.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            from_refs = ", ".join(render_read_binding(item) for item in spec_payload["reads"]) or "—"
            t.add_row(str(i), s.id, export_summaries.get(s.id, "—"), from_refs, s.plugin)
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
        if fmt == "json":
            raise typer.BadParameter("--format json is only supported with --list")
        _require_dataframe_records(decl, job_path, runtime=runtime)
        experiment_root = decl.experiment.root
        resources = decl.experiment_semantics.resources
        input_overrides = parse_input_overrides(inputs or [], root=experiment_root, resources=resources)
        set_overrides = parse_set_overrides(sets or [])
        selected = apply_step_overrides(
            selected,
            input_overrides=input_overrides,
            set_overrides=set_overrides,
            root=experiment_root,
            resources=resources,
        )
        parts = build_surface_command(
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
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
):
    try:
        job_path = _infer_job_path(job)
        _, decl = _load_job_models(job_path)
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
    fmt = _normalize_output_format(format)

    if fmt == "json":
        try:
            _emit_json(
                record_catalog_payload(
                    experiment=experiment_identity_payload(job_path=job_path, decl=decl),
                    store=store,
                    outputs_dir=outputs_dir,
                    base=decl.experiment.root,
                    include_history=all,
                )
            )
        except ReaderError as e:
            _handle_reader_error(e)
        return

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


@app.command(help="List pipeline steps and bindings for a config.")
def steps(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
):
    try:
        job_path = _infer_job_path(job)
        spec, decl = _load_job_models(job_path)
    except ReaderError as e:
        _handle_reader_error(e)
    runtime = builtin_runtime()
    fmt = _normalize_output_format(format)
    workbench = resolve_workbench(decl)
    pipeline = list(workbench.pipeline)
    payload = experiment_steps_payload(job_path=job_path, spec=spec, decl=decl, runtime=runtime)
    if fmt == "json":
        _emit_json(payload)
        return
    t = _table("Steps")
    t.add_column("#", justify="right", style="muted")
    t.add_column("stage", style="accent")
    t.add_column("id", style="accent", overflow="fold")
    t.add_column("plugin", overflow="fold")
    t.add_column("from", overflow="fold")
    t.add_column("writes", overflow="fold")
    for i, item in enumerate(payload["implementation"]["compiled"]["pipeline"], 1):
        from_refs = ", ".join(render_read_binding(entry) for entry in item["reads"]) or "—"
        writes = (
            ", ".join(
                (f"{entry['label']} -> {entry['display']}" if entry.get("kind") == "dataframe" else str(entry["label"]))
                for entry in item["writes"]
            )
            or "—"
        )
        t.add_row(str(i), str(item["stage"]), str(item["id"]), str(item["plugin"]), from_refs, writes)
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
    protocol: str | None = typer.Option(
        None,
        "--protocol",
        metavar="ID",
        help="Limit to plugins used by the named protocol's default compiled plan.",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
):
    protocol = protocol if isinstance(protocol, str) else None
    fmt = _normalize_output_format(format)
    try:
        runtime = builtin_runtime()
        reg = runtime.plugins
        descriptors = reg.catalog().filter(category=category, domain=domain, family=family)
        if protocol:
            bound_protocol = runtime.bind_protocol(ProtocolBinding(id=protocol))
            plan = bound_protocol.compile()
            allowed_plugins = {step.plugin for step in (*plan.pipeline, *plan.plots, *plan.exports)}
            descriptors = [descriptor for descriptor in descriptors if descriptor.plugin in allowed_plugins]
    except ReaderError as e:
        _handle_reader_error(e)
    if fmt == "json":
        _emit_json(
            plugin_registry_payload(
                descriptors=descriptors,
                category=category,
                domain=domain,
                family=family,
                protocol=protocol,
            )
        )
        return
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
            subtitle=(
                f"[muted]{len(descriptors)} plugin(s) discovered"
                f"{f' • protocol: {protocol}' if protocol else ''}[/muted]"
            ),
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


@app.command(help="Show a quick guided walkthrough.")
def demo():
    steps = [
        ("1", "Find experiments", "reader ls"),
        ("2", "Show experiment details", "reader inspect 1"),
        ("3", "List protocols", "reader protocols"),
        (
            "4",
            "Scaffold a new experiment",
            "reader init ./experiments/20260317_new_assay --protocol plate_reader/dual_reporter_screen",
        ),
        ("5", "Starter YAML for a protocol", "reader protocols plate_reader/dual_reporter_screen --example-config"),
        ("6", "Show pipeline chain", "reader steps 1"),
        ("7", "Explain plan", "reader explain 1"),
        ("8", "Validate config + inputs", "reader validate 1"),
        ("9", "Run pipeline (records)", "reader run 1"),
        ("10", "See records", "reader records 1"),
        ("11", "List plot specs", "reader plot 1 --list"),
        ("12", "Save plots", "reader plot 1"),
        ("13", "Run exports", "reader export 1"),
        ("14", "Notebook (marimo)", "reader notebook 1"),
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
