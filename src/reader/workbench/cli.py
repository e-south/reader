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
from copy import deepcopy
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
from reader.workbench.engine import run_job, run_spec, slice_pipeline_steps
from reader.workbench.engine import validate as validate_job
from reader.workbench.engine import validation_summary as validate_summary_job
from reader.workbench.engine._shared import pipeline_has_plugin
from reader.workbench.experiment import ResourceCatalog
from reader.workbench.experiments import discover_experiment_configs
from reader.workbench.graph import (
    FileRef,
    InputRef,
    OutputRef,
    RecordRef,
    ResourceRef,
    input_ref_display,
    input_ref_to_dict,
    materialize_workbench,
    output_ref_to_dict,
    resolve_workbench,
    select_workbench_specs,
)
from reader.workbench.notebooks import write_experiment_notebook
from reader.workbench.records import DataFrameArtifactRecord, record_to_dict
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


def _count_visible_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and not item.name.startswith("."))


def _count_visible_glob(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob(pattern) if item.is_file() and not item.name.startswith("."))


def _visible_relative_files(path: Path, *, base: Path, limit: int = 8) -> list[str]:
    if not path.exists():
        return []
    files = sorted(item for item in path.rglob("*") if item.is_file() and not item.name.startswith("."))
    relative: list[str] = []
    for item in files[:limit]:
        try:
            relative.append(str(item.relative_to(base)))
        except ValueError:
            relative.append(str(item))
    return relative


def _format_relative_path(path: Path, *, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _resolve_output_subdir(outputs_dir: Path, subdir: str) -> Path:
    return outputs_dir if subdir in ("", ".", "./") else outputs_dir / subdir


def _preview_output_files(path: Path, *, base: Path, limit: int = 4) -> str:
    files = _visible_relative_files(path, base=base, limit=limit)
    if not files:
        return "—"
    remaining = _count_visible_files(path) - len(files)
    preview = ", ".join(files)
    if remaining > 0:
        preview += f", … (+{remaining} more)"
    return preview


def _preview_identifiers(values: list[str], *, limit: int = 4) -> str:
    if not values:
        return "—"
    preview = ", ".join(values[:limit])
    remaining = len(values) - limit
    if remaining > 0:
        preview += f", … (+{remaining} more)"
    return preview


def _render_compact_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        value = list(value)
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return str(value)


def _flatten_binding_rows(value, *, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child_path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_binding_rows(value[key], prefix=child_path))
        return rows
    rows.append((prefix or "value", _render_compact_value(value)))
    return rows


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


def _summarize_outputs_dir(
    outputs_dir: Path,
    *,
    plots_subdir: str = "plots",
    exports_subdir: str = "exports",
    notebooks_subdir: str = "notebooks",
) -> dict[str, int]:
    plots_dir = outputs_dir if plots_subdir in ("", ".", "./") else outputs_dir / plots_subdir
    exports_dir = outputs_dir if exports_subdir in ("", ".", "./") else outputs_dir / exports_subdir
    notebooks_dir = outputs_dir if notebooks_subdir in ("", ".", "./") else outputs_dir / notebooks_subdir
    artifacts_dir = outputs_dir / "artifacts"
    return {
        "records": _count_visible_glob(artifacts_dir, "*.parquet"),
        "plots": _count_visible_files(plots_dir),
        "exports": _count_visible_files(exports_dir),
        "notebooks": _count_visible_files(notebooks_dir),
    }


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
                _emit_json(_protocol_descriptor_payload(descriptor, runtime=runtime))
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
            if descriptor.execution.plugin_defaults:
                table.add_row(
                    "Plugin defaults", ", ".join(item.plugin for item in descriptor.execution.plugin_defaults)
                )
            console.print(Panel(table, border_style="accent", box=box.ROUNDED))
            input_rows = _protocol_surface_rows(descriptor.input_fields)
            if input_rows:
                console.print(
                    Panel(_protocol_surface_table("Inputs Surface", input_rows), border_style="accent", box=box.ROUNDED)
                )
            analysis_rows = _protocol_surface_rows(descriptor.analysis_fields)
            if analysis_rows:
                console.print(
                    Panel(
                        _protocol_surface_table("Analysis Surface", analysis_rows),
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
                console.print(Panel(_semantic_program_table(semantic_program), border_style="accent", box=box.ROUNDED))
            if descriptor.plot_profiles:
                console.print(Panel(_protocol_plot_profiles_table(descriptor), border_style="accent", box=box.ROUNDED))
            if descriptor.figures:
                console.print(Panel(_protocol_plot_outputs_table(descriptor), border_style="accent", box=box.ROUNDED))
            if descriptor.artifacts:
                console.print(Panel(_protocol_artifacts_table(descriptor), border_style="accent", box=box.ROUNDED))
            if compiled_plan.pipeline:
                console.print(
                    Panel(_protocol_pipeline_table(compiled_plan.pipeline), border_style="accent", box=box.ROUNDED)
                )
            if compiled_plan.plots:
                console.print(
                    Panel(
                        _protocol_surface_impl_table(
                            "Plot Implementations",
                            compiled_plan.plots,
                            _plot_output_summaries(bound_protocol),
                        ),
                        border_style="accent",
                        box=box.ROUNDED,
                    )
                )
            if compiled_plan.exports:
                console.print(
                    Panel(
                        _protocol_surface_impl_table(
                            "Export Implementations",
                            compiled_plan.exports,
                            _export_output_summaries(bound_protocol),
                        ),
                        border_style="accent",
                        box=box.ROUNDED,
                    )
                )
            if example_config:
                console.print(
                    Panel(
                        _protocol_example_config(descriptor),
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
    example_document = _protocol_example_document(descriptor)
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


def _plot_output_summaries(bound_protocol) -> dict[str, str]:
    return {item.id: item.summary for item in bound_protocol.descriptor.figures}


def _export_output_summaries(bound_protocol) -> dict[str, str]:
    return {item.id: item.summary for item in bound_protocol.descriptor.artifacts}


def _selected_plan_payload(*, spec: ReaderSpec, decl: WorkbenchDecl, runtime: ReaderRuntime) -> dict[str, object]:
    bound_protocol = _bind_decl_protocol(decl=decl, runtime=runtime)
    workbench = resolve_workbench(decl)
    plot_ids = [spec_decl.id for spec_decl in workbench.plots]
    export_ids = [spec_decl.id for spec_decl in workbench.exports]
    notebook_templates = [notebook.template for notebook in workbench.notebooks]
    pipeline_ids = [step.id for step in workbench.pipeline]
    return {
        "plot_profile": spec.protocol.outputs.plots.profile or bound_protocol.default_plot_profile or "—",
        "notebook_template": spec.protocol.outputs.notebook.template or bound_protocol.default_notebook_template or "—",
        "pipeline": {
            "count": len(pipeline_ids),
            "ids": pipeline_ids,
        },
        "plots": {
            "count": len(plot_ids),
            "ids": plot_ids,
        },
        "exports": {
            "count": len(export_ids),
            "ids": export_ids,
        },
        "notebooks": {
            "count": len(notebook_templates),
            "templates": notebook_templates,
        },
    }


def _selected_plan_summary(selected: dict[str, object] | None) -> str:
    if not selected:
        return "—"
    pipeline = dict(selected["pipeline"])
    plots = dict(selected["plots"])
    exports = dict(selected["exports"])
    notebooks = dict(selected["notebooks"])
    profile = str(selected.get("plot_profile") or "—")
    return f"{profile} • {pipeline['count']} st • {plots['count']} pl • {exports['count']} ex • {notebooks['count']} nb"


def _generated_summary(generated: dict[str, int]) -> str:
    return (
        f"{generated['records']} rec • "
        f"{generated['plots']} pl • "
        f"{generated['exports']} ex • "
        f"{generated['notebooks']} nb"
    )


def _protocol_surface_rows(fields) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for field in fields:
        rows.extend(field.iter_rows())
    return rows


def _protocol_surface_table(title: str, rows: list[tuple[str, str, str, str, str]]) -> Table:
    table = _table(title)
    table.add_column("Path", style="accent")
    table.add_column("Type")
    table.add_column("Required", justify="center")
    table.add_column("Default")
    table.add_column("Summary")
    for path, kind, required, default, summary in rows:
        table.add_row(path, kind, required, default, summary)
    return table


def _semantic_node_payload(node) -> dict[str, object]:
    payload = {
        "id": node.id,
        "kind": node.kind,
        "summary": node.summary,
        "execution": {
            "status": node.execution.status,
            "step_ids": list(node.execution.step_ids),
            "plugin_ids": list(node.execution.plugin_ids),
            "record_ids": list(node.execution.record_ids),
            "config_paths": list(node.execution.config_paths),
            "note": node.execution.note,
        },
    }
    if node.kind == "control_rule":
        payload["match_on"] = list(node.match_on)
        payload["control_selector"] = node.control_selector
    if node.kind == "window":
        payload["anchor"] = node.anchor
        payload["selector"] = node.selector
        payload["params"] = dict(node.params)
    if node.kind == "metric":
        payload["stage"] = node.stage
        payload["formula"] = node.formula
        payload["depends_on"] = list(node.depends_on)
    if node.kind == "ranking":
        payload["primary_metric"] = node.primary_metric
        payload["direction"] = node.direction
        payload["penalties"] = list(node.penalties)
        payload["supporting_metrics"] = list(node.supporting_metrics)
    return payload


def _semantic_program_payload(program) -> dict[str, object]:
    return {
        "protocol": program.protocol,
        "controls": [_semantic_node_payload(node) for node in program.controls],
        "windows": [_semantic_node_payload(node) for node in program.windows],
        "metrics": [_semantic_node_payload(node) for node in program.metrics],
        "ranking": _semantic_node_payload(program.ranking) if program.ranking is not None else None,
    }


def _semantic_program_table(program) -> Table:
    table = _table("Semantic Program")
    table.add_column("kind", style="accent", width=13)
    table.add_column("id", style="accent", overflow="fold")
    table.add_column("status", width=18)
    table.add_column("compiled via", overflow="fold")
    table.add_column("summary", overflow="fold")

    def _add_node(kind: str, node) -> None:
        compiled_via = ", ".join(node.execution.step_ids) or "—"
        note = node.execution.note
        summary = node.summary if not note else f"{node.summary} ({note})"
        table.add_row(kind, node.id, node.execution.status, compiled_via, summary)

    for node in program.controls:
        _add_node("control_rule", node)
    for node in program.windows:
        _add_node("window", node)
    for node in program.metrics:
        _add_node("metric", node)
    if program.ranking is not None:
        _add_node("ranking", program.ranking)
    return table


def _protocol_plot_profiles_table(descriptor) -> Table:
    table = _table("Plot Profiles")
    table.add_column("id", style="accent")
    table.add_column("figures")
    table.add_column("summary")
    for item in descriptor.plot_profiles:
        table.add_row(item.id, ", ".join(item.figures), item.summary)
    return table


def _protocol_plot_outputs_table(descriptor) -> Table:
    table = _table("Plot Outputs")
    table.add_column("id", style="accent")
    table.add_column("kind")
    table.add_column("primary", justify="center")
    table.add_column("summary")
    for item in descriptor.figures:
        table.add_row(item.id, item.kind, "yes" if item.primary else "no", item.summary)
    return table


def _protocol_artifacts_table(descriptor) -> Table:
    table = _table("Export Artifacts")
    table.add_column("id", style="accent")
    table.add_column("summary")
    for item in descriptor.artifacts:
        table.add_row(item.id, item.summary)
    return table


def _default_protocol_plan(*, descriptor, runtime: ReaderRuntime):
    bound_protocol = runtime.bind_protocol(ProtocolBinding(id=descriptor.protocol))
    return bound_protocol, bound_protocol.compile()


def _protocol_pipeline_table(steps) -> Table:
    table = _table("Default Pipeline")
    table.add_column("#", justify="right", style="muted")
    table.add_column("stage", style="accent")
    table.add_column("id", overflow="fold")
    table.add_column("plugin", overflow="fold")
    for idx, step in enumerate(steps, 1):
        stage = step.plugin.split("/", 1)[0]
        table.add_row(str(idx), stage, step.id, step.plugin)
    return table


def _protocol_surface_impl_table(title: str, steps, summaries: dict[str, str]) -> Table:
    table = _table(title)
    table.add_column("id", style="accent", overflow="fold")
    table.add_column("plugin", overflow="fold")
    table.add_column("from", overflow="fold")
    table.add_column("summary", overflow="fold")
    for step in steps:
        from_refs = ", ".join(f"{label} <- {_binding_display(ref)}" for label, ref in (step.reads or {}).items()) or "—"
        table.add_row(step.id, step.plugin, from_refs, summaries.get(step.id, "—"))
    return table


_EXAMPLE_OMIT = object()


def _protocol_field_example_value(field) -> object:
    if field.kind == "mapping":
        example: dict[str, object] = {}
        for child in field.children:
            child_value = _protocol_field_example_value(child)
            if child_value is _EXAMPLE_OMIT:
                continue
            example[child.key] = child_value
        if example:
            return example
        if field.has_default:
            return deepcopy(field.default)
        if field.required:
            return {}
        return _EXAMPLE_OMIT
    if field.has_default:
        return deepcopy(field.default)
    if field.required:
        return "<required>"
    return _EXAMPLE_OMIT


def _protocol_surface_example(fields) -> dict[str, object]:
    example: dict[str, object] = {}
    for field in fields:
        value = _protocol_field_example_value(field)
        if value is _EXAMPLE_OMIT:
            continue
        example[field.key] = value
    return example


def _protocol_example_document(descriptor) -> dict[str, object]:
    protocol_block: dict[str, object] = {"id": descriptor.protocol}
    inputs = _protocol_surface_example(descriptor.input_fields)
    if inputs:
        protocol_block["inputs"] = inputs
    analysis = _protocol_surface_example(descriptor.analysis_fields)
    if analysis:
        protocol_block["analysis"] = analysis

    outputs: dict[str, object] = {
        "notebook": {"template": descriptor.execution.notebook.default_template},
    }
    if descriptor.plot_profiles or descriptor.figures:
        plots: dict[str, object] = {}
        if descriptor.default_plot_profile is not None:
            plots["profile"] = descriptor.default_plot_profile
        elif descriptor.figures:
            plots["include"] = [item.id for item in descriptor.figures if item.primary]
        if plots:
            outputs["plots"] = plots
    default_artifacts = [item.id for item in descriptor.artifacts if item.default]
    if default_artifacts:
        outputs["exports"] = {"include": default_artifacts}
    protocol_block["outputs"] = outputs

    return {
        "schema": "reader/v7",
        "experiment": {"id": "example_experiment"},
        "protocol": protocol_block,
        "resources": {},
        "annotations": {},
    }


def _protocol_example_config(descriptor) -> str:
    return yaml.safe_dump(_protocol_example_document(descriptor), sort_keys=False)


def _binding_display(ref) -> str:
    record_id = getattr(ref, "record_id", None)
    if isinstance(record_id, str) and record_id:
        return record_id
    resource_id = getattr(ref, "resource_id", None)
    if isinstance(resource_id, str) and resource_id:
        return f"resource({resource_id})"
    path = getattr(ref, "path", None)
    if path is not None:
        return str(path)
    return input_ref_display(ref)


def _output_surface_payload(port) -> dict[str, object] | None:
    surface = getattr(port, "contract_surface", None)
    if surface is None:
        return None
    return {
        "minimum": surface.minimum,
        "runtime_mode": surface.runtime_mode,
        "promoted": list(surface.promoted),
        "note": surface.note,
        "rendered": surface.render(),
    }


def _record_producer_map(steps, *, runtime: ReaderRuntime) -> dict[str, dict[str, object]]:
    producers: dict[str, dict[str, object]] = {}
    for step in steps:
        plugin_cls = runtime.plugins.resolve_descriptor(step.plugin).cls
        for output_name, port in plugin_cls.output_ports().items():
            if port.kind != "dataframe":
                continue
            record_ref = (step.writes or {}).get(output_name, OutputRef(record_id=f"{step.id}/{output_name}"))
            producers[record_ref.record_id] = {
                "producer": {
                    "id": step.id,
                    "plugin": step.plugin,
                    "stage": step.plugin.split("/", 1)[0],
                },
                "output": output_name,
                "contract": port.contract,
                "surface": _output_surface_payload(port),
            }
    return producers


def _plugin_semantics_payload(plugin: str, *, runtime: ReaderRuntime) -> dict[str, object]:
    descriptor = runtime.plugins.resolve_descriptor(plugin)
    return {
        "category": descriptor.category,
        "domain": descriptor.domain,
        "family": descriptor.family,
        "summary": descriptor.summary,
    }


def _serialize_reads(reads, *, declared_ports=None, record_producers=None) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for label, ref in (reads or {}).items():
        item: dict[str, object] = {
            "label": label,
            "display": _binding_display(ref),
        }
        record_id = getattr(ref, "record_id", None)
        resource_id = getattr(ref, "resource_id", None)
        path = getattr(ref, "path", None)
        if isinstance(record_id, str) and record_id:
            item["ref"] = {"record": record_id}
            if record_producers and record_id in record_producers:
                item["source"] = deepcopy(record_producers[record_id])
        elif isinstance(resource_id, str) and resource_id:
            item["ref"] = {"resource": resource_id}
        elif path is not None:
            item["ref"] = {"file": str(path)}
        else:
            item["ref"] = {"display": _binding_display(ref)}
        declared_port = (declared_ports or {}).get(label)
        if declared_port is not None:
            item["kind"] = declared_port.kind
            item["declared"] = declared_port.render()
            if declared_port.contract is not None:
                item["contract"] = declared_port.contract
            if declared_port.optional:
                item["optional"] = True
        payload.append(item)
    return payload


def _pipeline_writes_payload(step, *, runtime: ReaderRuntime) -> list[dict[str, object]]:
    plugin_cls = runtime.plugins.resolve_descriptor(step.plugin).cls
    outputs: list[dict[str, object]] = []
    for output_name, port in plugin_cls.output_ports().items():
        if port.kind == "dataframe":
            record_ref = (step.writes or {}).get(output_name, OutputRef(record_id=f"{step.id}/{output_name}"))
            outputs.append(
                {
                    "label": output_name,
                    "kind": port.kind,
                    "declared": port.render(),
                    "display": record_ref.record_id,
                    "contract": port.contract,
                    "ref": output_ref_to_dict(record_ref),
                }
            )
            surface = _output_surface_payload(port)
            if surface is not None:
                outputs[-1]["surface"] = surface
            continue
        outputs.append(
            {
                "label": output_name,
                "kind": port.kind,
                "declared": port.render(),
                "display": output_name,
            }
        )
    return outputs


def _pipeline_step_payload(step, *, runtime: ReaderRuntime, record_producers=None) -> dict[str, object]:
    plugin_cls = runtime.plugins.resolve_descriptor(step.plugin).cls
    return {
        "stage": step.plugin.split("/", 1)[0],
        "id": step.id,
        "plugin": step.plugin,
        "semantics": _plugin_semantics_payload(step.plugin, runtime=runtime),
        "reads": _serialize_reads(
            step.reads, declared_ports=plugin_cls.input_ports(), record_producers=record_producers
        ),
        "writes": _pipeline_writes_payload(step, runtime=runtime),
    }


def _spec_step_payload(step, *, summary: str, runtime: ReaderRuntime, record_producers=None) -> dict[str, object]:
    plugin_cls = runtime.plugins.resolve_descriptor(step.plugin).cls
    return {
        "id": step.id,
        "plugin": step.plugin,
        "summary": summary,
        "semantics": _plugin_semantics_payload(step.plugin, runtime=runtime),
        "reads": _serialize_reads(
            step.reads, declared_ports=plugin_cls.input_ports(), record_producers=record_producers
        ),
    }


def _render_read_binding(item: dict[str, object]) -> str:
    rendered = f"{item['label']} <- {item['display']}"
    source = item.get("source")
    if not isinstance(source, dict):
        return rendered
    surface = source.get("surface")
    if not isinstance(surface, dict):
        return rendered
    producer = source.get("producer")
    producer_id = producer.get("id") if isinstance(producer, dict) else None
    if not isinstance(producer_id, str) or not producer_id:
        return rendered
    mode = surface.get("runtime_mode")
    promoted = [str(value) for value in (surface.get("promoted") or []) if str(value).strip()]
    contract = item.get("contract")
    if mode == "promoted" and promoted:
        return f"{rendered} (via {producer_id}; may promote to {', '.join(promoted)})"
    if mode == "passthrough" and isinstance(contract, str) and contract in promoted:
        return f"{rendered} (via {producer_id}; preserves {contract})"
    return rendered


def _inventory_summary(entries: list[dict[str, object]]) -> dict[str, object]:
    status_counts: dict[str, int] = {}
    protocol_counts: dict[str, int] = {}
    with_outputs = 0
    for entry in entries:
        status = str(entry.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        protocol_value = entry.get("protocol")
        if isinstance(protocol_value, str) and protocol_value:
            protocol_counts[protocol_value] = protocol_counts.get(protocol_value, 0) + 1
        if bool(entry.get("has_outputs")):
            with_outputs += 1
    return {
        "status": dict(sorted(status_counts.items())),
        "protocols": dict(sorted(protocol_counts.items())),
        "with_outputs": with_outputs,
        "without_outputs": len(entries) - with_outputs,
    }


def _experiment_identity_payload(
    *,
    job_path: Path,
    decl: WorkbenchDecl,
    protocol_id: str | None = None,
) -> dict[str, object]:
    return {
        "id": decl.experiment.id,
        "title": decl.experiment.title,
        "protocol": protocol_id or decl.experiment_semantics.protocol.id,
        "config": str(job_path),
        "root": str(decl.experiment.root),
    }


def _surface_specs_payload(
    *,
    job_path: Path,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
    bound_protocol,
    selected,
    kind: str,
    only: list[str],
    exclude: list[str],
) -> dict[str, object]:
    workbench = resolve_workbench(decl)
    record_producers = _record_producer_map(workbench.plugin_steps(), runtime=runtime)
    if kind == "plot":
        summary_lookup = _plot_output_summaries(bound_protocol)
        payload_key = "plots"
    else:
        summary_lookup = _export_output_summaries(bound_protocol)
        payload_key = "exports"
    return {
        "experiment": _experiment_identity_payload(job_path=job_path, decl=decl, protocol_id=bound_protocol.id),
        "count": len(selected),
        "filters": {
            "only": list(only),
            "exclude": list(exclude),
        },
        payload_key: [
            _spec_step_payload(
                step,
                summary=summary_lookup.get(step.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            for step in selected
        ],
    }


def _record_detail_text(record, *, base: Path) -> str:
    if isinstance(record, DataFrameArtifactRecord):
        return f"{record.contract_id} • {_format_relative_path(record.path, base=base)}"
    return ", ".join(_format_relative_path(path, base=base) for path in record.files) or "—"


def _record_payload(record, *, outputs_dir: Path, base: Path, revision_count: int | None = None) -> dict[str, object]:
    payload = record_to_dict(record, outputs_dir=outputs_dir)
    payload["producer_label"] = f"{record.producer.kind}:{record.producer.id}"
    payload["detail"] = _record_detail_text(record, base=base)
    if revision_count is not None:
        payload["revision_count"] = revision_count
    return payload


def _iter_record_payloads(
    *,
    store,
    outputs_dir: Path,
    base: Path,
    include_history: bool = False,
) -> list[dict[str, object]]:
    latest_records = store.iter_latest_records()
    if not include_history:
        return [_record_payload(record, outputs_dir=outputs_dir, base=base) for record in latest_records]
    revision_counts = {record.record_id: len(store.record_history(record.record_id)) for record in latest_records}
    return [
        _record_payload(
            record,
            outputs_dir=outputs_dir,
            base=base,
            revision_count=revision_counts[record.record_id],
        )
        for record in latest_records
    ]


def _protocol_descriptor_payload(descriptor, *, runtime: ReaderRuntime) -> dict[str, object]:
    bound_protocol, compiled_plan = _default_protocol_plan(descriptor=descriptor, runtime=runtime)
    record_producers = _record_producer_map(compiled_plan.pipeline, runtime=runtime)
    semantic_program = compiled_plan.semantic_program or descriptor.semantic_program()
    semantic_payload = _semantic_program_payload(semantic_program)
    return {
        "protocol": descriptor.protocol,
        "domain": descriptor.domain,
        "family": descriptor.family,
        "summary": descriptor.summary,
        "tags": list(descriptor.tags),
        "semantic_program": semantic_payload,
        "factors": [
            {
                "name": item.name,
                "role": item.role,
                "summary": item.summary,
                "required": item.required,
                "repeatable": item.repeatable,
            }
            for item in descriptor.factors
        ],
        "control_rules": [
            {
                "id": item["id"],
                "summary": item["summary"],
                "match_on": item.get("match_on", []),
                "control_selector": item.get("control_selector"),
                "execution": item["execution"],
            }
            for item in semantic_payload["controls"]
        ],
        "windows": [
            {
                "id": item["id"],
                "summary": item["summary"],
                "anchor": item.get("anchor"),
                "selector": item.get("selector"),
                "params": item.get("params", {}),
                "execution": item["execution"],
            }
            for item in semantic_payload["windows"]
        ],
        "metrics": [
            {
                "id": item["id"],
                "summary": item["summary"],
                "stage": item.get("stage"),
                "formula": item.get("formula"),
                "depends_on": item.get("depends_on", []),
                "execution": item["execution"],
            }
            for item in semantic_payload["metrics"]
        ],
        "effect_signs": [
            {
                "target": item.target,
                "expected_sign": item.expected_sign,
                "summary": item.summary,
            }
            for item in descriptor.effect_signs
        ],
        "ranking": (
            {
                "primary_metric": semantic_program.ranking.primary_metric,
                "direction": semantic_program.ranking.direction,
                "summary": semantic_program.ranking.summary,
                "penalties": list(semantic_program.ranking.penalties),
                "supporting_metrics": list(semantic_program.ranking.supporting_metrics),
                "execution": _semantic_node_payload(semantic_program.ranking)["execution"],
            }
            if semantic_program.ranking is not None
            else None
        ),
        "notebook_policy": {
            "default_template": descriptor.execution.notebook.default_template,
            "allowed_templates": list(descriptor.execution.notebook.allowed_templates),
            "summary": descriptor.execution.notebook.summary,
        },
        "plugin_defaults": [
            {
                "plugin": item.plugin,
                "summary": item.summary,
                "with": dict(item.with_ or {}),
            }
            for item in descriptor.execution.plugin_defaults
        ],
        "input_surface": [
            {
                "path": path,
                "kind": kind,
                "required": required == "yes",
                "default": None if default == "—" else default,
                "summary": summary,
            }
            for path, kind, required, default, summary in _protocol_surface_rows(descriptor.input_fields)
        ],
        "analysis_surface": [
            {
                "path": path,
                "kind": kind,
                "required": required == "yes",
                "default": None if default == "—" else default,
                "summary": summary,
            }
            for path, kind, required, default, summary in _protocol_surface_rows(descriptor.analysis_fields)
        ],
        "plot_profiles": [
            {
                "id": item.id,
                "figures": list(item.figures),
                "summary": item.summary,
            }
            for item in descriptor.plot_profiles
        ],
        "figures": [
            {
                "id": item.id,
                "kind": item.kind,
                "primary": item.primary,
                "summary": item.summary,
            }
            for item in descriptor.figures
        ],
        "artifacts": [
            {
                "id": item.id,
                "summary": item.summary,
                "default": item.default,
            }
            for item in descriptor.artifacts
        ],
        "default_plot_profile": descriptor.default_plot_profile,
        "starter_config": _protocol_example_document(descriptor),
        "compiled": {
            "pipeline": [
                _pipeline_step_payload(step, runtime=runtime, record_producers=record_producers)
                for step in compiled_plan.pipeline
            ],
            "plots": [
                _spec_step_payload(
                    step,
                    summary=_plot_output_summaries(bound_protocol).get(step.id, "—"),
                    runtime=runtime,
                    record_producers=record_producers,
                )
                for step in compiled_plan.plots
            ],
            "exports": [
                _spec_step_payload(
                    step,
                    summary=_export_output_summaries(bound_protocol).get(step.id, "—"),
                    runtime=runtime,
                    record_producers=record_producers,
                )
                for step in compiled_plan.exports
            ],
            "notebooks": [
                {
                    "id": notebook.id,
                    "template": notebook.template,
                }
                for notebook in compiled_plan.notebooks
            ],
        },
    }


def _explain_payload(*, job_path: Path, decl: WorkbenchDecl, runtime: ReaderRuntime) -> dict[str, object]:
    bound_protocol = _bind_decl_protocol(decl=decl, runtime=runtime)
    workbench = resolve_workbench(decl)
    pipeline_steps = list(workbench.pipeline)
    plot_steps = list(workbench.plots)
    export_steps = list(workbench.exports)
    notebook_steps = list(workbench.notebooks)
    record_producers = _record_producer_map(workbench.plugin_steps(), runtime=runtime)
    return {
        "experiment": _experiment_identity_payload(job_path=job_path, decl=decl, protocol_id=bound_protocol.id),
        "semantic_program": (
            _semantic_program_payload(decl.experiment_semantics.protocol_program)
            if decl.experiment_semantics.protocol_program is not None
            else None
        ),
        "plan": {
            "protocol": bound_protocol.id,
            "input_sections": sorted(bound_protocol.inputs),
            "analysis_knobs": sorted(bound_protocol.analysis),
            "resources": sorted(decl.experiment_semantics.resources.by_id.keys()),
            "pipeline_flow": [step.id for step in pipeline_steps],
            "plots": [step.id for step in plot_steps],
            "exports": [step.id for step in export_steps],
            "notebooks": [step.template for step in notebook_steps],
        },
        "pipeline": [
            _pipeline_step_payload(step, runtime=runtime, record_producers=record_producers) for step in pipeline_steps
        ],
        "plots": [
            _spec_step_payload(
                step,
                summary=_plot_output_summaries(bound_protocol).get(step.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            for step in plot_steps
        ],
        "exports": [
            _spec_step_payload(
                step,
                summary=_export_output_summaries(bound_protocol).get(step.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            for step in export_steps
        ],
        "notebooks": [{"id": step.id, "template": step.template} for step in notebook_steps],
    }


def _run_dry_run_payload(
    *,
    job_path: Path,
    decl: WorkbenchDecl,
    runtime: ReaderRuntime,
    resume_from: str | None,
    until: str | None,
    only: str | None = None,
) -> dict[str, object]:
    workbench = resolve_workbench(decl)
    pipeline_steps = slice_pipeline_steps(
        list(workbench.pipeline),
        resume_from=only or resume_from,
        until=only or until,
    )
    record_producers = _record_producer_map(workbench.plugin_steps(), runtime=runtime)
    payload = _explain_payload(job_path=job_path, decl=decl, runtime=runtime)
    payload["dry_run"] = True
    payload["slice"] = {
        "from": resume_from,
        "until": until,
        "only": only,
    }
    payload["plan"]["pipeline_flow"] = [step.id for step in pipeline_steps]
    payload["plan"]["plots"] = []
    payload["plan"]["exports"] = []
    payload["pipeline"] = [
        _pipeline_step_payload(step, runtime=runtime, record_producers=record_producers) for step in pipeline_steps
    ]
    payload["plots"] = []
    payload["exports"] = []
    return payload


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

    def _inventory_payload(experiments: list[dict[str, object]]) -> dict[str, object]:
        return {
            "root": str(root_path),
            "count": len(experiments),
            "details": details,
            "summary": _inventory_summary(experiments),
            "filters": {
                "protocol": protocol_filter,
                "status": status_filter,
            },
            "experiments": experiments,
        }

    if str(root).strip() == "./experiments":
        root_path = _find_nearest_experiments_dir(Path.cwd())
    else:
        root_path = Path(root).resolve()
    jobs = _find_jobs(root_path, include_scaffolds=include_scaffolds)
    if not jobs:
        if fmt == "json":
            _emit_json(_inventory_payload([]))
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
                entry["selected"] = _selected_plan_payload(spec=spec, decl=decl, runtime=runtime)
            else:
                spec = ReaderSpec.load(p)
            entry["protocol"] = spec.protocol.id
            outputs_dir = (p.parent / spec.paths.outputs).resolve()
            output_counts = _summarize_outputs_dir(
                outputs_dir,
                plots_subdir=spec.paths.plots,
                exports_subdir=spec.paths.exports,
                notebooks_subdir=spec.paths.notebooks,
            )
            entry["generated"] = output_counts
            entry["has_outputs"] = any(output_counts.values())
            if details:
                entry["generated_examples"] = {
                    "records": _preview_output_files(outputs_dir / "artifacts", base=p.parent),
                    "plots": _preview_output_files(
                        _resolve_output_subdir(outputs_dir, spec.paths.plots), base=p.parent
                    ),
                    "exports": _preview_output_files(
                        _resolve_output_subdir(outputs_dir, spec.paths.exports),
                        base=p.parent,
                    ),
                    "notebooks": _preview_output_files(
                        _resolve_output_subdir(outputs_dir, spec.paths.notebooks),
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
            _emit_json(_inventory_payload([]))
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
        _emit_json(_inventory_payload(entries))
        return

    inventory_summary = _inventory_summary(entries)
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
                _selected_plan_summary(entry.get("selected") if isinstance(entry, dict) else None),
                _generated_summary(generated),
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
            f"{inventory_summary['with_outputs']} with outputs • {inventory_summary['without_outputs']} without outputs",
        )
        status_bits = [f"{key}={value}" for key, value in dict(inventory_summary["status"]).items()]
        summary.add_row("Status", ", ".join(status_bits) if status_bits else "—")
        protocol_bits = [f"{key}={value}" for key, value in dict(inventory_summary["protocols"]).items()]
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
    bound_protocol = _bind_decl_protocol(decl=decl, runtime=runtime)
    workbench = resolve_workbench(decl)
    exp_root = decl.experiment.root
    inputs_dir = exp_root / "inputs"
    outputs_dir = decl.experiment_semantics.layout.outputs_dir
    output_counts = _summarize_outputs_dir(
        outputs_dir,
        plots_subdir=decl.experiment_semantics.layout.plots_subdir,
        exports_subdir=decl.experiment_semantics.layout.exports_subdir,
        notebooks_subdir=decl.experiment_semantics.layout.notebooks_subdir,
    )
    plots_dir = _resolve_output_subdir(outputs_dir, decl.experiment_semantics.layout.plots_subdir)
    exports_dir = _resolve_output_subdir(outputs_dir, decl.experiment_semantics.layout.exports_subdir)
    notebooks_dir = _resolve_output_subdir(outputs_dir, decl.experiment_semantics.layout.notebooks_subdir)
    artifacts_dir = outputs_dir / "artifacts"
    store = runtime.record_store(
        outputs_dir,
        plots_subdir=decl.experiment_semantics.layout.plots_subdir,
        exports_subdir=decl.experiment_semantics.layout.exports_subdir,
        create=False,
    )
    input_files = _visible_relative_files(inputs_dir, base=exp_root, limit=8)
    resource_rows = [
        (resource_id, _format_relative_path(entry.path, base=exp_root))
        for resource_id, entry in sorted(decl.experiment_semantics.resources.by_id.items())
    ]
    plot_summaries = _plot_output_summaries(bound_protocol)
    export_summaries = _export_output_summaries(bound_protocol)
    record_producers = _record_producer_map(workbench.plugin_steps(), runtime=runtime)
    authoring_rows: list[tuple[str, str, str]] = []
    for section, values in (
        ("inputs", spec.protocol.inputs),
        ("analysis", spec.protocol.analysis),
        ("outputs", spec.protocol.outputs.model_dump(exclude_none=True)),
    ):
        for path, value in _flatten_binding_rows(values):
            authoring_rows.append((section, path, value))
    records_payload = (
        _iter_record_payloads(store=store, outputs_dir=outputs_dir, base=exp_root) if store.catalog_exists() else []
    )

    payload = {
        "experiment": {
            **_experiment_identity_payload(job_path=job_path, decl=decl, protocol_id=bound_protocol.id),
            "plot_profile": spec.protocol.outputs.plots.profile or bound_protocol.default_plot_profile,
            "notebook_template": spec.protocol.outputs.notebook.template or bound_protocol.default_notebook_template,
        },
        "semantic_program": (
            _semantic_program_payload(decl.experiment_semantics.protocol_program)
            if decl.experiment_semantics.protocol_program is not None
            else None
        ),
        "authoring": {
            "inputs": spec.protocol.inputs,
            "analysis": spec.protocol.analysis,
            "outputs": spec.protocol.outputs.model_dump(exclude_none=True),
        },
        "inputs": {
            "files": input_files,
            "resources": [
                {
                    "id": resource_id,
                    "path": path_text,
                }
                for resource_id, path_text in resource_rows
            ],
        },
        "generated": {
            "counts": output_counts,
            "examples": {
                "records": _preview_output_files(artifacts_dir, base=exp_root),
                "plots": _preview_output_files(plots_dir, base=exp_root),
                "exports": _preview_output_files(exports_dir, base=exp_root),
                "notebooks": _preview_output_files(notebooks_dir, base=exp_root),
            },
            "records": records_payload,
        },
        "pipeline": [
            _pipeline_step_payload(step, runtime=runtime, record_producers=record_producers)
            for step in workbench.pipeline
        ],
        "plots": [
            _spec_step_payload(
                spec_decl,
                summary=plot_summaries.get(spec_decl.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            for spec_decl in workbench.plots
        ],
        "exports": [
            _spec_step_payload(
                spec_decl,
                summary=export_summaries.get(spec_decl.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            for spec_decl in workbench.exports
        ],
        "notebooks": [
            {
                "id": notebook.id,
                "template": notebook.template,
            }
            for notebook in workbench.notebooks
        ],
    }

    if fmt == "json":
        _emit_json(payload)
        return

    overview = Table(box=box.ROUNDED, expand=True, show_header=False)
    overview.add_column("Field", style="accent", no_wrap=True)
    overview.add_column("Value")
    overview.add_row("Experiment", decl.experiment.id)
    overview.add_row("Protocol", bound_protocol.id)
    overview.add_row("Config", str(job_path))
    overview.add_row("Root", str(exp_root))
    overview.add_row(
        "Inputs",
        f"{_count_visible_files(inputs_dir)} file(s) under {_format_relative_path(inputs_dir, base=exp_root)}",
    )
    overview.add_row(
        "Generated",
        (
            f"{output_counts['records']} records, {output_counts['plots']} plots, "
            f"{output_counts['exports']} exports, {output_counts['notebooks']} notebooks"
        ),
    )
    overview.add_row("Plot profile", spec.protocol.outputs.plots.profile or bound_protocol.default_plot_profile or "—")
    overview.add_row(
        "Notebook",
        spec.protocol.outputs.notebook.template or bound_protocol.default_notebook_template or "—",
    )
    console.print(Panel(overview, title="Experiment overview", border_style="accent", box=box.ROUNDED))

    authoring = _table("Authoring bindings")
    authoring.add_column("section", style="accent", width=10)
    authoring.add_column("path", overflow="fold")
    authoring.add_column("value", overflow="fold")
    if authoring_rows:
        for section, path, value in authoring_rows:
            authoring.add_row(section, path, value)
    else:
        authoring.add_row("—", "—", "No explicit bindings; protocol defaults only.")
    console.print(Panel(authoring, border_style="accent", box=box.ROUNDED))

    if decl.experiment_semantics.protocol_program is not None:
        console.print(
            Panel(
                _semantic_program_table(decl.experiment_semantics.protocol_program),
                border_style="accent",
                box=box.ROUNDED,
            )
        )

    filesystem = _table("Inputs + resources")
    filesystem.add_column("kind", style="accent", width=10)
    filesystem.add_column("entry")
    filesystem.add_column("details")
    if input_files:
        for relpath in input_files:
            filesystem.add_row("input", relpath, "detected under inputs/")
        remaining_inputs = _count_visible_files(inputs_dir) - len(input_files)
        if remaining_inputs > 0:
            filesystem.add_row("input", "…", f"{remaining_inputs} more file(s)")
    else:
        filesystem.add_row("input", "—", "No visible files under inputs/")
    if resource_rows:
        for resource_id, path_text in resource_rows:
            filesystem.add_row("resource", resource_id, path_text)
    console.print(Panel(filesystem, border_style="accent", box=box.ROUNDED))

    generated = _table("Generated outputs")
    generated.add_column("kind", style="accent", width=10)
    generated.add_column("count", justify="right", width=7)
    generated.add_column("examples", overflow="fold")
    generated.add_row("records", str(output_counts["records"]), _preview_output_files(artifacts_dir, base=exp_root))
    generated.add_row("plots", str(output_counts["plots"]), _preview_output_files(plots_dir, base=exp_root))
    generated.add_row("exports", str(output_counts["exports"]), _preview_output_files(exports_dir, base=exp_root))
    generated.add_row("notebooks", str(output_counts["notebooks"]), _preview_output_files(notebooks_dir, base=exp_root))
    console.print(Panel(generated, border_style="accent", box=box.ROUNDED))

    records_table = _table("Record catalog")
    records_table.add_column("record", style="accent", overflow="fold")
    records_table.add_column("kind", width=18)
    records_table.add_column("producer", overflow="fold")
    records_table.add_column("detail", overflow="fold")
    if records_payload:
        for record in records_payload:
            records_table.add_row(
                str(record["record_id"]),
                str(record["kind"]),
                str(record["producer_label"]),
                str(record["detail"]),
            )
    else:
        records_table.add_row("—", "—", "—", "No records catalog found under outputs/manifests/records.json.")
    console.print(Panel(records_table, border_style="accent", box=box.ROUNDED))

    pipeline_table = _table("Pipeline chain")
    pipeline_table.add_column("#", justify="right", style="muted")
    pipeline_table.add_column("stage", style="accent")
    pipeline_table.add_column("id", overflow="fold")
    pipeline_table.add_column("plugin", overflow="fold")
    pipeline_table.add_column("from", overflow="fold")
    pipeline_table.add_column("writes", overflow="fold")
    for idx, step in enumerate(workbench.pipeline, 1):
        payload_row = _pipeline_step_payload(step, runtime=runtime, record_producers=record_producers)
        from_refs = ", ".join(_render_read_binding(item) for item in payload_row["reads"]) or "—"
        writes = (
            ", ".join(
                (f"{item['label']} -> {item['display']}" if item.get("kind") == "dataframe" else str(item["label"]))
                for item in payload_row["writes"]
            )
            or "—"
        )
        pipeline_table.add_row(str(idx), str(payload_row["stage"]), step.id, step.plugin, from_refs, writes)
    console.print(
        Panel(
            pipeline_table,
            border_style="accent",
            box=box.ROUNDED,
            subtitle=f"[muted]{len(tuple(workbench.pipeline))} step(s)[/muted]",
        )
    )

    plot_table = _table("Plot outputs")
    plot_table.add_column("#", justify="right", style="muted")
    plot_table.add_column("id", style="accent", overflow="fold")
    plot_table.add_column("summary", overflow="fold")
    plot_table.add_column("from", overflow="fold")
    if workbench.plots:
        for idx, spec_decl in enumerate(workbench.plots, 1):
            spec_payload = _spec_step_payload(
                spec_decl,
                summary=plot_summaries.get(spec_decl.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            from_refs = ", ".join(_render_read_binding(item) for item in spec_payload["reads"]) or "—"
            plot_table.add_row(str(idx), spec_decl.id, plot_summaries.get(spec_decl.id, "—"), from_refs)
    else:
        plot_table.add_row("—", "—", "No plot outputs selected.", "—")
    console.print(Panel(plot_table, border_style="accent", box=box.ROUNDED))

    export_table = _table("Export artifacts")
    export_table.add_column("#", justify="right", style="muted")
    export_table.add_column("id", style="accent", overflow="fold")
    export_table.add_column("summary", overflow="fold")
    export_table.add_column("from", overflow="fold")
    if workbench.exports:
        for idx, spec_decl in enumerate(workbench.exports, 1):
            spec_payload = _spec_step_payload(
                spec_decl,
                summary=export_summaries.get(spec_decl.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            from_refs = ", ".join(_render_read_binding(item) for item in spec_payload["reads"]) or "—"
            export_table.add_row(str(idx), spec_decl.id, export_summaries.get(spec_decl.id, "—"), from_refs)
    else:
        export_table.add_row("—", "—", "No export artifacts selected.", "—")
    console.print(Panel(export_table, border_style="accent", box=box.ROUNDED))

    notebook_table = _table("Notebooks")
    notebook_table.add_column("#", justify="right", style="muted")
    notebook_table.add_column("template", style="accent")
    notebook_table.add_column("status")
    if workbench.notebooks:
        for idx, notebook in enumerate(workbench.notebooks, 1):
            notebook_table.add_row(str(idx), notebook.template, "selected")
    else:
        notebook_table.add_row("—", "—", "No notebook template selected.")
    console.print(Panel(notebook_table, border_style="accent", box=box.ROUNDED))


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
        _, decl = _load_job_models(job_path)
        runtime = builtin_runtime()
        fmt = _normalize_output_format(format)
        if fmt == "json":
            _emit_json(_explain_payload(job_path=job_path, decl=decl, runtime=runtime))
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
            _emit_json(
                {
                    "experiment": _experiment_identity_payload(job_path=job_path, decl=decl),
                    "validation": validate_summary_job(
                        decl,
                        check_files=not no_files,
                        exp_root=decl.experiment.root,
                        runtime=runtime,
                    ),
                }
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
    payload = spec.model_dump(by_alias=True)
    materialized = materialize_workbench(decl)
    payload["compiled"] = {
        "pipeline": materialized["pipeline"],
        "plots": materialized["plots"],
        "exports": materialized["exports"],
        "notebooks": materialized["notebooks"],
    }
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
        _, decl = _load_job_models(job_path)
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
                    _run_dry_run_payload(
                        job_path=job_path,
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
                _run_dry_run_payload(
                    job_path=job_path,
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
    record_producers = _record_producer_map(workbench.plugin_steps(), runtime=runtime)
    if not plot_specs:
        if list_only:
            if fmt == "json":
                _emit_json(
                    _surface_specs_payload(
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
    selected = _select_steps(plot_specs, only=only or [], exclude=exclude or [], kind="plot spec")
    if list_only:
        if fmt == "json":
            _emit_json(
                _surface_specs_payload(
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
        plot_summaries = _plot_output_summaries(bound_protocol)
        for i, s in enumerate(selected, 1):
            spec_payload = _spec_step_payload(
                s,
                summary=plot_summaries.get(s.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            from_refs = ", ".join(_render_read_binding(item) for item in spec_payload["reads"]) or "—"
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
    record_producers = _record_producer_map(workbench.plugin_steps(), runtime=runtime)
    bound_protocol = _bind_decl_protocol(decl=decl, runtime=runtime)
    export_specs = list(workbench.exports)
    if not export_specs:
        if list_only:
            if fmt == "json":
                _emit_json(
                    _surface_specs_payload(
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
    selected = _select_steps(export_specs, only=only or [], exclude=exclude or [], kind="export spec")
    if list_only:
        if fmt == "json":
            _emit_json(
                _surface_specs_payload(
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
        export_summaries = _export_output_summaries(bound_protocol)
        for i, s in enumerate(selected, 1):
            spec_payload = _spec_step_payload(
                s,
                summary=export_summaries.get(s.id, "—"),
                runtime=runtime,
                record_producers=record_producers,
            )
            from_refs = ", ".join(_render_read_binding(item) for item in spec_payload["reads"]) or "—"
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
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
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
    fmt = _normalize_output_format(format)

    try:
        latest_records = store.iter_latest_records()
    except ReaderError as e:
        _handle_reader_error(e)

    if fmt == "json":
        _emit_json(
            {
                "all": all,
                "count": len(latest_records),
                "records": _iter_record_payloads(
                    store=store,
                    outputs_dir=outputs_dir,
                    base=decl.experiment.root,
                    include_history=all,
                ),
            }
        )
        return

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
        _, decl = _load_job_models(job_path)
    except ReaderError as e:
        _handle_reader_error(e)
    runtime = builtin_runtime()
    fmt = _normalize_output_format(format)
    workbench = resolve_workbench(decl)
    pipeline = list(workbench.pipeline)
    record_producers = _record_producer_map(workbench.plugin_steps(), runtime=runtime)
    payload = {
        "experiment": _experiment_identity_payload(job_path=job_path, decl=decl),
        "semantic_program": (
            _semantic_program_payload(decl.experiment_semantics.protocol_program)
            if decl.experiment_semantics.protocol_program is not None
            else None
        ),
        "count": len(pipeline),
        "pipeline": [
            _pipeline_step_payload(step, runtime=runtime, record_producers=record_producers) for step in pipeline
        ],
    }
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
    for i, item in enumerate(payload["pipeline"], 1):
        from_refs = ", ".join(_render_read_binding(entry) for entry in item["reads"]) or "—"
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
            {
                "protocol": protocol,
                "count": len(descriptors),
                "plugins": [
                    {
                        "category": descriptor.category,
                        "domain": descriptor.domain,
                        "family": descriptor.family,
                        "key": descriptor.key,
                        "plugin": descriptor.plugin,
                        "summary": descriptor.summary,
                        "class": f"{descriptor.cls.__module__}.{descriptor.cls.__name__}",
                    }
                    for descriptor in descriptors
                ],
            }
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
