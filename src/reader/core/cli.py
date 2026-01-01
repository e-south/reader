"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import difflib
import re  # used for friendly error parsing in run-step
import json
import yaml
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.traceback import install as rich_tracebacks

from reader.core.artifacts import ArtifactStore
from reader.core.config_model import ReaderSpec
from reader.core.contracts import BUILTIN
from reader.core.engine import explain as explain_job
from reader.core.engine import explain_steps
from reader.core.engine import run_job, run_reports
from reader.core.engine import validate as validate_job
from reader.core.errors import (
    ArtifactError,
    ConfigError,
    ContractError,
    ExecutionError,
    MergeError,
    ParseError,
    PlotError,
    ReaderError,
    RegistryError,
    SFXIError,
    TransformError,
)
from reader.core.presets import describe_preset, list_presets
from reader.core.registry import load_entry_points

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

APP_EPILOG = (
    "Examples:\n"
    "  reader ls\n"
    "  reader init experiments/2026/20260101_example --preset plate_reader/basic\n"
    "  reader config experiments/demo/config.yaml\n"
    "  reader validate -c experiments/demo/config.yaml\n"
    "  reader run 3 --step 2\n"
    "  reader run --resume-from merge --until sfxi_vec8\n"
    "  reader report experiments/demo/config.yaml\n"
    "Tip: CONFIG can be a path, an experiment dir, or an index from `reader ls`."
)

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help=(
        "reader — experiment workbench + pipeline runner.\n\n"
        "Run config.yaml pipelines (ingest → merge → transform → sfxi) and optional reports "
        "(plots/exports) inside experiment directories; outputs land under outputs/."
    ),
    epilog=APP_EPILOG,
)
console = Console(theme=THEME)
rich_tracebacks(show_locals=False)


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    """Show help when no command is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

_RUN_EXTRA_ARG = typer.Argument(
    None,
    metavar="[EXTRA]...",
    help=(
        "Optional tokens for the 'step N' form. "
        "Note: variadic positional arguments cannot have defaults (Click restriction)."
    ),
)

LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}

ERROR_EXIT_CODES: dict[type[Exception], int] = {
    ConfigError: 2,
    RegistryError: 2,
    ContractError: 3,
    ParseError: 4,
    MergeError: 4,
    TransformError: 4,
    SFXIError: 4,
    ArtifactError: 4,
    PlotError: 5,
    ExecutionError: 5,
}

ERROR_HINTS: dict[type[Exception], str] = {
    ConfigError: "Tip: run `reader validate` to see config and plugin checks.",
    ContractError: "Tip: run `reader contracts` to inspect expected schemas.",
    ParseError: "Tip: verify input paths and file formats.",
    MergeError: "Tip: verify merge keys and metadata columns.",
    TransformError: "Tip: check upstream channels/columns and align_on keys.",
    PlotError: "Tip: verify plot inputs and required columns; run with --log-level DEBUG for context.",
    ExecutionError: "Tip: re-run with --dry-run or --log-level DEBUG to inspect the plan/logs.",
}


def _normalize_log_level(level: str) -> str:
    lvl = level.strip().upper()
    if lvl not in LOG_LEVELS:
        raise typer.BadParameter(
            f"Invalid --log-level '{level}'. Use one of: DEBUG | INFO | WARNING | ERROR."
        )
    return lvl


def _exit_code_for(err: Exception) -> int:
    for cls, code in ERROR_EXIT_CODES.items():
        if isinstance(err, cls):
            return code
    return 5


def _render_error(err: Exception) -> None:
    msg = str(err).strip() or err.__class__.__name__
    hint = ""
    for cls, tip in ERROR_HINTS.items():
        if isinstance(err, cls):
            hint = tip
            break
    body = f"[error]✗ {msg}[/error]"
    if hint:
        body += f"\n[muted]{hint}[/muted]"
    body += f"\n[muted]Exit code: {_exit_code_for(err)}[/muted]"
    console.print(Panel.fit(body, border_style="error", box=box.ROUNDED))


def _handle_reader_errors(fn):
    try:
        return fn()
    except ReaderError as e:
        _render_error(e)
        raise typer.Exit(code=_exit_code_for(e)) from None


def _select_job_arg(job: str | None, config: str | None) -> str | None:
    if job and config:
        raise typer.BadParameter("Pass CONFIG either positionally or via --config/-c, not both.")
    return config or job


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
        "Run inside an experiment dir or pass a path to the config (or the experiment dir)."
    )


def _find_jobs(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.glob("**/config.yaml"):
        out.append(p.resolve())
    return sorted(out)


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


class MetadataKind(str, Enum):
    sample_map = "sample_map"
    sample_metadata = "sample_metadata"


def _write_metadata_template(path: Path, kind: MetadataKind) -> None:
    if kind == MetadataKind.sample_map:
        header = "position,design_id,treatment\n"
    else:
        header = "sample_id,design_id,treatment\n"
    path.write_text(header, encoding="utf-8")


@app.command(help="List built-in presets (or show details for one).")
def presets(
    name: str | None = typer.Argument(
        None,
        metavar="[PRESET]",
        help="Optional preset name to show expanded steps.",
    ),
    write: bool = typer.Option(
        False,
        "--write",
        help="Emit a minimal config.yaml using the given preset (prints to stdout unless --output is set).",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        metavar="PATH",
        help="Write the generated config to this path instead of stdout.",
    ),
):
    def _run():
        if write:
            if not name:
                raise typer.BadParameter("--write requires a PRESET name.")
            describe_preset(name)  # validate existence
            text = _render_config_yaml(presets=[name])
            if output:
                Path(output).write_text(text, encoding="utf-8")
            else:
                typer.echo(text)
            return

        if name:
            info = describe_preset(name)
            table = _table(f"Preset: {info['name']}")
            table.add_column("#", justify="right", style="muted")
            table.add_column("Step ID", style="accent")
            table.add_column("Plugin")
            for i, step in enumerate(info["steps"], 1):
                table.add_row(str(i), step.get("id", ""), step.get("uses", ""))
            console.print(Panel(table, border_style="accent", box=box.ROUNDED, subtitle=info["description"]))
            return

        table = _table("Presets")
        table.add_column("Name", style="accent")
        table.add_column("Description")
        for preset, desc in list_presets():
            table.add_row(preset, desc)
        console.print(Panel(table, border_style="accent", box=box.ROUNDED))

    return _handle_reader_errors(_run)


@app.command(
    help=(
        "Initialize a new experiment directory with inputs/ outputs/ notebooks/ and a minimal config.yaml. "
        "Use --preset to choose a workflow preset."
    )
)
def init(
    path: str = typer.Argument(..., metavar="PATH", help="Target experiment directory to create."),
    preset: list[str] = typer.Option(
        None,
        "--preset",
        "-p",
        metavar="NAME",
        help="Preset(s) to include in the config (repeatable).",
    ),
    report_preset: list[str] = typer.Option(
        None,
        "--report-preset",
        metavar="NAME",
        help="Report preset(s) to include in the config (repeatable).",
    ),
    metadata: MetadataKind | None = typer.Option(
        None,
        "--metadata",
        metavar="KIND",
        help="Optional metadata template: sample_map | sample_metadata.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing config/metadata files if present.",
    ),
):
    def _run():
        if not preset and not report_preset:
            raise typer.BadParameter(
                "Use --preset and/or --report-preset to choose at least one preset (see `reader presets`)."
            )
        for p in preset or []:
            describe_preset(p)
        for p in report_preset or []:
            describe_preset(p)

        target = Path(path).resolve()
        if target.exists() and any(target.iterdir()) and not force:
            raise typer.BadParameter(
                f"Target directory is not empty: {target}. Use --force to overwrite config files."
            )
        target.mkdir(parents=True, exist_ok=True)

        # Standard subdirs (always safe to create)
        for sub in ("inputs", "outputs", "notebooks"):
            (target / sub).mkdir(parents=True, exist_ok=True)

        cfg_path = target / "config.yaml"
        if cfg_path.exists() and not force:
            raise typer.BadParameter(f"{cfg_path} already exists. Use --force to overwrite.")
        cfg_text = _render_config_yaml(
            presets=list(preset or []),
            report_presets=list(report_preset or []),
            experiment_name=target.name,
        )
        cfg_path.write_text(cfg_text, encoding="utf-8")

        if metadata is not None:
            meta_path = target / "metadata.csv"
            if meta_path.exists() and not force:
                raise typer.BadParameter(f"{meta_path} already exists. Use --force to overwrite.")
            _write_metadata_template(meta_path, metadata)

        msg = (
            f"Initialized experiment in [path]{target}[/path]\n"
            f"Config: [path]{cfg_path}[/path]\n"
            f"Presets: {', '.join(preset or [])}"
        )
        if report_preset:
            msg += f"\nReport presets: {', '.join(report_preset)}"
        console.print(Panel.fit(msg, border_style="ok", box=box.ROUNDED))

    return _handle_reader_errors(_run)


@app.command(
    help=(
        "List experiments under --root (default: nearest ./experiments). "
        "Shows directory names and whether outputs/manifest.json exists."
    )
)
def ls(
    root: str = typer.Option(
        "./experiments",
        "--root",
        metavar="DIR",
        help="Directory to search recursively for **/config.yaml.",
    ),
):
    # If user didn't override --root, auto-detect nearest experiments/ so this
    # works from anywhere inside the repository.
    if str(root).strip() == "./experiments":
        root_path = _find_nearest_experiments_dir(Path.cwd())
    else:
        root_path = Path(root).resolve()
    if not root_path.exists():
        raise typer.BadParameter(f"--root directory not found: {root_path}")
    jobs = _find_jobs(root_path)
    if not jobs:
        console.print(
            Panel.fit(
                f"No experiments found under [path]{root_path}[/path] (looked for **/config.yaml).",
                border_style="warn",
                box=box.ROUNDED,
            )
        )
        return
    t = _table("Experiments")
    t.add_column("#", justify="right", style="muted")
    t.add_column("Name", style="accent")
    t.add_column("Outputs", justify="center")
    for i, p in enumerate(jobs, 1):
        name = p.parent.name
        man = p.parent / "outputs" / "manifest.json"
        t.add_row(str(i), name, _checkmark(man.exists()))
    console.print(
        Panel(
            t,
            border_style="accent",
            box=box.ROUNDED,
            subtitle=f"[muted]root: [path]{root_path}[/path] — {len(jobs)} found[/muted]",
        )
    )


@app.command(
    help=(
        "Print the execution plan for a config.yaml (no steps are run). "
        "Accepts a config path, experiment directory, or index from 'reader ls'."
    )
)
def explain(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        metavar="CONFIG",
        help="Same as CONFIG positional; useful in scripts.",
    ),
    reports: bool = typer.Option(
        False,
        "--reports",
        help="Show the report plan (plots/exports) instead of the pipeline.",
    ),
    all: bool = typer.Option(
        False,
        "--all",
        help="Show both pipeline and report plans.",
    ),
):
    def _run():
        job_path = _infer_job_path(_select_job_arg(job, config))
        _append_journal(job_path, f"reader explain {job_path}")
        spec = ReaderSpec.load(job_path)
        if all:
            explain_steps(steps=spec.steps, console=console, title="Pipeline plan")
            if spec.reports:
                explain_steps(steps=spec.reports, console=console, title="Report plan")
            else:
                console.print(Panel.fit("No reports defined.", border_style="warn", box=box.ROUNDED))
            return
        if reports:
            if not spec.reports:
                console.print(Panel.fit("No reports defined.", border_style="warn", box=box.ROUNDED))
                return
            explain_steps(steps=spec.reports, console=console, title="Report plan")
            return
        explain_job(spec, console=console)

    return _handle_reader_errors(_run)


@app.command(
    help=(
        "Validate a config.yaml (schema + plugin configs). No steps are executed. "
        "Accepts a config path, experiment directory, or index from 'reader ls'."
    )
)
def validate(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        metavar="CONFIG",
        help="Same as CONFIG positional; useful in scripts.",
    ),
):
    def _run():
        job_path = _infer_job_path(_select_job_arg(job, config))
        _append_journal(job_path, f"reader validate {job_path}")
        spec = ReaderSpec.load(job_path)
        validate_job(spec, console=console)

    return _handle_reader_errors(_run)


def _render_config_yaml(
    *, presets: list[str], report_presets: list[str] | None = None, experiment_name: str | None = None
) -> str:
    payload: dict[str, Any] = {"experiment": {"outputs": "./outputs"}}
    if experiment_name:
        payload["experiment"]["name"] = experiment_name
    if presets:
        payload["presets"] = presets
    if report_presets:
        payload["report_presets"] = report_presets
    return yaml.safe_dump(payload, sort_keys=False)


@app.command(
    help=(
        "Print the expanded config (presets + overrides applied) as YAML. "
        "Paths are resolved relative to the config file."
    )
)
def config(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        metavar="CONFIG",
        help="Same as CONFIG positional; useful in scripts.",
    ),
    schema: bool = typer.Option(
        False,
        "--schema",
        help="Print the JSON schema for config.yaml and exit (no CONFIG required).",
    ),
    template: bool = typer.Option(
        False,
        "--template",
        help="Print a minimal config template and exit (requires --preset).",
    ),
    preset: list[str] = typer.Option(
        None,
        "--preset",
        "-p",
        metavar="NAME",
        help="Preset(s) to include when using --template (repeatable).",
    ),
    report_preset: list[str] = typer.Option(
        None,
        "--report-preset",
        metavar="NAME",
        help="Report preset(s) to include when using --template (repeatable).",
    ),
):
    def _run():
        if schema:
            if job or config or template:
                raise typer.BadParameter("--schema cannot be combined with CONFIG or --template.")
            typer.echo(json.dumps(ReaderSpec.model_json_schema(), indent=2))
            return

        if template:
            if job or config or schema:
                raise typer.BadParameter("--template cannot be combined with CONFIG or --schema.")
            if not preset and not report_preset:
                raise typer.BadParameter("Use --preset and/or --report-preset to build a template config.")
            for p in preset or []:
                describe_preset(p)
            for p in report_preset or []:
                describe_preset(p)
            typer.echo(
                _render_config_yaml(presets=list(preset or []), report_presets=list(report_preset or []))
            )
            return

        job_path = _infer_job_path(_select_job_arg(job, config))
        spec = ReaderSpec.load(job_path)
        payload = spec.model_dump(by_alias=True)
        payload.get("experiment", {}).pop("root", None)
        # Hide empty collections for readability
        if not payload.get("collections"):
            payload.pop("collections", None)
        if not payload.get("reports"):
            payload.pop("reports", None)
        for key in ("presets", "overrides", "report_presets", "report_overrides"):
            payload.pop(key, None)
        yaml_text = yaml.safe_dump(payload, sort_keys=False)
        typer.echo(yaml_text)

    return _handle_reader_errors(_run)


@app.command(
    help=(
        "Run a pipeline from config.yaml. "
        "Use --dry-run to print the plan without running. "
        "Slice with --resume-from/--until, or use --step to run exactly one step. "
        "Logs go to outputs/reader.log."
    )
)
def run(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        metavar="CONFIG",
        help="Same as CONFIG positional; useful in scripts.",
    ),
    step: str | None = typer.Option(
        None,
        "--step",
        "-s",
        metavar="STEP",
        help="Run exactly this step by id or 1-based index (sugar for --resume-from/--until).",
    ),
    step_pos: str | None = typer.Argument(
        None,
        metavar="[STEP]",
        help="(optional) Step id or 1-based index. You can also write: 'reader run 14 step 11'.",
    ),
    extra: list[str] | None = _RUN_EXTRA_ARG,
    resume_from: str | None = typer.Option(
        None,
        "--resume-from",
        metavar="STEP_ID",
        help="Start from this step (inclusive). Use an exact id as declared in the config.",
    ),
    until: str | None = typer.Option(
        None,
        "--until",
        metavar="STEP_ID",
        help="Stop after this step (inclusive). Use an exact id as declared in the config.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan only: validate and print the plan without executing steps.",
    ),
    with_report: bool = typer.Option(
        False,
        "--with-report",
        help="After running the pipeline, run report steps (plots/exports) if defined.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        metavar="LEVEL",
        help="Logging level: DEBUG | INFO | WARNING | ERROR (default: INFO).",
    ),
):
    def _run():
        # Normalize args & support sugar: 'reader run 14 11' and 'reader run 14 step 11'
        job_path = _infer_job_path(_select_job_arg(job, config))
        norm_log = _normalize_log_level(log_level)
        parts = ["reader run", str(job_path)]

        # Interpret trailing STEP tokens if provided
        selected_step: str | None = None
        if step is not None:
            selected_step = step
        else:
            # Build tokens explicitly; 'extra' can be None for a variadic positional.
            tokens: list[str] = []
            if step_pos:
                tokens.append(step_pos)
            if extra:
                tokens.extend(extra)
            tokens = [t for t in tokens if t]
            if tokens:
                # accept: "11" • "sfxi_vec8" • "step 11"
                if tokens[0].lower() == "step" and len(tokens) >= 2:
                    selected_step = tokens[1]
                elif len(tokens) == 1:
                    selected_step = tokens[0]
                elif len(tokens) == 2 and tokens[0].lower() == "step":
                    selected_step = tokens[1]
                else:
                    raise typer.BadParameter(f"Unexpected extra arguments: {' '.join(tokens)}")

        if selected_step and (resume_from or until):
            raise typer.BadParameter("--step/STEP cannot be combined with --resume-from/--until")
        if selected_step and with_report:
            raise typer.BadParameter("--with-report cannot be combined with --step/STEP")

        if selected_step:
            # Convert possibly-numeric STEP into a concrete id
            spec = ReaderSpec.load(job_path)
            step_id = _resolve_step_id(spec, selected_step)
            parts += ["step", selected_step]
            if dry_run:
                parts += ["--dry-run"]
            if norm_log and norm_log != "INFO":
                parts += ["--log-level", norm_log]
            _append_journal(job_path, " ".join(parts))
            run_job(
                job_path,
                resume_from=step_id,
                until=step_id,
                dry_run=dry_run,
                log_level=norm_log,
                console=console,
            )
            return

        # Accept numeric indices for --resume-from/--until as well
        res = resume_from
        stop = until
        if res and res.isdigit():
            spec = ReaderSpec.load(job_path)
            res = _resolve_step_id(spec, res)
        if stop and stop.isdigit():
            spec = ReaderSpec.load(job_path)
            stop = _resolve_step_id(spec, stop)
        if res:
            parts += ["--resume-from", res]
        if stop:
            parts += ["--until", stop]
        if dry_run:
            parts += ["--dry-run"]
        if norm_log and norm_log != "INFO":
            parts += ["--log-level", norm_log]
        if with_report:
            parts += ["--with-report"]
        _append_journal(job_path, " ".join(parts))
        # Pass our themed console down so engine can render pretty output & progress
        run_job(job_path, resume_from=res, until=stop, dry_run=dry_run, log_level=norm_log, console=console)
        if with_report:
            spec = ReaderSpec.load(job_path)
            if not spec.reports:
                console.print(Panel.fit("No reports defined — skipping.", border_style="warn", box=box.ROUNDED))
                return
            run_reports(job_path, dry_run=dry_run, log_level=norm_log, console=console)

    return _handle_reader_errors(_run)


@app.command(
    help=(
        "Run report steps (plots/exports) from config.yaml. "
        "Reports read artifacts and write deliverables; they do not re-run the pipeline."
    )
)
def report(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        metavar="CONFIG",
        help="Same as CONFIG positional; useful in scripts.",
    ),
    step: str | None = typer.Option(
        None,
        "--step",
        "-s",
        metavar="STEP",
        help="Run exactly this report step by id or 1-based index.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan only: validate and print the report plan without executing.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        metavar="LEVEL",
        help="Logging level: DEBUG | INFO | WARNING | ERROR (default: INFO).",
    ),
):
    def _run():
        job_path = _infer_job_path(_select_job_arg(job, config))
        norm_log = _normalize_log_level(log_level)
        spec = ReaderSpec.load(job_path)
        resolved = _resolve_report_step_id(spec, step) if step else None
        _append_journal(job_path, f"reader report {job_path}")
        run_reports(job_path, step_id=resolved, dry_run=dry_run, log_level=norm_log, console=console)

    return _handle_reader_errors(_run)


@app.command(
    help=(
        "List artifacts from outputs/manifest.json. "
        "By default shows the latest revision; use --all for history counts."
    )
)
def artifacts(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        metavar="CONFIG",
        help="Same as CONFIG positional; useful in scripts.",
    ),
    all: bool = typer.Option(False, "--all", help="Show revision history counts instead of latest entries."),
):
    def _run():
        spec = ReaderSpec.load(_infer_job_path(_select_job_arg(job, config)))
        outputs_dir = Path(spec.experiment.outputs)
        manifest_path = outputs_dir / "manifest.json"
        if not manifest_path.exists():
            console.print(
                Panel.fit(
                    "No manifest.json found in outputs/. Run 'reader run' to materialize artifacts.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not man.get("artifacts") and not man.get("history"):
            console.print(
                Panel.fit(
                    "No artifacts recorded yet. Run 'reader run' to produce outputs.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        store = ArtifactStore(outputs_dir)
        if all:
            t = _table("Artifacts • history")
            t.add_column("Label")
            t.add_column("Revisions", justify="right")
            for k, hist in man["history"].items():
                t.add_row(k, str(len(hist)))
        else:
            t = _table("Artifacts • latest")
            t.add_column("Label")
            t.add_column("Contract", style="accent")
            t.add_column("Path", style="path")
            for k, entry in man["artifacts"].items():
                t.add_row(k, entry["contract"], str(store.artifacts_dir / entry["step_dir"] / entry["filename"]))
        console.print(Panel(t, border_style="accent", box=box.ROUNDED))

    return _handle_reader_errors(_run)


@app.command(help="Show steps declared in a config.yaml (index, id, plugin key).")
def steps(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        metavar="CONFIG",
        help="Same as CONFIG positional; useful in scripts.",
    ),
):
    def _run():
        spec = ReaderSpec.load(_infer_job_path(_select_job_arg(job, config)))
        t = _table("Steps")
        t.add_column("#", justify="right", style="muted")
        t.add_column("id", style="accent")
        t.add_column("uses")
        for i, s in enumerate(spec.steps, 1):
            t.add_row(str(i), s.id, s.uses)
        console.print(
            Panel(t, border_style="accent", box=box.ROUNDED, subtitle=f"[muted]{len(spec.steps)} total[/muted]")
        )

    return _handle_reader_errors(_run)


@app.command(
    help=(
        "List discovered plugins (built-ins + entry points). Use --category to filter."
    )
)
def plugins(
    category: str | None = typer.Option(
        None,
        "--category",
        metavar="NAME",
        help="Filter by category: ingest | merge | transform | plot | export | validator",
    ),
):
    def _run():
        reg = load_entry_points()
        cats = reg.categories()
        if category and category not in cats:
            valid = " | ".join(sorted(cats))
            raise typer.BadParameter(f"Unknown category '{category}'. Valid: {valid}.")
        if category:
            cats = {category: cats.get(category, {})}
        t = _table("Plugins")
        t.add_column("category", style="accent")
        t.add_column("key")
        t.add_column("class", style="muted", overflow="fold")
        total = 0
        for cat, m in cats.items():
            for key, cls in sorted(m.items(), key=lambda kv: kv[0]):
                t.add_row(cat, key, f"{cls.__module__}.{cls.__name__}")
            total += len(m)
        console.print(
            Panel(
                t,
                border_style="accent",
                box=box.ROUNDED,
                subtitle=f"[muted]{total} plugin(s) discovered[/muted]",
            )
        )

    return _handle_reader_errors(_run)


@app.command(help="List built-in contract schemas (or show details for one).")
def contracts(
    name: str | None = typer.Argument(
        None,
        metavar="[CONTRACT_ID]",
        help="Optional contract id to show full column rules.",
    )
):
    def _run():
        if name:
            if name not in BUILTIN:
                available = ", ".join(sorted(BUILTIN.keys()))
                raise typer.BadParameter(f"Unknown contract '{name}'. Available: {available}")
            c = BUILTIN[name]
            t = _table(f"Contract: {c.id}")
            t.add_column("column", style="accent")
            t.add_column("dtype")
            t.add_column("required", justify="center")
            t.add_column("allow_nan", justify="center")
            t.add_column("nonneg", justify="center")
            t.add_column("allowed_values", overflow="fold")
            for rule in c.columns:
                allowed = ",".join(rule.allowed_values) if rule.allowed_values else ""
                t.add_row(
                    rule.name,
                    rule.dtype,
                    "yes" if rule.required else "no",
                    "yes" if rule.allow_nan else "no",
                    "yes" if rule.nonnegative else "no",
                    allowed,
                )
            console.print(Panel(t, border_style="accent", box=box.ROUNDED, subtitle=c.description))
            return

        t = _table("Contracts")
        t.add_column("id", style="accent")
        t.add_column("description")
        for cid, contract in sorted(BUILTIN.items(), key=lambda kv: kv[0]):
            t.add_row(cid, contract.description)
        console.print(Panel(t, border_style="accent", box=box.ROUNDED))

    return _handle_reader_errors(_run)


# --------------------------- run one step ---------------------------


def _resolve_step_id(spec: ReaderSpec, which: str) -> str:
    """
    Accept either a step id or a 1-based index string; return the step id.
    """
    which_str = str(which).strip()
    if which_str.isdigit():
        idx = int(which_str)
        if idx < 1 or idx > len(spec.steps):
            raise typer.BadParameter(f"Step index out of range: {idx} (valid: 1..{len(spec.steps)})")
        return spec.steps[idx - 1].id
    # by id
    try:
        _ = next(s for s in spec.steps if s.id == which_str)
    except StopIteration:
        ids = [s.id for s in spec.steps]
        options = ", ".join(ids[:12])
        close = difflib.get_close_matches(which_str, ids, n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise typer.BadParameter(
            f"Unknown step id '{which_str}'.{hint} Tip: use 'reader steps' to list ids "
            f"(first few: {options}{' …' if len(spec.steps) > 12 else ''})."
        ) from None
    return which_str


def _resolve_report_step_id(spec: ReaderSpec, which: str) -> str:
    which_str = str(which).strip()
    reports = list(spec.reports or [])
    if which_str.isdigit():
        idx = int(which_str)
        if idx < 1 or idx > len(reports):
            raise typer.BadParameter(f"Report step index out of range: {idx} (valid: 1..{len(reports)})")
        return reports[idx - 1].id
    try:
        _ = next(s for s in reports if s.id == which_str)
    except StopIteration:
        ids = [s.id for s in reports]
        options = ", ".join(ids[:12])
        close = difflib.get_close_matches(which_str, ids, n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise typer.BadParameter(
            f"Unknown report step id '{which_str}'.{hint} Tip: use 'reader explain --reports' to list ids "
            f"(first few: {options}{' …' if len(reports) > 12 else ''})."
        ) from None
    return which_str


def _run_one_step_cli(which: str, job: str | None, *, dry_run: bool, log_level: str) -> None:
    def _run():
        job_path = _infer_job_path(job)
        norm_log = _normalize_log_level(log_level)
        spec = ReaderSpec.load(job_path)
        step_id = _resolve_step_id(spec, which)
        _append_journal(job_path, f"reader run-step {which} --config {job_path}")

        # Nice preflight: tell the user if some upstream artifacts are clearly missing
        # (we only check presence; contracts are still asserted by the engine).
        store = ArtifactStore(Path(spec.experiment.outputs))
        selected = next(s for s in spec.steps if s.id == step_id)
        missing = []
        for _, target in (selected.reads or {}).items():
            if isinstance(target, str) and not target.startswith("file:") and store.latest(target) is None:
                missing.append(target)
        if missing and not dry_run:
            lines = []
            for lab in missing:
                prod = lab.split("/", 1)[0]
                lines.append(f"   - {lab}   (produced by step “{prod}”)")
            msg = (
                f"[error]✗ Cannot run step “{step_id}”: missing upstream artifact(s)[/error]\n"
                + "\n".join(lines)
                + "\n[muted]Tip: materialize prerequisites with:[/muted]\n"
                f"   [accent]reader run --until {missing[0].split('/', 1)[0]}[/accent]\n"
                "[muted]Then re-run just this step with:[/muted]\n"
                f"   [accent]reader run-step {step_id}[/accent]"
            )
            console.print(Panel.fit(msg, border_style="error", box=box.ROUNDED))
            raise typer.Exit(code=_exit_code_for(ArtifactError(msg)))

        # Delegate to the engine using (resume_from == until == selected step)
        try:
            run_job(
                job_path,
                resume_from=step_id,
                until=step_id,
                dry_run=dry_run,
                log_level=norm_log,
                console=console,
            )
        except ExecutionError as e:
            # Make common “artifact missing” errors friendlier.
            m = re.search(r"Artifact '([^']+)' missing", str(e))
            if m:
                art = m.group(1)
                prod = art.split("/", 1)[0]
                msg = (
                    f"[error]✗ Cannot run step “{step_id}”: missing upstream artifact[/error]\n"
                    f"   - {art}   (produced by step “{prod}”)\n"
                    "[muted]Tip: materialize prerequisites with:[/muted]\n"
                    f"   [accent]reader run --until {prod}[/accent]\n"
                    "[muted]Then re-run just this step with:[/muted]\n"
                    f"   [accent]reader run-step {step_id}[/accent]"
                )
                console.print(Panel.fit(msg, border_style="error", box=box.ROUNDED))
                raise typer.Exit(code=_exit_code_for(e)) from None
            raise

    return _handle_reader_errors(_run)


@app.command(
    name="run-step",
    help=(
        "Run exactly one step by ID or 1-based index using existing artifacts. "
        "Does not run prior steps. Use --dry-run to print the plan slice."
    ),
)
def run_step(
    which: str = typer.Argument(..., metavar="STEP", help="Step id (e.g. 'ingest') or 1-based index (e.g. '1')."),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        metavar="CONFIG",
        help="Path to config.yaml • experiment dir • or numeric index from 'reader ls' (defaults to nearest ./config.yaml).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan only: validate and print the plan slice without executing."
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", metavar="LEVEL", help="Logging level: DEBUG | INFO | WARNING | ERROR."
    ),
):
    _run_one_step_cli(which, config, dry_run=dry_run, log_level=log_level)


@app.command(name="step", help="[alias] Same as 'run-step'. Run exactly one step by ID or index.")
def step_alias(
    which: str = typer.Argument(..., metavar="STEP"),
    config: str | None = typer.Option(None, "--config", "-c", metavar="CONFIG"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    log_level: str = typer.Option("INFO", "--log-level", metavar="LEVEL"),
):
    _run_one_step_cli(which, config, dry_run=dry_run, log_level=log_level)
