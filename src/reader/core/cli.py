"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import re  # used for friendly error parsing in run-step
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

from reader.core.artifacts import ArtifactStore
from reader.core.config_model import ReaderSpec
from reader.core.engine import explain as explain_job
from reader.core.engine import run_job
from reader.core.engine import validate as validate_job
from reader.core.errors import ConfigError, ExecutionError, ReaderError
from reader.core.notebooks import list_notebook_presets, write_experiment_notebook
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

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help=(
        "reader — experiment pipeline runner.\n\n"
        "Run pipelines defined by config.yaml (steps + optional deliverables). "
        "Use 'reader deliverables' for plots/exports and 'reader explore' for notebooks. "
        "Start with 'reader demo' or 'reader ls'."
    ),
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


@app.command(help="List presets or describe one.")
def presets(
    name: str | None = typer.Argument(
        None,
        metavar="[NAME]",
        help="Optional preset name to describe (e.g., plate_reader/synergy_h1).",
    ),
):
    try:
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
    except ConfigError as e:
        raise typer.BadParameter(str(e)) from e


@app.command(help="Scaffold an interactive marimo notebook (no execution).")
def explore(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help="Experiment config path, directory, or index from 'reader ls'.",
    ),
    name: str = typer.Option(
        "eda.py",
        "--name",
        help="Notebook filename (created under experiments/<exp>/notebooks).",
    ),
    preset: str = typer.Option(
        "eda/basic",
        "--preset",
        help="Notebook preset (use --list-presets to see options).",
    ),
    list_presets: bool = typer.Option(
        False,
        "--list-presets",
        help="List notebook presets and exit.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite the notebook if it already exists.",
    ),
):
    try:
        if list_presets:
            table = _table("Notebook presets")
            table.add_column("Name", style="accent")
            table.add_column("Description")
            for preset_name, desc in list_notebook_presets():
                table.add_row(preset_name, desc)
            console.print(Panel(table, border_style="accent", box=box.ROUNDED))
            return
        job_path = _infer_job_path(job)
        exp_dir = job_path.parent
        nb_dir = exp_dir / "notebooks"
        target = nb_dir / name
        has_fcs = any(p.suffix.lower() == ".fcs" for p in exp_dir.rglob("*.fcs"))
        write_experiment_notebook(target, preset=preset, overwrite=force)
        console.print(
            Panel.fit(
                f"✓ Notebook created: [path]{target}[/path]\n[muted]preset[/muted]: {preset}",
                border_style="ok",
                box=box.ROUNDED,
            )
        )
        sync_cmd = "uv sync --locked --group notebooks"
        if has_fcs:
            sync_cmd = "uv sync --locked --group notebooks --group cytometry"
        console.print(
            Panel.fit(
                "Next:\n"
                f"  1) {sync_cmd}\n"
                "     (Note: uv sync removes undeclared packages; include extra groups you use)\n"
                "  2) uv run marimo edit " + str(target),
                border_style="accent",
                box=box.ROUNDED,
            )
        )
    except ConfigError as e:
        raise typer.BadParameter(str(e)) from e


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


@app.command(
    help="List experiments under a root (default: ./experiments)."
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
    help="Show planned steps and contracts (no execution)."
)
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
        spec = ReaderSpec.load(job_path)
        explain_job(spec, console=console)
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(
    help="Validate config and plugin params (no data I/O)."
)
def validate(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
):
    try:
        job_path = _infer_job_path(job)
        _append_journal(job_path, f"reader validate {job_path}")
        spec = ReaderSpec.load(job_path)
        validate_job(spec, console=console)
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(help="Print the expanded config (presets + overrides applied).")
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
        spec = ReaderSpec.load(job_path)
    except ReaderError as e:
        _handle_reader_error(e)
    fmt = str(format).strip().lower()
    payload = spec.model_dump(by_alias=True)
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if fmt == "yaml":
        typer.echo(yaml.safe_dump(payload, sort_keys=False))
        return
    raise typer.BadParameter("format must be 'yaml' or 'json'")


@app.command(help="Check that file inputs declared in reads: file: exist.")
def check_inputs(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
):
    try:
        job_path = _infer_job_path(job)
        spec = ReaderSpec.load(job_path)
    except ReaderError as e:
        _handle_reader_error(e)

    exp_root = Path(spec.experiment.get("root", job_path.parent))
    entries: list[tuple[str, str, str, Path]] = []
    for label, items in (("step", spec.steps), ("deliverable", spec.deliverables or [])):
        for step in items:
            for key, target in (step.reads or {}).items():
                if isinstance(target, str) and target.startswith("file:"):
                    path = Path(target.split("file:", 1)[1])
                    entries.append((label, step.id, key, path))

    if not entries:
        console.print(Panel.fit("No file inputs declared in reads: file:", border_style="warn", box=box.ROUNDED))
        return

    missing = []
    for label, step_id, key, path in entries:
        if not path.exists():
            missing.append((label, step_id, key, path))

    if missing:
        lines = ["[error]✗ Missing input files[/error]"]
        for label, step_id, key, path in missing:
            rel = None
            try:
                rel = path.relative_to(exp_root)
            except Exception:
                rel = path
            lines.append(f"- {label}:{step_id} • {key} → [path]{rel}[/path]")
        lines.append("[muted]tip[/muted]: update reads: file: paths or place files under the experiment directory")
        console.print(Panel.fit("\n".join(lines), border_style="error", box=box.ROUNDED))
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"✓ Inputs found ({len(entries)} file input(s))",
            border_style="ok",
            box=box.ROUNDED,
        )
    )


@app.command(
    help="Run pipeline. Deliverables (plots/exports) run by default; use --no-deliverables for pipeline-only."
)
def run(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
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
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        metavar="LEVEL",
        help="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL (default: INFO).",
    ),
    deliverables: bool | None = typer.Option(
        None,
        "--deliverables/--no-deliverables",
        help="Run deliverable steps after the pipeline (default: on for full runs).",
    ),
):
    # Normalize args & support sugar: 'reader run 14 11' and 'reader run 14 step 11'
    job_path = _infer_job_path(job)
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
    if selected_step and deliverables is not None:
        raise typer.BadParameter("--deliverables/--no-deliverables cannot be used with --step/STEP.")

    if selected_step:
        # Convert possibly-numeric STEP into a concrete id
        try:
            spec = ReaderSpec.load(job_path)
        except ReaderError as e:
            _handle_reader_error(e)
        step_id = _resolve_step_id(spec, selected_step)
        parts += ["step", selected_step]
        if dry_run:
            parts += ["--dry-run"]
        if log_level and log_level != "INFO":
            parts += ["--log-level", log_level]
        if deliverables is not None:
            parts += ["--deliverables" if deliverables else "--no-deliverables"]
        _append_journal(job_path, " ".join(parts))
        try:
            run_job(
                job_path,
                resume_from=step_id,
                until=step_id,
                dry_run=dry_run,
                log_level=log_level,
                console=console,
                include_pipeline=True,
                include_deliverables=False,
            )
        except ReaderError as e:
            _handle_reader_error(e)
        return

    # Accept numeric indices for --resume-from/--until as well
    if resume_from and resume_from.isdigit():
        try:
            spec = ReaderSpec.load(job_path)
        except ReaderError as e:
            _handle_reader_error(e)
        resume_from = _resolve_step_id(spec, resume_from)
    if until and until.isdigit():
        try:
            spec = ReaderSpec.load(job_path)
        except ReaderError as e:
            _handle_reader_error(e)
        until = _resolve_step_id(spec, until)
    if resume_from:
        parts += ["--resume-from", resume_from]
    if until:
        parts += ["--until", until]
    if dry_run:
        parts += ["--dry-run"]
    if log_level and log_level != "INFO":
        parts += ["--log-level", log_level]
    if deliverables is not None:
        parts += ["--deliverables" if deliverables else "--no-deliverables"]
    _append_journal(job_path, " ".join(parts))
    # Pass our themed console down so engine can render pretty output & progress
    include_deliverables = True if deliverables is None else deliverables
    if deliverables is None and (resume_from or until):
        include_deliverables = False
    try:
        run_job(
            job_path,
            resume_from=resume_from,
            until=until,
            dry_run=dry_run,
            log_level=log_level,
            console=console,
            include_pipeline=True,
            include_deliverables=include_deliverables,
        )
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(
    help="Render deliverable steps only (static plots/exports) using existing artifacts. Use --list to show steps."
)
def deliverables(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    list_only: bool = typer.Option(
        False,
        "--list",
        help="List deliverable steps for this config and exit.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan only: validate and print the deliverables plan without executing.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        metavar="LEVEL",
        help="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL (default: INFO).",
    ),
):
    try:
        job_path = _infer_job_path(job)
        spec = ReaderSpec.load(job_path)
    except ReaderError as e:
        _handle_reader_error(e)
    if list_only:
        if not spec.deliverables:
            console.print(Panel.fit("No deliverables configured.", border_style="warn", box=box.ROUNDED))
            return
        t = _table("Deliverables")
        t.add_column("#", justify="right", style="muted")
        t.add_column("id", style="accent")
        t.add_column("uses")
        for i, s in enumerate(spec.deliverables, 1):
            t.add_row(str(i), s.id, s.uses)
        console.print(
            Panel(
                t,
                border_style="accent",
                box=box.ROUNDED,
                subtitle=f"[muted]{len(spec.deliverables)} total[/muted]",
            )
        )
        return
    if not spec.deliverables:
        raise typer.BadParameter(
            "No deliverables configured in this experiment. Add a deliverables section or deliverable_presets."
        )
    parts = ["reader deliverables", str(job_path)]
    if dry_run:
        parts += ["--dry-run"]
    if log_level and log_level != "INFO":
        parts += ["--log-level", log_level]
    _append_journal(job_path, " ".join(parts))
    try:
        run_job(
            job_path,
            dry_run=dry_run,
            log_level=log_level,
            console=console,
            include_pipeline=False,
            include_deliverables=True,
        )
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(
    help="List emitted artifacts from outputs/manifest.json."
)
def artifacts(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help="Path to config.yaml • experiment directory • or numeric index from 'reader ls' (defaults to nearest ./config.yaml)",
    ),
    all: bool = typer.Option(False, "--all", help="Show revision history counts instead of latest entries."),
):
    try:
        spec = ReaderSpec.load(_infer_job_path(job))
        outputs_dir = Path(spec.experiment["outputs"])
        manifest_path = outputs_dir / "manifest.json"
        if not manifest_path.exists():
            _abort("No outputs/manifest.json found. Run 'reader run' first to produce artifacts.")
        store = ArtifactStore(outputs_dir)
        man = store._read_manifest()
    except ReaderError as e:
        _handle_reader_error(e)

    if all:
        if not man.get("history"):
            console.print(
                Panel.fit(
                    "No artifact history listed in outputs/manifest.json. Run 'reader run' first.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        t = _table("Artifacts • history")
        t.add_column("Label")
        t.add_column("Revisions", justify="right")
        for k, hist in man["history"].items():
            t.add_row(k, str(len(hist)))
    else:
        if not man.get("artifacts"):
            console.print(
                Panel.fit(
                    "No artifacts listed in outputs/manifest.json. Run 'reader run' first.",
                    border_style="warn",
                    box=box.ROUNDED,
                )
            )
            return
        t = _table("Artifacts • latest")
        t.add_column("Label")
        t.add_column("Contract", style="accent")
        t.add_column("Path", style="path")
        for k, entry in man["artifacts"].items():
            t.add_row(k, entry["contract"], str(store.artifacts_dir / entry["step_dir"] / entry["filename"]))
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
        spec = ReaderSpec.load(_infer_job_path(job))
    except ReaderError as e:
        _handle_reader_error(e)
    t = _table("Steps")
    t.add_column("#", justify="right", style="muted")
    t.add_column("id", style="accent")
    t.add_column("uses")
    for i, s in enumerate(spec.steps, 1):
        t.add_row(str(i), s.id, s.uses)
    console.print(Panel(t, border_style="accent", box=box.ROUNDED, subtitle=f"[muted]{len(spec.steps)} total[/muted]"))


@app.command(
    help="List plugins by category."
)
def plugins(
    category: str | None = typer.Option(
        None,
        "--category",
        metavar="NAME",
        help="Filter by category: ingest | merge | transform | plot | export | validator",
    ),
):
    try:
        reg = load_entry_points()
    except ReaderError as e:
        _handle_reader_error(e)
    cats = reg.categories()
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


# --------------------------- run one step ---------------------------


def _resolve_step_id(spec: ReaderSpec, which: str) -> str:
    """
    Accept either a step id or a 1-based index string; return the step id.
    """
    which_str = str(which).strip()
    if which_str.isdigit():
        idx = int(which_str)
        if idx < 1 or idx > len(spec.steps):
            raise typer.BadParameter(
                f"Step index out of range: {idx} (valid: 1..{len(spec.steps)}). "
                "Tip: use 'reader steps <config>' to list ids."
            )
        return spec.steps[idx - 1].id
    # by id
    try:
        _ = next(s for s in spec.steps if s.id == which_str)
    except StopIteration:
        options = ", ".join(s.id for s in spec.steps[:12])
        raise typer.BadParameter(
            f"Unknown step id '{which_str}'. Tip: use 'reader steps' to list ids "
            f"(first few: {options}{' …' if len(spec.steps) > 12 else ''})."
        ) from None
    return which_str


def _run_one_step_cli(which: str, job: str | None, *, dry_run: bool, log_level: str) -> None:
    job_path = _infer_job_path(job)
    spec = ReaderSpec.load(job_path)
    step_id = _resolve_step_id(spec, which)
    _append_journal(job_path, f"reader run-step {which} --config {job_path}")

    # Nice preflight: tell the user if some upstream artifacts are clearly missing
    # (we only check presence; contracts are still asserted by the engine).
    store = ArtifactStore(Path(spec.experiment["outputs"]))
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
        raise typer.Exit(code=1)

    # Delegate to the engine using (resume_from == until == selected step)
    try:
        run_job(
            job_path,
            resume_from=step_id,
            until=step_id,
            dry_run=dry_run,
            log_level=log_level,
            console=console,
            include_pipeline=True,
            include_deliverables=False,
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
            raise typer.Exit(code=1) from None
        raise


def _print_missing_step_hint(command: str) -> None:
    msg = (
        "[error]✗ Missing STEP[/error]\n"
        "Provide a step id or a 1-based index.\n"
        "Examples:\n"
        f"  {command} 1\n"
        f"  {command} ratio_yfp_od600\n"
        f"  {command} 3 --config path/to/config.yaml\n"
        "[muted]Tip: use 'reader steps <config>' to list ids.[/muted]"
    )
    console.print(Panel.fit(msg, border_style="error", box=box.ROUNDED))


@app.command(
    name="run-step",
    help=(
        "Run a single step using existing artifacts (no prior steps)."
    ),
)
def run_step(
    which: str | None = typer.Argument(
        None,
        metavar="STEP",
        help="Step id (e.g. 'ingest') or 1-based index (e.g. '1').",
    ),
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
        "INFO", "--log-level", metavar="LEVEL", help="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL."
    ),
):
    if not which:
        _print_missing_step_hint("reader run-step")
        raise typer.Exit(code=2)
    try:
        _run_one_step_cli(which, config, dry_run=dry_run, log_level=log_level)
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(name="step", help="[alias] Same as 'run-step'. Run exactly one step by ID or index.")
def step_alias(
    which: str | None = typer.Argument(None, metavar="STEP"),
    config: str | None = typer.Option(None, "--config", "-c", metavar="CONFIG"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    log_level: str = typer.Option("INFO", "--log-level", metavar="LEVEL"),
):
    if not which:
        _print_missing_step_hint("reader step")
        raise typer.Exit(code=2)
    try:
        _run_one_step_cli(which, config, dry_run=dry_run, log_level=log_level)
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(help="Show a quick guided walkthrough.")
def demo():
    steps = [
        ("1", "Find experiments", "reader ls"),
        ("2", "List presets", "reader presets"),
        ("3", "Explain plan", "reader explain 1"),
        ("4", "Validate config", "reader validate 1"),
        ("5", "Check file inputs", "reader check-inputs 1"),
        ("6", "Run pipeline + deliverables", "reader run 1"),
        ("7", "Deliverables only (plots/exports)", "reader deliverables 1"),
        ("8", "List deliverable steps", "reader deliverables --list 1"),
        ("9", "Notebook scaffold (marimo)", "reader explore 1"),
        ("10", "See artifacts", "reader artifacts 1"),
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
