"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/cli.py

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

from reader.core.artifacts import ArtifactStore
from reader.core.config_model import ReaderSpec
from reader.core.engine import explain as explain_job
from reader.core.engine import run_job, run_spec
from reader.core.engine import validate as validate_job
from reader.core.errors import ArtifactError, ConfigError, ReaderError
from reader.core.notebooks import list_notebook_presets, normalize_notebook_preset, write_experiment_notebook
from reader.core.presets import describe_preset, list_presets
from reader.core.registry import load_entry_points
from reader.core.specs import materialize_specs, resolve_export_specs, resolve_plot_specs

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
        "Run pipelines to generate artifacts, then render plots, exports, or notebooks. "
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
    help="Filter plot ids when using --preset notebook/eda (repeatable).",
)
NOTEBOOK_PLOT_EXCLUDE_OPTION = typer.Option(
    None,
    "--exclude",
    help="Exclude plot ids when using --preset notebook/eda (repeatable).",
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


@app.command(help="List presets or describe one.")
def presets(
    name: str | None = typer.Argument(
        None,
        metavar="[NAME]",
        help="Optional preset name to describe (e.g., plate_reader/synergy_h1).",
    ),
    category: str | None = typer.Option(
        None,
        "--category",
        metavar="NAME",
        help="Filter presets by category: pipeline | plot | export | notebook.",
    ),
):
    try:
        if name:
            if category == "notebook":
                raise typer.BadParameter("Use 'reader notebook --list-presets' for notebook templates.")
            info = describe_preset(name)
            table = _table(f"Preset: {info['name']}")
            table.add_column("#", justify="right", style="muted")
            table.add_column("Step ID", style="accent")
            table.add_column("Plugin")
            for i, step in enumerate(info["steps"], 1):
                table.add_row(str(i), step.get("id", ""), step.get("uses", ""))
            console.print(Panel(table, border_style="accent", box=box.ROUNDED, subtitle=info["description"]))
            return

        if category == "notebook":
            table = _table("Notebook presets")
            table.add_column("Name", style="accent")
            table.add_column("Description")
            for preset_name, desc in list_notebook_presets():
                table.add_row(preset_name, desc)
            console.print(Panel(table, border_style="accent", box=box.ROUNDED))
            return
        table = _table("Presets")
        table.add_column("Name", style="accent")
        table.add_column("Description")
        for preset, desc in list_presets(category=category):
            table.add_row(preset, desc)
        console.print(Panel(table, border_style="accent", box=box.ROUNDED))
    except ConfigError as e:
        raise typer.BadParameter(str(e)) from e


def _config_has_notebooks_override(path: Path) -> bool:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    paths = data.get("paths")
    return isinstance(paths, dict) and "notebooks" in paths


def _has_sfxi_step(spec: ReaderSpec) -> bool:
    return any(str(getattr(step, "uses", "")) == "transform/sfxi" for step in spec.pipeline.steps)


def _has_sfxi_artifacts(outputs_dir: Path) -> bool:
    manifest_path = outputs_dir / "manifests" / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return False
    for entry in artifacts.values():
        if not isinstance(entry, dict):
            continue
        contract = str(entry.get("contract", ""))
        if contract == "tidy+map.v1" or contract.startswith("sfxi.vec8."):
            return True
    return False


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
    preset: str | None,
    list_presets: bool,
    overwrite: bool,
    new: bool,
    refresh: bool,
    mode: str,
    plot_only: list[str] | None,
    plot_exclude: list[str] | None,
) -> None:
    try:
        if list_presets:
            table = _table("Notebook presets")
            table.add_column("Name", style="accent")
            table.add_column("Description")
            for preset_name, desc in list_notebook_presets():
                table.add_row(preset_name, desc)
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
        spec = ReaderSpec.load(job_path)
        selected_preset = preset
        if not selected_preset and getattr(spec, "notebook", None) and getattr(spec.notebook, "preset", None):
            selected_preset = spec.notebook.preset
        if not selected_preset:
            selected_preset = "notebook/eda" if resolve_plot_specs(spec) else "notebook/basic"
        selected_preset = normalize_notebook_preset(selected_preset)
        if (plot_only or plot_exclude) and selected_preset != "notebook/eda":
            raise typer.BadParameter("--only/--exclude are only supported with --preset notebook/eda.")
        outputs_dir = Path(spec.paths.outputs)
        if selected_preset == "notebook/sfxi_eda" and not (
            _has_sfxi_step(spec) or _has_sfxi_artifacts(outputs_dir)
        ):
            raise typer.BadParameter(
                "Preset notebook/sfxi_eda requires a transform/sfxi step in config.yaml "
                "or existing SFXI artifacts (tidy+map or vec8). "
                "See docs/sfxi_vec8_in_reader.md."
            )
        notebooks_cfg = spec.paths.notebooks
        nb_dir = outputs_dir if notebooks_cfg in ("", ".", "./") else outputs_dir / str(notebooks_cfg)
        if not _config_has_notebooks_override(job_path):
            legacy_dir = exp_dir / "notebooks"
            if legacy_dir.exists() and not nb_dir.exists():
                nb_dir = legacy_dir
                console.print(
                    Panel.fit(
                        f"Legacy notebooks/ detected; using [path]{legacy_dir}[/path].",
                        border_style="warn",
                        box=box.ROUNDED,
                    )
                )
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
        if selected_preset == "notebook/eda":
            plot_specs = resolve_plot_specs(spec)
            selected = _select_steps(plot_specs, only=plot_only or [], exclude=plot_exclude or [], kind="plot spec")
            plot_specs_payload = [_spec_to_dict(s) for s in selected]
        target, created = write_experiment_notebook(
            target,
            preset=selected_preset,
            overwrite=overwrite,
            plot_specs=plot_specs_payload,
        )
        if created:
            if existed and overwrite:
                status = f"✓ Notebook overwritten: [path]{target}[/path]\n[muted]preset[/muted]: {selected_preset}"
            else:
                status = f"✓ Notebook created: [path]{target}[/path]\n[muted]preset[/muted]: {selected_preset}"
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
    preset: str | None = typer.Option(
        None,
        "--preset",
        help="Notebook preset (defaults to config notebook.preset, else auto-picks).",
    ),
    list_presets: bool = typer.Option(
        False,
        "--list-presets",
        help="List notebook presets and exit.",
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
    mode: str = NOTEBOOK_MODE_OPTION,
    only: list[str] = NOTEBOOK_PLOT_ONLY_OPTION,
    exclude: list[str] = NOTEBOOK_PLOT_EXCLUDE_OPTION,
):
    _scaffold_notebook(
        job=job,
        name=name,
        preset=preset,
        list_presets=list_presets,
        overwrite=overwrite,
        new=new,
        refresh=refresh,
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


def _find_jobs(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.glob("**/config.yaml"):
        out.append(p.resolve())
    return sorted(out)


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


def _require_plot_artifacts(spec: ReaderSpec, job_path: Path) -> None:
    outputs_dir = Path(spec.paths.outputs)
    manifest_path = outputs_dir / "manifests" / "manifest.json"
    if not manifest_path.exists():
        raise ArtifactError(
            f"No outputs/manifests/manifest.json found. Run 'reader run {job_path}' first to generate artifacts."
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactError(
            f"Could not read artifacts manifest at {manifest_path}. Run 'reader run {job_path}' first."
        ) from exc
    artifacts = payload.get("artifacts") or {}
    if not artifacts:
        raise ArtifactError(
            f"No artifacts listed in outputs/manifests/manifest.json. Run 'reader run {job_path}' first."
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
            spec = ReaderSpec.load(p)
            man = Path(spec.paths.outputs) / "manifests" / "manifest.json"
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
    help="Validate config, plugin params, reads wiring, and input files."
)
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
        spec = ReaderSpec.load(job_path)
        exp_root = Path(spec.experiment.root or job_path.parent)
        validate_job(spec, console=console, check_files=not no_files, exp_root=exp_root)
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
    materialized = materialize_specs(spec)
    payload.setdefault("plots", {})
    payload.setdefault("exports", {})
    payload["plots"]["specs"] = materialized["plots"]
    payload["exports"]["specs"] = materialized["exports"]
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if fmt == "yaml":
        typer.echo(yaml.safe_dump(payload, sort_keys=False))
        return
    raise typer.BadParameter("format must be 'yaml' or 'json'")




@app.command(help="Run pipeline to generate artifacts.")
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
        spec = ReaderSpec.load(job_path)
    except ReaderError as e:
        _handle_reader_error(e)

    if only:
        _resolve_pipeline_step_id(spec, only)
        parts += ["--only", only]
        if dry_run:
            parts += ["--dry-run"]
        if log_level and log_level != "INFO":
            parts += ["--log-level", log_level]
        if compact:
            parts += ["--compact"]
        _append_journal(job_path, " ".join(parts))
        try:
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
            )
        except ReaderError as e:
            _handle_reader_error(e)
        return

    if from_step:
        _resolve_pipeline_step_id(spec, from_step)
        parts += ["--from", from_step]
    if until:
        _resolve_pipeline_step_id(spec, until)
        parts += ["--until", until]
    if dry_run:
        parts += ["--dry-run"]
    if log_level and log_level != "INFO":
        parts += ["--log-level", log_level]
    if compact:
        parts += ["--compact"]
    _append_journal(job_path, " ".join(parts))
    try:
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
    spec = ReaderSpec.load(job_path)
    if not list_only:
        _require_plot_artifacts(spec, job_path)
    plot_specs = resolve_plot_specs(spec)
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
        t.add_column("uses")
        for i, s in enumerate(selected, 1):
            t.add_row(str(i), s.id, s.uses)
        console.print(
            Panel(
                t,
                border_style="accent",
                box=box.ROUNDED,
                subtitle=f"[muted]{len(selected)} total[/muted]",
            )
        )
        return
    input_overrides = _parse_input_overrides(inputs or [])
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
    selected = _apply_step_overrides(selected, input_overrides=input_overrides, set_overrides=set_overrides)
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
        spec,
        dry_run=dry_run,
        log_level=log_level,
        console=console,
        include_pipeline=False,
        include_plots=True,
        include_exports=False,
        plot_specs=selected,
    )
    if not dry_run:
        job_hint = _format_job_arg(job_hint)
        outputs_dir = Path(spec.paths.outputs)
        plots_cfg = spec.paths.plots
        plots_dir = outputs_dir if plots_cfg in ("", ".", "./") else outputs_dir / str(plots_cfg)

        def _cmd(base: str, tail: str = "") -> str:
            return f"{base} {job_hint}{tail}" if job_hint else f"{base}{tail}"

        lines = [f"Plots saved in [path]{plots_dir}[/path]", "", "Next steps:"]
        lines.append(f"  {_cmd('reader notebook')}")
        if resolve_export_specs(spec):
            lines.append(f"  {_cmd('reader export')}")
        console.print(Panel.fit("\n".join(lines), border_style="green", box=box.ROUNDED))


@app.command(help="Save plot files from plot specs using existing artifacts.")
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


@app.command(help="Run export specs using existing artifacts.")
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
        spec = ReaderSpec.load(job_path)
    except ReaderError as e:
        _handle_reader_error(e)
    export_specs = resolve_export_specs(spec)
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
        t.add_column("uses")
        for i, s in enumerate(selected, 1):
            t.add_row(str(i), s.id, s.uses)
        console.print(
            Panel(
                t,
                border_style="accent",
                box=box.ROUNDED,
                subtitle=f"[muted]{len(selected)} total[/muted]",
            )
        )
        return
    input_overrides = _parse_input_overrides(inputs or [])
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
    selected = _apply_step_overrides(selected, input_overrides=input_overrides, set_overrides=set_overrides)
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
    try:
        run_spec(
            spec,
            dry_run=dry_run,
            log_level=log_level,
            console=console,
            include_pipeline=False,
            include_plots=False,
            include_exports=True,
            export_specs=selected,
        )
        if not dry_run:
            job_hint = _format_job_arg(job)
            outputs_dir = Path(spec.paths.outputs)
            exports_cfg = spec.paths.exports
            exports_dir = outputs_dir if exports_cfg in ("", ".", "./") else outputs_dir / str(exports_cfg)

            def _cmd(base: str, tail: str = "") -> str:
                return f"{base} {job_hint}{tail}" if job_hint else f"{base}{tail}"

            lines = [f"Exports saved in [path]{exports_dir}[/path]", "", "Next steps:"]
            if resolve_plot_specs(spec):
                lines.append(f"  {_cmd('reader plot')}")
            lines.append(f"  {_cmd('reader notebook')}")
            console.print(Panel.fit("\n".join(lines), border_style="green", box=box.ROUNDED))
    except ReaderError as e:
        _handle_reader_error(e)


@app.command(
    help="List emitted artifacts from outputs/manifests/manifest.json."
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
        outputs_dir = Path(spec.paths.outputs)
        manifest_path = outputs_dir / "manifests" / "manifest.json"
        if not manifest_path.exists():
            _abort("No outputs/manifests/manifest.json found. Run 'reader run' first to produce artifacts.")
        store = ArtifactStore(outputs_dir, plots_subdir=spec.paths.plots, exports_subdir=spec.paths.exports)
        man = store._read_manifest()
    except ReaderError as e:
        _handle_reader_error(e)

    if all:
        if not man.get("history"):
            console.print(
                Panel.fit(
                    "No artifact history listed in outputs/manifests/manifest.json. Run 'reader run' first.",
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
                    "No artifacts listed in outputs/manifests/manifest.json. Run 'reader run' first.",
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
    for i, s in enumerate(spec.pipeline.steps, 1):
        t.add_row(str(i), s.id, s.uses)
    console.print(
        Panel(
            t,
            border_style="accent",
            box=box.ROUNDED,
            subtitle=f"[muted]{len(spec.pipeline.steps)} total[/muted]",
        )
    )


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


# --------------------------- helpers ---------------------------


def _resolve_pipeline_step_id(spec: ReaderSpec, which: str) -> str:
    which_str = str(which).strip()
    if any(s.id == which_str for s in spec.pipeline.steps):
        return which_str
    options = ", ".join(s.id for s in spec.pipeline.steps[:12])
    raise typer.BadParameter(
        f"Unknown pipeline step id '{which_str}'. Tip: use 'reader steps' to list ids "
        f"(first few: {options}{' …' if len(spec.pipeline.steps) > 12 else ''})."
    )


def _spec_to_dict(spec_obj) -> dict:
    return {
        "id": spec_obj.id,
        "uses": spec_obj.uses,
        "reads": dict(spec_obj.reads or {}),
        "with": dict(spec_obj.with_ or {}),
        "writes": dict(spec_obj.writes or {}),
    }


def _select_steps(steps, *, only: list[str], exclude: list[str], kind: str):
    ids = [s.id for s in steps]
    if only:
        missing = sorted(set(only) - set(ids))
        if missing:
            raise typer.BadParameter(f"Unknown {kind} id(s): {missing}. Use --list to see valid ids.")
        selected = [s for s in steps if s.id in set(only)]
    else:
        selected = list(steps)
    if exclude:
        missing = sorted(set(exclude) - set(ids))
        if missing:
            raise typer.BadParameter(f"Unknown {kind} id(s): {missing}. Use --list to see valid ids.")
        selected = [s for s in selected if s.id not in set(exclude)]
    return selected


def _parse_input_overrides(raw_inputs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
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
        overrides[key] = value
    return overrides


def _set_nested(mapping: dict, keys: list[str], value) -> None:
    cur = mapping
    for k in keys[:-1]:
        if k not in cur:
            cur[k] = {}
        if not isinstance(cur[k], dict):
            raise typer.BadParameter(f"--set path invalid (non-mapping at '{k}')")
        cur = cur[k]
    cur[keys[-1]] = value


def _apply_step_overrides(steps, *, input_overrides: dict[str, str], set_overrides: list[tuple[str, object]]):
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
            root = parts[0]
            if root not in {"reads", "with", "writes"}:
                raise typer.BadParameter("--set path must start with reads., with., or writes.")
            if root in {"reads", "writes"}:
                if len(parts) != 2:
                    raise typer.BadParameter(f"--set {root} expects a single key (e.g., {root}.foo=bar)")
                target = reads if root == "reads" else writes
                target[parts[1]] = value
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
            updated.append(
                step.__class__(
                    id=step.id,
                    uses=step.uses,
                    reads=reads,
                    with_=with_block,
                    writes=writes,
                    preset_meta=getattr(step, "preset_meta", None),
                )
            )
    return updated


@app.command(help="Show a quick guided walkthrough.")
def demo():
    steps = [
        ("1", "Find experiments", "reader ls"),
        ("2", "List presets", "reader presets"),
        ("3", "Explain plan", "reader explain 1"),
        ("4", "Validate config + inputs", "reader validate 1"),
        ("5", "Run pipeline (artifacts)", "reader run 1"),
        ("6", "See artifacts", "reader artifacts 1"),
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
