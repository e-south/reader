"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import re  # used for friendly error parsing in run-step
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
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
from reader.core.errors import ExecutionError
from reader.core.registry import load_entry_points

THEME = Theme({
    "title":  "bold cyan",
    "accent": "cyan",
    "ok":     "bold green",
    "warn":   "bold yellow",
    "error":  "bold red",
    "muted":  "dim",
    "path":   "magenta",
})

app = typer.Typer(
    add_completion=False,
    help=(
        "reader — instrument-agnostic pipeline runner.\n\n"
        "Run data pipelines described by a config.yaml. Pipelines are sequences of steps "
        "(ingest → merge → transform → plot) implemented by plugins. "
        "Each experiment is a directory with a config.yaml; results are written under outputs/."
    ),
)
console = Console(theme=THEME)
rich_tracebacks(show_locals=False)


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
        cand = (base / "experiments")
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    return (start / "experiments").resolve()

def _infer_job_path(job: Optional[str]) -> Path:
    """
    Resolve CONFIG argument with convenient defaults:
      - If provided and points to a directory, assume <dir>/config.yaml
      - If provided and points to a file, use it as-is
      - If omitted, search for the nearest 'config.yaml' in CWD or any parent
    """
    if job:
        p = Path(job)
        if p.is_dir():
            candidate = p / "config.yaml"
            if candidate.exists():
                return candidate.resolve()
            # allow explicit dir even if it doesn't have config.yaml; fall back to p
        return p.resolve()

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
        "JOURNAL.md"
        if (exp_dir / "JOURNAL.md").exists() or not (exp_dir / "journal.md").exists()
        else "journal.md"
    )
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = "" if journal.exists() else "# Experiment Journal\n\n"
    entry = f"### {ts}\n\n```\n{command_line}\n```\n\n"
    journal.write_text(
        header + (journal.read_text(encoding="utf-8") if journal.exists() else "") + entry, encoding="utf-8"
    )


@app.command(
    help=(
        "List experiments found under --root (default: ./experiments). "
        "Shows the experiment *names* (directory names). The absolute experiments root "
        "is shown once in the panel subtitle."
    )
)
def ls(
    root: str = typer.Option(
        "./experiments",
        "--root",
        metavar="DIR",
        help="Directory to search recursively for **/config.yaml.",
    )
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
    help=(
        "Render the execution plan for a config.yaml: step order, plugins, and input/output contracts. "
        "This does not run the pipeline."
    )
)
def explain(
    job: Optional[str] = typer.Argument(
        None, metavar="[CONFIG]", help="Path to config.yaml or an experiment directory (defaults to nearest ./config.yaml)"
    )
):
    job_path = _infer_job_path(job)
    _append_journal(job_path, f"reader explain {job_path}")
    spec = ReaderSpec.load(job_path)
    explain_job(spec, console=console)


@app.command(
    help=(
        "Validate a config.yaml: schema, plugin availability, and each plugin's configuration. "
        "No steps are executed."
    )
)
def validate(
    job: Optional[str] = typer.Argument(
        None, metavar="[CONFIG]", help="Path to config.yaml or an experiment directory (defaults to nearest ./config.yaml)"
    )
):
    job_path = _infer_job_path(job)
    _append_journal(job_path, f"reader validate {job_path}")
    spec = ReaderSpec.load(job_path)
    validate_job(spec, console=console)


@app.command(
    help=(
        "Execute a pipeline described by config.yaml. Use --dry-run to print the plan without running. "
        "Use --resume-from/--until to run a slice of steps. Logs go to outputs/reader.log."
    )
)
def run(
    job: Optional[str] = typer.Argument(
        None, metavar="[CONFIG]", help="Path to config.yaml or an experiment directory (defaults to nearest ./config.yaml)"
    ),
    resume_from: Optional[str] = typer.Option(
        None,
        "--resume-from",
        metavar="STEP_ID",
        help="Start from this step (inclusive). Use an exact id as declared in the config.",
    ),
    until: Optional[str] = typer.Option(
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
        help="Logging level: DEBUG | INFO | WARNING | ERROR (default: INFO).",
    ),
):
    job_path = _infer_job_path(job)
    parts = ["reader run", str(job_path)]
    if resume_from:
        parts += ["--resume-from", resume_from]
    if until:
        parts += ["--until", until]
    if dry_run:
        parts += ["--dry-run"]
    if log_level and log_level != "INFO":
        parts += ["--log-level", log_level]
    _append_journal(job_path, " ".join(parts))
    # Pass our themed console down so engine can render pretty output & progress
    run_job(job_path, resume_from=resume_from, until=until, dry_run=dry_run, log_level=log_level, console=console)


@app.command(
    help=(
        "List materialized artifacts for a run (from outputs/manifest.json). "
        "By default shows the latest revision per artifact. Use --all for history counts."
    )
)
def artifacts(
    job: Optional[str] = typer.Argument(
        None, metavar="[CONFIG]", help="Path to config.yaml or an experiment directory (defaults to nearest ./config.yaml)"
    ),
    all: bool = typer.Option(False, "--all", help="Show revision history counts instead of latest entries."),
):
    spec = ReaderSpec.load(_infer_job_path(job))
    store = ArtifactStore(Path(spec.experiment["outputs"]))
    man = store._read_manifest()
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


@app.command(help="Show the steps declared in a config.yaml (index, id, plugin key).")
def steps(
    job: Optional[str] = typer.Argument(
        None, metavar="[CONFIG]", help="Path to config.yaml or an experiment directory (defaults to nearest ./config.yaml)"
    )
):
    spec = ReaderSpec.load(_infer_job_path(job))
    t = _table("Steps")
    t.add_column("#", justify="right", style="muted")
    t.add_column("id", style="accent")
    t.add_column("uses")
    for i, s in enumerate(spec.steps, 1):
        t.add_row(str(i), s.id, s.uses)
    console.print(Panel(t, border_style="accent", box=box.ROUNDED, subtitle=f"[muted]{len(spec.steps)} total[/muted]"))


@app.command(
    help=(
        "Show discovered plugins by category. Built-ins are scanned from reader.plugins.*; "
        "external plugins are loaded via entry points. Use --category to filter."
    )
)
def plugins(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        metavar="NAME",
        help="Filter by category: ingest | merge | transform | plot",
    )
):
    reg = load_entry_points()
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
            raise typer.BadParameter(f"Step index out of range: {idx} (valid: 1..{len(spec.steps)})")
        return spec.steps[idx - 1].id
    # by id
    try:
        _ = next(s for s in spec.steps if s.id == which_str)
    except StopIteration:
        options = ", ".join(s.id for s in spec.steps[:12])
        raise typer.BadParameter(
            f"Unknown step id '{which_str}'. Tip: use 'reader steps' to list ids "
            f"(first few: {options}{' …' if len(spec.steps)>12 else ''})."
        )
    return which_str


def _run_one_step_cli(which: str, job: Optional[str], *, dry_run: bool, log_level: str) -> None:
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
        if isinstance(target, str) and not target.startswith("file:"):
            if store.latest(target) is None:
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
            f"   [accent]reader run --until {missing[0].split('/',1)[0]}[/accent]\n"
            "[muted]Then re-run just this step with:[/muted]\n"
            f"   [accent]reader run-step {step_id}[/accent]"
        )
        console.print(Panel.fit(msg, border_style="error", box=box.ROUNDED))
        raise typer.Exit(code=1)

    # Delegate to the engine using (resume_from == until == selected step)
    try:
        run_job(job_path, resume_from=step_id, until=step_id, dry_run=dry_run, log_level=log_level, console=console)
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
            raise typer.Exit(code=1)
        raise


@app.command(
    name="run-step",
    help=(
        "Execute exactly one step by ID or 1-based index, using existing artifacts for inputs. "
        "Does not run prior steps. Use --dry-run to print the plan slice without executing."
    ),
)
def run_step(
    which: str = typer.Argument(..., metavar="STEP", help="Step id (e.g. 'ingest') or 1-based index (e.g. '1')."),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        metavar="CONFIG",
        help="Path to config.yaml or an experiment dir (defaults to nearest ./config.yaml).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only: validate and print the plan slice without executing."),
    log_level: str = typer.Option("INFO", "--log-level", metavar="LEVEL", help="Logging level: DEBUG | INFO | WARNING | ERROR."),
):
    _run_one_step_cli(which, config, dry_run=dry_run, log_level=log_level)


@app.command(name="step", help="[alias] Same as 'run-step'. Run exactly one step by ID or index.")
def step_alias(
    which: str = typer.Argument(..., metavar="STEP"),
    config: Optional[str] = typer.Option(None, "--config", "-c", metavar="CONFIG"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    log_level: str = typer.Option("INFO", "--log-level", metavar="LEVEL"),
):
    _run_one_step_cli(which, config, dry_run=dry_run, log_level=log_level)
