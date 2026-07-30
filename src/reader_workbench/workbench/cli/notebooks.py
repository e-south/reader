from __future__ import annotations

import os
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel

from reader_workbench.errors import ConfigError, RecordError
from reader_workbench.workbench.paths import resolve_confined_sink_root, resolve_path_within_root

from . import shared
from ._lazy import load as _load
from .helpers import (
    default_notebook_name,
    infer_job_path,
    load_job_models,
    next_available_path,
)
from .shared import (
    NOTEBOOK_MODE_OPTION,
    app,
)


def render_marimo_help(target: Path, *, mode: str, has_fcs: bool) -> None:
    marimo_cmd = f"{shared.sys.executable} -m marimo {mode} {target}"
    shared.console.print(
        Panel.fit(
            "Could not launch marimo automatically.\n\n"
            "Reader does not publish a notebook dependency extra while the released Marimo and "
            "PyMdown constraints remain unresolved. Use a separately managed and audited "
            "environment that provides Marimo, Altair, and DuckDB, then run:\n"
            f"  {marimo_cmd}\n\n"
            f"Notebook: [path]{target}[/path]",
            border_style="warn",
            box=box.ROUNDED,
        )
    )


def render_marimo_routes(*, target: Path, url: str, runtime_root: Path) -> None:
    check_cmd = f"uv run marimo check {target}"
    shared.console.print(
        Panel.fit(
            "Review routes:\n"
            f"  Static check: {check_cmd}\n"
            f"  Browser review: {url}\n"
            "  In-app browser: open the URL in a fresh isolated page.\n\n"
            f"Managed runtime root: [path]{runtime_root}[/path]",
            border_style="accent",
            box=box.ROUNDED,
        )
    )


def _notebook_filename(raw: str) -> str:
    candidate = Path(raw)
    if (
        not raw
        or raw != raw.strip()
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate in {Path("."), Path("..")}
        or candidate.suffix != ".py"
    ):
        raise ConfigError("--name must be a non-empty .py filename with no directory components")
    return raw


def _launch_marimo(
    mode: str,
    target: Path,
    *,
    has_fcs: bool,
    headless: bool = False,
    port: int | None = None,
    repo_root: Path | None = None,
) -> None:
    launch = _load("reader_workbench.workbench.notebooks.launch")
    plan = launch.plan_marimo_launch(
        mode=mode,
        target=target,
        headless=headless,
        preferred_port=port,
        base_env=os.environ.copy(),
        repo_root=repo_root,
    )
    if plan.terminated_sessions:
        shared.console.print(
            Panel.fit(
                f"Pruned {len(plan.terminated_sessions)} existing reader-managed Marimo session(s) "
                "for this experiment before launch.",
                border_style="warn",
                box=box.ROUNDED,
            )
        )
    if plan.reused_session is not None:
        shared.console.print(
            Panel.fit(
                f"Notebook already running: [path]{target}[/path]\n[muted]url[/muted]: {plan.url}",
                border_style="ok",
                box=box.ROUNDED,
            )
        )
        render_marimo_routes(target=target, url=plan.url, runtime_root=plan.runtime_paths.root)
        if not headless:
            launch.open_url(plan.url)
        return
    shared.console.print(
        Panel.fit(
            f"Launching: {' '.join(plan.cmd)}\n[muted]url[/muted]: {plan.url}",
            border_style="accent",
            box=box.ROUNDED,
        )
    )
    render_marimo_routes(target=target, url=plan.url, runtime_root=plan.runtime_paths.root)
    try:
        proc = shared.subprocess.Popen(plan.cmd, env=plan.env)
    except FileNotFoundError:
        render_marimo_help(target, mode=mode, has_fcs=has_fcs)
        raise typer.Exit(code=1) from None
    launch.register_managed_session(
        registry_path=plan.runtime_paths.registry_path,
        pid=proc.pid,
        port=plan.port,
        host=plan.host,
        mode=mode,
        target=target,
        repo_root=plan.repo_root,
    )
    try:
        returncode = proc.wait()
    finally:
        launch.unregister_managed_session(registry_path=plan.runtime_paths.registry_path, pid=proc.pid)
    if returncode != 0:
        render_marimo_help(target, mode=mode, has_fcs=has_fcs)
        raise typer.Exit(code=1)


def _scaffold_notebook(
    *,
    job: str | None,
    name: str | None,
    overwrite: bool,
    new: bool,
    refresh: bool,
    mode: str,
    headless: bool,
    port: int | None,
) -> None:
    try:
        if overwrite and new:
            raise typer.BadParameter("--overwrite cannot be combined with --new.")
        if refresh:
            overwrite = True
        mode_value = (mode or "").strip().lower()
        if mode_value not in {"edit", "run", "none"}:
            raise typer.BadParameter("--mode must be one of: edit, run, none.")
        job_path = infer_job_path(job)
        exp_dir = job_path.parent
        _, decl = load_job_models(job_path)
        layout = decl.experiment_semantics.layout
        outputs_dir = layout.outputs_dir
        notebooks_cfg = layout.notebooks_subdir
        notebook_root = outputs_dir if notebooks_cfg in ("", ".", "./") else outputs_dir / str(notebooks_cfg)
        try:
            nb_dir = resolve_confined_sink_root(notebook_root, root=outputs_dir, label="notebooks")
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        target_name = default_notebook_name() if name is None else _notebook_filename(name)
        target_candidate = nb_dir / target_name
        if target_candidate.is_symlink():
            raise ConfigError(f"Notebook target must not be a symlink: {target_candidate}")
        try:
            target = resolve_path_within_root(target_candidate, root=nb_dir)
        except ValueError as exc:
            raise ConfigError(f"Notebook target must stay within the notebooks sink root: {target_candidate}") from exc
        if new:
            target = next_available_path(target)
        elif overwrite and target.exists():
            confirm = typer.confirm(f"Notebook already exists at {target}. Overwrite?", default=False)
            if not confirm:
                overwrite = False
        has_fcs = any(path.suffix.lower() == ".fcs" for path in exp_dir.rglob("*.fcs"))
        existed = target.exists()
        target, created = _load("reader_workbench.workbench.notebooks").write_experiment_notebook(
            target,
            experiment_root=decl.experiment.root,
            notebooks_root=nb_dir,
            overwrite=overwrite,
        )
        if created:
            if existed and overwrite:
                status = f"✓ Notebook overwritten: [path]{target}[/path]"
            else:
                status = f"✓ Notebook created: [path]{target}[/path]"
            border_style = "ok"
        else:
            action = "opening existing" if mode_value != "none" else "using existing"
            status = f"Notebook already exists: [path]{target}[/path] {action}."
            border_style = "warn"
        shared.console.print(Panel.fit(status, border_style=border_style, box=box.ROUNDED))
        if mode_value == "none":
            shared.console.print(str(target))
            return
        _launch_marimo(mode_value, target, has_fcs=has_fcs, headless=headless, port=port)
    except (ConfigError, RecordError) as err:
        raise typer.BadParameter(str(err)) from err


@app.command(help="Scaffold an interactive marimo notebook and open it.")
def notebook(
    job: str | None = typer.Argument(
        None,
        metavar="CONFIG|DIR|INDEX",
        help=shared.JOB_ARG_HELP_SHORT,
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="Notebook filename (created under outputs/notebooks). Defaults to EDA_YYYYMMDD.py.",
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
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Launch without opening a browser. Reader prints a loopback URL suitable for in-app review.",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help="Preferred loopback port. Defaults to a reader-managed clean port starting at 2718.",
    ),
    mode: str = NOTEBOOK_MODE_OPTION,
):
    _scaffold_notebook(
        job=job,
        name=name,
        overwrite=overwrite,
        new=new,
        refresh=refresh,
        headless=headless,
        port=port,
        mode=mode,
    )
