from __future__ import annotations

import json
import subprocess  # noqa: F401
import sys  # noqa: F401
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.traceback import install as rich_tracebacks

from reader.errors import ReaderError
from reader.protocols.model import ProtocolBindingValueRef

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
        "Start with 'uv run reader demo', 'uv run reader ls', or 'uv run reader protocols'."
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


def checkmark(cond: bool) -> str:
    return "[ok]✓[/ok]" if cond else "[muted]—[/muted]"


def table(title: str) -> Table:
    return Table(
        title=f"[title]{title}[/title]",
        title_justify="left",
        header_style="bold",
        box=box.ROUNDED,
        expand=True,
        show_lines=False,
        show_edge=True,
    )


def abort(msg: str, *, code: int = 1) -> None:
    console.print(Panel.fit(f"[error]✗ {msg}[/error]", border_style="error", box=box.ROUNDED))
    raise typer.Exit(code=code)


def handle_reader_error(err: ReaderError) -> None:
    abort(str(err))


def normalize_output_format(
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


def normalize_flag(value: bool | object, *, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def normalize_status_filter(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    allowed = {"ok", "config_error"}
    if normalized not in allowed:
        raise typer.BadParameter(f"status must be one of: {', '.join(sorted(allowed))}")
    return normalized


def normalize_lifecycle_filter(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    allowed = {"active", "draft", "template"}
    if normalized not in allowed:
        raise typer.BadParameter(f"lifecycle must be one of: {', '.join(sorted(allowed))}")
    return normalized


def json_friendly(value):
    if isinstance(value, ProtocolBindingValueRef):
        payload = {"binding_value": value.key}
        if value.has_default:
            payload["default"] = json_friendly(value.default)
        return payload
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_friendly(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_friendly(item) for item in value]
    if isinstance(value, list):
        return [json_friendly(item) for item in value]
    return value


def emit_json(payload: object) -> None:
    typer.echo(json.dumps(json_friendly(payload), indent=2, sort_keys=True))
