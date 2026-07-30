from __future__ import annotations

import subprocess  # noqa: F401
import sys  # noqa: F401
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.traceback import install as rich_tracebacks

from reader._version import package_version
from reader.errors import ReaderError

from .automation import (
    emit_document,
    error_envelope,
    json_requested,
    reader_error_details,
    success_envelope,
)
from .pagination import Page, PageRequestError, page_collection

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
        "Discover assays and experiments, inspect workflow plans, validate YAML, "
        "run pipelines, and produce plots, exports, or notebooks. "
        "Start with 'reader demo', 'reader ls', or 'reader protocols'."
    ),
)
console = Console(theme=THEME)
rich_tracebacks(show_locals=False)

JOB_INDEX_SCOPE_NOTE = "resolved against the nearest experiments/ root from the current working directory"
JOB_ARG_HELP = (
    f"Path to config.yaml • experiment directory • or numeric index from the default 'reader ls' inventory, "
    f"{JOB_INDEX_SCOPE_NOTE}."
)
JOB_ARG_HELP_WITH_DEFAULT = f"{JOB_ARG_HELP[:-1]} (defaults to nearest ./config.yaml)"
JOB_ARG_HELP_SHORT = (
    f"Experiment config path, directory, or index from the default 'reader ls' inventory, {JOB_INDEX_SCOPE_NOTE}."
)

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


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", is_eager=True, help="Show the installed Reader version."),
) -> None:
    """Show help when no command is provided."""
    if version:
        typer.echo(package_version())
        raise typer.Exit()
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
    if json_requested():
        emit_document(
            error_envelope(
                code="command_rejected",
                field="command",
                reason=msg,
                remediation="Correct the reported command state, then retry.",
                retryable=False,
            )
        )
        raise typer.Exit(code=code)
    console.print(Panel.fit(f"[error]✗ {msg}[/error]", border_style="error", box=box.ROUNDED))
    raise typer.Exit(code=code)


def handle_reader_error(err: ReaderError) -> None:
    if json_requested():
        emit_document(error_envelope(**reader_error_details(err)))
        raise typer.Exit(code=1)
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


def normalize_semantic_section(
    value: str | None,
    *,
    allowed: Sequence[str],
    format: str,
) -> str | None:
    if not isinstance(value, str):
        return None
    section = value.strip().lower()
    if format != "json":
        raise typer.BadParameter("--section requires --format json", param_hint="--section")
    if section not in allowed:
        raise typer.BadParameter(
            f"section must be one of: {', '.join(allowed)}",
            param_hint="--section",
        )
    return section


def normalize_paging_options(
    limit: int | object,
    continuation: str | object,
) -> tuple[int | None, str | None]:
    normalized_limit = limit if type(limit) is int else None
    normalized_continuation = continuation.strip() if isinstance(continuation, str) else None
    return normalized_limit, normalized_continuation or None


def page_json_collection[T](
    items: Sequence[T],
    *,
    key: Callable[[T], str],
    surface: str,
    selection: Mapping[str, object],
    limit: int | None,
    continuation: str | None,
) -> Page[T]:
    try:
        return page_collection(
            items,
            key=key,
            surface=surface,
            selection=selection,
            limit=limit,
            continuation=continuation,
        )
    except PageRequestError as exc:
        option = f"--{exc.field.replace('_', '-')}"
        raise typer.BadParameter(str(exc), param_hint=option) from exc


def require_json_paging(*, format: str, limit: int | None, continuation: str | None) -> None:
    if format == "json" or (limit is None and continuation is None):
        return
    option = "--continuation" if continuation is not None else "--limit"
    raise typer.BadParameter(f"{option} requires --format json", param_hint=option)


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


def _is_protocol_binding_value_ref(value: object) -> bool:
    value_type = type(value)
    # Keep plain CLI imports off the protocol bootstrap path.
    return value_type.__module__ == "reader.protocols.model" and value_type.__name__ == "ProtocolBindingValueRef"


def json_friendly(value):
    if _is_protocol_binding_value_ref(value):
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


def emit_json(
    payload: object,
    *,
    projection: str = "full",
    truncated: bool = False,
    continuation: str | None = None,
) -> None:
    emit_document(
        success_envelope(
            json_friendly(payload),
            projection=projection,
            truncated=truncated,
            continuation=continuation,
        )
    )


def emit_json_error(
    *,
    code: str,
    field: str,
    reason: str,
    remediation: str,
    retryable: bool = False,
) -> None:
    emit_document(
        error_envelope(
            code=code,
            field=field,
            reason=reason,
            remediation=remediation,
            retryable=retryable,
        )
    )
