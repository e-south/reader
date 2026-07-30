from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Protocol, cast

import click

SCHEMA = "reader.cli/v1"
_DEFAULT_META = {"projection": "full", "truncated": False, "continuation": None}


@dataclass(frozen=True)
class AutomationRequest:
    json_requested: bool
    command: str


_REQUEST: ContextVar[AutomationRequest | None] = ContextVar("reader_cli_automation_request", default=None)
_DOCUMENT_EMITTED: ContextVar[bool] = ContextVar("reader_cli_json_document_emitted", default=False)


class ClickError(Protocol):
    """The stable behavior Reader needs from a Click-compatible usage error."""

    exit_code: int

    def format_message(self) -> str: ...

    def show(self) -> None: ...


def request_from_argv(argv: Sequence[str]) -> AutomationRequest:
    args = list(argv)
    json_requested = any(
        (item == "--format" and index + 1 < len(args) and args[index + 1].strip().lower() == "json")
        or item.strip().lower() == "--format=json"
        for index, item in enumerate(args)
    )
    positional = [item for item in args if item and not item.startswith("-")]
    command = positional[0] if positional else "reader"
    if command == "response-window" and len(positional) > 1:
        command = f"{command} {positional[1]}"
    return AutomationRequest(json_requested=json_requested, command=command)


def begin_request(argv: Sequence[str]) -> tuple[Token, Token]:
    request_token = _REQUEST.set(request_from_argv(argv))
    emitted_token = _DOCUMENT_EMITTED.set(False)
    return request_token, emitted_token


def end_request(tokens: tuple[Token, Token]) -> None:
    request_token, emitted_token = tokens
    _DOCUMENT_EMITTED.reset(emitted_token)
    _REQUEST.reset(request_token)


def _click_context() -> click.Context | None:
    try:
        return click.get_current_context(silent=True)
    except RuntimeError:
        return None


def json_requested() -> bool:
    request = _REQUEST.get()
    if request is not None:
        return request.json_requested
    context = _click_context()
    while context is not None:
        value = context.params.get("format")
        if isinstance(value, str) and value.strip().lower() == "json":
            return True
        context = context.parent
    return request_from_argv(sys.argv[1:]).json_requested


def command_name() -> str:
    context = _click_context()
    if context is not None:
        parts = context.command_path.split()
        if len(parts) > 1:
            return " ".join(parts[1:])
        if context.info_name:
            return context.info_name
    request = _REQUEST.get()
    if request is not None:
        return request.command
    return request_from_argv(sys.argv[1:]).command


def success_envelope(
    data: object,
    *,
    command: str | None = None,
    projection: str = "full",
    truncated: bool = False,
    continuation: str | None = None,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "ok": True,
        "command": command or command_name(),
        "data": data,
        "error": None,
        "meta": {
            "projection": projection,
            "truncated": truncated,
            "continuation": continuation,
        },
    }


def error_envelope(
    *,
    code: str,
    field: str,
    reason: str,
    remediation: str,
    retryable: bool,
    command: str | None = None,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "command": command or command_name(),
        "data": None,
        "error": {
            "code": code,
            "field": field,
            "reason": reason,
            "remediation": remediation,
            "retryable": retryable,
        },
        "meta": dict(_DEFAULT_META),
    }


def emit_document(payload: dict[str, object]) -> None:
    managed_request = _REQUEST.get() is not None
    if managed_request and _DOCUMENT_EMITTED.get():
        return
    click.echo(json.dumps(payload, indent=2, sort_keys=True), err=False)
    if managed_request:
        _DOCUMENT_EMITTED.set(True)


def _has_typer_click_base(exc: Exception, name: str) -> bool:
    return any(base.__name__ == name and base.__module__.partition(".")[0] == "typer" for base in type(exc).__mro__)


def as_click_error(exc: Exception) -> ClickError | None:
    """Adapt external Click and Typer-owned Click errors at the CLI boundary.

    Newer Typer releases may own their Click implementation, so exception class
    identity is not guaranteed to match the separately installed ``click``
    package. Reader depends only on the public usage-error behavior here and
    does not import Typer's private compatibility modules.
    """

    if not isinstance(exc, click.ClickException) and not _has_typer_click_base(exc, "ClickException"):
        return None
    if type(getattr(exc, "exit_code", None)) is not int:
        return None
    if not callable(getattr(exc, "format_message", None)) or not callable(getattr(exc, "show", None)):
        return None
    return cast(ClickError, exc)


def _bad_parameter_field(exc: ClickError) -> str:
    param = getattr(exc, "param", None)
    if param is not None and getattr(param, "name", None):
        return str(param.name)
    param_hint = getattr(exc, "param_hint", None)
    if param_hint:
        hint = param_hint[0] if isinstance(param_hint, tuple) else param_hint
        return str(hint).lstrip("-").replace("-", "_")
    reason = str(exc).lower()
    if "experiment" in reason and "root" in reason:
        return "root"
    if "protocol" in reason:
        return "name"
    if "log level" in reason:
        return "log_level"
    if "format" in reason:
        return "format"
    if "section" in reason:
        return "section"
    if "continuation" in reason:
        return "continuation"
    if "limit" in reason:
        return "limit"
    return "parameter"


def click_error_details(exc: ClickError) -> dict[str, Any]:
    if isinstance(exc, click.BadParameter) or (
        isinstance(exc, Exception) and _has_typer_click_base(exc, "BadParameter")
    ):
        field = _bad_parameter_field(exc)
        return {
            "code": "invalid_parameter",
            "field": field,
            "reason": exc.format_message(),
            "remediation": f"Correct '{field}' and retry the command.",
            "retryable": False,
        }
    return {
        "code": "usage_error",
        "field": "arguments",
        "reason": exc.format_message(),
        "remediation": "Correct the command arguments and retry. Use --help to inspect the accepted interface.",
        "retryable": False,
    }


def reader_error_details(exc: Exception) -> dict[str, Any]:
    reason = str(exc)
    lowered = reason.lower()
    field = "log_level" if "log level" in lowered else "experiment"
    return {
        "code": "reader_error",
        "field": field,
        "reason": reason,
        "remediation": "Correct the reported Reader input or state, then retry the command.",
        "retryable": False,
    }
