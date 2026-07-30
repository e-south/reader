from __future__ import annotations

import sys
from collections.abc import Sequence

import click
import typer

from reader_workbench.errors import ReaderError

from .automation import (
    as_click_error,
    begin_request,
    click_error_details,
    emit_document,
    end_request,
    error_envelope,
    json_requested,
    reader_error_details,
)
from .shared import app


def main(args: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if args is None else args)
    tokens = begin_request(argv)
    try:
        command = typer.main.get_command(app)
        result = command.main(args=argv, prog_name="reader", standalone_mode=False)
        return int(result) if isinstance(result, int) else 0
    except (click.exceptions.Exit, typer.Exit) as exc:
        return int(exc.exit_code)
    except ReaderError as exc:
        if json_requested():
            emit_document(error_envelope(**reader_error_details(exc)))
            return 1
        raise
    except Exception as exc:
        click_error = as_click_error(exc)
        if click_error is not None:
            if json_requested():
                emit_document(error_envelope(**click_error_details(click_error)))
            else:
                click_error.show()
            return int(click_error.exit_code)
        if not json_requested():
            raise
        emit_document(
            error_envelope(
                code="internal_error",
                field="command",
                reason="Reader encountered an unexpected internal error.",
                remediation="Report this Reader defect with the command and error code; do not retry unchanged.",
                retryable=False,
            )
        )
        return 1
    finally:
        end_request(tokens)
