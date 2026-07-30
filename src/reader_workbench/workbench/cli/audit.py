from __future__ import annotations

import json
from pathlib import Path

import typer

from . import shared
from ._lazy import load as _load
from .shared import app, emit_json, emit_json_error, normalize_output_format

audit_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Pressure-test experiment workbenches without mutating their outputs.",
)
EXPERIMENTS_ROOT_OPTION = typer.Option(Path("experiments"), "--root", metavar="DIR", help="Experiments root.")
EXPERIMENT_YEARS_OPTION = typer.Option(
    None,
    "--years",
    metavar="YYYY",
    help="Year directory to include (repeatable; defaults to every numeric year).",
)
AUDIT_REPORT_PATH_OPTION = typer.Option(
    None,
    "--report-path",
    metavar="FILE",
    help="Optional path for the raw JSON audit report.",
)


@audit_app.command(
    "experiments", help="Run active experiments from isolated temporary copies and verify their outputs."
)
def experiments(
    root: Path = EXPERIMENTS_ROOT_OPTION,
    years: list[str] | None = EXPERIMENT_YEARS_OPTION,
    include_non_active: bool = typer.Option(
        False,
        "--include-non-active",
        help="Run draft and other non-active experiments instead of reporting them as skipped.",
    ),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop after the first failed experiment."),
    report_path: Path | None = AUDIT_REPORT_PATH_OPTION,
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
) -> None:
    fmt = normalize_output_format(format)
    audit_module = _load("reader_workbench.workbench.audit.experiments")
    try:
        payload = audit_module.audit_experiments(
            root,
            years=None if years is None else tuple(years),
            include_non_active=include_non_active,
            fail_fast=fail_fast,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--root/--years") from exc

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    failed = int(payload["summary"]["failed"])
    if fmt == "json" and failed:
        first_failed = next(item for item in payload["results"] if item["status"] == "failed")
        detail = str(first_failed["detail"] or "no failure detail")
        bounded_detail = detail if len(detail) <= 400 else f"{detail[:399]}…"
        emit_json_error(
            code="experiment_audit_failed",
            field="experiments",
            reason=(
                f"{failed} experiment audit(s) failed. First: {first_failed['name']} "
                f"during {first_failed['phase']}: {bounded_detail}"
            ),
            remediation=(
                "Inspect the named experiment, correct its validation, execution, or verification failure, "
                "then rerun the audit. Use --report-path when the complete machine report is required."
            ),
        )
    elif fmt == "json":
        emit_json(payload)
    else:
        results = [audit_module.AuditResult(**item) for item in payload["results"]]
        shared.console.print(audit_module.render_text(results))

    if failed:
        raise typer.Exit(code=1)


app.add_typer(audit_app, name="audit")
