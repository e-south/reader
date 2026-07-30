from __future__ import annotations

import typer
from rich import box
from rich.panel import Panel

from reader_workbench.errors import ReaderError

from . import shared
from ._lazy import load as _load
from .helpers import infer_job_path, load_job_models
from .shared import app, emit_json, emit_json_error, normalize_output_format, table


@app.command(help="Verify current records against source, config, build, upstream revision, and artifact evidence.")
def verify(
    job: str | None = typer.Argument(
        None,
        metavar="[CONFIG]",
        help=shared.JOB_ARG_HELP_WITH_DEFAULT,
    ),
    format: str = typer.Option(
        "table", "--format", metavar="FMT", help="Output format: table | json (default: table)."
    ),
) -> None:
    fmt = normalize_output_format(format)
    try:
        job_path = infer_job_path(job)
        _spec, decl = load_job_models(job_path)
        runtime = _load("reader_workbench.runtime").builtin_runtime()
        workbench = _load("reader_workbench.workbench.graph").resolve_workbench(decl)
        layout = decl.experiment_semantics.layout
        store = runtime.record_store(
            layout.outputs_dir,
            plots_subdir=layout.plots_subdir,
            exports_subdir=layout.exports_subdir,
            experiment_root=decl.experiment.root,
            create=False,
        )
        report = _load("reader_workbench.workbench.records").verify_record_store(
            store,
            experiment_root=decl.experiment.root,
            expected_config_digest=decl.config_digest,
            scope=_load("reader_workbench.workbench.inspection.runtime").workbench_record_verification_scope(
                workbench,
                runtime=runtime,
            ),
        )
    except ReaderError as exc:
        report = {
            "schema": "reader.verify/v1",
            "status": "failed",
            "summary": {
                "checked": 0,
                "failed": 1,
                "unverifiable": 0,
                "invocations_checked": 0,
                "invocation_failures": 0,
            },
            "issues": [
                {
                    "code": "experiment.invalid",
                    "field": "config",
                    "reason": str(exc),
                    "remediation": "Correct the experiment path or configuration, then verify again.",
                    "retryable": False,
                }
            ],
            "records": [],
        }

    if fmt == "json":
        if report["status"] == "ok":
            emit_json(report)
        else:
            issues = list(report.get("issues") or [])
            for row in list(report.get("records") or []):
                issues.extend(row.get("issues") or [])
            issue = issues[0] if issues else {}
            emit_json_error(
                code=str(issue.get("code") or "verification_failed"),
                field=str(issue.get("field") or "records"),
                reason=str(issue.get("reason") or "Record verification failed."),
                remediation=str(
                    issue.get("remediation") or "Repair or regenerate the affected records, then verify again."
                ),
                retryable=bool(issue.get("retryable", False)),
            )
    else:
        _render_report(report)
    if report["status"] != "ok":
        raise typer.Exit(code=1)


def _render_report(report: dict[str, object]) -> None:
    summary = report["summary"]
    status = str(report["status"])
    color = "ok" if status == "ok" else ("warn" if status == "unverifiable" else "error")
    shared.console.print(
        Panel.fit(
            f"[{color}]{status}[/{color}] • checked={summary['checked']} • "
            f"failed={summary['failed']} • unverifiable={summary['unverifiable']} • "
            f"invocations={summary['invocations_checked']} • "
            f"invocation failures={summary['invocation_failures']}",
            title="Record verification",
            border_style=color,
            box=box.ROUNDED,
        )
    )
    rows = list(report.get("records") or [])
    issues = list(report.get("issues") or [])
    for row in rows:
        issues.extend(row.get("issues") or [])
    if not issues:
        return
    listing = table("Verification issues")
    listing.add_column("Code", style="accent")
    listing.add_column("Field")
    listing.add_column("Reason", overflow="fold")
    listing.add_column("Remediation", overflow="fold")
    for issue in issues:
        listing.add_row(
            str(issue["code"]),
            str(issue["field"]),
            str(issue["reason"]),
            str(issue["remediation"]),
        )
    shared.console.print(listing)
