from __future__ import annotations

from pathlib import Path

import typer

from . import shared
from ._lazy import load as _load
from .shared import app, emit_json, emit_json_error, normalize_output_format

maintain_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run checks that maintain the Reader source repository.",
)
REPO_ROOT_OPTION = typer.Option(Path("."), "--repo-root", metavar="DIR", help="Reader source checkout root.")


def _emit_report(report, *, format: str) -> None:
    if format == "json":
        if report.ok:
            emit_json(report.to_payload())
        else:
            first_error = report.errors[0] if report.errors else "unknown maintenance check failure"
            emit_json_error(
                code="maintenance_check_failed",
                field="repo_root",
                reason=f"{report.check} check found {len(report.errors)} problem(s). First: {first_error}",
                remediation=f"Correct the reported {report.check} problem(s), then rerun this command.",
            )
        return
    status = "ok" if report.ok else "failed"
    shared.console.print(f"{report.check} integrity {status}: {report.checked} checked")
    for error in report.errors:
        shared.console.print(f"- {error}")


def _run_check(name: str, *, repo_root: Path, format: str) -> None:
    fmt = normalize_output_format(format)
    if (
        not repo_root.is_dir()
        or not (repo_root / "pyproject.toml").is_file()
        or not (repo_root / "src" / "reader").is_dir()
    ):
        raise typer.BadParameter(
            f"Reader source checkout not found: {repo_root}",
            param_hint="--repo-root",
        )
    module = _load(f"reader.maintenance.{name}")
    report = getattr(module, f"check_{name}")(repo_root)
    _emit_report(report, format=fmt)
    if not report.ok:
        raise typer.Exit(code=1)


@maintain_app.command("docs", help="Check documentation links, routes, anchors, and front matter.")
def docs(
    repo_root: Path = REPO_ROOT_OPTION,
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
) -> None:
    _run_check("docs", repo_root=repo_root, format=format)


@maintain_app.command("skills", help="Check repo-local skill structure and source references.")
def skills(
    repo_root: Path = REPO_ROOT_OPTION,
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
) -> None:
    _run_check("skills", repo_root=repo_root, format=format)


app.add_typer(maintain_app, name="maintain")
