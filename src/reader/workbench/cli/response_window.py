"""CLI lifecycle for manifest-backed response-window bundles."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich import box
from rich.panel import Panel

from reader.errors import ReaderError
from reader.workbench.paths import resolve_confined_sink_root

from .helpers import resolve_output_experiment
from .notebooks import _launch_marimo
from .shared import app, console, emit_json, emit_json_error, normalize_output_format, table

if TYPE_CHECKING:
    from reader.response_window import ResponseWindowBundle, ResponseWindowPreflight

response_window_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Preflight, build, verify, and review event-relative plate-reader handoffs.",
)


def preflight_response_window_request(**kwargs):
    return import_module("reader.response_window").preflight_response_window_request(**kwargs)


def build_response_window_bundle(**kwargs):
    return import_module("reader.response_window").build_response_window_bundle(**kwargs)


def verify_response_window_bundle(path: Path):
    return import_module("reader.response_window").verify_response_window_bundle(path)


@response_window_app.command(
    "preflight",
    help="Verify sources and non-publishing reduction, aggregation, and QC readiness.",
)
def preflight(
    request: Annotated[Path, typer.Argument(help="Path to a reader.response_window.request.v3 YAML file.")],
    reader_root: Annotated[Path, typer.Option("--reader-root", help="Reader repository root.")] = Path("."),
    output_format: Annotated[
        str,
        typer.Option("--format", metavar="FMT", help="Output format: table | json (default: table)."),
    ] = "table",
) -> None:
    try:
        result = preflight_response_window_request(reader_root=reader_root, request_path=request)
    except (FileNotFoundError, ReaderError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    fmt = normalize_output_format(output_format)
    if fmt == "json":
        if result.ready:
            emit_json(result.to_payload())
        else:
            missing = ", ".join(result.missing_display_examples)
            emit_json_error(
                code="preflight_failed",
                field="missing_display_examples",
                reason=f"Configured response-window display examples were not observed: {missing}.",
                remediation="Correct request.display.examples or produce the missing designs, then run preflight again.",
            )
    else:
        _render_preflight(result)
    if not result.ready:
        raise typer.Exit(code=1)


@response_window_app.command("build", help="Materialize an atomic, verified response-window bundle.")
def build(
    request: Annotated[Path, typer.Argument(help="Path to a reader.response_window.request.v3 YAML file.")],
    output_experiment: Annotated[
        str,
        typer.Option(
            "--output-experiment",
            metavar="CONFIG|DIR|INDEX",
            help="Experiment that owns the generated response-window bundle.",
        ),
    ],
    reader_root: Annotated[Path, typer.Option("--reader-root", help="Reader repository root.")] = Path("."),
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Atomically replace an existing bundle.")] = False,
    output_format: Annotated[
        str,
        typer.Option("--format", metavar="FMT", help="Output format: table | json (default: table)."),
    ] = "table",
) -> None:
    output_format = normalize_output_format(output_format)
    try:
        out_dir = _bundle_destination(output_experiment, bundle_kind="response-window")
        bundle = build_response_window_bundle(
            reader_root=reader_root,
            request_path=request,
            out_dir=out_dir,
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, ReaderError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit_bundle(bundle, output_format=output_format, heading="Response-window bundle built")


@response_window_app.command("verify", help="Verify bundle schemas, counts, provenance, and artifact digests.")
def verify(
    bundle_root: Annotated[Path, typer.Argument(help="Published response-window bundle directory.")],
    output_format: Annotated[
        str,
        typer.Option("--format", metavar="FMT", help="Output format: table | json (default: table)."),
    ] = "table",
) -> None:
    try:
        bundle = verify_response_window_bundle(bundle_root)
    except (FileNotFoundError, ReaderError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit_bundle(bundle, output_format=output_format, heading="Response-window bundle verified")


@response_window_app.command("review", help="Verify a bundle, then open its generated Marimo review notebook.")
def review(
    bundle_root: Annotated[Path, typer.Argument(help="Published response-window bundle directory.")],
    reader_root: Annotated[Path, typer.Option("--reader-root", help="Reader repository root.")] = Path("."),
    mode: Annotated[str, typer.Option("--mode", help="Marimo mode: run | edit (default: run).")] = "run",
    headless: Annotated[bool, typer.Option("--headless", help="Do not open a browser window.")] = False,
    port: Annotated[int | None, typer.Option("--port", help="Preferred local Marimo port.")] = None,
) -> None:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"run", "edit"}:
        raise typer.BadParameter("--mode must be one of: run, edit.")
    try:
        bundle = verify_response_window_bundle(bundle_root)
    except (FileNotFoundError, ReaderError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _launch_marimo(
        normalized_mode,
        bundle.notebook_path,
        has_fcs=False,
        headless=headless,
        port=port,
        repo_root=reader_root,
    )


def _emit_bundle(bundle: ResponseWindowBundle, *, output_format: str, heading: str) -> None:
    payload = {
        "schema_version": bundle.manifest["schema_version"],
        "study_id": bundle.manifest["study_id"],
        "request_id": bundle.manifest["request_id"],
        "bundle_root": str(bundle.root),
        "manifest": str(bundle.manifest_path),
        "notebook": str(bundle.notebook_path),
        "counts": bundle.counts,
    }
    if normalize_output_format(output_format) == "json":
        emit_json(payload)
        return
    summary = table(heading)
    summary.add_column("Field", style="accent")
    summary.add_column("Value")
    summary.add_row("Request", str(payload["request_id"]))
    summary.add_row("Study", str(payload["study_id"]))
    summary.add_row("Schema", str(payload["schema_version"]))
    summary.add_row("Bundle", str(bundle.root))
    summary.add_row("Notebook", str(bundle.notebook_path))
    summary.add_row("Counts", ", ".join(f"{key}={value}" for key, value in sorted(bundle.counts.items())))
    console.print(Panel(summary, border_style="ok", box=box.ROUNDED))


def _render_preflight(result: ResponseWindowPreflight) -> None:
    summary = table("Response-window source preflight")
    summary.add_column("Experiment", style="accent")
    summary.add_column("Response designs", justify="right")
    summary.add_column("Fluorescence designs", justify="right")
    summary.add_column("Post-event coverage (h)", justify="right")
    summary.add_column("Event interval (h)")
    for experiment in result.experiments:
        summary.add_row(
            experiment.experiment_id,
            str(experiment.response_designs),
            str(experiment.magnitude_designs),
            f"{experiment.post_event_coverage_h:.2f}",
            f"{experiment.event_interval_start_assay_h:.2f} to {experiment.event_interval_end_assay_h:.2f}",
        )
    missing = ", ".join(result.missing_display_examples) or "none"
    status = "ready" if result.ready else "blocked"
    console.print(
        Panel(summary, title=f"{status}: {result.request_id}", border_style="ok" if result.ready else "error")
    )
    console.print(f"[muted]Primary reduction:[/muted] {result.primary_reduction_id}")
    console.print(f"[muted]Missing review examples:[/muted] {missing}")


def _bundle_destination(output_experiment: str, *, bundle_kind: str) -> Path:
    """Confine an aggregate publication to its owning experiment's outputs."""

    outputs_dir = resolve_output_experiment(output_experiment)
    return resolve_confined_sink_root(
        outputs_dir / "bundles" / bundle_kind,
        root=outputs_dir,
        label=f"{bundle_kind} bundle",
    )


app.add_typer(response_window_app, name="response-window")

__all__ = [
    "build",
    "preflight",
    "response_window_app",
    "review",
    "verify",
]
