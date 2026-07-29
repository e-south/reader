from __future__ import annotations

import typer
from rich import box
from rich.panel import Panel

from reader.errors import ConfigError

from . import shared
from ._lazy import load as _load
from .shared import app, emit_json, normalize_output_format

dop_app = typer.Typer(
    add_completion=False,
    help="Inspect the reader-local Data Operations Plan registry.",
)


def _registry():
    return _load("reader.workbench.dop").builtin_dop_registry()


def _validate_registry_protocol_refs(registry, runtime) -> None:
    registry.validate_protocol_refs(descriptor.protocol for descriptor in runtime.protocols.all())


@dop_app.command("classes", help="List DOP data classes and their reader protocol candidates.")
def data_classes(
    name: str | None = typer.Argument(None, metavar="[DATA_CLASS]", help="Optional DOP data class id to describe."),
    protocol: str | None = typer.Option(
        None,
        "--protocol",
        metavar="ID",
        help="Only show data classes that include the given reader protocol id.",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
):
    registry = _registry()
    runtime = _load("reader.runtime").builtin_runtime()
    inspection_dop = _load("reader.workbench.inspection.dop")
    fmt = normalize_output_format(format)
    try:
        _validate_registry_protocol_refs(registry, runtime)
        selected = registry.data_classes()
        if name is not None:
            selected = (registry.data_class(name),)
        if protocol is not None and protocol.strip():
            resolved_protocol = runtime.protocols.resolve(protocol.strip()).protocol
            selected = tuple(item for item in selected if resolved_protocol in item.protocol_candidates)
    except (ConfigError, ValueError) as err:
        raise typer.BadParameter(str(err)) from err
    if fmt == "json":
        emit_json(inspection_dop.data_classes_payload(selected))
        return
    shared.console.print(Panel(inspection_dop.data_classes_table(selected), border_style="accent", box=box.ROUNDED))


@dop_app.command("ready-specs", help="List DOP readiness gates and their reader evidence requirements.")
def ready_specs(
    name: str | None = typer.Argument(None, metavar="[READY_SPEC]", help="Optional DOP ready spec id to describe."),
    format: str = typer.Option(
        "table",
        "--format",
        metavar="FMT",
        help="Output format: table | json (default: table).",
    ),
):
    registry = _registry()
    inspection_dop = _load("reader.workbench.inspection.dop")
    readiness = _load("reader.workbench.inspection.readiness")
    fmt = normalize_output_format(format)
    try:
        registry.validate_ready_refs(
            readiness_states=readiness.READINESS_STATES,
            capability_keys=readiness.READINESS_CAPABILITY_KEYS,
        )
        selected = registry.ready_specs() if name is None else (registry.ready_spec(name),)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err
    if fmt == "json":
        emit_json(inspection_dop.ready_specs_payload(selected))
        return
    shared.console.print(Panel(inspection_dop.ready_specs_table(selected), border_style="accent", box=box.ROUNDED))


app.add_typer(dop_app, name="dop")
