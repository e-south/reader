from __future__ import annotations

from rich import box
from rich.table import Table

from reader_workbench.workbench.dop import DOP_SCHEMA, DataClassSpec, ReadySpec


def data_classes_payload(data_classes: tuple[DataClassSpec, ...]) -> dict[str, object]:
    return {
        "schema": DOP_SCHEMA,
        "data_classes": [item.to_payload() for item in data_classes],
    }


def ready_specs_payload(ready_specs: tuple[ReadySpec, ...]) -> dict[str, object]:
    return {
        "schema": DOP_SCHEMA,
        "ready_specs": [item.to_payload() for item in ready_specs],
    }


def _table(title: str) -> Table:
    return Table(
        title=f"[title]{title}[/title]",
        title_justify="left",
        header_style="bold",
        box=box.ROUNDED,
        expand=True,
        show_lines=False,
        show_edge=True,
    )


def data_classes_table(data_classes: tuple[DataClassSpec, ...]) -> Table:
    table = _table("DOP Data Classes")
    table.add_column("ID", style="accent", overflow="fold")
    table.add_column("Protocols", overflow="fold")
    table.add_column("Minimum capture", overflow="fold")
    table.add_column("Stop when", overflow="fold")
    for item in data_classes:
        table.add_row(
            item.id,
            ", ".join(item.protocol_candidates),
            "; ".join(item.minimum_capture),
            "; ".join(item.stop_conditions),
        )
    return table


def ready_specs_table(ready_specs: tuple[ReadySpec, ...]) -> Table:
    table = _table("DOP Ready Specs")
    table.add_column("ID", style="accent", overflow="fold")
    table.add_column("Evidence", overflow="fold")
    table.add_column("Reader states", overflow="fold")
    table.add_column("Capabilities", overflow="fold")
    for item in ready_specs:
        table.add_row(
            item.id,
            "; ".join(item.required_evidence),
            ", ".join(item.accepted_readiness_states) or "—",
            ", ".join(item.required_capabilities) or "—",
        )
    return table
