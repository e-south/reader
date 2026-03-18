from __future__ import annotations

from rich import box
from rich.table import Table


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


def semantic_node_payload(node) -> dict[str, object]:
    payload = {
        "id": node.id,
        "kind": node.kind,
        "summary": node.summary,
        "execution": {
            "status": node.execution.status,
            "step_ids": list(node.execution.step_ids),
            "plugin_ids": list(node.execution.plugin_ids),
            "record_ids": list(node.execution.record_ids),
            "config_paths": list(node.execution.config_paths),
            "note": node.execution.note,
        },
    }
    if node.kind == "control_rule":
        payload["match_on"] = list(node.match_on)
        payload["control_selector"] = node.control_selector
    if node.kind == "window":
        payload["anchor"] = node.anchor
        payload["selector"] = node.selector
        payload["params"] = dict(node.params)
    if node.kind == "metric":
        payload["stage"] = node.stage
        payload["formula"] = node.formula
        payload["depends_on"] = list(node.depends_on)
    if node.kind == "ranking":
        payload["primary_metric"] = node.primary_metric
        payload["direction"] = node.direction
        payload["penalties"] = list(node.penalties)
        payload["supporting_metrics"] = list(node.supporting_metrics)
    return payload


def semantic_program_summary(program) -> dict[str, object]:
    summary = {
        "total": 0,
        "compiled": 0,
        "descriptive_only": 0,
        "by_kind": {
            "control_rule": {"total": 0, "compiled": 0, "descriptive_only": 0},
            "window": {"total": 0, "compiled": 0, "descriptive_only": 0},
            "metric": {"total": 0, "compiled": 0, "descriptive_only": 0},
            "ranking": {"total": 0, "compiled": 0, "descriptive_only": 0},
        },
    }

    def _record(node) -> None:
        status = str(node.execution.status)
        kind = str(node.kind)
        summary["total"] += 1
        summary[status] += 1
        kind_counts = dict(summary["by_kind"][kind])
        kind_counts["total"] += 1
        kind_counts[status] += 1
        summary["by_kind"][kind] = kind_counts

    for node in program.controls:
        _record(node)
    for node in program.windows:
        _record(node)
    for node in program.metrics:
        _record(node)
    if program.ranking is not None:
        _record(program.ranking)
    return summary


def semantic_program_payload(program) -> dict[str, object]:
    return {
        "protocol": program.protocol,
        "summary": semantic_program_summary(program),
        "controls": [semantic_node_payload(node) for node in program.controls],
        "windows": [semantic_node_payload(node) for node in program.windows],
        "metrics": [semantic_node_payload(node) for node in program.metrics],
        "ranking": semantic_node_payload(program.ranking) if program.ranking is not None else None,
    }


def semantic_program_table(program) -> Table:
    coverage = semantic_program_summary(program)
    table = _table(
        "Semantic Program"
        f" • {coverage['compiled']}/{coverage['total']} compiled"
        f" • {coverage['descriptive_only']} descriptive"
    )
    table.add_column("kind", style="accent", width=13)
    table.add_column("id", style="accent", overflow="fold")
    table.add_column("status", width=18)
    table.add_column("compiled via", overflow="fold")
    table.add_column("summary", overflow="fold")

    def _add_node(kind: str, node) -> None:
        compiled_via = ", ".join(node.execution.step_ids) or "—"
        note = node.execution.note
        summary = node.summary if not note else f"{node.summary} ({note})"
        table.add_row(kind, node.id, node.execution.status, compiled_via, summary)

    for node in program.controls:
        _add_node("control_rule", node)
    for node in program.windows:
        _add_node("window", node)
    for node in program.metrics:
        _add_node("metric", node)
    if program.ranking is not None:
        _add_node("ranking", program.ranking)
    return table
