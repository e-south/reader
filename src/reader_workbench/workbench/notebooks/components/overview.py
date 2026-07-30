from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reader_workbench.workbench.notebooks.presentation import experiment_display_title


@dataclass(frozen=True)
class NotebookOverview:
    experiment_id: str
    experiment_title: str
    protocol_id: str
    experiment_root: Path
    outputs_dir: Path
    notebooks_dir: Path
    pipeline_rows: tuple[dict[str, Any], ...]


def build_notebook_overview(
    *,
    experiment_id: str,
    experiment_title: str | None,
    protocol_id: str,
    experiment_root: Path,
    outputs_dir: Path,
    notebooks_dir: Path,
    pipeline_step_ids: Sequence[str],
) -> NotebookOverview:
    resolved_experiment_id = _required_text(experiment_id, "experiment_id")
    resolved_protocol_id = _required_text(protocol_id, "protocol_id")
    title = experiment_display_title(
        experiment_id=resolved_experiment_id,
        authored_title=experiment_title,
    )
    pipeline_rows = tuple(
        {"Order": idx, "Step ID": str(step_id)} for idx, step_id in enumerate(pipeline_step_ids, start=1)
    )
    return NotebookOverview(
        experiment_id=resolved_experiment_id,
        experiment_title=title,
        protocol_id=resolved_protocol_id,
        experiment_root=Path(experiment_root).expanduser().resolve(),
        outputs_dir=Path(outputs_dir).expanduser().resolve(),
        notebooks_dir=Path(notebooks_dir).expanduser().resolve(),
        pipeline_rows=pipeline_rows,
    )


def render_notebook_overview_panel(
    mo: Any,
    overview: NotebookOverview,
    *,
    detail_sections: Mapping[str, Any] | None = None,
    include_heading: bool = True,
) -> Any:
    at_a_glance_rows = (
        {"Field": "Experiment ID", "Value": overview.experiment_id},
        {"Field": "Protocol", "Value": overview.protocol_id},
        {"Field": "Pipeline steps", "Value": len(overview.pipeline_rows)},
    )
    path_rows = (
        {
            "Path": "Experiment root",
            "Value": _path_label(overview.experiment_root, base=overview.experiment_root),
        },
        {
            "Path": "Outputs",
            "Value": _path_label(overview.outputs_dir, base=overview.experiment_root),
        },
        {
            "Path": "Generated notebooks",
            "Value": _path_label(overview.notebooks_dir, base=overview.experiment_root),
        },
    )
    rendered_detail_sections = dict(detail_sections or {})
    reserved_sections = {"Pipeline", "Paths"}
    conflicts = sorted(reserved_sections.intersection(rendered_detail_sections))
    if conflicts:
        raise ValueError(f"Notebook overview detail sections use reserved names: {', '.join(conflicts)}")
    rendered_detail_sections.update(
        {
            "Pipeline": _render_table(
                mo,
                overview.pipeline_rows,
                "No pipeline steps are declared.",
            ),
            "Paths": _render_table(mo, path_rows, "No output paths are available."),
        }
    )
    items = [
        mo.ui.table(list(at_a_glance_rows), page_size=len(at_a_glance_rows)),
        mo.accordion(rendered_detail_sections, multiple=True, lazy=True),
    ]
    if include_heading:
        items.insert(0, mo.md(f"# {overview.experiment_title}"))
    return mo.vstack(items)


def _render_table(mo: Any, rows: tuple[dict[str, Any], ...], empty_text: str) -> Any:
    if not rows:
        return mo.md(empty_text)
    return mo.ui.table(list(rows), page_size=min(12, max(1, len(rows))))


def _path_label(path: Path, *, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _required_text(value: str, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required for notebook overview rendering.")
    return text
