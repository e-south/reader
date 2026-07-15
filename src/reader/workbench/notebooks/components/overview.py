from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reader.notebook_presentation import experiment_display_title


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


def build_design_treatment_summary_rows(df: Any) -> tuple[tuple[dict[str, str], ...], str]:
    if df is None:
        return (), "No dataset selected yet."
    columns = set(getattr(df, "columns", []))
    missing = [col for col in ("design_id", "treatment") if col not in columns]
    if missing:
        return (), f"Missing column(s): {', '.join(missing)}."

    rows = [{"Category": "Design ID", "Value": value} for value in _unique_column_values(df, "design_id")]
    rows.extend({"Category": "Treatment", "Value": value} for value in _unique_column_values(df, "treatment"))
    if not rows:
        return (), "No non-empty design or treatment values found."
    return tuple(rows), ""


def render_notebook_overview_panel(
    mo: Any,
    overview: NotebookOverview,
    *,
    design_treatment_rows: Sequence[Mapping[str, Any]],
    design_treatment_note: str = "",
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
    detail_sections = {
        "Design/treatment scope": _render_table(
            mo,
            tuple(dict(row) for row in design_treatment_rows),
            design_treatment_note or "No design/treatment summary is available.",
        ),
        "Pipeline": _render_table(
            mo,
            overview.pipeline_rows,
            "No pipeline steps are declared.",
        ),
        "Paths": _render_table(mo, path_rows, "No output paths are available."),
    }
    items = [
        mo.ui.table(list(at_a_glance_rows), page_size=len(at_a_glance_rows)),
        mo.accordion(detail_sections, multiple=True, lazy=True),
    ]
    if include_heading:
        items.insert(0, mo.md(f"# {overview.experiment_title}"))
    return mo.vstack(items)


def _unique_column_values(df: Any, col: str) -> tuple[str, ...]:
    values: list[Any] = []
    if hasattr(df, "get_column"):
        series = df.get_column(col)
        if hasattr(series, "drop_nulls"):
            series = series.drop_nulls()
        if hasattr(series, "unique"):
            series = series.unique()
        if hasattr(series, "to_list"):
            values = series.to_list()
    elif hasattr(df, "__getitem__"):
        series = df[col]
        if hasattr(series, "dropna"):
            series = series.dropna()
        if hasattr(series, "unique"):
            series = series.unique()
        if hasattr(series, "tolist"):
            values = series.tolist()
        elif hasattr(series, "to_list"):
            values = series.to_list()
    text_values = [str(value) for value in values if value is not None and str(value).strip()]
    return tuple(sorted(set(text_values)))


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
