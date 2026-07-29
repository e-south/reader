"""Validated facade for response-window review figures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reader.domains.plate_reader.analysis.response_window.display import validate_display_manifest

from .plot_style import apply_publication_style
from .review_collection import cross_experiment_design_rows
from .review_cross_experiment import cross_experiment_state_figure
from .review_endpoint_plots import (
    measured_response_examples_figure,
    quality_figure,
    reduction_sensitivity_figure,
    state_summary_figure,
)
from .review_time_series import time_series_figure
from .review_views import (
    REVIEW_VIEW_SPECS,
    VIEW_LABELS,
    ReviewViewSpec,
    review_view_spec,
)
from .visual_labels import STATE_COLORS, response_summary_label


def load_review_tables(bundle_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(bundle_root).resolve()
    paths = tuple(
        root / "tables" / name for name in ("designs.parquet", "wells.parquet", "traces.parquet", "events.parquet")
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"response-window review bundle is incomplete: {missing}")
    return tuple(pd.read_parquet(path) for path in paths)  # type: ignore[return-value]


def render_review_figure(
    *,
    view_id: str,
    reduction_id: str,
    designs: pd.DataFrame,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    events: pd.DataFrame,
    display: dict[str, object],
    experiment_id: str | None = None,
    design_id: str | None = None,
    state: str | None = None,
    experiment_labels: Mapping[str, str] | None = None,
) -> plt.Figure:
    display = validate_display_manifest(display)
    review_view_spec(view_id, display=display)
    if view_id == "multi_experiment_evidence":
        if design_id is None or state is None or experiment_labels is None:
            raise ValueError(
                "multi-experiment response-window review requires a Reader design, condition, and experiment labels."
            )
        selected = cross_experiment_design_rows(
            designs,
            design_id=design_id,
            reduction_id=reduction_id,
        )
        figure = cross_experiment_state_figure(
            selected=selected,
            state=state,
            experiment_labels=experiment_labels,
            wells=wells,
            traces=traces,
            events=events,
            display=display,
        )
    elif view_id == "measured_response_examples":
        figure = measured_response_examples_figure(
            rows=measured_response_example_rows(designs, display=display, reduction_id=reduction_id),
            display=display,
        )
    elif view_id == "reduction_sensitivity":
        experiment_id, design_id = _required_local_selection(experiment_id, design_id)
        rows = designs.loc[
            designs["experiment_id"].astype(str).eq(experiment_id) & designs["design_id"].astype(str).eq(design_id)
        ]
        figure = reduction_sensitivity_figure(rows=rows, display=display)
    else:
        experiment_id, design_id = _required_local_selection(experiment_id, design_id)
        selected = selected_handoff_row(
            designs,
            experiment_id=experiment_id,
            design_id=design_id,
            reduction_id=reduction_id,
        ).iloc[0]
        if view_id == "time_series":
            figure = time_series_figure(
                experiment_id=experiment_id,
                design_id=design_id,
                reduction_id=reduction_id,
                selected=selected,
                wells=wells,
                traces=traces,
                events=events,
                display=display,
            )
        elif view_id == "state_summary":
            figure = state_summary_figure(
                experiment_id=experiment_id,
                design_id=design_id,
                reduction_id=reduction_id,
                selected=selected,
                wells=wells,
                display=display,
            )
        elif view_id == "quality":
            selected_wells = wells.loc[
                wells["experiment_id"].astype(str).eq(experiment_id)
                & wells["design_id"].astype(str).eq(design_id)
                & wells["reduction_id"].astype(str).eq(reduction_id)
            ]
            figure = quality_figure(selected=selected, selected_wells=selected_wells, display=display)
        else:
            raise AssertionError(f"unhandled validated review view: {view_id!r}.")
    return apply_publication_style(figure)


def _required_local_selection(experiment_id: str | None, design_id: str | None) -> tuple[str, str]:
    if experiment_id is None or design_id is None:
        raise ValueError("experiment-local response-window review requires an experiment and Reader design.")
    return experiment_id, design_id


def selected_handoff_row(
    designs: pd.DataFrame,
    *,
    experiment_id: str,
    design_id: str,
    reduction_id: str,
) -> pd.DataFrame:
    selected = designs.loc[
        designs["experiment_id"].astype(str).eq(experiment_id)
        & designs["design_id"].astype(str).eq(design_id)
        & designs["reduction_id"].astype(str).eq(reduction_id)
    ].copy()
    if len(selected) != 1:
        raise ValueError(
            "response-window selection must resolve to one handoff row: "
            f"experiment={experiment_id!r}, design={design_id!r}, reduction={reduction_id!r}."
        )
    return selected


def measured_response_example_rows(
    designs: pd.DataFrame,
    *,
    display: dict[str, object],
    reduction_id: str,
) -> pd.DataFrame:
    examples = display["examples"]
    if not isinstance(examples, list):
        raise ValueError("validated display examples must be a list.")
    metadata = pd.DataFrame.from_records(examples)
    rows = designs.loc[
        designs["reduction_id"].astype(str).eq(reduction_id)
        & designs["design_id"].astype(str).isin(metadata["design_id"].astype(str))
    ].merge(metadata, on="design_id", how="inner", validate="many_to_one")
    missing = sorted(set(metadata["design_id"].astype(str)) - set(rows["design_id"].astype(str)))
    if missing:
        raise ValueError(f"response-window reduction lacks configured measured-example designs: {missing}.")
    rows = rows.rename(columns={"label": "example_label", "role": "example_role"})
    response_experiments = set(rows.loc[rows["example_role"].eq("response_example"), "experiment_id"].astype(str))
    rows = rows.loc[
        rows["example_role"].eq("response_example") | rows["experiment_id"].astype(str).isin(response_experiments)
    ].copy()
    rows["role_order"] = rows["example_role"].map({"reference_anchor": 0, "response_example": 1})
    return rows.sort_values(["role_order", "example_label", "experiment_id"], kind="mergesort").reset_index(drop=True)


def response_summary_options(available_reductions: pd.DataFrame) -> dict[str, str]:
    """Return display labels mapped to stable reduction IDs for one design."""

    required = {
        "reduction_id",
        "window_start_event_h",
        "window_end_event_h",
        "reduction_method",
        "response_basis",
        "reduction_role",
    }
    missing = sorted(required - set(available_reductions.columns))
    if missing:
        raise ValueError(f"response-summary options are missing columns: {missing}.")
    rows = available_reductions.loc[:, sorted(required)].drop_duplicates()
    if rows.empty or rows["reduction_id"].astype(str).duplicated().any():
        raise ValueError("response-summary options require one definition per reduction ID.")
    rows["role_order"] = rows["reduction_role"].astype(str).map({"primary": 0, "sensitivity": 1})
    if rows["role_order"].isna().any():
        raise ValueError("response-summary options contain an unknown reduction role.")
    rows = rows.sort_values(["role_order", "reduction_id"], kind="mergesort")
    options = {response_summary_label(row._asdict()): str(row.reduction_id) for row in rows.itertuples(index=False)}
    if len(options) != len(rows):
        raise ValueError("response-summary display labels must be unique.")
    return options


__all__ = [
    "REVIEW_VIEW_SPECS",
    "STATE_COLORS",
    "VIEW_LABELS",
    "ReviewViewSpec",
    "load_review_tables",
    "measured_response_example_rows",
    "render_review_figure",
    "response_summary_options",
    "review_view_spec",
    "selected_handoff_row",
]
