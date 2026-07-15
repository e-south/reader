"""Validated facade for response-window review figures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .display import validate_display_manifest
from .plot_style import apply_publication_style
from .review_endpoint_plots import (
    measured_response_examples_figure,
    quality_figure,
    reduction_sensitivity_figure,
    state_summary_figure,
)
from .review_time_series import time_series_figure
from .visual_labels import (
    STATE_COLORS,
    anchored_fluorescence_axis_label,
    channels,
    response_axis_label,
    response_summary_label,
)


@dataclass(frozen=True)
class ReviewViewSpec:
    view_id: str
    label: str
    premise: str
    decision_value: str
    interpretation: str
    alt_text: str
    non_claim_boundary: str


REVIEW_VIEW_SPECS = (
    ReviewViewSpec(
        view_id="time_series",
        label="Time series and window",
        premise="The selected post-stress interval connects observed trajectories to the eight-value handoff",
        decision_value="Shows whether the reduced response and fluorescence values are supported by sustained, replicate-observed trajectories.",
        interpretation="Solid curves are replicate medians, translucent bands are the central 90% of design wells, and the dashed curve is the same-state {reference_id} fluorescence anchor. Gray shading marks event-time uncertainty; amber marks the selected response window. The lower panels keep bootstrap intervals and event-time sensitivity separate, with observed well points shown only for rᵢ.",
        alt_text="Three aligned square trajectory panels show {growth} growth, {response_axis} response, and {magnitude_ratio} fluorescence in four conditions using replicate medians and central 90% replicate intervals. Two square dot-and-whisker panels show the four response and four {reference_id}-relative fluorescence values produced by the selected post-stress window. Hollow points are observed response-well reductions; anchored fluorescence compares independent design and reference aggregates and therefore has no per-well b points. A compact card records the window and assay support.",
        non_claim_boundary="The selected interval is a prespecified assay summary, not proof of when biology begins to respond.",
    ),
    ReviewViewSpec(
        view_id="state_summary",
        label="State handoff values",
        premise="The selected window preserves response and anchored fluorescence by condition",
        decision_value="Shows the eight measured values and their assay-derived uncertainty before study scoring.",
        interpretation="Response is {response_axis}; fluorescence is {anchored_axis}. Hollow points are observed response-well reductions. Colored bootstrap intervals and gray event-time sensitivity marks remain separate; anchored fluorescence has no fabricated per-well b points.",
        alt_text="Two dot-and-whisker panels show four condition-specific {response_axis} responses and four {anchored_axis} fluorescence summaries. The response panel includes observed well reductions. Both panels show asymmetric bootstrap intervals separately from gray event-time sensitivity marks.",
        non_claim_boundary="These assay summaries are not campaign scores or validated responsive promoters.",
    ),
    ReviewViewSpec(
        view_id="measured_response_examples",
        label="Measured response examples",
        premise="Measured response examples provide direction checks across all four conditions",
        decision_value="Checks that familiar response examples retain their expected signed directions under the selected reduction.",
        interpretation="Each panel preserves raw condition summaries; no per-design min-max transform is applied.",
        alt_text="Two heatmaps show four-condition {response_axis} response and {anchored_axis} fluorescence for the configured measured response examples and their anchor rows across Reader experiments.",
        non_claim_boundary="SpyP and sulAp are interpretation references, not required or optimal campaign archetypes.",
    ),
    ReviewViewSpec(
        view_id="reduction_sensitivity",
        label="Reduction sensitivity",
        premise="Prespecified reductions retain the same condition-level response structure",
        decision_value="Reveals whether a window, integration, or pre-event subtraction choice changes the handoff materially.",
        interpretation="Rows are prespecified reductions; columns are the four assay conditions on explicit response and fluorescence scales.",
        alt_text="Two heatmaps compare the selected design's four {response_axis} responses and four {anchored_axis} fluorescence values across prespecified post-event windows, reduction methods, and response bases.",
        non_claim_boundary="Agreement among reductions does not establish which interval is biologically optimal.",
    ),
    ReviewViewSpec(
        view_id="quality",
        label="Quality and uncertainty",
        premise="The handoff exposes uncertainty and replicate support for every condition",
        decision_value="Separates replicate-bootstrap variation, event-bound sensitivity, and independent-well support.",
        interpretation="Uncertainty is reported separately for {response_axis} response and {anchored_axis} fluorescence.",
        alt_text="Three grouped-bar panels show replicate-bootstrap variation, event-bound sensitivity, and independent-well counts for each of the four assay conditions.",
        non_claim_boundary="These empirical intervals are not calibrated biological effect thresholds.",
    ),
)
VIEW_LABELS = {spec.label: spec.view_id for spec in REVIEW_VIEW_SPECS}
_VIEW_SPECS_BY_ID = {spec.view_id: spec for spec in REVIEW_VIEW_SPECS}


def review_view_spec(view_id: str, *, display: dict[str, object]) -> ReviewViewSpec:
    display = validate_display_manifest(display)
    try:
        spec = _VIEW_SPECS_BY_ID[view_id]
    except KeyError as exc:
        raise ValueError(f"unknown response-window review view: {view_id!r}.") from exc
    values = channels(display)
    context = {
        "event_label_lower": str(display["event_label"]).lower(),
        "growth": values["growth"],
        "magnitude_ratio": values["magnitude_ratio"],
        "reference_id": values["reference_design_id"],
        "response_axis": response_axis_label(display),
        "anchored_axis": anchored_fluorescence_axis_label(display),
    }
    return replace(
        spec,
        premise=spec.premise.format_map(context),
        decision_value=spec.decision_value.format_map(context),
        interpretation=spec.interpretation.format_map(context),
        alt_text=spec.alt_text.format_map(context),
        non_claim_boundary=spec.non_claim_boundary.format_map(context),
    )


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
    experiment_id: str,
    design_id: str,
    reduction_id: str,
    designs: pd.DataFrame,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    events: pd.DataFrame,
    display: dict[str, object],
) -> plt.Figure:
    display = validate_display_manifest(display)
    review_view_spec(view_id, display=display)
    if view_id == "measured_response_examples":
        figure = measured_response_examples_figure(
            rows=measured_response_example_rows(designs, display=display, reduction_id=reduction_id),
            display=display,
        )
    elif view_id == "reduction_sensitivity":
        rows = designs.loc[
            designs["experiment_id"].astype(str).eq(experiment_id) & designs["design_id"].astype(str).eq(design_id)
        ]
        figure = reduction_sensitivity_figure(rows=rows, display=display)
    else:
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
