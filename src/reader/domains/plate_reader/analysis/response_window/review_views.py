"""Declarative navigation and interpretation contracts for review views."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .display import validate_display_manifest
from .visual_labels import (
    anchored_fluorescence_axis_label,
    channels,
    response_axis_label,
)

SelectionScope = Literal["experiment_design", "multi_experiment_design", "review_collection"]
ReductionMode = Literal["selected", "all"]
ConditionMode = Literal["selected", "all"]


@dataclass(frozen=True)
class ReviewViewSpec:
    view_id: str
    label: str
    selection_scope: SelectionScope
    reduction_mode: ReductionMode
    condition_mode: ConditionMode
    premise: str
    decision_value: str
    interpretation: str
    alt_text: str
    non_claim_boundary: str


REVIEW_VIEW_SPECS = (
    ReviewViewSpec(
        view_id="time_series",
        label="Time series and window",
        selection_scope="experiment_design",
        reduction_mode="selected",
        condition_mode="all",
        premise="The selected post-stress interval connects observed trajectories to the eight-value handoff",
        decision_value="Shows whether the reduced response and fluorescence values are supported by sustained, replicate-observed trajectories.",
        interpretation="Solid curves are replicate medians, translucent bands are the central 90% of design wells, and the dashed curve is the same-state {reference_id} fluorescence anchor. Gray shading marks event-time uncertainty; amber marks the selected response window. The lower panels keep bootstrap intervals and event-time sensitivity separate, with observed well points shown only for rᵢ.",
        alt_text="Three aligned square trajectory panels show {growth} growth, {response_axis} response, and {magnitude_ratio} fluorescence in four conditions using replicate medians and central 90% replicate intervals. Two square dot-and-whisker panels show the four response and four {reference_id}-relative fluorescence values produced by the selected post-stress window. Hollow points are observed response-well reductions; anchored fluorescence compares independent design and reference aggregates and therefore has no per-well b points. A compact card records the window and assay support.",
        non_claim_boundary="The selected interval is a prespecified assay summary, not proof of when biology begins to respond.",
    ),
    ReviewViewSpec(
        view_id="multi_experiment_evidence",
        label="Across experiments",
        selection_scope="multi_experiment_design",
        reduction_mode="selected",
        condition_mode="selected",
        premise="The same exact Reader design appears in multiple experiments",
        decision_value="Shows whether experiment-level trajectories and handoff values agree or disagree before the study decides comparability or aggregation.",
        interpretation="Each line style and summary row is one Reader experiment. Trajectories remain on their own sampling grids. Thin colored intervals are replicate-bootstrap uncertainty; thick gray marks are event-time sensitivity. Hollow points are observed response-well reductions, while anchored fluorescence has no fabricated paired-well points.",
        alt_text="Six square panels compare one condition for an exact Reader design across experiments: growth, {response_axis} response, {magnitude_ratio} fluorescence with the {reference_id} anchor, experiment-level response and anchored-fluorescence summaries, and an evidence-boundary card.",
        non_claim_boundary="Reader presents separate experiment evidence and makes no cross-experiment aggregation or comparability decision.",
    ),
    ReviewViewSpec(
        view_id="state_summary",
        label="State handoff values",
        selection_scope="experiment_design",
        reduction_mode="selected",
        condition_mode="all",
        premise="The selected window preserves response and anchored fluorescence by condition",
        decision_value="Shows the eight measured values and their assay-derived uncertainty before study scoring.",
        interpretation="Response is {response_axis}; fluorescence is {anchored_axis}. Hollow points are observed response-well reductions. Colored bootstrap intervals and gray event-time sensitivity marks remain separate; anchored fluorescence has no fabricated per-well b points.",
        alt_text="Two dot-and-whisker panels show four condition-specific {response_axis} responses and four {anchored_axis} fluorescence summaries. The response panel includes observed well reductions. Both panels show asymmetric bootstrap intervals separately from gray event-time sensitivity marks.",
        non_claim_boundary="These assay summaries are not campaign scores or validated responsive promoters.",
    ),
    ReviewViewSpec(
        view_id="measured_response_examples",
        label="Measured response examples",
        selection_scope="review_collection",
        reduction_mode="selected",
        condition_mode="all",
        premise="Measured response examples provide direction checks across all four conditions",
        decision_value="Checks that familiar response examples retain their expected signed directions under the selected reduction.",
        interpretation="Each panel preserves raw condition summaries; no per-design min-max transform is applied.",
        alt_text="Two heatmaps show four-condition {response_axis} response and {anchored_axis} fluorescence for the configured measured response examples and their anchor rows across Reader experiments.",
        non_claim_boundary="SpyP and sulAp are interpretation references, not required or optimal campaign archetypes.",
    ),
    ReviewViewSpec(
        view_id="reduction_sensitivity",
        label="Reduction sensitivity",
        selection_scope="experiment_design",
        reduction_mode="all",
        condition_mode="all",
        premise="Prespecified reductions retain the same condition-level response structure",
        decision_value="Reveals whether a window, integration, or pre-event subtraction choice changes the handoff materially.",
        interpretation="Rows are prespecified reductions; columns are the four assay conditions on explicit response and fluorescence scales.",
        alt_text="Two heatmaps compare the selected design's four {response_axis} responses and four {anchored_axis} fluorescence values across prespecified post-event windows, reduction methods, and response bases.",
        non_claim_boundary="Agreement among reductions does not establish which interval is biologically optimal.",
    ),
    ReviewViewSpec(
        view_id="quality",
        label="Quality and uncertainty",
        selection_scope="experiment_design",
        reduction_mode="selected",
        condition_mode="all",
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


__all__ = ["REVIEW_VIEW_SPECS", "VIEW_LABELS", "ReviewViewSpec", "review_view_spec"]
