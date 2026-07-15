"""Publication figure for objective-neutral promoter response evidence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reader.domains.promoter.candidate_bindings import PromoterCandidateBinding
from reader.domains.promoter.sequence_panel import render_candidate_sequence_panel

from .plot_style import apply_publication_style
from .promoter_evidence_bundle_contract import PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION
from .promoter_evidence_cards import draw_header_axis, draw_provenance_axis
from .promoter_evidence_components import (
    draw_eight_value_handoff_axis,
    draw_trajectory_axis,
)
from .promoter_evidence_overlay import ObjectiveDisplayOverlay
from .review_replicates import reference_replicate_counts, response_replicate_rows
from .visual_labels import response_summary_label


def promoter_evidence_figure(
    *,
    experiment_id: str,
    design_id: str,
    reduction_id: str,
    selected: pd.Series,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    events: pd.DataFrame,
    display: dict[str, object],
    binding: PromoterCandidateBinding,
    experiment_title: str | None = None,
    objective_overlay: ObjectiveDisplayOverlay | None = None,
) -> tuple[plt.Figure, object]:
    """Render trajectories, handoff values, provenance, and one BaseRender panel."""

    _verify_selection(
        experiment_id=experiment_id,
        design_id=design_id,
        reduction_id=reduction_id,
        binding=binding,
        objective_overlay=objective_overlay,
    )
    event = events.loc[events["experiment_id"].astype(str).eq(experiment_id)]
    if len(event) != 1:
        raise ValueError(f"experiment {experiment_id!r} must have exactly one event record.")
    channels = display.get("channels")
    state_labels = display.get("state_labels")
    if not isinstance(channels, dict) or not isinstance(state_labels, dict):
        raise ValueError("validated response-window display is malformed.")
    confidence_level = float(selected["confidence_level"])
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("response-window confidence level must lie strictly between zero and one.")
    reference_id = str(selected["reference_design_id"])
    uncertainty = float(event.iloc[0]["event_time_uncertainty_h"])
    experiment_traces = traces.loc[traces["experiment_id"].astype(str).eq(experiment_id)].copy()
    replicate_rows = response_replicate_rows(
        selected=selected,
        wells=wells,
        experiment_id=experiment_id,
        design_id=design_id,
        reduction_id=reduction_id,
    )
    reference_counts = reference_replicate_counts(
        selected=selected,
        wells=wells,
        experiment_id=experiment_id,
        reduction_id=reduction_id,
    )

    figure_height, lower_row_height, evidence_heights = _layout_for_overlay(objective_overlay)
    figure = plt.figure(figsize=(13.2, figure_height), constrained_layout=True)
    figure.set_gid(PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION)
    grid = figure.add_gridspec(3, 3, height_ratios=(0.44, 2.7, lower_row_height))
    header_axis = figure.add_subplot(grid[0, :])
    trajectories = [figure.add_subplot(grid[1, index]) for index in range(3)]
    handoff_axis = figure.add_subplot(grid[2, 0])
    evidence_grid = grid[2, 1:].subgridspec(2, 1, height_ratios=evidence_heights)
    sequence_axis = figure.add_subplot(evidence_grid[0, 0])
    provenance_axis = figure.add_subplot(evidence_grid[1, 0])
    draw_header_axis(
        header_axis,
        experiment_label=experiment_title or experiment_id,
        reduction_label=response_summary_label(selected),
        state_labels=state_labels,
        reference_id=reference_id,
    )
    specs = (
        ("growth", "A  Growth by condition", str(channels["growth"])),
        ("response", "B  YFP / CFP response", f"log2({_spaced(channels['response_ratio'])})"),
        (
            "magnitude",
            f"C  YFP / OD600 with {reference_id} anchor",
            f"log2({_spaced(channels['magnitude_ratio'])})",
        ),
    )
    for index, (axis, (signal_kind, title, ylabel)) in enumerate(zip(trajectories, specs, strict=True)):
        draw_trajectory_axis(
            axis,
            traces=experiment_traces,
            signal_kind=signal_kind,
            design_id=design_id,
            reference_id=reference_id,
            confidence_level=confidence_level,
            uncertainty=uncertainty,
            selected=selected,
            event_label=str(display["event_label"]),
            title=title,
            ylabel=ylabel,
            annotate_spans=index == 0,
        )
    draw_eight_value_handoff_axis(
        handoff_axis,
        selected=selected,
        display=display,
        replicate_rows=replicate_rows,
        reference_counts=reference_counts,
    )
    draw_provenance_axis(
        provenance_axis,
        binding=binding,
        experiment_id=experiment_id,
        design_id=design_id,
        reduction_id=reduction_id,
        objective_overlay=objective_overlay,
    )
    diagnostics = _draw_sequence_axis(sequence_axis, binding=binding)
    figure.suptitle(
        _compact_title(binding=binding),
        fontsize=12,
        fontweight="semibold",
    )
    return apply_publication_style(figure), diagnostics


def _verify_selection(
    *,
    experiment_id: str,
    design_id: str,
    reduction_id: str,
    binding: PromoterCandidateBinding,
    objective_overlay: ObjectiveDisplayOverlay | None,
) -> None:
    if binding.reader_design_id != design_id:
        raise ValueError(
            f"promoter-evidence binding {binding.reader_design_id!r} disagrees with Reader design {design_id!r}."
        )
    if objective_overlay is not None and (
        objective_overlay.experiment_id != experiment_id
        or objective_overlay.reader_design_id != design_id
        or objective_overlay.reduction_id != reduction_id
    ):
        raise ValueError("objective display overlay selection disagrees with the promoter-evidence selection.")


def _draw_sequence_axis(axis: plt.Axes, *, binding: PromoterCandidateBinding) -> object:
    rendered = render_candidate_sequence_panel(
        binding,
        style_profile="promoter_compact_slide.v1",
        target_width_px=2200,
        target_height_px=310,
        vertical_anchor="center",
        canvas_top_pad_px=0,
    )
    diagnostics = rendered.diagnostics
    image = np.asarray(rendered.image)
    axis.imshow(image)
    _focus_sequence_content(axis, image)
    axis.set_axis_off()
    title = {
        "densegen_tfbs": "E  DenseGen TFBS annotation",
        "usr_genbank_annotations_v1": "E  GenBank source annotation",
    }[binding.baserender_adapter_kind]
    axis.set_title(title, loc="left", fontsize=10, fontweight="semibold")
    return diagnostics


def _focus_sequence_content(axis: plt.Axes, image: np.ndarray) -> None:
    """Remove BaseRender canvas padding without resampling sequence evidence."""

    if image.ndim != 3 or image.shape[2] < 3:
        return
    visible = np.any(image[..., :3] < 248, axis=2)
    if image.shape[2] >= 4:
        visible &= image[..., 3] > 8
    min_row_ink = max(12, int(round(image.shape[1] * 0.01)))
    rows = np.flatnonzero(visible.sum(axis=1) >= min_row_ink)
    if not len(rows):
        rows = np.flatnonzero(visible.any(axis=1))
    if not len(rows):
        return
    height = image.shape[0]
    y_pad = max(8, int(round(height * 0.04)))
    top = max(0, int(rows.min()) - y_pad)
    bottom = min(height - 1, int(rows.max()) + y_pad)
    axis.set_ylim(bottom + 0.5, top - 0.5)


def _spaced(value: object) -> str:
    return str(value).replace("/", " / ")


def _compact_title(*, binding: PromoterCandidateBinding) -> str:
    return f"Promoter response evidence · {binding.display_label}"


def _layout_for_overlay(
    overlay: ObjectiveDisplayOverlay | None,
) -> tuple[float, float, tuple[float, float]]:
    if overlay is None:
        return 9.4, 2.7, (1.55, 0.45)
    if len(overlay.components) <= 3:
        return 9.8, 3.0, (1.65, 1.0)
    return 10.6, 3.6, (1.55, 2.05)


__all__ = ["PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION", "promoter_evidence_figure"]
