"""Publication figure for objective-neutral promoter response evidence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reader.domains.promoter.candidate_bindings import PromoterCandidateBinding
from reader.domains.promoter.sequence_panel import render_candidate_sequence_panel

from .plot_style import LEGEND_SIZE, PANEL_TITLE_SIZE, apply_publication_style
from .promoter_evidence_bundle_contract import PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION
from .promoter_evidence_cards import draw_header_axis
from .promoter_evidence_components import (
    draw_eight_value_handoff_axis,
    draw_trajectory_axis,
)
from .promoter_evidence_handoff_annotations import (
    draw_handoff_family_axis,
    handoff_legend_handles,
)
from .promoter_evidence_overlay import ObjectiveDisplayOverlay
from .review_replicates import response_replicate_rows
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
    objective_overlay: ObjectiveDisplayOverlay | None = None,
) -> tuple[plt.Figure, object]:
    """Render assay trajectories, handoff values, and one BaseRender panel.

    Exact provenance, QC, and any study-issued objective overlay remain in the
    verified bundle manifest instead of competing with assay evidence for
    static figure space.
    """

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
    figure = plt.figure(figsize=(13.2, 9.5), layout="compressed")
    figure.set_gid(PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION)
    grid = figure.add_gridspec(3, 3, height_ratios=(0.30, 2.7, 2.7))
    header_axis = figure.add_subplot(grid[0, :])
    top_grid = grid[1, :].subgridspec(1, 3, width_ratios=(1.0, 1.0, 1.0), wspace=0.015)
    trajectories = [figure.add_subplot(top_grid[0, index]) for index in range(3)]
    lower_grid = grid[2, :].subgridspec(1, 3, width_ratios=(1.0, 0.30, 1.70), wspace=0.0)
    handoff_axis = figure.add_subplot(lower_grid[0, 0])
    handoff_axis.set_anchor("E")
    handoff_family_axis = figure.add_subplot(lower_grid[0, 1])
    sequence_axis = figure.add_subplot(lower_grid[0, 2])
    draw_header_axis(
        header_axis,
        state_labels=state_labels,
        reference_id=reference_id,
    )
    specs = (
        ("growth", "Growth trajectory across conditions", str(channels["growth"])),
        ("response", "Reporter response across conditions", f"log₂({_spaced(channels['response_ratio'])})"),
        (
            "magnitude",
            f"Fluorescence relative to {reference_id}",
            f"log₂({_spaced(channels['magnitude_ratio'])})",
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
            annotate_spans=index == 1,
        )
    draw_eight_value_handoff_axis(
        handoff_axis,
        selected=selected,
        replicate_rows=replicate_rows,
    )
    draw_handoff_family_axis(handoff_family_axis)
    diagnostics = _draw_sequence_axis(sequence_axis, binding=binding)
    figure.legend(
        handles=handoff_legend_handles(
            replicate_stat=str(selected["replicate_stat"]),
            confidence_level=confidence_level,
            event_label=str(display["event_label"]),
        ),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.6,
        handletextpad=0.55,
    )
    figure.suptitle(
        f"Promoter response evidence · {_title_response_summary(selected)}",
        fontsize=13,
        fontweight="semibold",
    )
    figure = apply_publication_style(figure)
    figure.get_layout_engine().set(h_pad=0.025, w_pad=0.025, hspace=0.04, wspace=0.04)
    return figure, diagnostics


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
        target_height_px=480,
        vertical_anchor="center",
        canvas_top_pad_px=0,
    )
    diagnostics = rendered.diagnostics
    image = np.asarray(rendered.image)
    axis.imshow(image)
    axis.set_axis_off()
    axis.set_title(
        f"Measured promoter sequence · {_sequence_title(binding=binding)}",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="semibold",
        pad=5,
    )
    return diagnostics


def _spaced(value: object) -> str:
    return str(value).replace("/", " / ")


def _sequence_title(*, binding: PromoterCandidateBinding) -> str:
    return binding.display_label


def _title_response_summary(selected: pd.Series) -> str:
    return response_summary_label(selected).replace("-", "–", 1)


__all__ = ["PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION", "promoter_evidence_figure"]
