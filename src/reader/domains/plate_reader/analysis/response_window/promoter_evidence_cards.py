"""Header and provenance cards for the promoter-evidence figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from reader.domains.promoter.candidate_bindings import PromoterCandidateBinding

from .promoter_evidence_overlay import ObjectiveDisplayOverlay
from .sources import STATE_ORDER
from .visual_labels import STATE_COLORS, STATE_MARKERS


def draw_header_axis(
    axis: plt.Axes,
    *,
    experiment_id: str,
    reduction_id: str,
    binding: PromoterCandidateBinding,
    state_labels: dict[str, object],
    reference_id: str,
    confidence_level: float,
) -> None:
    axis.set_axis_off()
    axis.set_gid("promoter-evidence-header")
    axis.text(
        0.01,
        0.12,
        f"Experiment  {experiment_id}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#334155",
    )
    axis.text(
        0.99,
        0.12,
        f"Reduction  {reduction_id}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#334155",
    )
    axis.legend(
        handles=_figure_legend(
            state_labels=state_labels,
            reference_id=reference_id,
            confidence_level=confidence_level,
        ),
        loc="upper center",
        ncol=8,
        frameon=False,
        fontsize=7.2,
        borderaxespad=0,
    )


def draw_provenance_axis(
    axis: plt.Axes,
    *,
    binding: PromoterCandidateBinding,
    reduction_id: str,
    objective_overlay: ObjectiveDisplayOverlay | None,
) -> None:
    axis.set_axis_off()
    axis.set_title("E  Provenance and QC", loc="left", fontsize=10, fontweight="semibold")
    axis.add_patch(
        Rectangle((0.01, 0.03), 0.98, 0.90, transform=axis.transAxes, fill=False, edgecolor="#cbd5e1", linewidth=0.9)
    )
    lines = [
        "Objective-neutral evidence",
        f"Candidate  {_compact(binding.candidate_id, width=22)}",
        f"Source  {_compact(f'{binding.source_class} · {binding.design_family}', width=36)}",
        f"Reduction  {_compact(reduction_id, width=34)}",
        f"Binding  {binding.binding_method.replace('_', ' ')}",
        f"Sequence  {binding.sequence_sha256[:12]}…",
        "RMF is not calculated by Reader",
    ]
    if objective_overlay is not None:
        label = (
            "RMF"
            if objective_overlay.objective_id == "response_magnitude_feasibility_v1"
            else objective_overlay.objective_id
        )
        lines.append(f"{label} raw components · {objective_overlay.claim_status.replace('_', ' ')}")
        lines.extend(f"{item.label}  {item.value:g} {item.unit}" for item in objective_overlay.components)
    axis.text(
        0.06,
        0.86,
        "\n".join(lines),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=6.8 if objective_overlay is not None else 8,
        linespacing=1.28 if objective_overlay is not None else 1.45,
    )


def _figure_legend(
    *,
    state_labels: dict[str, object],
    reference_id: str,
    confidence_level: float,
) -> list[Line2D]:
    handles = [
        Line2D(
            [],
            [],
            color=STATE_COLORS[state],
            marker=STATE_MARKERS[state],
            linewidth=1.8,
            label=str(state_labels[state]),
        )
        for state in STATE_ORDER
    ]
    handles.extend(
        [
            Line2D([], [], color="#111827", linestyle="--", linewidth=1.3, label=f"{reference_id} anchor"),
            Line2D([], [], color="#2563eb", linewidth=6, alpha=0.18, label=f"Central {confidence_level:.0%}"),
            Line2D([], [], color="#2563eb", marker="|", markersize=9, linestyle="", label="Bootstrap SD"),
            Line2D([], [], color="#9ca3af", linewidth=6, alpha=0.38, label="Event-time sensitivity"),
        ]
    )
    return handles


def _compact(value: str, *, width: int) -> str:
    return value if len(value) <= width else value[: width - 1] + "…"


__all__ = ["draw_header_axis", "draw_provenance_axis"]
