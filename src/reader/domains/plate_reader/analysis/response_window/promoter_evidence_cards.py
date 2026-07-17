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
    experiment_label: str,
    reduction_label: str,
    state_labels: dict[str, object],
    reference_id: str,
) -> None:
    axis.set_axis_off()
    axis.set_gid("promoter-evidence-header")
    axis.text(
        0.01,
        0.12,
        f"Experiment  {experiment_label}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#334155",
    )
    axis.text(
        0.99,
        0.12,
        f"Response summary  {reduction_label}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#334155",
    )
    axis.legend(
        handles=_figure_legend(state_labels=state_labels),
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=7.2,
        borderaxespad=0,
    )
    axis.text(
        0.5,
        0.42,
        f"Trajectory style  solid/filled = selected design · dashed/hollow = {reference_id} anchor",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color="#475569",
    )


def draw_provenance_axis(
    axis: plt.Axes,
    *,
    binding: PromoterCandidateBinding,
    experiment_id: str,
    design_id: str,
    reduction_id: str,
    objective_overlay: ObjectiveDisplayOverlay | None,
) -> None:
    axis.set_axis_off()
    axis.set_title("F  Provenance and QC", loc="left", fontsize=10, fontweight="semibold")
    axis.add_patch(
        Rectangle((0.01, 0.04), 0.98, 0.88, transform=axis.transAxes, fill=False, edgecolor="#cbd5e1", linewidth=0.9)
    )
    lines = [
        " · ".join(
            (
                f"Reader experiment  {_compact(experiment_id, width=32)}",
                f"design  {_compact(design_id, width=28)}",
            )
        ),
        " · ".join(
            (
                "Objective-neutral evidence",
                f"Candidate  {_compact(binding.candidate_id, width=24)}",
                f"Source  {_compact(f'{binding.source_class} / {binding.design_family}', width=34)}",
                f"Binding  {binding.binding_method.replace('_', ' ')}",
            )
        ),
        " · ".join(
            (
                f"Reduction  {_compact(reduction_id, width=32)}",
                f"Sequence  {binding.sequence_sha256[:12]}…",
                "Objective scoring stays outside Reader",
            )
        ),
    ]
    if objective_overlay is not None:
        label = (
            "RMF"
            if objective_overlay.objective_id == "response_magnitude_feasibility_v1"
            else objective_overlay.objective_id.replace("_", " ")
        )
        lines.append(f"{label} raw components · {objective_overlay.claim_status.replace('_', ' ')}")
        lines.extend(
            f"{_compact(item.label, width=34)}  {item.value:g} {_compact(item.unit, width=54)}"
            for item in objective_overlay.components
        )
    axis.text(
        0.04,
        0.84,
        "\n".join(lines),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=5.9 if objective_overlay is not None else 7.2,
        linespacing=1.18 if objective_overlay is not None else 1.34,
    )


def _figure_legend(
    *,
    state_labels: dict[str, object],
) -> list[Line2D]:
    return [
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


def _compact(value: str, *, width: int) -> str:
    return value if len(value) <= width else value[: width - 1] + "…"


__all__ = ["draw_header_axis", "draw_provenance_axis"]
