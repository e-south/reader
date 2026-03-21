from __future__ import annotations

import io
import textwrap
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from reader.domains.plate_reader.plots.common import annotate_points_smart, shared_numeric_limits
from reader.domains.plate_reader.plots.retron_sponge import (
    build_retron_decomposition_frame,
    plot_retron_sponge_summary,
    plot_retron_sponge_trace,
)
from reader.domains.plate_reader.plots.time_series import plot_time_series
from reader.workbench.notebooks import context as notebook_context
from reader.workbench.records import discover_dataframe_records

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "relevant", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "irrelevant", "off"}
_PLOT_STAGE_ORDER = ("1. QC", "2. Assay kinetics", "3. Ranking and overview")
_FAMILY_ORDER = {"mono": 0, "bi": 1, "tri": 2, "quad": 3, "control": 4}
_FAMILY_COLOR_MAP = {
    "mono": "#0072B2",
    "bi": "#E69F00",
    "tri": "#009E73",
    "quad": "#CC79A7",
    "control": "#6f6f6f",
    "other": "#56B4E9",
}
_FINGERPRINT_FRAME_COLUMNS = [
    "selected_sponge",
    "sensor",
    "stress_condition",
    "source_experiment_id",
    "source_label",
    "comparison_group",
    "sponge",
    "sponge_family_size",
    "value",
]

load_notebook_workbench_context = notebook_context.load_notebook_workbench_context

_RETRON_EXPERIMENT_PLOT_CACHE: dict[
    tuple[str, tuple[tuple[str, int], ...]],
    RetronNotebookPlotResult,
] = {}
_RETRON_AGGREGATE_PLOT_CACHE: dict[
    tuple[str, int, tuple[tuple[str, tuple[str, ...]], ...], str, str, str, str | None],
    RetronAggregatePlotResult,
] = {}

_RETRON_SOURCE_PLOT_SELECTOR_TAGS = {
    "control_burden_panel": "R,mu",
    "baseline_shifted_kinetics": "B",
    "matched_control_kinetics": "C",
    "induced_effect_kinetics": "D",
    "absolute_effect_kinetics": "D_abs",
    "control_anchored_decomposition": "R/P_pre/D_abs",
    "interaction_summary": "C_AUC/C_END",
    "library_heatmaps": "S_abs_AUC/S_AUC/P_pre",
    "stress_modulation_scores": "M_AUC",
    "pareto_ranking": "S_abs_AUC vs burden",
}

_RETRON_SOURCE_PLOT_SELECTOR_TITLES = {
    "raw_kinetics": "QC traces",
    "control_burden_panel": "tetO burden",
    "baseline_shifted_kinetics": "Baseline-shifted kinetics",
    "matched_control_kinetics": "tetO-subtracted kinetics",
    "induced_effect_kinetics": "IPTG-state effect",
    "absolute_effect_kinetics": "Absolute matched-control effect",
    "control_anchored_decomposition": "Decision cards",
    "interaction_summary": "IPTG x stress summary",
    "library_heatmaps": "Library heatmaps",
    "stress_modulation_scores": "Stress-gated score",
    "pareto_ranking": "Pareto ranking",
}

_RETRON_PLOT_GUIDE = {
    "raw_kinetics": {
        "title": "QC traces",
        "stage": "1. QC",
        "question": "Are the raw channels and support ratios internally consistent before any matched-control subtraction?",
        "math": "Raw OD600(t), YFP(t), and CFP(t) from overflow/df, shown beside YFP/OD600, CFP/OD600, and YFP/CFP from the downstream ratio table.",
        "record": "overflow/df for raw OD600/YFP/CFP; ratio_yfp_od600/df for support ratios",
        "meaning": "Use this view to spot plate-junction shifts, saturated channels, or reporter-specific artifacts before reading any normalized summary.",
    },
    "support_kinetics": {
        "title": "QC traces",
        "stage": "1. QC",
        "question": "Do both fluorescence channels move with biomass, or is one channel showing a sheet-boundary offset or reporter-specific shift?",
        "math": "Support ratios (YFP/OD600, CFP/OD600, YFP/CFP) from ratio_yfp_od600/df, interpreted beside raw OD600/YFP/CFP from overflow/df.",
        "record": "ratio_yfp_od600/df for support ratios; overflow/df for raw OD600/YFP/CFP context",
        "meaning": "Read support ratios beside the raw channels to separate broad physiology shifts from reporter-specific movement. Overflow-capped fluorescence rows can flatten a channel without implying a ratio bug.",
    },
    "control_burden_panel": {
        "title": "tetO burden panel",
        "stage": "1. QC",
        "question": "How much movement comes from IPTG-driven retron expression alone?",
        "math": "tetO-only traces over R(t)=log2(YFP/CFP) and mu(t)=d ln(OD600) / dt.",
        "record": "semantic_metrics/trace",
        "meaning": "Use the tetO traces as the burden baseline across the full run before attributing movement to a real sponge site.",
    },
    "baseline_shifted_kinetics": {
        "title": "Baseline-shifted kinetics",
        "stage": "2. Assay kinetics",
        "question": "After removing each well's own pre-stress offset, how do the trajectories compare over time?",
        "math": "B(t)=R(t)-R_pre, where R_pre is the mean of the last three pre-stress reads.",
        "record": "semantic_metrics/trace",
        "meaning": "Helpful for comparing post-stress motion, but it intentionally hides absolute preload before stress.",
    },
    "matched_control_kinetics": {
        "title": "Matched-control-normalized kinetics",
        "stage": "2. Assay kinetics",
        "question": "After same-sensor tetO subtraction, which sponge arms still depart from the matched control across the full run?",
        "math": "C(t)=B(t)-mean(B matched tetO at same sensor, plate, stress, IPTG, and time).",
        "record": "semantic_metrics/trace",
        "meaning": "Compare pre-stress offsets and post-stress movement arm by arm after tetO subtraction; this is the clearest view for seeing whether a deviation is already present before stress.",
    },
    "induced_effect_kinetics": {
        "title": "IPTG-state effect kinetics",
        "stage": "2. Assay kinetics",
        "question": "Within each sponge arm, how does the +IPTG versus -IPTG gap evolve once baseline shift and tetO matching are applied?",
        "math": "D(t)=mean(C +IPTG)-mean(C -IPTG) within each sensor and stress state.",
        "record": "semantic_metrics/trace",
        "meaning": "Use this view for incremental effects over the displayed trace window. The small sidecar bar chart reduces the same trace to D_AUC by stress state, so the figure shows both the trajectory and the score it feeds.",
    },
    "absolute_effect_kinetics": {
        "title": "Absolute matched-control effect kinetics",
        "stage": "2. Assay kinetics",
        "question": "If the sample already differs before stress, does an absolute matched-control IPTG gap remain over the full run?",
        "math": "D_abs(t)=mean(R-R_tetO,matched)(+IPTG)-mean(R-R_tetO,matched)(-IPTG) within each sensor and stress state.",
        "record": "semantic_metrics/trace",
        "meaning": "Keeps same-sensor tetO correction while preserving pre-stress offsets that the incremental D(t) view removes. The sidecar bar chart shows D_abs_AUC directly so preload-sensitive traces and their reduction stay linked.",
    },
    "control_anchored_decomposition": {
        "title": "Decision cards",
        "stage": "2. Assay kinetics",
        "question": "Does the real sponge move the reporter beyond matched tetO, and is that signal preload-driven, post-stress, or costly?",
        "math": "Relevant-stress and H2O R(t)=log2(YFP/CFP) traces are shown beside summary intervals for P_pre, D_abs_AUC, D_AUC, and D_growth_AUC.",
        "record": "semantic_metrics/trace",
        "meaning": "Use this as the primary decision surface. It keeps the observed traces, the total matched-control effect, the post-stress increment, the preload shift, and the burden cost in one place.",
    },
    "interaction_summary": {
        "title": "IPTG and stress state summary",
        "stage": "3. Ranking and overview",
        "question": "Across the four assay states, is the signal dominated by IPTG state, stress state, or their combination?",
        "math": "C_AUC or C_END across the four IPTG/stress states: H2O/-IPTG, H2O/+IPTG, stress/-IPTG, stress/+IPTG.",
        "record": "semantic_metrics/trace + semantic_metrics/summary",
        "meaning": "Compact state summary after matched-control normalization. Compare it with the full-time trace views before overinterpreting one state pattern.",
    },
    "library_heatmaps": {
        "title": "Library heatmaps",
        "stage": "3. Ranking and overview",
        "question": "Across the library, which pairs have strong absolute on-target effect, how much of that is post-stress incremental, and which hits are preload-heavy?",
        "math": "Relevant-stress heatmaps over S_abs_AUC, S_AUC, and P_pre.",
        "record": "semantic_metrics/summary",
        "meaning": "Lead with the absolute score to answer whether a sponge works at all, then use the incremental and preload panels to separate stress-gated hits from preload-driven hits.",
    },
    "stress_modulation_scores": {
        "title": "Stress modulation scores",
        "stage": "3. Ranking and overview",
        "question": "Which on-target effects become materially stronger once the relevant pathway is stressed?",
        "math": "M_AUC=AUC(D(relevant stress)-D(H2O)).",
        "record": "semantic_metrics/summary",
        "meaning": "Ranks how strongly stress unmasks a sponge effect.",
    },
    "pareto_ranking": {
        "title": "Pareto ranking",
        "stage": "3. Ranking and overview",
        "question": "Which candidates balance strong absolute on-target effect with low burden and low leakiness?",
        "math": "Absolute on-target score S_abs_AUC versus construct-specific burden (D_growth_AUC by default), with |L_pre| encoded as point size.",
        "record": "semantic_metrics/summary",
        "meaning": "Balances total effect against burden and leakiness for candidate selection.",
    },
}

_RETRON_AGGREGATE_PLOT_GUIDE = {
    "specificity_matrix": {
        "title": "Target activity matrix",
        "question": "Across the tested on-target sensor arms, where is activity concentrated for each sponge design?",
        "math": "Relevant-stress on-target pivot over S_AUC or O_AUC across tested sponge x sensor pairs.",
        "meaning": "Shows how mono, bi, tri, and quad sponges distribute activity across the sensor arms they were actually tested against, without implying exhaustive off-target specificity coverage.",
    },
    "pareto_ranking": {
        "title": "Pareto ranking",
        "question": "Which sponge designs stay strong after burden and leakiness are considered across the review set?",
        "math": "Mean on-target score versus mean construct-specific burden, with absolute leakiness encoded as point size.",
        "meaning": "Ranks candidates across the full review set instead of within a single source experiment.",
    },
    "architecture_plot": {
        "title": "Architecture plot",
        "question": "Does adding extra motifs preserve, dilute, or redistribute the intended sponge effect?",
        "math": "Relevant-stress S_AUC or O_AUC versus motif_count or irrelevant_motif_count, faceted by sensor.",
        "meaning": "Tests whether extra motifs preserve, dilute, or redistribute the relevant sponge arm.",
    },
    "expected_vs_observed": {
        "title": "Expected vs observed multifunction performance",
        "question": "Do multifunctional designs behave additively, better than additive, or worse than additive?",
        "math": "Observed multifunction score versus the mono-derived expected_sum or expected_best_single baseline.",
        "meaning": "Separates additive multifunction behavior from dilution or synergy.",
    },
    "sponge_fingerprint": {
        "title": "Sponge fingerprint",
        "question": "For one multi-functional sponge, which intended sensor arms are strong and which are weak?",
        "math": "Selected multifunction sponge plotted beside the matched tetO reference across relevant sensors over S_AUC or O_AUC, with source-level points when available.",
        "meaning": "Shows whether a multi-functional sponge is balanced across its intended sensor arms and how far each arm moves away from the matched tetO baseline.",
    },
}

_RETRON_ASSAY_CONTEXT = (
    {
        "Topic": "Design unit",
        "Details": "Each genotype is one sensor plasmid plus one sponge plasmid measured in a 2x2 H2O or stress by -IPTG or +IPTG design.",
    },
    {
        "Topic": "Timing",
        "Details": "IPTG is present in the starting media and sets the retron-expression state from the start of the run. The t=0 boundary in kinetics plots marks stress addition and the plate-sheet junction, not IPTG addition or sponge induction.",
    },
    {
        "Topic": "Matched control",
        "Details": "Every real sponge row is normalized to the same-sensor tetO control on the same plate, stress state, IPTG state, and timepoint.",
    },
    {
        "Topic": "Primary score",
        "Details": "The reader-facing ladder is observed R(t)=log2(YFP/CFP), then preload shift P_pre, total effect D_abs_AUC, post-stress increment D_AUC, burden D_growth_AUC, and absolute scaled ranking S_abs_AUC.",
    },
    {
        "Topic": "Decision logic",
        "Details": "Strong candidates combine positive on-target effect with low burden and low leakiness instead of relying on one summary value alone.",
    },
)

_RETRON_TRANSFORM_LADDER = (
    {
        "Step": "Raw channels",
        "Formula": "OD600(t), CFP(t), YFP(t)",
        "Output": "raw QC only",
        "Meaning": "Check growth, saturation, drift, and failed wells before assay scoring.",
    },
    {
        "Step": "Support channels",
        "Formula": "YFP/OD600 and CFP/OD600",
        "Output": "support QC only",
        "Meaning": "Contextualize growth-linked channel shifts without replacing the dual-reporter score.",
    },
    {
        "Step": "Primary ratio",
        "Formula": "R(t)=log2(YFP/CFP)",
        "Output": "trace metric R",
        "Meaning": "Primary within-well reporter score for dual-reporter retron screens.",
    },
    {
        "Step": "Pre-stress baseline",
        "Formula": "R_pre=mean(last 3 pre-stress reads of R)",
        "Output": "summary metric R_pre",
        "Meaning": "Defines each well's baseline before the stress pulse.",
    },
    {
        "Step": "Preload shift",
        "Formula": "P_pre=delta_IPTG[R_pre-R_pre,tetO,matched]",
        "Output": "summary metric P_pre",
        "Meaning": "Captures the matched-control preload already present before stress addition.",
    },
    {
        "Step": "Baseline shift",
        "Formula": "B(t)=R(t)-R_pre",
        "Output": "trace metric B",
        "Meaning": "Removes pre-stress offsets while preserving post-stress dynamics.",
    },
    {
        "Step": "Matched tetO normalization",
        "Formula": "C(t)=B(t)-mean(B matched tetO at same sensor, plate, stress, IPTG, and time)",
        "Output": "trace metric C",
        "Meaning": "Subtracts same-sensor control behavior on the same plate.",
    },
    {
        "Step": "IPTG-state effect",
        "Formula": "D(t)=mean(C +IPTG)-mean(C -IPTG)",
        "Output": "trace metric D; summary D_AUC and D_END",
        "Meaning": "Compares matched-control-normalized +IPTG and -IPTG states within each stress condition; because IPTG is present from the start, this is a state contrast rather than a t=0 induction pulse.",
    },
    {
        "Step": "Absolute matched-control effect",
        "Formula": "D_abs(t)=mean(R-R_tetO,matched)(+IPTG)-mean(R-R_tetO,matched)(-IPTG)",
        "Output": "trace metric D_abs; summary D_abs_AUC and D_abs_END",
        "Meaning": "Keeps same-sensor tetO subtraction but preserves pre-stress preload differences that D(t) intentionally removes.",
    },
    {
        "Step": "Stress modulation",
        "Formula": "M(t)=D(relevant stress)-D(H2O)",
        "Output": "trace metric M; summary M_AUC and M_END",
        "Meaning": "Measures how much stress reveals or amplifies the sponge effect.",
    },
    {
        "Step": "Sign correction",
        "Formula": "O(t)=expected_decoy_sign * D(t)",
        "Output": "trace metric O; summary O_AUC",
        "Meaning": "Makes stronger expected effects point in the same direction across sensors.",
    },
    {
        "Step": "Absolute sign correction",
        "Formula": "O_abs(t)=expected_decoy_sign * D_abs(t)",
        "Output": "trace metric O_abs; summary O_abs_AUC",
        "Meaning": "Keeps preload-sensitive total effects aligned in the expected direction across sensors.",
    },
    {
        "Step": "Cross-sensor scaling",
        "Formula": "S_abs_AUC=O_abs_AUC / abs(G_sensor); S_AUC=O_AUC / abs(G_sensor)",
        "Output": "summary metrics S_abs_AUC and S_AUC",
        "Meaning": "Separates the total absolute effect from the specifically post-stress incremental component while keeping cross-sensor comparisons comparable.",
    },
    {
        "Step": "Leakiness and burden",
        "Formula": "L_pre, L_post_AUC, D_growth_AUC, T_ratio_AUC, T_growth_AUC, T_finalOD",
        "Output": "summary metrics",
        "Meaning": "Separates strong hits from leaky constructs and distinguishes sponge-specific burden from assay-context tetO burden.",
    },
)

_RETRON_AGGREGATE_FIGURES = (
    {
        "Figure": "Target activity matrix",
        "Math": "Cross-run pivot over relevant-stress S_AUC or O_AUC for tested on-target sensor/sponge pairs.",
        "Why": "Shows how mono, bi, tri, and quad sponges distribute activity across the sensor arms they were actually tested against.",
    },
    {
        "Figure": "Pareto ranking",
        "Math": "Mean on-target score versus mean construct-specific burden, with absolute leakiness encoded as point size.",
        "Why": "Ranks candidate sponge designs across the full review set instead of within one source experiment.",
    },
    {
        "Figure": "Architecture plot",
        "Math": "Relevant-stress S_AUC or O_AUC versus motif_count or irrelevant_motif_count.",
        "Why": "Tests whether extra motifs preserve, dilute, or redistribute on-target activity.",
    },
    {
        "Figure": "Observed vs expected multifunction",
        "Math": "Observed multi-site score versus mono-derived expected score (best single or sum of relevant mono arms).",
        "Why": "Separates additive behavior from dilution or synergy in multifunctional sponges.",
    },
    {
        "Figure": "Sponge fingerprint",
        "Math": "Selected sponge versus matched tetO reference across sensors over relevant-stress S_AUC or O_AUC, with source-level points when available.",
        "Why": "Shows whether a multi-functional sponge is balanced across its intended sensor arms and whether that signal sits above the matched tetO baseline.",
    },
)

_RETRON_FIGURE_COVERAGE = (
    {
        "Figure": "Figure 1 — Raw kinetics QC",
        "Scope": "Per experiment",
        "Surface": "raw_kinetics",
        "Coverage": "Exact compiled plot",
        "Math": "OD600(t), YFP(t), CFP(t), YFP/OD600, CFP/OD600, and YFP/CFP.",
    },
    {
        "Figure": "Figure 2 — Support ratios per OD",
        "Scope": "Per experiment",
        "Surface": "raw_kinetics",
        "Coverage": "Combined into the QC trace surface",
        "Math": "Support ratios are shown in the same QC panel as the raw channels.",
    },
    {
        "Figure": "Figure 3 — tetO burden panel",
        "Scope": "Per experiment",
        "Surface": "control_burden_panel",
        "Coverage": "Exact compiled plot",
        "Math": "tetO-only R(t) and mu(t) traces.",
    },
    {
        "Figure": "Figure 4 — Pre-stress baseline plot",
        "Scope": "Per experiment",
        "Surface": "assay summary review",
        "Coverage": "Derived from assay tables",
        "Math": "R_pre and L_pre from the derived assay summary table.",
    },
    {
        "Figure": "Figure 5 — Raw ratio kinetics by sensor",
        "Scope": "Per experiment",
        "Surface": "assay trace review",
        "Coverage": "Derived from assay tables",
        "Math": "R(t)=log2(YFP/CFP) from the derived assay trace table.",
    },
    {
        "Figure": "Figure 6 — Baseline-shifted kinetics",
        "Scope": "Per experiment",
        "Surface": "baseline_shifted_kinetics",
        "Coverage": "Exact compiled plot",
        "Math": "B(t)=R(t)-R_pre.",
    },
    {
        "Figure": "Figure 7 — tetO-normalized kinetics",
        "Scope": "Per experiment",
        "Surface": "matched_control_kinetics",
        "Coverage": "Exact compiled plot",
        "Math": "C(t)=B(t)-mean(B matched tetO).",
    },
    {
        "Figure": "Figure 8 — IPTG-state effect over time",
        "Scope": "Per experiment",
        "Surface": "induced_effect_kinetics",
        "Coverage": "Exact compiled plot",
        "Math": "D(t)=mean(C +IPTG)-mean(C -IPTG).",
    },
    {
        "Figure": "Figure 9 — Control-anchored traces",
        "Scope": "Per experiment",
        "Surface": "control_anchored_decomposition",
        "Coverage": "Exact compiled plot",
        "Math": "Relevant-stress R(t) traces for the selected sponge and matched tetO under +/-IPTG, with the supporting table reducing the primary window to D_abs_AUC.",
    },
    {
        "Figure": "Figure 10 — 2x2 interaction summary",
        "Scope": "Per experiment",
        "Surface": "interaction_summary",
        "Coverage": "Exact compiled plot",
        "Math": "C_AUC or C_END across the four IPTG/stress states.",
    },
    {
        "Figure": "Figure 11 — Library heatmaps",
        "Scope": "Per experiment",
        "Surface": "library_heatmaps",
        "Coverage": "Exact compiled plot",
        "Math": "D_AUC, M_AUC, and scaled ranking heatmaps.",
    },
    {
        "Figure": "Figure 12 — Leakiness panel",
        "Scope": "Per experiment",
        "Surface": "assay summary review",
        "Coverage": "Derived from assay tables",
        "Math": "L_pre and L_post_AUC from the derived assay summary table.",
    },
    {
        "Figure": "Figure 13 — Target activity matrix",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "Relevant-stress O_AUC or S_AUC pivoted over sponge x sensor.",
    },
    {
        "Figure": "Figure 14 — Pareto ranking",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "On-target score versus burden, with leakiness encoded.",
    },
    {
        "Figure": "Figure 15 — Architecture plot",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "Relevant-stress O_AUC or S_AUC versus motif complexity.",
    },
    {
        "Figure": "Figure 16 — Observed versus expected multi-functional performance",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "Observed multi-site score versus mono-derived expected score.",
    },
    {
        "Figure": "Figure 17 — Sponge-centric fingerprint plots",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "Selected multifunction sponge versus matched tetO reference across relevant sensors over O_AUC or S_AUC.",
    },
    {
        "Figure": "Figure 18 — Growth impact summary",
        "Scope": "Per experiment",
        "Surface": "control_burden_panel plus assay summary review",
        "Coverage": "Derived from assay tables",
        "Math": "mu(t), T_growth_AUC, and T_finalOD burden summaries.",
    },
    {
        "Figure": "Figure 18 — Plate-position heatmaps",
        "Scope": "Per experiment",
        "Surface": "raw well layout follow-on",
        "Coverage": "Not first-class compiled yet",
        "Math": "Endpoint OD600, CFP, R(t), or C_AUC mapped to well coordinates.",
    },
)


@dataclass(frozen=True)
class RetronReviewSource:
    label: str
    experiment_id: str
    experiment_root: Path | None
    config_path: Path | None
    summary_path: Path
    trace_path: Path


@dataclass(frozen=True)
class RetronReviewBundle:
    manifest_path: Path
    sources: tuple[RetronReviewSource, ...]
    summary_df: pd.DataFrame
    trace_df: pd.DataFrame
    relevant_stress_map: dict[str, str]
    sensor_target_map: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class RetronReviewSourceSurface:
    experiment_title: str
    protocol_id: str
    plot_specs: tuple[dict[str, Any], ...]
    plot_selector_rows: tuple[dict[str, str], ...]
    plot_catalog_rows: tuple[dict[str, str], ...]
    record_paths: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RetronNotebookPlotResult:
    plot_id: str
    title: str
    stage: str
    question: str
    math: str
    meaning: str
    source_record: str
    figures: tuple[Any, ...]
    supporting_table: pd.DataFrame
    supporting_table_title: str


@dataclass(frozen=True)
class RetronAggregatePlotResult:
    plot_id: str
    title: str
    question: str
    math: str
    meaning: str
    figure: Any | None
    supporting_table: pd.DataFrame
    supporting_table_title: str


@dataclass(frozen=True)
class _AggregatePlotContext:
    summary_df: pd.DataFrame
    sensor_target_map: Mapping[str, tuple[str, ...]]
    score_metric: str
    architecture_x: str
    expected_mode: str
    fingerprint_sponge: str | None


@dataclass(frozen=True)
class _AggregatePlotPayload:
    figure: Any | None
    supporting_table: pd.DataFrame
    supporting_table_title: str


@dataclass(frozen=True)
class _FingerprintFigurePayload:
    selected_sponge: str
    sensor_levels: tuple[str, ...]
    comparison_order: tuple[str, ...]
    stats: pd.DataFrame
    y_limits: tuple[float, float]
    max_sources: int
    width: float
    offsets: dict[str, float]
    x_positions: dict[str, float]
    comparison_colors: dict[str, str]
    edge_colors: dict[str, str]
    point_facecolors: dict[str, str]


@dataclass(frozen=True)
class _AggregateParetoFigurePayload:
    family_levels: tuple[str, ...]
    color_map: dict[str, Any]
    sizes: pd.Series


@dataclass(frozen=True)
class _SourceSurfacePlotRows:
    selector_row: dict[str, str]
    catalog_row: dict[str, str]


@dataclass(frozen=True)
class _ResolvedSourcePaths:
    experiment_root: Path | None
    config_path: Path | None
    summary_path: Path
    trace_path: Path


def retron_transform_ladder_rows() -> list[dict[str, str]]:
    return [dict(row) for row in _RETRON_TRANSFORM_LADDER]


def retron_aggregate_figure_rows() -> list[dict[str, str]]:
    return [dict(row) for row in _RETRON_AGGREGATE_FIGURES]


def retron_aggregate_plot_rows(plot_ids: list[str] | None = None) -> list[dict[str, str]]:
    selected = list(plot_ids or _RETRON_AGGREGATE_PLOT_GUIDE)
    rows: list[dict[str, str]] = []
    for plot_id in selected:
        guide = _RETRON_AGGREGATE_PLOT_GUIDE.get(str(plot_id))
        rows.append(
            {
                "Plot id": str(plot_id),
                "Figure": (guide or {}).get("title", str(plot_id)),
                "Math / transform": (guide or {}).get("math", "Notebook-local aggregate view."),
                "How to read": (guide or {}).get(
                    "meaning", "Interpret against the direct-ratio matched-control workflow."
                ),
            }
        )
    return rows


def build_label_value_options(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_key: str,
    value_key: str,
    disambiguator_key: str | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(label_key, "")).strip()
        counts[label] = counts.get(label, 0) + 1
    options: dict[str, Any] = {}
    for row in rows:
        label = str(row.get(label_key, "")).strip()
        if counts.get(label, 0) > 1:
            disambiguator = str(row.get(disambiguator_key or value_key, "")).strip()
            label = f"{label} [{disambiguator}]"
        if label in options:
            raise ValueError(f"retron_review: duplicate selector label {label!r}")
        options[label] = row.get(value_key)
    return options


def retron_table_kwargs(
    *,
    page_size: int | None = None,
    pagination: bool | None = None,
    wrapped_columns: Sequence[str] | None = None,
    max_height: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "selection": None,
        "show_column_summaries": False,
        "show_data_types": False,
        "show_download": False,
    }
    if page_size is not None:
        kwargs["page_size"] = page_size
    if pagination is not None:
        kwargs["pagination"] = pagination
    if wrapped_columns:
        kwargs["wrapped_columns"] = list(wrapped_columns)
    if max_height is not None:
        kwargs["max_height"] = max_height
    return kwargs


def retron_figure_coverage_rows() -> list[dict[str, str]]:
    return [dict(row) for row in _RETRON_FIGURE_COVERAGE]


def retron_plot_guide_rows(plot_ids: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for plot_id in plot_ids:
        guide = _RETRON_PLOT_GUIDE.get(str(plot_id))
        rows.append(
            {
                "Stage": (guide or {}).get("stage", _PLOT_STAGE_ORDER[-1]),
                "Plot": (guide or {}).get("title", str(plot_id).replace("_", " ")),
                "Plot id": str(plot_id),
                "Math / transform": (guide or {}).get("math", "Protocol-specific transform guide not registered."),
                "Source record": (guide or {}).get("record", "see compiled plot spec"),
                "How to read": (guide or {}).get(
                    "meaning", "Interpret in the context of the compiled assay semantics."
                ),
            }
        )
    rows.sort(key=lambda row: (_plot_stage_rank(row["Stage"]), row["Plot"]))
    return rows


def retron_plot_rendered_files(plots_dir: Path, *, plot_id: str, plugin: str) -> list[str]:
    patterns = [f"{plot_id}*.pdf"]
    if str(plot_id) == "raw_kinetics" and str(plugin) == "plot/time_series":
        # Support legacy raw-kinetics filenames generated before the explicit filename contract.
        patterns.append("ts_*.pdf")
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(path.name for path in plots_dir.glob(pattern))
    return sorted(set(matches))


def retron_assay_context_rows() -> list[dict[str, str]]:
    return [dict(row) for row in _RETRON_ASSAY_CONTEXT]


def load_retron_semantic_maps_from_config(
    config_path: Path,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    analysis = ((payload.get("protocol") or {}).get("analysis") or {}).get("semantic_metrics") or {}
    if not isinstance(analysis, dict):
        raise ValueError("retron_review: protocol.analysis.semantic_metrics must be a mapping")
    return (
        _normalize_relevant_stress_map(analysis.get("relevant_stress_map") or {}),
        _normalize_sensor_target_map(analysis.get("sensor_target_map") or {}),
    )


def load_cached_parquet_frame(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _load_cached_parquet_frame(str(resolved), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=64)
def _load_cached_parquet_frame(path: str, mtime_ns: int, size_bytes: int) -> pd.DataFrame:
    del mtime_ns, size_bytes
    return pd.read_parquet(path)


def retron_sensor_context_rows(
    relevant_stress_map: Mapping[str, str],
    sensor_target_map: Mapping[str, tuple[str, ...]],
) -> list[dict[str, str]]:
    sensors = sorted({str(key) for key in relevant_stress_map} | {str(key) for key in sensor_target_map})
    rows: list[dict[str, str]] = []
    for sensor in sensors:
        motifs = sensor_target_map.get(sensor, ())
        rows.append(
            {
                "Sensor": sensor,
                "Relevant stress": str(relevant_stress_map.get(sensor, "not declared")),
                "Relevant motifs": ", ".join(motifs) if motifs else "not declared",
            }
        )
    return rows


def load_retron_source_surface(source: RetronReviewSource) -> RetronReviewSourceSurface:
    if source.config_path is None:
        raise ValueError(f"retron_review: source {source.label!r} has no config path for scoped review")
    config_path = source.config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"retron_review: source config not found for scoped review: {config_path}")
    return _load_retron_source_surface(str(config_path))


def retron_visible_plot_specs(plot_specs: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    redundant_plot_ids = _redundant_retron_surface_plot_ids(plot_specs)
    return tuple(dict(spec) for spec in plot_specs if str(spec.get("id", "")) not in redundant_plot_ids)


def _load_retron_source_surface(path: str) -> RetronReviewSourceSurface:
    source_context = load_notebook_workbench_context(Path(path))
    plot_specs = _visible_source_plot_specs(source_context)
    plot_selector_rows, plot_catalog_rows = _source_surface_plot_rows(
        source_context=source_context, plot_specs=plot_specs
    )
    record_paths = _source_surface_record_paths(source_context.outputs_dir)
    return RetronReviewSourceSurface(
        experiment_title=source_context.decl.experiment.title or source_context.decl.experiment.id,
        protocol_id=source_context.decl.experiment_semantics.protocol.id,
        plot_specs=plot_specs,
        plot_selector_rows=tuple(plot_selector_rows),
        plot_catalog_rows=tuple(plot_catalog_rows),
        record_paths=record_paths,
    )


def _visible_source_plot_specs(source_context: Any) -> tuple[dict[str, Any], ...]:
    return retron_visible_plot_specs(tuple(spec.to_dict() for spec in source_context.workbench.plots))


def _source_surface_plot_rows(
    *,
    source_context: Any,
    plot_specs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    plot_guides = {row["Plot id"]: row for row in retron_plot_guide_rows([spec.get("id", "") for spec in plot_specs])}
    rows = [
        _source_surface_plot_row(source_context=source_context, plot_spec=plot_spec, plot_guides=plot_guides)
        for plot_spec in plot_specs
    ]
    plot_selector_rows = sorted((row.selector_row for row in rows), key=lambda item: (item["Stage"], item["Plot"]))
    plot_catalog_rows = sorted((row.catalog_row for row in rows), key=lambda item: (item["Stage"], item["Plot"]))
    return plot_selector_rows, plot_catalog_rows


def _source_surface_plot_row(
    *,
    source_context: Any,
    plot_spec: Mapping[str, Any],
    plot_guides: Mapping[str, Mapping[str, str]],
) -> _SourceSurfacePlotRows:
    plot_id = str(plot_spec.get("id", ""))
    guide = plot_guides.get(plot_id, {})
    stage = _source_surface_plot_stage(guide)
    title = _source_surface_plot_title(plot_id=plot_id, plot_spec=plot_spec, guide=guide)
    rendered = _source_surface_rendered(source_context=source_context, plot_id=plot_id, plot_spec=plot_spec)
    return _SourceSurfacePlotRows(
        selector_row={
            "Selector label": _retron_source_plot_selector_label(plot_id=plot_id, title=title),
            "Stage": stage,
            "Plot": title,
            "Plot id": plot_id,
        },
        catalog_row={
            "Stage": stage,
            "Plot": title,
            "Plot id": plot_id,
            "Rendered": rendered,
            "Math / transform": str(
                guide.get("Math / transform", "Interpret against the direct-ratio matched-control workflow.")
            ),
            "How to read": str(guide.get("How to read", "See the transform ladder for the exact semantics.")),
        },
    )


def _source_surface_plot_stage(guide: Mapping[str, str]) -> str:
    return str(guide.get("Stage", "3. Ranking and overview"))


def _source_surface_plot_title(
    *,
    plot_id: str,
    plot_spec: Mapping[str, Any],
    guide: Mapping[str, str],
) -> str:
    return str(guide.get("Plot", (plot_spec.get("with") or {}).get("title", plot_id)))


def _source_surface_rendered(
    *,
    source_context: Any,
    plot_id: str,
    plot_spec: Mapping[str, Any],
) -> str:
    rendered_files = retron_plot_rendered_files(
        source_context.plots_dir,
        plot_id=plot_id,
        plugin=str(plot_spec.get("plugin", "")),
    )
    return "yes" if rendered_files else "no"


def _source_surface_record_paths(outputs_dir: Path) -> tuple[tuple[str, str], ...]:
    record_info, _, _, _ = discover_dataframe_records(outputs_dir, allow_scan=False)
    return tuple(
        sorted(
            (
                str(info.get("record_id")),
                str(Path(info["path"]).expanduser().resolve()),
            )
            for info in record_info.values()
            if info.get("record_id") and info.get("path")
        )
    )


def _redundant_retron_surface_plot_ids(plot_specs: Sequence[Mapping[str, Any]]) -> set[str]:
    plot_ids = {str(spec.get("id", "")) for spec in plot_specs}
    redundant: set[str] = set()
    if {"raw_kinetics", "support_kinetics"}.issubset(plot_ids):
        redundant.add("support_kinetics")
    redundant.update({"baseline_shifted_kinetics", "stress_modulation_scores", "pareto_ranking"} & plot_ids)
    return redundant


def load_retron_review_bundle(
    manifest_path: Path,
    *,
    source_root: Path | None = None,
) -> RetronReviewBundle:
    payload = _load_manifest_payload(manifest_path)
    sources = _resolve_sources(
        manifest_path,
        payload,
        source_root=source_root.expanduser().resolve() if source_root is not None else None,
    )
    if not sources:
        raise ValueError("retron_review: review manifest must declare at least one source entry")
    relevant_stress_map, sensor_target_map = _resolve_semantic_maps(payload, sources=sources)
    summary_frames = []
    trace_frames = []
    for source in sources:
        summary_frame = _read_semantic_table(source.summary_path, kind="summary")
        trace_frame = _read_semantic_table(source.trace_path, kind="trace")
        summary_frames.append(_annotate_source(summary_frame, source=source))
        trace_frames.append(_annotate_source(trace_frame, source=source))
    return RetronReviewBundle(
        manifest_path=manifest_path.resolve(),
        sources=tuple(sources),
        summary_df=pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
        trace_df=pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame(),
        relevant_stress_map=relevant_stress_map,
        sensor_target_map=sensor_target_map,
    )


def retron_source_selector_rows(bundle: RetronReviewBundle) -> list[dict[str, str | int]]:
    counts: dict[str, int] = {}
    for source in bundle.sources:
        label = str(source.label)
        counts[label] = counts.get(label, 0) + 1
    rows: list[dict[str, str | int]] = []
    for idx, source in enumerate(bundle.sources):
        label = str(source.label)
        selector_label = label if counts.get(label, 0) == 1 else f"{label} • {source.experiment_id}"
        rows.append({"Selector label": selector_label, "Index": idx})
    return rows


def source_rows(bundle: RetronReviewBundle) -> list[dict[str, str]]:
    return [
        {
            "Label": source.label,
            "Experiment": source.experiment_id,
            "Config": str(source.config_path) if source.config_path is not None else "manifest-only",
            "Summary export": str(source.summary_path),
            "Trace export": str(source.trace_path),
        }
        for source in bundle.sources
    ]


def retron_source_surface_overview_rows(
    source: RetronReviewSource,
    surface: RetronReviewSourceSurface,
) -> list[dict[str, str]]:
    return [
        {"Field": "Source label", "Value": source.label},
        {"Field": "Experiment", "Value": surface.experiment_title or source.experiment_id},
        {"Field": "Protocol", "Value": surface.protocol_id},
        {"Field": "Compiled plots", "Value": str(len(surface.plot_catalog_rows))},
        {"Field": "Dataframe records", "Value": str(len(surface.record_paths))},
    ]


def _retron_source_plot_selector_label(*, plot_id: str, title: str) -> str:
    display_title = str(_RETRON_SOURCE_PLOT_SELECTOR_TITLES.get(str(plot_id), title)).strip() or str(title)
    tag = str(_RETRON_SOURCE_PLOT_SELECTOR_TAGS.get(str(plot_id), "")).strip()
    if not tag:
        return display_title
    return f"[{tag}] {display_title}"


def retron_figure_option_rows(figures: list[Any] | tuple[Any, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for figure in figures:
        filename = str(getattr(figure, "filename", ""))
        rows.append(
            {
                "Filename": filename,
                "Label": retron_figure_label(filename),
            }
        )
    return rows


def retron_figure_label(filename: str) -> str:
    value = str(filename or "").strip()
    if "__sensor=" in value:
        return value.split("__sensor=", 1)[1].replace("_", " ")
    if "__design_id_alias=" in value:
        return value.split("__design_id_alias=", 1)[1].replace("_", " ")
    if "__design_id=" in value:
        return value.split("__design_id=", 1)[1].replace("_", " ")
    return value.replace("_", " ")


def filter_supporting_table_for_figure(table: pd.DataFrame, *, filename: str | None) -> pd.DataFrame:
    if table.empty or not filename:
        return table
    frame = table.copy()
    scope_tokens = {
        "sensor": _scope_token(filename, key="sensor"),
        "design_id_alias": _scope_token(filename, key="design_id_alias"),
        "design_id": _scope_token(filename, key="design_id"),
    }
    for column, token in scope_tokens.items():
        if token is None or column not in frame.columns:
            continue
        mask = frame[column].astype(str).map(_slug) == _slug(token)
        if not mask.any():
            mask = frame[column].astype(str) == token
        if mask.any():
            frame = frame.loc[mask].copy()
    return frame.reset_index(drop=True)


def retron_notebook_table_preview(table: pd.DataFrame | None, *, max_rows: int = 500) -> pd.DataFrame | None:
    if table is None:
        return None
    frame = table.reset_index(drop=True)
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    return frame.head(max_rows).copy()


def contextualize_retron_plot_copy(
    *,
    question: str,
    math: str,
    meaning: str,
    supporting_table: pd.DataFrame | None,
    relevant_stress_map: Mapping[str, str] | None = None,
    no_stress_label: str = "H2O",
) -> dict[str, str]:
    stress_phrase = _infer_relevant_stress_phrase(
        supporting_table=supporting_table,
        relevant_stress_map=relevant_stress_map,
        no_stress_label=no_stress_label,
    )
    if stress_phrase is None:
        return {"question": question, "math": math, "meaning": meaning}
    return {
        "question": _replace_relevant_stress_text(question, stress_phrase),
        "math": _replace_relevant_stress_text(math, stress_phrase),
        "meaning": _replace_relevant_stress_text(meaning, stress_phrase),
    }


def _infer_relevant_stress_phrase(
    *,
    supporting_table: pd.DataFrame | None,
    relevant_stress_map: Mapping[str, str] | None,
    no_stress_label: str,
) -> str | None:
    sensor_stress_pairs: dict[str, str] = {}
    if supporting_table is not None and not supporting_table.empty:
        frame = supporting_table.copy()
        if "stress_condition" in frame.columns:
            stress_rows = frame[frame["stress_condition"].notna()].copy()
            stress_rows["stress_condition"] = stress_rows["stress_condition"].astype(str)
            stress_rows = stress_rows[
                stress_rows["stress_condition"].str.strip().ne("")
                & stress_rows["stress_condition"].str.strip().ne(str(no_stress_label))
            ]
            if "sensor" in stress_rows.columns:
                for sensor, values in stress_rows.groupby(stress_rows["sensor"].astype(str), sort=True)[
                    "stress_condition"
                ]:
                    unique_values = sorted(
                        {value for value in values.astype(str) if value and value != str(no_stress_label)}
                    )
                    if len(unique_values) == 1:
                        sensor_stress_pairs[str(sensor)] = unique_values[0]
            unique_stresses = sorted({value for value in stress_rows["stress_condition"].astype(str) if value})
            if len(unique_stresses) == 1:
                return unique_stresses[0]
        if not sensor_stress_pairs and {"treatment", "treatment_alias"}.issubset(frame.columns):
            stress_rows = frame[
                frame["treatment_alias"].astype(str).str.contains(r"\+stress", regex=True, na=False)
            ].copy()
            stress_rows["__stress_label"] = stress_rows["treatment"].map(_extract_stress_from_treatment)
            stress_rows = stress_rows[stress_rows["__stress_label"].notna()].copy()
            if "sensor" in stress_rows.columns:
                for sensor, values in stress_rows.groupby(stress_rows["sensor"].astype(str), sort=True)[
                    "__stress_label"
                ]:
                    unique_values = sorted({str(value) for value in values if pd.notna(value)})
                    if len(unique_values) == 1:
                        sensor_stress_pairs[str(sensor)] = unique_values[0]
            unique_stresses = sorted({str(value) for value in stress_rows["__stress_label"] if pd.notna(value)})
            if len(unique_stresses) == 1:
                return unique_stresses[0]
    if not sensor_stress_pairs and relevant_stress_map:
        sensor_stress_pairs = {
            str(sensor): str(stress) for sensor, stress in relevant_stress_map.items() if str(stress).strip()
        }
    if not sensor_stress_pairs:
        return None
    unique_stresses = sorted(set(sensor_stress_pairs.values()))
    if len(unique_stresses) == 1:
        return unique_stresses[0]
    ordered_pairs = "; ".join(f"{sensor}: {sensor_stress_pairs[sensor]}" for sensor in sorted(sensor_stress_pairs))
    return f"sensor-matched stress ({ordered_pairs})"


def _extract_stress_from_treatment(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split(",")]
    stress_parts = [part for part in parts if "IPTG" not in part]
    if not stress_parts:
        return None
    stress = ", ".join(part for part in stress_parts if part and part != "H2O").strip()
    return stress or None


def _replace_relevant_stress_text(text: str, stress_phrase: str) -> str:
    updated = str(text)
    replacements = (
        ("Relevant-stress", stress_phrase),
        ("relevant-stress", stress_phrase),
        ("Relevant stress", stress_phrase),
        ("relevant stress", stress_phrase),
    )
    for source, target in replacements:
        updated = updated.replace(source, target)
    return updated


def figure_to_download_bytes(figure: Any, *, fmt: str) -> bytes:
    buffer = io.BytesIO()
    export_format = str(fmt).lower()
    figure.savefig(
        buffer,
        format=export_format,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
        dpi=240 if export_format == "png" else None,
    )
    return buffer.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def download_safe_stem(value: str) -> str:
    stem = str(value or "").strip()
    if not stem:
        return "retron_review"
    return _slug(stem).replace("/", "_") or "retron_review"


def render_retron_experiment_plot(
    plot_spec: Mapping[str, Any],
    *,
    datasets: Mapping[str, pd.DataFrame],
) -> RetronNotebookPlotResult:
    plot_id = str(plot_spec.get("id") or "").strip()
    if not plot_id:
        raise ValueError("retron_review: plot spec is missing an id")
    metadata = _plot_guide_metadata(plot_id)
    with_cfg = dict(plot_spec.get("with") or {})
    figures, supporting_table, supporting_table_title = _render_experiment_plot_payload(
        plot_spec=plot_spec,
        datasets=datasets,
    )

    styled_figures = tuple(_prepare_notebook_plot_figure(item) for item in figures)
    return RetronNotebookPlotResult(
        plot_id=plot_id,
        title=str(with_cfg.get("title") or metadata["title"]),
        stage=metadata["stage"],
        question=metadata["question"],
        math=metadata["math"],
        meaning=metadata["meaning"],
        source_record=metadata["record"],
        figures=styled_figures,
        supporting_table=supporting_table.reset_index(drop=True),
        supporting_table_title=supporting_table_title,
    )


def render_retron_experiment_plot_cached(
    plot_spec: Mapping[str, Any],
    *,
    datasets: Mapping[str, pd.DataFrame],
) -> RetronNotebookPlotResult:
    plot_id = str(plot_spec.get("id") or "").strip()
    cache_key = (
        plot_id,
        tuple(sorted((str(record_id), id(frame)) for record_id, frame in datasets.items())),
    )
    cached = _RETRON_EXPERIMENT_PLOT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = render_retron_experiment_plot(plot_spec, datasets=datasets)
    _RETRON_EXPERIMENT_PLOT_CACHE[cache_key] = result
    return result


def render_retron_source_plot_cached(
    source: RetronReviewSource,
    *,
    plot_id: str,
) -> RetronNotebookPlotResult:
    surface = load_retron_source_surface(source)
    plot_id_value = str(plot_id).strip()
    plot_spec = next((spec for spec in surface.plot_specs if str(spec.get("id", "")) == plot_id_value), None)
    if plot_spec is None:
        raise ValueError(f"retron_review: unknown scoped plot id {plot_id_value!r} for source {source.label!r}")
    record_paths = dict(surface.record_paths)
    datasets: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for read_ref in dict(plot_spec.get("reads") or {}).values():
        record_id = str((read_ref or {}).get("record", "")).strip()
        if not record_id or record_id in datasets:
            continue
        record_path = record_paths.get(record_id)
        if record_path is None:
            errors.append(f"Missing dataframe record `{record_id}` for the selected source plot.")
            continue
        try:
            datasets[record_id] = load_cached_parquet_frame(record_path)
        except Exception as exc:
            errors.append(f"Failed to load `{record_id}`: {exc}")
    if plot_id_value in {"raw_kinetics", "support_kinetics"}:
        overflow_path = record_paths.get("overflow/df")
        if overflow_path is not None and "overflow/df" not in datasets:
            try:
                datasets["overflow/df"] = load_cached_parquet_frame(overflow_path)
            except Exception as exc:
                errors.append(f"Failed to load `overflow/df`: {exc}")
    if errors:
        raise ValueError(" ".join(errors))
    return render_retron_experiment_plot_cached(plot_spec, datasets=datasets)


def render_retron_aggregate_plot(
    plot_id: str,
    *,
    summary_df: pd.DataFrame,
    sensor_target_map: Mapping[str, tuple[str, ...]],
    score_metric: str,
    architecture_x: str,
    expected_mode: str,
    fingerprint_sponge: str | None,
) -> RetronAggregatePlotResult:
    selected_plot_id = str(plot_id)
    guide = _aggregate_plot_guide_metadata(selected_plot_id)
    figure, supporting_table, supporting_table_title = _render_aggregate_plot_payload(
        plot_id=selected_plot_id,
        summary_df=summary_df,
        sensor_target_map=sensor_target_map,
        score_metric=score_metric,
        architecture_x=architecture_x,
        expected_mode=expected_mode,
        fingerprint_sponge=fingerprint_sponge,
    )

    return RetronAggregatePlotResult(
        plot_id=selected_plot_id,
        title=guide["title"],
        question=guide["question"],
        math=guide["math"],
        meaning=guide["meaning"],
        figure=_style_notebook_figure(figure) if figure is not None else None,
        supporting_table=supporting_table.reset_index(drop=True),
        supporting_table_title=supporting_table_title,
    )


def _render_experiment_plot_payload(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> tuple[list[Any], pd.DataFrame, str]:
    plugin = str(plot_spec.get("plugin") or "").strip()
    if plugin == "plot/time_series":
        return _render_time_series_notebook_payload(plot_spec=plot_spec, datasets=datasets)
    if plugin == "plot/retron_trace":
        return _render_trace_notebook_payload(plot_spec=plot_spec, datasets=datasets)
    if plugin == "plot/retron_summary":
        return _render_summary_notebook_payload(plot_spec=plot_spec, datasets=datasets)
    raise ValueError(f"retron_review: unsupported notebook plot plugin {plugin!r}")


def _render_time_series_notebook_payload(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> tuple[list[Any], pd.DataFrame, str]:
    plot_id = str(plot_spec.get("id") or "").strip()
    with_cfg = dict(plot_spec.get("with") or {})
    if plot_id in {"raw_kinetics", "support_kinetics"}:
        qc_df = _retron_qc_dataframe(plot_spec=plot_spec, datasets=datasets)
        figures = _render_retron_qc_plot_spec(plot_spec=plot_spec, datasets=datasets)
        channels = ["OD600", "YFP", "CFP", "YFP/OD600", "CFP/OD600", "YFP/CFP"]
        supporting_table = _time_series_supporting_table(qc_df, channels=channels)
        title = "Underlying overflow-handled raw channel rows plus derived support-ratio rows for the selected QC view"
        return figures, supporting_table, title
    figures = _render_time_series_plot_spec(plot_spec=plot_spec, datasets=datasets)
    channels = _normalize_optional_str_list(with_cfg.get("y")) or _normalize_optional_str_list(with_cfg.get("channels"))
    supporting_table = _time_series_supporting_table(
        _require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="df"),
        channels=channels,
    )
    return figures, supporting_table, "Underlying tidy rows for the selected raw or support channels"


def _render_trace_notebook_payload(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> tuple[list[Any], pd.DataFrame, str]:
    with_cfg = dict(plot_spec.get("with") or {})
    figures = _render_retron_trace_plot_spec(plot_spec=plot_spec, datasets=datasets)
    supporting_table = _trace_supporting_table(
        _require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="trace"),
        metrics=_normalize_optional_str_list(with_cfg.get("metrics")) or [],
    )
    return figures, supporting_table, "Underlying assay trace rows for the selected kinetic transform"


def _render_summary_notebook_payload(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> tuple[list[Any], pd.DataFrame, str]:
    with_cfg = dict(plot_spec.get("with") or {})
    figures = _render_retron_summary_plot_spec(plot_spec=plot_spec, datasets=datasets)
    view_id = str(with_cfg.get("view") or "")
    if view_id == "decomposition":
        supporting_table = build_retron_decomposition_frame(
            _require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="trace"),
            control_name=str(with_cfg.get("control_name") or "tetO"),
            relevant_only=bool(with_cfg.get("relevant_only", True)),
        )
        return figures, supporting_table, "Primary-window decomposition behind the decision-card view"
    supporting_table = _summary_supporting_table(
        _require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="summary"),
        view=view_id,
        metric=str(with_cfg.get("metric") or ""),
        burden_metric=str(with_cfg.get("burden_metric") or "D_growth_AUC"),
    )
    return figures, supporting_table, "Underlying assay summary rows for the selected ranking view"


def _render_aggregate_plot_payload(
    *,
    plot_id: str,
    summary_df: pd.DataFrame,
    sensor_target_map: Mapping[str, tuple[str, ...]],
    score_metric: str,
    architecture_x: str,
    expected_mode: str,
    fingerprint_sponge: str | None,
) -> tuple[Any | None, pd.DataFrame, str]:
    payload = _aggregate_payload_builder(plot_id)(
        _AggregatePlotContext(
            summary_df=summary_df,
            sensor_target_map=sensor_target_map,
            score_metric=score_metric,
            architecture_x=architecture_x,
            expected_mode=expected_mode,
            fingerprint_sponge=fingerprint_sponge,
        )
    )
    return payload.figure, payload.supporting_table, payload.supporting_table_title


def _aggregate_payload_builder(plot_id: str) -> Callable[[_AggregatePlotContext], _AggregatePlotPayload]:
    builders: dict[str, Callable[[_AggregatePlotContext], _AggregatePlotPayload]] = {
        "specificity_matrix": _specificity_matrix_payload,
        "pareto_ranking": _aggregate_pareto_payload,
        "architecture_plot": _architecture_plot_payload,
        "expected_vs_observed": _expected_vs_observed_payload,
        "sponge_fingerprint": _sponge_fingerprint_payload,
    }
    try:
        return builders[str(plot_id)]
    except KeyError as exc:
        raise ValueError(f"retron_review: unknown aggregate plot id {plot_id!r}") from exc


def _specificity_matrix_payload(context: _AggregatePlotContext) -> _AggregatePlotPayload:
    matrix = build_specificity_matrix(context.summary_df, score_metric=context.score_metric)
    supporting_table = matrix.reset_index().rename(columns={"index": "sponge"})
    return _AggregatePlotPayload(
        figure=_build_specificity_matrix_figure(matrix=matrix, score_metric=context.score_metric),
        supporting_table=supporting_table,
        supporting_table_title="Relevant-stress on-target matrix behind the heatmap",
    )


def _aggregate_pareto_payload(context: _AggregatePlotContext) -> _AggregatePlotPayload:
    supporting_table = build_aggregate_pareto_frame(context.summary_df, score_metric=context.score_metric)
    return _AggregatePlotPayload(
        figure=_build_aggregate_pareto_figure(pareto_df=supporting_table, score_metric=context.score_metric),
        supporting_table=supporting_table,
        supporting_table_title="Aggregate on-target, burden, and leakiness table for candidate ranking",
    )


def _architecture_plot_payload(context: _AggregatePlotContext) -> _AggregatePlotPayload:
    supporting_table = build_architecture_frame(
        context.summary_df,
        sensor_target_map=dict(context.sensor_target_map),
        score_metric=context.score_metric,
    )
    return _AggregatePlotPayload(
        figure=_build_architecture_figure(
            architecture_df=supporting_table,
            score_metric=context.score_metric,
            architecture_x=context.architecture_x,
        ),
        supporting_table=supporting_table,
        supporting_table_title="Architecture score table behind the sensor-faceted scatter plots",
    )


def _expected_vs_observed_payload(context: _AggregatePlotContext) -> _AggregatePlotPayload:
    supporting_table = build_expected_vs_observed_frame(
        context.summary_df,
        sensor_target_map=dict(context.sensor_target_map),
        score_metric=context.score_metric,
    )
    return _AggregatePlotPayload(
        figure=_build_expected_vs_observed_figure(
            expected_vs_observed_df=supporting_table,
            score_metric=context.score_metric,
            expected_mode=context.expected_mode,
        ),
        supporting_table=supporting_table,
        supporting_table_title="Observed and mono-derived expected scores for multifunctional sponges",
    )


def _sponge_fingerprint_payload(context: _AggregatePlotContext) -> _AggregatePlotPayload:
    supporting_table = build_fingerprint_frame(
        context.summary_df,
        score_metric=context.score_metric,
        fingerprint_sponge=context.fingerprint_sponge,
    )
    return _AggregatePlotPayload(
        figure=_build_fingerprint_figure(
            fingerprint_df=supporting_table,
            score_metric=context.score_metric,
        ),
        supporting_table=supporting_table,
        supporting_table_title="Relevant-sensor score table for the selected multifunctional sponge and its source-matched tetO references",
    )


def render_retron_aggregate_plot_cached(
    plot_id: str,
    *,
    summary_df: pd.DataFrame,
    sensor_target_map: Mapping[str, tuple[str, ...]],
    score_metric: str,
    architecture_x: str,
    expected_mode: str,
    fingerprint_sponge: str | None,
) -> RetronAggregatePlotResult:
    cache_key = (
        str(plot_id),
        id(summary_df),
        tuple(
            sorted((str(sensor), tuple(str(item) for item in targets)) for sensor, targets in sensor_target_map.items())
        ),
        str(score_metric),
        str(architecture_x),
        str(expected_mode),
        fingerprint_sponge,
    )
    cached = _RETRON_AGGREGATE_PLOT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = render_retron_aggregate_plot(
        plot_id,
        summary_df=summary_df,
        sensor_target_map=sensor_target_map,
        score_metric=score_metric,
        architecture_x=architecture_x,
        expected_mode=expected_mode,
        fingerprint_sponge=fingerprint_sponge,
    )
    _RETRON_AGGREGATE_PLOT_CACHE[cache_key] = result
    return result


def build_specificity_matrix(
    summary_df: pd.DataFrame,
    *,
    score_metric: str = "S_AUC",
) -> pd.DataFrame:
    scores = aggregate_on_target_scores(summary_df, score_metric=score_metric)
    if scores.empty:
        return pd.DataFrame()
    pivot = scores.pivot_table(index="sensor", columns="sponge", values="value", aggfunc="mean")
    if pivot.empty:
        return pivot
    row_order = sorted(pivot.index.tolist())
    col_order = sorted(pivot.columns.tolist(), key=_sponge_sort_key)
    return pivot.reindex(index=row_order, columns=col_order)


def build_architecture_frame(
    summary_df: pd.DataFrame,
    *,
    sensor_target_map: dict[str, tuple[str, ...]],
    score_metric: str = "S_AUC",
) -> pd.DataFrame:
    scores = aggregate_on_target_scores(summary_df, score_metric=score_metric)
    if scores.empty:
        return scores
    frame = scores.copy()
    frame["motif_count"] = frame["sponge"].map(_motif_count)
    frame["relevant_motif_count"] = frame.apply(
        lambda row: _relevant_motif_count(str(row["sensor"]), str(row["sponge"]), sensor_target_map=sensor_target_map),
        axis=1,
    )
    frame["irrelevant_motif_count"] = frame["motif_count"] - frame["relevant_motif_count"]
    return frame.sort_values(["sensor", "motif_count", "sponge"], kind="stable").reset_index(drop=True)


def build_expected_vs_observed_frame(
    summary_df: pd.DataFrame,
    *,
    sensor_target_map: dict[str, tuple[str, ...]],
    score_metric: str = "S_AUC",
) -> pd.DataFrame:
    scores = aggregate_on_target_scores(summary_df, score_metric=score_metric)
    if scores.empty:
        return pd.DataFrame()
    mono_lookup = {
        (str(row["sensor"]), str(row["sponge"])): float(row["value"])
        for _, row in scores[scores["sponge_family_size"].astype(str) == "mono"].iterrows()
    }
    rows: list[dict[str, Any]] = []
    multi = scores[scores["sponge_family_size"].astype(str).isin({"bi", "tri", "quad"})]
    for _, row in multi.iterrows():
        sensor = str(row["sensor"])
        sponge = str(row["sponge"])
        relevant_motifs = _relevant_motifs(sensor=sensor, sponge=sponge, sensor_target_map=sensor_target_map)
        mono_scores = [mono_lookup[(sensor, motif)] for motif in relevant_motifs if (sensor, motif) in mono_lookup]
        if not mono_scores:
            continue
        rows.append(
            {
                "sensor": sensor,
                "sponge": sponge,
                "observed": float(row["value"]),
                "expected_best_single": max(mono_scores),
                "expected_sum": float(sum(mono_scores)),
                "relevant_motif_count": len(relevant_motifs),
                "sponge_family_size": row["sponge_family_size"],
            }
        )
    return pd.DataFrame(rows).sort_values(["sensor", "sponge"], kind="stable").reset_index(drop=True)


def build_fingerprint_frame(
    summary_df: pd.DataFrame,
    *,
    score_metric: str = "S_AUC",
    fingerprint_sponge: str | None = None,
    control_name: str = "tetO",
) -> pd.DataFrame:
    frame = _normalized_retron_summary_frame(
        summary_df,
        required={"sensor", "sponge", "metric", "value", "is_relevant_stress", "sponge_family_size"},
    )
    sample_rows = _fingerprint_sample_rows(frame, score_metric=score_metric)
    available = sorted({str(value) for value in sample_rows["sponge"].dropna()}, key=_sponge_sort_key)
    if not available:
        return pd.DataFrame(columns=_FINGERPRINT_FRAME_COLUMNS)
    selected_sponge = _select_fingerprint_sponge(available, fingerprint_sponge=fingerprint_sponge)
    sample_rows = _group_fingerprint_rows(sample_rows[sample_rows["sponge"] == selected_sponge].copy())
    if sample_rows.empty:
        return pd.DataFrame(columns=_FINGERPRINT_FRAME_COLUMNS)
    control_rows = _group_fingerprint_rows(
        frame[
            (frame["metric"] == str(score_metric))
            & frame["is_relevant_stress"].fillna(False)
            & (frame["sponge"] == str(control_name))
            & frame["sensor"].isin(sample_rows["sensor"].astype(str))
        ].copy()
    ).rename(
        columns={
            "value": "control_value",
            "sponge": "control_sponge",
            "sponge_family_size": "control_family_size",
        }
    )
    paired_rows = _pair_fingerprint_rows(sample_rows, control_rows)
    out = _build_fingerprint_long_frame(
        paired_rows,
        selected_sponge=selected_sponge,
        control_name=control_name,
    )
    return _sorted_fingerprint_frame(out)


def _normalized_retron_summary_frame(summary_df: pd.DataFrame, *, required: set[str]) -> pd.DataFrame:
    missing = sorted(required - set(summary_df.columns))
    if missing:
        raise ValueError(f"retron_review: summary dataframe is missing required columns: {missing}")
    frame = summary_df.copy()
    for column in ("metric", "sponge", "sensor", "sponge_family_size"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    for column in ("relevant_sensor_pair", "is_relevant_stress"):
        if column in frame.columns:
            frame[column] = _coerce_optional_bool_series(frame[column], label=column)
    return frame


def _fingerprint_sample_rows(frame: pd.DataFrame, *, score_metric: str) -> pd.DataFrame:
    sample_rows = frame[
        (frame["metric"] == str(score_metric))
        & frame["is_relevant_stress"].fillna(False)
        & frame["sponge_family_size"].isin({"bi", "tri", "quad"})
    ].copy()
    if "relevant_sensor_pair" in sample_rows.columns:
        sample_rows = sample_rows[sample_rows["relevant_sensor_pair"].fillna(False)]
    return sample_rows


def _select_fingerprint_sponge(available: Sequence[str], *, fingerprint_sponge: str | None) -> str:
    if fingerprint_sponge is None:
        return str(available[0])
    selected = str(fingerprint_sponge)
    if selected not in set(available):
        raise ValueError(
            f"retron_review: requested fingerprint sponge {selected!r} is not available; available: {list(available)!r}"
        )
    return selected


def _group_fingerprint_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    group_columns = [
        column
        for column in (
            "source_experiment_id",
            "source_label",
            "sensor",
            "stress_condition",
            "sponge",
            "sponge_family_size",
        )
        if column in frame.columns
    ]
    if not group_columns:
        return frame[["value"]].copy()
    return frame.groupby(group_columns, dropna=False)["value"].mean().reset_index()


def _pair_fingerprint_rows(sample_rows: pd.DataFrame, control_rows: pd.DataFrame) -> pd.DataFrame:
    match_columns = [
        column
        for column in ("source_experiment_id", "source_label", "sensor", "stress_condition")
        if column in sample_rows.columns and column in control_rows.columns
    ]
    control_columns = match_columns + ["control_value", "control_sponge", "control_family_size"]
    return sample_rows.merge(control_rows[control_columns], on=match_columns, how="left")


def _build_fingerprint_long_frame(
    paired_rows: pd.DataFrame,
    *,
    selected_sponge: str,
    control_name: str,
) -> pd.DataFrame:
    long_rows: list[dict[str, Any]] = []
    has_source_experiment_id = "source_experiment_id" in paired_rows.columns
    has_source_label = "source_label" in paired_rows.columns
    has_stress_condition = "stress_condition" in paired_rows.columns
    has_sample_family_size = "sponge_family_size" in paired_rows.columns
    has_control_sponge = "control_sponge" in paired_rows.columns
    has_control_family_size = "control_family_size" in paired_rows.columns
    for row in paired_rows.itertuples(index=False):
        source_experiment_id = row.source_experiment_id if has_source_experiment_id else pd.NA
        source_label = row.source_label if has_source_label else pd.NA
        stress_condition = row.stress_condition if has_stress_condition else pd.NA
        sensor = str(row.sensor)
        long_rows.append(
            {
                "selected_sponge": selected_sponge,
                "sensor": sensor,
                "stress_condition": stress_condition,
                "source_experiment_id": source_experiment_id,
                "source_label": source_label,
                "comparison_group": "Selected sponge",
                "sponge": str(row.sponge),
                "sponge_family_size": str(row.sponge_family_size) if has_sample_family_size else "other",
                "value": float(row.value),
            }
        )
        control_value = getattr(row, "control_value", pd.NA)
        if pd.notna(control_value):
            long_rows.append(
                {
                    "selected_sponge": selected_sponge,
                    "sensor": sensor,
                    "stress_condition": stress_condition,
                    "source_experiment_id": source_experiment_id,
                    "source_label": source_label,
                    "comparison_group": "tetO reference",
                    "sponge": str(row.control_sponge) if has_control_sponge else str(control_name),
                    "sponge_family_size": str(row.control_family_size) if has_control_family_size else "control",
                    "value": float(control_value),
                }
            )
    return pd.DataFrame(long_rows, columns=_FINGERPRINT_FRAME_COLUMNS)


def _sorted_fingerprint_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    sensor_order = sorted(out["sensor"].dropna().astype(str).unique())
    sensor_order_map = {sensor: idx for idx, sensor in enumerate(sensor_order)}
    out["__sensor_order"] = out["sensor"].map(sensor_order_map)
    out["__group_order"] = out["comparison_group"].map({"tetO reference": 0, "Selected sponge": 1}).fillna(99)
    order = ["__sensor_order", "__group_order", "source_experiment_id", "source_label", "sponge"]
    return (
        out.sort_values(order, kind="stable").drop(columns=["__sensor_order", "__group_order"]).reset_index(drop=True)
    )


def available_multifunctional_sponges(summary_df: pd.DataFrame) -> list[str]:
    scores = aggregate_on_target_scores(summary_df, score_metric="S_AUC")
    if scores.empty:
        return []
    multi = scores[scores["sponge_family_size"].astype(str).isin({"bi", "tri", "quad"})]
    return sorted({str(value) for value in multi["sponge"].dropna()}, key=_sponge_sort_key)


def aggregate_on_target_scores(summary_df: pd.DataFrame, *, score_metric: str) -> pd.DataFrame:
    frame = _normalized_retron_summary_frame(
        summary_df,
        required={"sensor", "sponge", "metric", "value", "relevant_sensor_pair", "is_relevant_stress"},
    )
    filtered = frame[
        (frame["metric"] == str(score_metric))
        & frame["relevant_sensor_pair"].fillna(False)
        & frame["is_relevant_stress"].fillna(False)
        & (frame["sponge"] != "tetO")
    ].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "sensor",
                "sponge",
                "sponge_family_size",
                "value",
                "n_rows",
                "n_experiments",
            ]
        )
    agg_kwargs: dict[str, tuple[str, str]] = {
        "value": ("value", "mean"),
        "n_rows": ("value", "size"),
    }
    if "source_experiment_id" in filtered.columns:
        agg_kwargs["n_experiments"] = ("source_experiment_id", "nunique")
    grouped = filtered.groupby(["sensor", "sponge", "sponge_family_size"], dropna=False).agg(**agg_kwargs).reset_index()
    if "n_experiments" not in grouped.columns:
        grouped["n_experiments"] = 1
    return grouped.sort_values(["sensor", "sponge"], kind="stable").reset_index(drop=True)


def build_aggregate_pareto_frame(
    summary_df: pd.DataFrame,
    *,
    score_metric: str = "S_AUC",
    burden_metric: str = "D_growth_AUC",
) -> pd.DataFrame:
    scores = aggregate_on_target_scores(summary_df, score_metric=score_metric)
    if scores.empty:
        return pd.DataFrame(
            columns=["sponge", "on_target", "burden", "leakiness", "sponge_family_size", "n_experiments"]
        )
    score_by_sponge = scores.groupby("sponge", dropna=False)["value"].mean().rename("on_target")
    experiment_count = scores.groupby("sponge", dropna=False)["n_experiments"].max().rename("n_experiments")
    family = scores.groupby("sponge", dropna=False)["sponge_family_size"].first()
    leak_rows = summary_df[
        (summary_df["metric"].astype(str) == "L_pre")
        & (summary_df["sponge"].astype(str) != "tetO")
        & _coerce_optional_bool_series(summary_df["relevant_sensor_pair"], label="relevant_sensor_pair").fillna(False)
    ].copy()
    leak_rows["value"] = pd.to_numeric(leak_rows["value"], errors="coerce").abs()
    leakiness = leak_rows.groupby("sponge", dropna=False)["value"].mean().rename("leakiness")
    burden_rows = summary_df[
        (summary_df["metric"].astype(str) == burden_metric)
        & (summary_df["sponge"].astype(str) != "tetO")
        & _coerce_optional_bool_series(summary_df["relevant_sensor_pair"], label="relevant_sensor_pair").fillna(False)
    ][["sponge", "value"]].copy()
    burden_rows["value"] = pd.to_numeric(burden_rows["value"], errors="coerce")
    burden = burden_rows.groupby("sponge", dropna=False)["value"].mean().rename("burden")
    table = (
        pd.concat([score_by_sponge, burden, leakiness, family.rename("sponge_family_size"), experiment_count], axis=1)
        .reset_index()
        .dropna(subset=["on_target", "burden"])
    )
    if table.empty:
        return table
    table["__family_order"] = table["sponge_family_size"].map(lambda value: _FAMILY_ORDER.get(str(value), 99))
    table["__sponge_order"] = table["sponge"].map(_sponge_sort_key)
    return (
        table.sort_values(["__family_order", "__sponge_order"], kind="stable")
        .drop(columns=["__family_order", "__sponge_order"])
        .reset_index(drop=True)
    )


def _plot_guide_metadata(plot_id: str) -> dict[str, str]:
    guide = _RETRON_PLOT_GUIDE.get(str(plot_id), {})
    return {
        "title": str(guide.get("title", plot_id.replace("_", " ").title())),
        "stage": str(guide.get("stage", _PLOT_STAGE_ORDER[-1])),
        "question": str(guide.get("question", "What does this plot contribute to the retron sponge decision path?")),
        "math": str(guide.get("math", "Protocol-specific transform guide not registered.")),
        "record": str(guide.get("record", "see compiled plot spec")),
        "meaning": str(guide.get("meaning", "Interpret in the context of the compiled assay semantics.")),
    }


def _aggregate_plot_guide_metadata(plot_id: str) -> dict[str, str]:
    guide = _RETRON_AGGREGATE_PLOT_GUIDE.get(str(plot_id), {})
    return {
        "title": str(guide.get("title", plot_id.replace("_", " ").title())),
        "question": str(guide.get("question", "What cross-run question does this aggregate plot answer?")),
        "math": str(guide.get("math", "Notebook-local aggregate figure.")),
        "meaning": str(guide.get("meaning", "Interpret against the direct-ratio matched-control workflow.")),
    }


def _prepare_notebook_plot_figure(item: Any) -> Any:
    fig = getattr(item, "fig", None)
    if fig is None:
        return item
    return replace(item, fig=_style_notebook_figure(fig))


def _style_notebook_figure(fig: Any) -> Any:
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    with suppress(Exception):
        fig.set_dpi(max(160, int(fig.get_dpi())))
    for axis in getattr(fig, "axes", ()):
        if hasattr(axis, "set_facecolor"):
            axis.set_facecolor("white")
        if hasattr(axis, "tick_params"):
            axis.tick_params(colors="#111111")
        for spine in getattr(axis, "spines", {}).values():
            spine.set_color("#111111")
        if hasattr(axis, "xaxis") and hasattr(axis.xaxis, "label"):
            axis.xaxis.label.set_color("#111111")
        if hasattr(axis, "yaxis") and hasattr(axis.yaxis, "label"):
            axis.yaxis.label.set_color("#111111")
        if hasattr(axis, "title"):
            axis.title.set_color("#111111")
        for text in getattr(axis, "texts", ()):
            text.set_color("#111111")
        if hasattr(axis, "legend"):
            legend = axis.legend_
            if legend is not None:
                _style_notebook_legend(legend)
    for legend in getattr(fig, "legends", ()):
        _style_notebook_legend(legend)
    for text in getattr(fig, "texts", ()):
        text.set_color("#111111")
    return fig


def _style_notebook_legend(legend: Any) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    frame.set_edgecolor("#d0d0d0")
    for text in legend.get_texts():
        text.set_color("#111111")
        with suppress(Exception):
            text.set_fontsize(8)
    legend.get_title().set_color("#111111")
    with suppress(Exception):
        legend.get_title().set_fontsize(8)


def _render_time_series_plot_spec(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> list[Any]:
    with_cfg = dict(plot_spec.get("with") or {})
    partition = dict(with_cfg.get("partition") or {})
    df = _require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="df")
    blanks = _optional_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="blanks")
    return plot_time_series(
        df=df,
        blanks=blanks if blanks is not None else df.iloc[0:0].copy(),
        output_dir=None,
        x=str(with_cfg.get("x") or "time"),
        xlabel=_normalize_optional_str(with_cfg.get("xlabel")),
        y=_normalize_optional_str_list(with_cfg.get("y")),
        ylabel_map=dict(with_cfg.get("ylabel_map") or {}),
        hue_label_map=dict(with_cfg.get("hue_label_map") or {}),
        hue=str(with_cfg.get("hue") or "treatment"),
        channels=_normalize_optional_str_list(with_cfg.get("channels")),
        subplots=None,
        group_on=_normalize_optional_str(partition.get("by")),
        pool_sets=partition.get("collection_items"),
        pool_match=str(partition.get("match") or "exact"),
        fig_kwargs=dict(with_cfg.get("fig") or {}),
        add_sheet_line=bool(with_cfg.get("add_sheet_line", False)),
        sheet_line_kwargs=dict(with_cfg.get("sheet_line_kwargs") or {}),
        log_transform=with_cfg.get("log_transform", False),
        time_window=_normalize_optional_float_list(with_cfg.get("time_window")),
        palette_book=None,
        ci=float(with_cfg.get("ci", 95.0)),
        ci_alpha=float(with_cfg.get("ci_alpha", 0.15)),
        ci_boot=int(with_cfg.get("ci_boot", 100)),
        ci_seed=int(with_cfg.get("ci_seed", 0)),
        legend_loc=str(with_cfg.get("legend_loc") or "upper left"),
        show_replicates=bool(with_cfg.get("show_replicates", False)),
        shared_legend=bool(with_cfg.get("shared_legend", False)),
        filename=_normalize_optional_str(with_cfg.get("filename")),
    )


def _render_retron_qc_plot_spec(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> list[Any]:
    with_cfg = dict(plot_spec.get("with") or {})
    partition = dict(with_cfg.get("partition") or {})
    df = _retron_qc_dataframe(plot_spec=plot_spec, datasets=datasets)
    blanks = _optional_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="blanks")
    preferred_channels = ["OD600", "YFP", "CFP", "YFP/CFP", "YFP/OD600", "CFP/OD600"]
    available = set(df["channel"].astype(str).tolist()) if "channel" in df.columns else set()
    channels = [channel for channel in preferred_channels if channel in available]
    ylabel_map = {
        "OD600": "OD600",
        "YFP": "YFP",
        "CFP": "CFP",
        "YFP/OD600": "YFP/OD600",
        "CFP/OD600": "CFP/OD600",
        "YFP/CFP": "YFP/CFP",
    }
    return plot_time_series(
        df=df,
        blanks=blanks if blanks is not None else df.iloc[0:0].copy(),
        output_dir=None,
        x=str(with_cfg.get("x") or "time"),
        xlabel=str(with_cfg.get("xlabel") or "Time from stress addition (h)"),
        y=channels,
        ylabel_map={key: value for key, value in ylabel_map.items() if key in channels},
        hue_label_map=dict(with_cfg.get("hue_label_map") or {}),
        hue=str(with_cfg.get("hue") or "treatment"),
        channels=None,
        subplots=None,
        group_on=_normalize_optional_str(partition.get("by")),
        pool_sets=partition.get("collection_items"),
        pool_match=str(partition.get("match") or "exact"),
        fig_kwargs=dict(with_cfg.get("fig") or {}),
        add_sheet_line=True,
        sheet_line_kwargs={"color": "#9E9E9E", "linestyle": "--", "linewidth": 0.9, "alpha": 0.95},
        log_transform=with_cfg.get("log_transform", False),
        time_window=_normalize_optional_float_list(with_cfg.get("time_window")),
        palette_book=None,
        ci=float(with_cfg.get("ci", 95.0)),
        ci_alpha=float(with_cfg.get("ci_alpha", 0.15)),
        ci_boot=int(with_cfg.get("ci_boot", 100)),
        ci_seed=int(with_cfg.get("ci_seed", 0)),
        legend_loc=str(with_cfg.get("legend_loc") or "upper left"),
        show_replicates=bool(with_cfg.get("show_replicates", False)),
        shared_legend=True,
        filename=_normalize_optional_str(with_cfg.get("filename")),
    )


def _retron_qc_dataframe(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    ratio_df = _require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="df")
    raw_df = datasets.get("overflow/df")
    if raw_df is None or raw_df.empty or "channel" not in raw_df.columns:
        return ratio_df
    raw_channels = [channel for channel in raw_df["channel"].astype(str).unique().tolist() if "/" not in channel]
    if not raw_channels:
        return ratio_df
    support_channels = (
        [channel for channel in ratio_df["channel"].astype(str).unique().tolist() if "/" in channel]
        if "channel" in ratio_df.columns
        else []
    )
    frames = [raw_df[raw_df["channel"].astype(str).isin(raw_channels)].copy()]
    if support_channels:
        frames.append(ratio_df[ratio_df["channel"].astype(str).isin(support_channels)].copy())
    all_columns = list(dict.fromkeys([*raw_df.columns.tolist(), *ratio_df.columns.tolist()]))
    aligned = [frame.reindex(columns=all_columns) for frame in frames if not frame.empty]
    if not aligned:
        return ratio_df
    return pd.concat(aligned, ignore_index=True)


def _render_retron_trace_plot_spec(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> list[Any]:
    with_cfg = dict(plot_spec.get("with") or {})
    return plot_retron_sponge_trace(
        trace=_require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="trace"),
        output_dir=None,
        metrics=_normalize_optional_str_list(with_cfg.get("metrics")) or [],
        title=str(with_cfg.get("title") or "Retron sponge trace"),
        filename=_normalize_optional_str(with_cfg.get("filename")),
        palette_book=None,
        control_name=str(with_cfg.get("control_name") or "tetO"),
        include_control=bool(with_cfg.get("include_control", False)),
        only_control=bool(with_cfg.get("only_control", False)),
        relevant_only=bool(with_cfg.get("relevant_only", False)),
        stress_order=_normalize_optional_str_list(with_cfg.get("stress_order")),
        panel_by=str(with_cfg.get("panel_by") or "stress"),
        metric_label_map=dict(with_cfg.get("metric_label_map") or {}),
        fig_kwargs=dict(with_cfg.get("fig") or {}),
    )


def _render_retron_summary_plot_spec(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
) -> list[Any]:
    with_cfg = dict(plot_spec.get("with") or {})
    return plot_retron_sponge_summary(
        summary=_require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="summary"),
        trace=_optional_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="trace"),
        output_dir=None,
        view=str(with_cfg.get("view") or "heatmap"),
        title=str(with_cfg.get("title") or "Retron sponge summary"),
        filename=_normalize_optional_str(with_cfg.get("filename")),
        palette_book=None,
        control_name=str(with_cfg.get("control_name") or "tetO"),
        no_stress_label=str(with_cfg.get("no_stress_label") or "H2O"),
        relevant_only=bool(with_cfg.get("relevant_only", True)),
        metric=_normalize_optional_str(with_cfg.get("metric")),
        state_order=_normalize_optional_str_list(with_cfg.get("state_order")),
        burden_metric=str(with_cfg.get("burden_metric") or "T_growth_AUC"),
        fig_kwargs=dict(with_cfg.get("fig") or {}),
    )


def _require_plot_dataset(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    record_id = _read_record_id(plot_spec=plot_spec, label=label)
    try:
        return datasets[record_id]
    except KeyError as exc:
        raise ValueError(
            f"retron_review: plot {plot_spec.get('id')!r} requires record {record_id!r} for input {label!r}"
        ) from exc


def _optional_plot_dataset(
    *,
    plot_spec: Mapping[str, Any],
    datasets: Mapping[str, pd.DataFrame],
    label: str,
) -> pd.DataFrame | None:
    record_id = _read_record_id(plot_spec=plot_spec, label=label, optional=True)
    if record_id is None:
        return None
    return datasets.get(record_id)


def _read_record_id(
    *,
    plot_spec: Mapping[str, Any],
    label: str,
    optional: bool = False,
) -> str | None:
    reads = dict(plot_spec.get("reads") or {})
    ref = dict(reads.get(label) or {})
    record_id = _normalize_optional_str(ref.get("record"))
    if record_id is None:
        if optional:
            return None
        raise ValueError(f"retron_review: plot {plot_spec.get('id')!r} is missing a record binding for {label!r}")
    return record_id


def _time_series_supporting_table(df: pd.DataFrame, *, channels: list[str] | None) -> pd.DataFrame:
    frame = df.copy()
    if channels and "channel" in frame.columns:
        frame = frame[frame["channel"].astype(str).isin(channels)].copy()
    keep = [
        column
        for column in (
            "design_id_alias",
            "design_id",
            "treatment_alias",
            "treatment",
            "time",
            "sheet_index",
            "overflow",
            "channel",
            "value",
        )
        if column in frame.columns
    ]
    out = frame[keep]
    order = [column for column in ("design_id_alias", "design_id", "time", "channel") if column in keep]
    if order:
        out = out.sort_values(order, kind="stable")
    return out


def _trace_supporting_table(trace_df: pd.DataFrame, *, metrics: list[str]) -> pd.DataFrame:
    frame = trace_df.copy()
    if metrics:
        frame = frame[frame["metric"].astype(str).isin(metrics)].copy()
    keep = [
        column
        for column in (
            "sensor",
            "sponge",
            "stress_condition",
            "IPTG",
            "time_from_stress",
            "metric",
            "value",
            "configured_max_post_stress_hours",
            "relevant_sensor_pair",
            "is_relevant_stress",
            "sponge_family_size",
        )
        if column in frame.columns
    ]
    order = [
        column
        for column in ("sensor", "sponge", "stress_condition", "IPTG", "time_from_stress", "metric")
        if column in keep
    ]
    out = frame[keep]
    if order:
        out = out.sort_values(order, kind="stable")
    return out


def _summary_supporting_table(
    summary_df: pd.DataFrame,
    *,
    view: str,
    metric: str,
    burden_metric: str,
) -> pd.DataFrame:
    frame = summary_df.copy()
    view_id = str(view)
    if view_id == "interaction":
        metric_names = [metric or "C_AUC"]
    elif view_id == "stress_modulation":
        metric_names = [metric or "M_AUC"]
    elif view_id == "pareto":
        metric_names = ["S_abs_AUC", burden_metric or "D_growth_AUC", "L_pre", "P_pre"]
    elif view_id == "heatmap":
        metric_names = ["S_abs_AUC", "S_AUC", "P_pre"]
    else:
        metric_names = [metric] if metric else []
    if metric_names:
        frame = frame[frame["metric"].astype(str).isin(metric_names)].copy()
    keep = [
        column
        for column in (
            "sensor",
            "sponge",
            "stress_condition",
            "IPTG",
            "metric",
            "value",
            "relevant_sensor_pair",
            "is_relevant_stress",
            "sponge_family_size",
        )
        if column in frame.columns
    ]
    order = [column for column in ("sensor", "sponge", "stress_condition", "IPTG", "metric") if column in keep]
    out = frame[keep]
    if order:
        out = out.sort_values(order, kind="stable")
    return out


def _build_specificity_matrix_figure(*, matrix: pd.DataFrame, score_metric: str) -> Any | None:
    if matrix.empty:
        return None

    n_rows = max(1, len(matrix.index))
    n_cols = max(1, len(matrix.columns))
    figure, axis = plt.subplots(
        figsize=(max(7.4, 2.2 + 0.72 * n_cols), max(3.2, 1.6 + 0.72 * n_rows)),
        constrained_layout=True,
    )
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="vlag",
        center=0.0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        cbar=True,
        square=True,
        linewidths=0.3,
        linecolor="#f0f0f0",
        cbar_kws={"label": _metric_axis_label(score_metric), "shrink": 0.80},
    )
    axis.set_title("Relevant-stress target activity matrix", pad=10, fontweight="normal", fontsize=11)
    axis.set_xlabel("Sponge design", fontsize=10)
    axis.set_ylabel("")
    axis.tick_params(axis="x", labelrotation=0, labelsize=8)
    axis.tick_params(axis="y", labelrotation=0, labelsize=9)
    axis.set_xticklabels(
        [_wrap_hyphenated_plot_label(label.get_text(), max_parts_per_line=2) for label in axis.get_xticklabels()]
    )
    for label in axis.get_xticklabels():
        label.set_ha("center")
    return figure


def _build_aggregate_pareto_figure(*, pareto_df: pd.DataFrame, score_metric: str) -> Any | None:
    if pareto_df.empty:
        return None
    payload = _aggregate_pareto_figure_payload(pareto_df)
    figure, axis = plt.subplots(figsize=(8.2, 6.0), constrained_layout=False)
    _plot_aggregate_pareto_points(axis, pareto_df=pareto_df, payload=payload)
    annotate_points_smart(
        ax=axis,
        points=[(float(row["on_target"]), float(row["burden"])) for _, row in pareto_df.iterrows()],
        labels=[
            _wrap_hyphenated_plot_label(str(row["sponge"]), max_parts_per_line=2) for _, row in pareto_df.iterrows()
        ],
        text_kwargs={"fontsize": 8},
    )
    axis.axvline(0.0, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
    axis.axhline(0.0, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
    axis.set_xlabel(f"Mean on-target effect ({score_metric})", fontsize=11)
    axis.set_ylabel("Mean growth burden (D_growth_AUC)", fontsize=11)
    axis.tick_params(axis="both", labelsize=7)
    with suppress(Exception):
        axis.set_box_aspect(1.0)
    legend_handles = _aggregate_pareto_legend_handles(payload)
    if legend_handles:
        axis.legend(
            handles=legend_handles,
            frameon=False,
            title=None,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
        )
    figure.suptitle("Aggregate pareto ranking", y=0.97, x=0.5, ha="center", fontweight="normal", fontsize=13)
    figure.text(
        0.5,
        0.93,
        "Mean on-target score versus mean construct-specific burden across the review set; point size encodes |L_pre|.",
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
    )
    figure.subplots_adjust(top=0.85, right=0.79, left=0.12, bottom=0.12)
    return figure


def _aggregate_pareto_figure_payload(pareto_df: pd.DataFrame) -> _AggregateParetoFigurePayload:
    family_levels = tuple(_ordered_text(pareto_df["sponge_family_size"].fillna("other").astype(str).tolist()))
    color_values = sns.color_palette("colorblind", n_colors=max(1, len(family_levels)))
    color_map = {family: color_values[idx % len(color_values)] for idx, family in enumerate(family_levels)}
    sizes = 90.0 + 260.0 * pareto_df["leakiness"].fillna(0.0)
    return _AggregateParetoFigurePayload(
        family_levels=family_levels,
        color_map=color_map,
        sizes=sizes,
    )


def _plot_aggregate_pareto_points(
    axis: Any,
    *,
    pareto_df: pd.DataFrame,
    payload: _AggregateParetoFigurePayload,
) -> None:
    axis.scatter(
        pareto_df["on_target"],
        pareto_df["burden"],
        s=payload.sizes,
        c=[payload.color_map.get(str(item), "#4c72b0") for item in pareto_df["sponge_family_size"].fillna("other")],
        alpha=0.85,
        edgecolors="#222222",
        linewidths=0.5,
        zorder=2,
    )


def _aggregate_pareto_legend_handles(payload: _AggregateParetoFigurePayload) -> list[Any]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=family,
            markerfacecolor=payload.color_map[family],
            markeredgecolor="#222222",
            markersize=7,
        )
        for family in payload.family_levels
    ]


def _build_architecture_figure(
    *,
    architecture_df: pd.DataFrame,
    score_metric: str,
    architecture_x: str,
) -> Any | None:
    if architecture_df.empty:
        return None

    sensors = sorted(architecture_df["sensor"].dropna().astype(str).unique())
    palette = _family_palette(architecture_df)
    figure, axes = _sensor_subplot_figure(sensors=sensors, height=4.9)
    x_limits = shared_numeric_limits(
        architecture_df[architecture_x].to_numpy(dtype=float, copy=False),
        pad_fraction=0.10,
        min_span=1.0,
    )
    y_limits = shared_numeric_limits(
        architecture_df["value"].to_numpy(dtype=float, copy=False),
        center=0.0,
        pad_fraction=0.10,
        min_span=0.10,
    )
    for axis, sensor in zip(axes[0], sensors, strict=True):
        sensor_df = architecture_df[architecture_df["sensor"].astype(str) == sensor].copy()
        _plot_sensor_family_scatter(
            axis,
            sensor_df=sensor_df,
            x_column=architecture_x,
            y_column="value",
            palette=palette,
            point_size=88,
        )
        axis.axhline(0.0, color="#777777", linestyle=":", linewidth=1.0)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_title(str(sensor), pad=8, fontweight="normal")
        axis.set_xlabel("")
        axis.set_ylabel("")
    _finalize_sensor_scatter_figure(
        figure,
        axes[0],
        xlabel=_architecture_axis_label(architecture_x),
        ylabel=_metric_axis_label(score_metric),
    )
    return figure


def _build_expected_vs_observed_figure(
    *,
    expected_vs_observed_df: pd.DataFrame,
    score_metric: str,
    expected_mode: str,
) -> Any | None:
    if expected_vs_observed_df.empty:
        return None

    sensors = sorted(expected_vs_observed_df["sensor"].dropna().astype(str).unique())
    palette = _family_palette(expected_vs_observed_df)
    figure, axes = _sensor_subplot_figure(sensors=sensors, height=5.1)
    combined_limits = shared_numeric_limits(
        pd.concat(
            [
                expected_vs_observed_df[expected_mode],
                expected_vs_observed_df["observed"],
            ],
            ignore_index=True,
        ).to_numpy(dtype=float, copy=False),
        center=0.0,
        pad_fraction=0.10,
        min_span=0.10,
    )
    for axis, sensor in zip(axes[0], sensors, strict=True):
        sensor_df = expected_vs_observed_df[expected_vs_observed_df["sensor"].astype(str) == sensor].copy()
        _plot_sensor_family_scatter(
            axis,
            sensor_df=sensor_df,
            x_column=expected_mode,
            y_column="observed",
            palette=palette,
            point_size=92,
        )
        axis.plot(
            [combined_limits[0], combined_limits[1]],
            [combined_limits[0], combined_limits[1]],
            color="#777777",
            linestyle=":",
            linewidth=1.0,
        )
        axis.set_xlim(combined_limits)
        axis.set_ylim(combined_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(str(sensor), pad=8, fontweight="normal")
        axis.set_xlabel("")
        axis.set_ylabel("")
    _finalize_sensor_scatter_figure(
        figure,
        axes[0],
        xlabel=_expected_axis_label(expected_mode, score_metric=score_metric),
        ylabel=f"Observed multifunction score ({score_metric})",
    )
    return figure


def _build_fingerprint_figure(
    *,
    fingerprint_df: pd.DataFrame,
    score_metric: str,
) -> Any | None:
    if fingerprint_df.empty:
        return None
    payload = _fingerprint_figure_payload(fingerprint_df)
    figure, axis = plt.subplots(
        figsize=(max(6.0, 1.8 + 1.45 * len(payload.sensor_levels)), 4.9),
        constrained_layout=False,
    )
    _plot_fingerprint_bars(
        axis,
        stats=payload.stats,
        sensor_levels=payload.sensor_levels,
        comparison_order=payload.comparison_order,
        x_positions=payload.x_positions,
        offsets=payload.offsets,
        width=payload.width,
        comparison_colors=payload.comparison_colors,
        edge_colors=payload.edge_colors,
    )
    _plot_fingerprint_points(
        axis,
        fingerprint_df=fingerprint_df,
        sensor_levels=payload.sensor_levels,
        comparison_order=payload.comparison_order,
        x_positions=payload.x_positions,
        offsets=payload.offsets,
        point_facecolors=payload.point_facecolors,
        edge_colors=payload.edge_colors,
    )
    axis.axhline(0.0, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
    axis.set_xlim(-0.55, max(0.55, len(payload.sensor_levels) - 0.45))
    axis.set_ylim(payload.y_limits)
    axis.set_xticks([payload.x_positions[sensor] for sensor in payload.sensor_levels])
    axis.set_xticklabels(payload.sensor_levels)
    axis.set_xlabel("Relevant sensor arm", fontsize=11)
    axis.set_ylabel(_metric_axis_label(score_metric), fontsize=11)
    axis.tick_params(axis="x", labelsize=9)
    axis.tick_params(axis="y", labelsize=8)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.50)
    axis.set_title(
        _wrap_hyphenated_plot_label(payload.selected_sponge, max_parts_per_line=2),
        pad=10,
        fontweight="normal",
        fontsize=11,
    )
    axis.legend(frameon=False, title=None, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    figure.suptitle("Sponge fingerprint", y=0.97, x=0.5, ha="center", fontweight="normal", fontsize=13)
    figure.text(
        0.5,
        0.92,
        _fingerprint_support_text(max_sources=payload.max_sources),
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
    )
    figure.subplots_adjust(bottom=0.16, left=0.11, right=0.80, top=0.84)
    return figure


def _fingerprint_figure_payload(fingerprint_df: pd.DataFrame) -> _FingerprintFigurePayload:
    sensor_levels = tuple(sorted(fingerprint_df["sensor"].dropna().astype(str).unique().tolist()))
    comparison_order = tuple(_fingerprint_comparison_order(fingerprint_df))
    comparison_colors, edge_colors, point_facecolors = _fingerprint_comparison_styles()
    source_counts = (
        fingerprint_df.groupby(["sensor", "comparison_group"], dropna=False)["value"]
        .size()
        .reset_index(name="n_sources")
    )
    width = 0.34
    offsets = _group_offsets(comparison_order, width=width)
    return _FingerprintFigurePayload(
        selected_sponge=str(fingerprint_df["selected_sponge"].dropna().astype(str).iloc[0]),
        sensor_levels=sensor_levels,
        comparison_order=comparison_order,
        stats=_fingerprint_group_stats(fingerprint_df),
        y_limits=shared_numeric_limits(
            fingerprint_df["value"].to_numpy(dtype=float, copy=False),
            center=0.0,
            pad_fraction=0.12,
            min_span=0.10,
        ),
        max_sources=int(source_counts["n_sources"].max()) if not source_counts.empty else 0,
        width=width,
        offsets=offsets,
        x_positions={sensor: float(idx) for idx, sensor in enumerate(sensor_levels)},
        comparison_colors=comparison_colors,
        edge_colors=edge_colors,
        point_facecolors=point_facecolors,
    )


def _family_palette(frame: pd.DataFrame) -> dict[str, str]:
    family_levels = _ordered_text(frame["sponge_family_size"].fillna("other").astype(str).tolist())
    return {level: _FAMILY_COLOR_MAP.get(level, _FAMILY_COLOR_MAP["other"]) for level in family_levels}


def _sensor_subplot_figure(*, sensors: Sequence[str], height: float) -> tuple[Any, Any]:
    return plt.subplots(
        1,
        len(sensors),
        figsize=(5.3 * len(sensors), height),
        constrained_layout=False,
        squeeze=False,
        sharex=True,
        sharey=True,
    )


def _plot_sensor_family_scatter(
    axis: Any,
    *,
    sensor_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    palette: Mapping[str, str],
    point_size: float,
) -> None:
    sns.scatterplot(
        data=sensor_df,
        x=x_column,
        y=y_column,
        hue="sponge_family_size",
        palette=palette,
        s=point_size,
        edgecolor="black",
        linewidth=0.45,
        ax=axis,
    )
    annotate_points_smart(
        ax=axis,
        points=[(float(row[x_column]), float(row[y_column])) for _, row in sensor_df.iterrows()],
        labels=[
            _wrap_hyphenated_plot_label(str(row["sponge"]), max_parts_per_line=2) for _, row in sensor_df.iterrows()
        ],
        text_kwargs={"fontsize": 8},
    )


def _finalize_sensor_scatter_figure(
    figure: Any,
    axes: Sequence[Any],
    *,
    xlabel: str,
    ylabel: str,
) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(0.98, 0.5),
            ncol=1,
            title=None,
        )
    for axis in axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
        with suppress(Exception):
            axis.set_box_aspect(1.0)
    figure.supxlabel(xlabel, y=0.09)
    figure.supylabel(ylabel, x=0.02)
    figure.subplots_adjust(bottom=0.18, left=0.10, right=0.84, top=0.88, wspace=0.24)


def _fingerprint_comparison_order(fingerprint_df: pd.DataFrame) -> list[str]:
    available = set(fingerprint_df["comparison_group"].astype(str))
    return [group for group in ("tetO reference", "Selected sponge") if group in available]


def _fingerprint_group_stats(fingerprint_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fingerprint_df.groupby(["sensor", "comparison_group"], dropna=False)["value"]
        .agg(mean="mean", sd="std", n="size")
        .reset_index()
    )


def _fingerprint_comparison_styles() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    return (
        {
            "tetO reference": "#f3ebe7",
            "Selected sponge": "#4c72b0",
        },
        {
            "tetO reference": "#9b7d72",
            "Selected sponge": "#1f3552",
        },
        {
            "tetO reference": "#ffffff",
            "Selected sponge": "#4c72b0",
        },
    )


def _group_offsets(groups: Sequence[str], *, width: float) -> dict[str, float]:
    return {group: ((idx - (len(groups) - 1) / 2.0) * width) for idx, group in enumerate(groups)}


def _plot_fingerprint_bars(
    axis: Any,
    *,
    stats: pd.DataFrame,
    sensor_levels: Sequence[str],
    comparison_order: Sequence[str],
    x_positions: Mapping[str, float],
    offsets: Mapping[str, float],
    width: float,
    comparison_colors: Mapping[str, str],
    edge_colors: Mapping[str, str],
) -> None:
    for group in comparison_order:
        group_stats = stats[stats["comparison_group"].astype(str) == group].copy()
        group_stats = group_stats.set_index("sensor").reindex(sensor_levels).reset_index()
        x_values = [x_positions[str(sensor)] + offsets[group] for sensor in group_stats["sensor"].astype(str)]
        mean_values = pd.to_numeric(group_stats["mean"], errors="coerce").to_numpy(dtype=float)
        error_values = pd.to_numeric(group_stats["sd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        axis.bar(
            x_values,
            mean_values,
            width=width * 0.92,
            color=comparison_colors.get(group, "#4c72b0"),
            edgecolor=edge_colors.get(group, "#222222"),
            linewidth=0.9,
            label=group,
            zorder=2,
        )
        replicate_counts = pd.to_numeric(group_stats["n"], errors="coerce").fillna(0).to_numpy(dtype=int)
        error_mask = (replicate_counts > 1) & np.isfinite(error_values)
        if error_mask.any():
            axis.errorbar(
                np.asarray(x_values, dtype=float)[error_mask],
                mean_values[error_mask],
                yerr=error_values[error_mask],
                fmt="none",
                ecolor=edge_colors.get(group, "#222222"),
                elinewidth=1.0,
                capsize=3.0,
                capthick=1.0,
                zorder=3,
            )


def _plot_fingerprint_points(
    axis: Any,
    *,
    fingerprint_df: pd.DataFrame,
    sensor_levels: Sequence[str],
    comparison_order: Sequence[str],
    x_positions: Mapping[str, float],
    offsets: Mapping[str, float],
    point_facecolors: Mapping[str, str],
    edge_colors: Mapping[str, str],
) -> None:
    for group in comparison_order:
        group_points = fingerprint_df[fingerprint_df["comparison_group"].astype(str) == group].copy()
        for sensor in sensor_levels:
            sensor_points = group_points[group_points["sensor"].astype(str) == sensor].copy()
            if sensor_points.empty:
                continue
            count = len(sensor_points)
            jitters = [0.0] if count == 1 else np.linspace(-0.06, 0.06, count).tolist()
            x_center = x_positions[str(sensor)] + offsets[group]
            for jitter, (_, point_row) in zip(jitters, sensor_points.iterrows(), strict=False):
                axis.scatter(
                    x_center + float(jitter),
                    float(point_row["value"]),
                    s=28,
                    facecolor=point_facecolors.get(group, "#4c72b0"),
                    edgecolor=edge_colors.get(group, "#222222"),
                    linewidth=0.8,
                    zorder=4,
                )


def _fingerprint_support_text(*, max_sources: int) -> str:
    message = (
        "Bars show means across source experiments; points show source-level replicates against the matched tetO reference."
        if max_sources > 1
        else "Bars show the single available source experiment per arm (n=1 in this view); points mark that source-level estimate against the matched tetO reference."
    )
    return _wrap_plot_label(message, width=110)


def _wrap_plot_label(text: str, *, width: int) -> str:
    value = str(text or "").strip()
    if not value or len(value) <= width:
        return value
    return textwrap.fill(value, width=width, break_long_words=False, break_on_hyphens=False)


def _wrap_hyphenated_plot_label(text: str, *, max_parts_per_line: int = 2) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    parts = [part for part in value.split("-") if part]
    if len(parts) <= max_parts_per_line:
        return value
    lines = ["-".join(parts[index : index + max_parts_per_line]) for index in range(0, len(parts), max_parts_per_line)]
    return "\n".join(lines)


def _metric_axis_label(metric: str) -> str:
    labels = {
        "P_pre": "Preload shift (P_pre)",
        "S_abs_AUC": "Scaled absolute effect (S_abs_AUC)",
        "S_AUC": "Scaled effect (S_AUC)",
        "O_abs_AUC": "Signed absolute effect (O_abs_AUC)",
        "O_AUC": "Signed effect (O_AUC)",
        "C": "tetO-subtracted ratio (C)",
        "D": "IPTG-state effect (D)",
        "M": "Stress-gated effect (M)",
    }
    return labels.get(str(metric), f"Retron sponge score ({metric})")


def _architecture_axis_label(architecture_x: str) -> str:
    if str(architecture_x) == "irrelevant_motif_count":
        return "Extra non-cognate motifs (irrelevant_motif_count)"
    return "Total motifs in the sponge design (motif_count)"


def _expected_axis_label(expected_mode: str, *, score_metric: str) -> str:
    if str(expected_mode) == "expected_best_single":
        return f"Best mono baseline ({score_metric})"
    return f"Sum of mono baselines ({score_metric})"


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_optional_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _ordered_text(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values})


def _load_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"retron_review: review manifest not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("retron_review: review manifest must be a mapping")
    return payload


def _resolve_sources(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    source_root: Path | None = None,
) -> list[RetronReviewSource]:
    return [
        _resolve_source_entry(
            manifest_path=manifest_path,
            idx=idx,
            raw_source=raw_source,
            source_root=source_root,
        )
        for idx, raw_source in _validated_manifest_sources(payload)
    ]


def _validated_manifest_sources(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("retron_review: manifest 'sources' must be a list")
    validated: list[tuple[int, dict[str, Any]]] = []
    for idx, raw_source in enumerate(raw_sources, start=1):
        if not isinstance(raw_source, dict):
            raise ValueError(f"retron_review: sources[{idx}] must be a mapping")
        validated.append((idx, raw_source))
    return validated


def _resolve_source_entry(
    *,
    manifest_path: Path,
    idx: int,
    raw_source: dict[str, Any],
    source_root: Path | None,
) -> RetronReviewSource:
    label = _source_label(raw_source, idx=idx)
    paths = _resolve_source_paths(
        manifest_path=manifest_path,
        raw_source=raw_source,
        source_root=source_root,
    )
    _ensure_source_exports_exist(label=label, paths=paths)
    return RetronReviewSource(
        label=label,
        experiment_id=_source_experiment_id(raw_source, experiment_root=paths.experiment_root, label=label),
        experiment_root=paths.experiment_root,
        config_path=paths.config_path,
        summary_path=paths.summary_path.resolve(),
        trace_path=paths.trace_path.resolve(),
    )


def _source_label(raw_source: Mapping[str, Any], *, idx: int) -> str:
    return str(raw_source.get("label") or raw_source.get("family") or f"source_{idx}").strip()


def _source_experiment_id(
    raw_source: Mapping[str, Any],
    *,
    experiment_root: Path | None,
    label: str,
) -> str:
    return str(raw_source.get("experiment_id") or (experiment_root.name if experiment_root is not None else label))


def _resolve_source_paths(
    *,
    manifest_path: Path,
    raw_source: Mapping[str, Any],
    source_root: Path | None,
) -> _ResolvedSourcePaths:
    experiment_root, config_path = _resolve_source_scope(
        manifest_path=manifest_path,
        raw_source=raw_source,
        source_root=source_root,
    )
    return _ResolvedSourcePaths(
        experiment_root=experiment_root,
        config_path=config_path,
        summary_path=_resolve_source_export_path(
            manifest_path=manifest_path,
            raw_source=raw_source,
            field="summary",
            experiment_root=experiment_root,
            record_id="semantic_metrics/summary",
            export_name="semantic_summary.csv",
        ),
        trace_path=_resolve_source_export_path(
            manifest_path=manifest_path,
            raw_source=raw_source,
            field="trace",
            experiment_root=experiment_root,
            record_id="semantic_metrics/trace",
            export_name="semantic_trace.csv",
        ),
    )


def _resolve_source_scope(
    *,
    manifest_path: Path,
    raw_source: Mapping[str, Any],
    source_root: Path | None,
) -> tuple[Path | None, Path | None]:
    experiment_raw = raw_source.get("experiment")
    config_raw = raw_source.get("config")
    if experiment_raw is not None:
        experiment_root = _resolve_manifest_path(
            manifest_path,
            str(experiment_raw),
            relative_to=source_root,
        )
        return experiment_root, experiment_root / "config.yaml"
    if config_raw is not None:
        config_path = _resolve_manifest_path(
            manifest_path,
            str(config_raw),
            relative_to=source_root,
        )
        return config_path.parent, config_path
    return None, None


def _resolve_source_export_path(
    *,
    manifest_path: Path,
    raw_source: Mapping[str, Any],
    field: str,
    experiment_root: Path | None,
    record_id: str,
    export_name: str,
) -> Path:
    raw_value = raw_source.get(field)
    if raw_value is not None:
        return _resolve_manifest_path(manifest_path, str(raw_value))
    if experiment_root is None:
        raise ValueError(
            "retron_review: each source must declare either 'experiment' or explicit 'summary'/'trace' paths"
        )
    return _resolve_default_semantic_table(
        experiment_root,
        record_id=record_id,
        export_name=export_name,
    )


def _ensure_source_exports_exist(*, label: str, paths: _ResolvedSourcePaths) -> None:
    missing = [str(path) for path in (paths.summary_path, paths.trace_path) if not path.exists()]
    if not missing:
        return
    command = ""
    if paths.config_path is not None:
        command = f" Run 'uv run reader run {paths.config_path}' and 'uv run reader export {paths.config_path}'."
    raise FileNotFoundError(f"retron_review: source exports are missing for {label}: {missing}.{command}")


def _resolve_semantic_maps(
    payload: dict[str, Any],
    *,
    sources: list[RetronReviewSource],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    manifest_relevant = payload.get("relevant_stress_map")
    manifest_targets = payload.get("sensor_target_map")
    relevant_stress_map = _normalize_relevant_stress_map(manifest_relevant or {})
    sensor_target_map = _normalize_sensor_target_map(manifest_targets or {})
    if relevant_stress_map and sensor_target_map:
        return relevant_stress_map, sensor_target_map
    derived_relevant: dict[str, str] = {}
    derived_targets: dict[str, tuple[str, ...]] = {}
    for source in sources:
        if source.config_path is None or not source.config_path.exists():
            continue
        config = yaml.safe_load(source.config_path.read_text(encoding="utf-8")) or {}
        analysis = ((config.get("protocol") or {}).get("analysis") or {}).get("semantic_metrics") or {}
        if not isinstance(analysis, dict):
            continue
        candidate_relevant = _normalize_relevant_stress_map(analysis.get("relevant_stress_map") or {})
        candidate_targets = _normalize_sensor_target_map(analysis.get("sensor_target_map") or {})
        derived_relevant = _merge_semantic_map(
            derived_relevant,
            candidate_relevant,
            label="relevant_stress_map",
        )
        derived_targets = _merge_semantic_map(
            derived_targets,
            candidate_targets,
            label="sensor_target_map",
        )
    relevant_stress_map = relevant_stress_map or derived_relevant
    sensor_target_map = sensor_target_map or derived_targets
    if not relevant_stress_map:
        raise ValueError("retron_review: manifest must declare relevant_stress_map or point to source configs that do")
    if not sensor_target_map:
        raise ValueError("retron_review: manifest must declare sensor_target_map or point to source configs that do")
    return relevant_stress_map, sensor_target_map


def _merge_semantic_map(existing: dict[str, Any], candidate: dict[str, Any], *, label: str) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in candidate.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"retron_review: inconsistent {label} for {key!r}: {merged[key]!r} vs {value!r}")
        merged[key] = value
    return merged


def _normalize_relevant_stress_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("retron_review: relevant_stress_map must be a mapping when provided")
    return {str(key): str(item) for key, item in value.items()}


def _normalize_sensor_target_map(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise ValueError("retron_review: sensor_target_map must be a mapping when provided")
    normalized: dict[str, tuple[str, ...]] = {}
    for key, items in value.items():
        if not isinstance(items, list):
            raise ValueError("retron_review: sensor_target_map values must be lists of motif labels")
        normalized[str(key)] = tuple(str(item) for item in items)
    return normalized


def _resolve_manifest_path(
    manifest_path: Path,
    raw: str,
    *,
    relative_to: Path | None = None,
) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        anchor = relative_to if relative_to is not None else manifest_path.parent
        path = (anchor / path).resolve()
    return path


def _resolve_default_semantic_table(experiment_root: Path, *, record_id: str, export_name: str) -> Path:
    export_path = experiment_root / "outputs" / "exports" / "retron" / export_name
    if export_path.exists():
        return export_path
    record_info, _, _, _ = discover_dataframe_records(experiment_root / "outputs", allow_scan=False)
    for info in record_info.values():
        if str(info.get("record_id")) == str(record_id):
            return Path(info["path"]).resolve()
    return export_path


def _read_semantic_table(path: Path, *, kind: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"retron_review: unsupported semantic table format for {path}")
    if "metric" in frame.columns:
        frame["metric"] = frame["metric"].astype(str)
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    for column in ("relevant_sensor_pair", "is_relevant_stress"):
        if column in frame.columns:
            frame[column] = _coerce_optional_bool_series(frame[column], label=column)
    frame["source_kind"] = kind
    return frame


def _annotate_source(frame: pd.DataFrame, *, source: RetronReviewSource) -> pd.DataFrame:
    out = frame.copy()
    out["source_label"] = source.label
    out["source_experiment_id"] = source.experiment_id
    out["source_summary_path"] = str(source.summary_path)
    out["source_trace_path"] = str(source.trace_path)
    return out


def _coerce_optional_bool_series(series: pd.Series, *, label: str) -> pd.Series:
    return series.map(lambda value: _coerce_optional_bool(value, label=label))


def _coerce_optional_bool(value: Any, *, label: str) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().casefold()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"retron_review: {label} contains unsupported boolean value {value!r}")


def _relevant_motifs(
    *,
    sensor: str,
    sponge: str,
    sensor_target_map: dict[str, tuple[str, ...]],
) -> list[str]:
    targets = set(sensor_target_map.get(sensor, ()))
    motifs = _split_motifs(sponge)
    return [motif for motif in motifs if motif in targets]


def _relevant_motif_count(sensor: str, sponge: str, *, sensor_target_map: dict[str, tuple[str, ...]]) -> int:
    return len(_relevant_motifs(sensor=sensor, sponge=sponge, sensor_target_map=sensor_target_map))


def _split_motifs(sponge: str) -> list[str]:
    return [part for part in str(sponge).split("-") if part]


def _motif_count(sponge: str) -> int:
    return len(_split_motifs(sponge))


def _plot_stage_rank(stage: str) -> int:
    try:
        return _PLOT_STAGE_ORDER.index(stage)
    except ValueError:
        return len(_PLOT_STAGE_ORDER)


def _sponge_sort_key(value: str) -> tuple[int, int, str]:
    motif_count = _motif_count(value)
    family = _family_label(value)
    return (_FAMILY_ORDER.get(family, 99), motif_count, str(value))


def _family_label(sponge: str) -> str:
    motif_count = _motif_count(sponge)
    if str(sponge) == "tetO":
        return "control"
    if motif_count <= 1:
        return "mono"
    if motif_count == 2:
        return "bi"
    if motif_count == 3:
        return "tri"
    if motif_count == 4:
        return "quad"
    return f"{motif_count}-site"


def _scope_token(filename: str, *, key: str) -> str | None:
    marker = f"__{key}="
    value = str(filename or "")
    if marker not in value:
        return None
    token = value.split(marker, 1)[1]
    return token.strip() or None


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value)).strip("_").lower()
