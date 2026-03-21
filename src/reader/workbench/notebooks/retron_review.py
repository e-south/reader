from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
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
    "interaction_summary": "C_AUC/C_END",
    "library_heatmaps": "D_AUC/M_AUC/S_AUC",
    "stress_modulation_scores": "M_AUC",
    "pareto_ranking": "S_AUC vs burden",
}

_RETRON_SOURCE_PLOT_SELECTOR_TITLES = {
    "raw_kinetics": "QC traces",
    "control_burden_panel": "tetO burden",
    "baseline_shifted_kinetics": "Baseline-shifted kinetics",
    "matched_control_kinetics": "tetO-subtracted kinetics",
    "induced_effect_kinetics": "IPTG-state effect",
    "absolute_effect_kinetics": "Absolute matched-control effect",
    "interaction_summary": "IPTG x stress summary",
    "library_heatmaps": "Library heatmaps",
    "stress_modulation_scores": "Stress-gated score",
    "pareto_ranking": "Pareto ranking",
}

_RETRON_PLOT_GUIDE = {
    "raw_kinetics": {
        "title": "QC traces",
        "stage": "1. QC",
        "question": "Do any wells or channels show plate-junction offsets or fail basic assay sanity checks before normalization?",
        "math": "Notebook QC surface: raw OD600(t), YFP(t), and CFP(t) from overflow/df, plus YFP/OD600, CFP/OD600, and YFP/CFP from the downstream ratio table.",
        "record": "overflow/df for raw OD600/YFP/CFP; ratio_yfp_od600/df for support ratios",
        "meaning": "Check growth, fluorescence, support ratios, the raw reporter ratio, and any sheet-boundary channel shifts. If CFP flattens at a repeated ceiling, inspect overflow=True rows before blaming the ingest parser.",
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
        "question": "How much of the response comes from IPTG-driven retron expression burden rather than a real sponge site?",
        "math": "tetO-only traces over R(t)=log2(YFP/CFP) and mu(t)=d ln(OD600) / dt.",
        "record": "semantic_metrics/trace",
        "meaning": "Measures IPTG-state burden without a cognate sponge site.",
    },
    "baseline_shifted_kinetics": {
        "title": "Baseline-shifted kinetics",
        "stage": "2. Assay kinetics",
        "question": "Once each well is centered on its own pre-stress state, how do the post-stress trajectories compare?",
        "math": "B(t)=R(t)-R_pre, where R_pre is the mean of the last three pre-stress reads.",
        "record": "semantic_metrics/trace",
        "meaning": "Removes pre-stress offsets so post-stress motion is comparable well to well.",
    },
    "matched_control_kinetics": {
        "title": "Matched-control-normalized kinetics",
        "stage": "2. Assay kinetics",
        "question": "After same-sensor tetO subtraction, which trajectories still look sponge-specific?",
        "math": "C(t)=B(t)-mean(B matched tetO at same sensor, plate, stress, IPTG, and time).",
        "record": "semantic_metrics/trace",
        "meaning": "Shows sponge-specific deviation after same-sensor tetO control subtraction.",
    },
    "induced_effect_kinetics": {
        "title": "IPTG-state effect kinetics",
        "stage": "2. Assay kinetics",
        "question": "After matched-control normalization, how does the +IPTG versus -IPTG contrast evolve within each stress state?",
        "math": "D(t)=mean(C +IPTG)-mean(C -IPTG) within each sensor and stress state.",
        "record": "semantic_metrics/trace",
        "meaning": "Isolates the IPTG-state effect after matched-control normalization. IPTG is present from the start of the assay; the dashed line marks stress addition and the plate-sheet junction, not a new induction event.",
    },
    "absolute_effect_kinetics": {
        "title": "Absolute matched-control effect kinetics",
        "stage": "2. Assay kinetics",
        "question": "If a sponge preloads the reporter before stress, does a same-sensor tetO-matched absolute IPTG effect still remain?",
        "math": "D_abs(t)=mean(R-R_tetO,matched)(+IPTG)-mean(R-R_tetO,matched)(-IPTG) within each sensor and stress state.",
        "record": "semantic_metrics/trace",
        "meaning": "Keeps the same-sensor tetO correction but does not remove pre-stress preload effects. This is the companion view to D(t) when baseline subtraction could undercall derepression-heavy sensors such as sulAp/LexA.",
    },
    "interaction_summary": {
        "title": "IPTG and stress state summary",
        "stage": "3. Ranking and overview",
        "question": "Across the four assay states, is the phenotype linked to IPTG state, stress state, or both?",
        "math": "C_AUC or C_END across the four IPTG/stress states: H2O/-IPTG, H2O/+IPTG, stress/-IPTG, stress/+IPTG.",
        "record": "semantic_metrics/trace + semantic_metrics/summary",
        "meaning": "Shows whether activity tracks IPTG state, requires stress addition, leaks without IPTG, or is burden-dominated.",
    },
    "library_heatmaps": {
        "title": "Library heatmaps",
        "stage": "3. Ranking and overview",
        "question": "Across the library, which sponge-sensor pairs are on-target, stress-dependent, or strong after cross-sensor scaling?",
        "math": "Heatmaps over D_AUC(H2O), D_AUC(relevant stress), M_AUC, and S_AUC.",
        "record": "semantic_metrics/summary",
        "meaning": "Summarizes library-wide on-target, stress-dependent, and cross-sensor scaled effects.",
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
        "question": "Which candidates balance on-target effect with low burden and low leakiness?",
        "math": "On-target score S_AUC versus construct-specific burden (D_growth_AUC by default), with |L_pre| encoded as point size.",
        "record": "semantic_metrics/summary",
        "meaning": "Balances efficacy against burden and leakiness for candidate selection.",
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
        "math": "Selected multifunction sponge plotted beside the matched tetO reference across relevant sensors over S_AUC or O_AUC, with source-experiment replicate points.",
        "meaning": "Shows whether a multi-functional sponge is balanced across its intended sensor arms and how far each arm moves away from the matched tetO control.",
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
        "Details": "The direct-ratio core is R(t)=log2(YFP/CFP), then B(t), C(t), D(t) for incremental post-stress effect, D_abs(t) for preload-sensitive absolute effect, M(t), O(t), and S_AUC for cross-sensor ranking.",
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
        "Step": "Cross-sensor scaling",
        "Formula": "S_AUC=O_AUC / abs(G_sensor), where G_sensor is the native tetO stress response",
        "Output": "summary metric S_AUC",
        "Meaning": "Compares sponge effect size as a fraction of the native sensor response.",
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
        "Math": "Selected sponge versus matched tetO reference across sensors over relevant-stress S_AUC or O_AUC, with source-experiment replicate points.",
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
        "Figure": "Figure 9 — 2x2 interaction summary",
        "Scope": "Per experiment",
        "Surface": "interaction_summary",
        "Coverage": "Exact compiled plot",
        "Math": "C_AUC or C_END across the four IPTG/stress states.",
    },
    {
        "Figure": "Figure 10 — Library heatmaps",
        "Scope": "Per experiment",
        "Surface": "library_heatmaps",
        "Coverage": "Exact compiled plot",
        "Math": "D_AUC, M_AUC, and scaled ranking heatmaps.",
    },
    {
        "Figure": "Figure 11 — Leakiness panel",
        "Scope": "Per experiment",
        "Surface": "assay summary review",
        "Coverage": "Derived from assay tables",
        "Math": "L_pre and L_post_AUC from the derived assay summary table.",
    },
    {
        "Figure": "Figure 12 — Target activity matrix",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "Relevant-stress O_AUC or S_AUC pivoted over sponge x sensor.",
    },
    {
        "Figure": "Figure 13 — Pareto ranking",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "On-target score versus burden, with leakiness encoded.",
    },
    {
        "Figure": "Figure 14 — Architecture plot",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "Relevant-stress O_AUC or S_AUC versus motif complexity.",
    },
    {
        "Figure": "Figure 15 — Observed versus expected multi-functional performance",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "Observed multi-site score versus mono-derived expected score.",
    },
    {
        "Figure": "Figure 16 — Sponge-centric fingerprint plots",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "Selected multifunction sponge versus matched tetO reference across relevant sensors over O_AUC or S_AUC.",
    },
    {
        "Figure": "Figure 17 — Growth impact summary",
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
    plot_specs = retron_visible_plot_specs(tuple(spec.to_dict() for spec in source_context.workbench.plots))
    plot_guides = {row["Plot id"]: row for row in retron_plot_guide_rows([spec.get("id", "") for spec in plot_specs])}
    plot_selector_rows: list[dict[str, str]] = []
    plot_catalog_rows: list[dict[str, str]] = []
    for plot_spec in plot_specs:
        plot_id = str(plot_spec.get("id", ""))
        guide = plot_guides.get(plot_id, {})
        stage = str(guide.get("Stage", "3. Ranking and overview"))
        title = str(
            guide.get(
                "Plot",
                (plot_spec.get("with") or {}).get("title", plot_id),
            )
        )
        plot_selector_rows.append(
            {
                "Selector label": _retron_source_plot_selector_label(plot_id=plot_id, title=title),
                "Stage": stage,
                "Plot": title,
                "Plot id": plot_id,
            }
        )
        rendered_files = retron_plot_rendered_files(
            source_context.plots_dir,
            plot_id=plot_id,
            plugin=str(plot_spec.get("plugin", "")),
        )
        plot_catalog_rows.append(
            {
                "Stage": stage,
                "Plot": title,
                "Plot id": plot_id,
                "Rendered": "yes" if rendered_files else "no",
                "Math / transform": str(
                    guide.get("Math / transform", "Interpret against the direct-ratio matched-control workflow.")
                ),
                "How to read": str(guide.get("How to read", "See the transform ladder for the exact semantics.")),
            }
        )
    plot_selector_rows.sort(key=lambda item: (item["Stage"], item["Plot"]))
    plot_catalog_rows.sort(key=lambda row: (row["Stage"], row["Plot"]))
    record_info, _, _, _ = discover_dataframe_records(source_context.outputs_dir, allow_scan=False)
    record_paths = tuple(
        sorted(
            (
                str(info.get("record_id")),
                str(Path(info["path"]).expanduser().resolve()),
            )
            for info in record_info.values()
            if info.get("record_id") and info.get("path")
        )
    )
    return RetronReviewSourceSurface(
        experiment_title=source_context.decl.experiment.title or source_context.decl.experiment.id,
        protocol_id=source_context.decl.experiment_semantics.protocol.id,
        plot_specs=plot_specs,
        plot_selector_rows=tuple(plot_selector_rows),
        plot_catalog_rows=tuple(plot_catalog_rows),
        record_paths=record_paths,
    )


def _redundant_retron_surface_plot_ids(plot_specs: Sequence[Mapping[str, Any]]) -> set[str]:
    plot_ids = {str(spec.get("id", "")) for spec in plot_specs}
    redundant: set[str] = set()
    if {"raw_kinetics", "support_kinetics"}.issubset(plot_ids):
        redundant.add("support_kinetics")
    redundant.update({"stress_modulation_scores", "pareto_ranking"} & plot_ids)
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
    plugin = str(plot_spec.get("plugin") or "").strip()
    metadata = _plot_guide_metadata(plot_id)
    with_cfg = dict(plot_spec.get("with") or {})

    if plugin == "plot/time_series":
        if plot_id in {"raw_kinetics", "support_kinetics"}:
            qc_df = _retron_qc_dataframe(plot_spec=plot_spec, datasets=datasets)
            figures = _render_retron_qc_plot_spec(plot_spec=plot_spec, datasets=datasets)
            channels = ["OD600", "YFP", "CFP", "YFP/OD600", "CFP/OD600", "YFP/CFP"]
            supporting_table = _time_series_supporting_table(qc_df, channels=channels)
            supporting_table_title = (
                "Underlying overflow-handled raw channel rows plus derived support-ratio rows for the selected QC view"
            )
        else:
            figures = _render_time_series_plot_spec(plot_spec=plot_spec, datasets=datasets)
            channels = _normalize_optional_str_list(with_cfg.get("y")) or _normalize_optional_str_list(
                with_cfg.get("channels")
            )
            supporting_table = _time_series_supporting_table(
                _require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="df"),
                channels=channels,
            )
            supporting_table_title = "Underlying tidy rows for the selected raw or support channels"
    elif plugin == "plot/retron_trace":
        figures = _render_retron_trace_plot_spec(plot_spec=plot_spec, datasets=datasets)
        supporting_table = _trace_supporting_table(
            _require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="trace"),
            metrics=_normalize_optional_str_list(with_cfg.get("metrics")) or [],
        )
        supporting_table_title = "Underlying assay trace rows for the selected kinetic transform"
    elif plugin == "plot/retron_summary":
        figures = _render_retron_summary_plot_spec(plot_spec=plot_spec, datasets=datasets)
        supporting_table = _summary_supporting_table(
            _require_plot_dataset(plot_spec=plot_spec, datasets=datasets, label="summary"),
            view=str(with_cfg.get("view") or ""),
            metric=str(with_cfg.get("metric") or ""),
            burden_metric=str(with_cfg.get("burden_metric") or "D_growth_AUC"),
        )
        supporting_table_title = "Underlying assay summary rows for the selected ranking view"
    else:
        raise ValueError(f"retron_review: unsupported notebook plot plugin {plugin!r}")

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
    if selected_plot_id == "specificity_matrix":
        matrix = build_specificity_matrix(summary_df, score_metric=score_metric)
        figure = _build_specificity_matrix_figure(matrix=matrix, score_metric=score_metric)
        supporting_table = matrix.reset_index().rename(columns={"index": "sponge"})
        supporting_table_title = "Relevant-stress on-target matrix behind the heatmap"
    elif selected_plot_id == "pareto_ranking":
        supporting_table = build_aggregate_pareto_frame(summary_df, score_metric=score_metric)
        figure = _build_aggregate_pareto_figure(pareto_df=supporting_table, score_metric=score_metric)
        supporting_table_title = "Aggregate on-target, burden, and leakiness table for candidate ranking"
    elif selected_plot_id == "architecture_plot":
        supporting_table = build_architecture_frame(
            summary_df,
            sensor_target_map=dict(sensor_target_map),
            score_metric=score_metric,
        )
        figure = _build_architecture_figure(
            architecture_df=supporting_table,
            score_metric=score_metric,
            architecture_x=architecture_x,
        )
        supporting_table_title = "Architecture score table behind the sensor-faceted scatter plots"
    elif selected_plot_id == "expected_vs_observed":
        supporting_table = build_expected_vs_observed_frame(
            summary_df,
            sensor_target_map=dict(sensor_target_map),
            score_metric=score_metric,
        )
        figure = _build_expected_vs_observed_figure(
            expected_vs_observed_df=supporting_table,
            score_metric=score_metric,
            expected_mode=expected_mode,
        )
        supporting_table_title = "Observed and mono-derived expected scores for multifunctional sponges"
    elif selected_plot_id == "sponge_fingerprint":
        supporting_table = build_fingerprint_frame(
            summary_df,
            score_metric=score_metric,
            fingerprint_sponge=fingerprint_sponge,
        )
        figure = _build_fingerprint_figure(
            fingerprint_df=supporting_table,
            score_metric=score_metric,
        )
        supporting_table_title = (
            "Relevant-sensor score table for the selected multifunctional sponge and its source-matched tetO references"
        )
    else:
        raise ValueError(f"retron_review: unknown aggregate plot id {selected_plot_id!r}")

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
    empty_columns = [
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
    required = {"sensor", "sponge", "metric", "value", "is_relevant_stress", "sponge_family_size"}
    missing = sorted(required - set(summary_df.columns))
    if missing:
        raise ValueError(f"retron_review: summary dataframe is missing required columns: {missing}")
    frame = summary_df.copy()
    frame["metric"] = frame["metric"].astype(str)
    frame["sponge"] = frame["sponge"].astype(str)
    frame["sensor"] = frame["sensor"].astype(str)
    frame["sponge_family_size"] = frame["sponge_family_size"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["is_relevant_stress"] = _coerce_optional_bool_series(frame["is_relevant_stress"], label="is_relevant_stress")
    if "relevant_sensor_pair" in frame.columns:
        frame["relevant_sensor_pair"] = _coerce_optional_bool_series(
            frame["relevant_sensor_pair"], label="relevant_sensor_pair"
        )

    sample_rows = frame[
        (frame["metric"] == str(score_metric))
        & frame["is_relevant_stress"].fillna(False)
        & frame["sponge_family_size"].isin({"bi", "tri", "quad"})
    ].copy()
    if "relevant_sensor_pair" in sample_rows.columns:
        sample_rows = sample_rows[sample_rows["relevant_sensor_pair"].fillna(False)]
    available = sorted({str(value) for value in sample_rows["sponge"].dropna()}, key=_sponge_sort_key)
    if not available:
        return pd.DataFrame(columns=empty_columns)
    selected_sponge = (
        str(fingerprint_sponge)
        if fingerprint_sponge is not None and str(fingerprint_sponge) in set(available)
        else available[0]
    )
    sample_rows = sample_rows[sample_rows["sponge"] == selected_sponge].copy()
    if sample_rows.empty:
        return pd.DataFrame(columns=empty_columns)

    source_group_columns = [
        column
        for column in (
            "source_experiment_id",
            "source_label",
            "sensor",
            "stress_condition",
            "sponge",
            "sponge_family_size",
        )
        if column in sample_rows.columns
    ]
    sample_rows = (
        sample_rows.groupby(source_group_columns, dropna=False)["value"].mean().reset_index()
        if source_group_columns
        else sample_rows[["value"]].copy()
    )

    control_rows = frame[
        (frame["metric"] == str(score_metric))
        & frame["is_relevant_stress"].fillna(False)
        & (frame["sponge"] == str(control_name))
        & frame["sensor"].isin(sample_rows["sensor"].astype(str))
    ].copy()
    control_group_columns = [
        column
        for column in (
            "source_experiment_id",
            "source_label",
            "sensor",
            "stress_condition",
            "sponge",
            "sponge_family_size",
        )
        if column in control_rows.columns
    ]
    control_rows = (
        control_rows.groupby(control_group_columns, dropna=False)["value"].mean().reset_index()
        if control_group_columns
        else control_rows[["value"]].copy()
    )
    control_rows = control_rows.rename(
        columns={
            "value": "control_value",
            "sponge": "control_sponge",
            "sponge_family_size": "control_family_size",
        }
    )

    match_columns = [
        column
        for column in ("source_experiment_id", "source_label", "sensor", "stress_condition")
        if column in sample_rows.columns and column in control_rows.columns
    ]
    paired_rows = sample_rows.merge(
        control_rows[match_columns + ["control_value", "control_sponge", "control_family_size"]],
        on=match_columns,
        how="left",
    )

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
        sponge = str(row.sponge)
        sponge_family_size = str(row.sponge_family_size) if has_sample_family_size else "other"
        sample_value = float(row.value)
        long_rows.append(
            {
                "selected_sponge": selected_sponge,
                "sensor": sensor,
                "stress_condition": stress_condition,
                "source_experiment_id": source_experiment_id,
                "source_label": source_label,
                "comparison_group": "Selected sponge",
                "sponge": sponge,
                "sponge_family_size": sponge_family_size,
                "value": sample_value,
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

    out = pd.DataFrame(long_rows)
    if out.empty:
        return out
    sensor_order = sorted(out["sensor"].dropna().astype(str).unique())
    sensor_order_map = {sensor: idx for idx, sensor in enumerate(sensor_order)}
    out["__sensor_order"] = out["sensor"].map(sensor_order_map)
    out["__group_order"] = out["comparison_group"].map({"tetO reference": 0, "Selected sponge": 1}).fillna(99)
    order = ["__sensor_order", "__group_order", "source_experiment_id", "source_label", "sponge"]
    out = out.sort_values(order, kind="stable").drop(columns=["__sensor_order", "__group_order"])
    return out.reset_index(drop=True)


def available_multifunctional_sponges(summary_df: pd.DataFrame) -> list[str]:
    scores = aggregate_on_target_scores(summary_df, score_metric="S_AUC")
    if scores.empty:
        return []
    multi = scores[scores["sponge_family_size"].astype(str).isin({"bi", "tri", "quad"})]
    return sorted({str(value) for value in multi["sponge"].dropna()}, key=_sponge_sort_key)


def aggregate_on_target_scores(summary_df: pd.DataFrame, *, score_metric: str) -> pd.DataFrame:
    required = {"sensor", "sponge", "metric", "value", "relevant_sensor_pair", "is_relevant_stress"}
    missing = sorted(required - set(summary_df.columns))
    if missing:
        raise ValueError(f"retron_review: summary dataframe is missing required columns: {missing}")
    frame = summary_df.copy()
    frame["metric"] = frame["metric"].astype(str)
    frame["sponge"] = frame["sponge"].astype(str)
    frame["sensor"] = frame["sensor"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["relevant_sensor_pair"] = _coerce_optional_bool_series(
        frame["relevant_sensor_pair"], label="relevant_sensor_pair"
    )
    frame["is_relevant_stress"] = _coerce_optional_bool_series(frame["is_relevant_stress"], label="is_relevant_stress")
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
        metric_names = ["S_AUC", burden_metric or "D_growth_AUC", "L_pre"]
    elif view_id == "heatmap":
        metric_names = ["D_AUC", "M_AUC", "S_AUC"]
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
    axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    axis.tick_params(axis="y", labelrotation=0, labelsize=9)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    return figure


def _build_aggregate_pareto_figure(*, pareto_df: pd.DataFrame, score_metric: str) -> Any | None:
    if pareto_df.empty:
        return None
    family_levels = _ordered_text(pareto_df["sponge_family_size"].fillna("other").astype(str).tolist())
    color_values = sns.color_palette("colorblind", n_colors=max(1, len(family_levels)))
    color_map = {family: color_values[idx % len(color_values)] for idx, family in enumerate(family_levels)}
    sizes = 90.0 + 260.0 * pareto_df["leakiness"].fillna(0.0)
    figure, axis = plt.subplots(figsize=(8.2, 6.0), constrained_layout=False)
    axis.scatter(
        pareto_df["on_target"],
        pareto_df["burden"],
        s=sizes,
        c=[color_map.get(str(item), "#4c72b0") for item in pareto_df["sponge_family_size"].fillna("other")],
        alpha=0.85,
        edgecolors="#222222",
        linewidths=0.5,
        zorder=2,
    )
    annotate_points_smart(
        ax=axis,
        points=[(float(row["on_target"]), float(row["burden"])) for _, row in pareto_df.iterrows()],
        labels=[str(row["sponge"]) for _, row in pareto_df.iterrows()],
        text_kwargs={"fontsize": 8},
    )
    axis.axvline(0.0, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
    axis.axhline(0.0, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
    axis.set_xlabel(f"Mean on-target effect ({score_metric})", fontsize=11)
    axis.set_ylabel("Mean construct-specific growth burden (D_growth_AUC)", fontsize=11)
    axis.tick_params(axis="both", labelsize=7)
    with suppress(Exception):
        axis.set_box_aspect(1.0)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=family,
            markerfacecolor=color_map[family],
            markeredgecolor="#222222",
            markersize=7,
        )
        for family in family_levels
    ]
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


def _build_architecture_figure(
    *,
    architecture_df: pd.DataFrame,
    score_metric: str,
    architecture_x: str,
) -> Any | None:
    if architecture_df.empty:
        return None

    sensors = sorted(architecture_df["sensor"].dropna().astype(str).unique())
    family_levels = _ordered_text(architecture_df["sponge_family_size"].fillna("other").astype(str).tolist())
    palette = {level: _FAMILY_COLOR_MAP.get(level, _FAMILY_COLOR_MAP["other"]) for level in family_levels}
    figure, axes = plt.subplots(
        1,
        len(sensors),
        figsize=(5.3 * len(sensors), 4.9),
        constrained_layout=False,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
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
        sns.scatterplot(
            data=sensor_df,
            x=architecture_x,
            y="value",
            hue="sponge_family_size",
            palette=palette,
            s=88,
            edgecolor="black",
            linewidth=0.45,
            ax=axis,
        )
        annotate_points_smart(
            ax=axis,
            points=[(float(row[architecture_x]), float(row["value"])) for _, row in sensor_df.iterrows()],
            labels=[str(row["sponge"]) for _, row in sensor_df.iterrows()],
            text_kwargs={"fontsize": 8},
        )
        axis.axhline(0.0, color="#777777", linestyle=":", linewidth=1.0)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_title(str(sensor), pad=8, fontweight="normal")
        axis.set_xlabel("")
        axis.set_ylabel("")
    handles, labels = axes[0][0].get_legend_handles_labels()
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
    for axis in axes[0]:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
        with suppress(Exception):
            axis.set_box_aspect(1.0)
    figure.supxlabel(_architecture_axis_label(architecture_x), y=0.09)
    figure.supylabel(_metric_axis_label(score_metric), x=0.02)
    figure.subplots_adjust(bottom=0.18, left=0.10, right=0.84, top=0.88, wspace=0.24)
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
    family_levels = _ordered_text(expected_vs_observed_df["sponge_family_size"].fillna("other").astype(str).tolist())
    palette = {level: _FAMILY_COLOR_MAP.get(level, _FAMILY_COLOR_MAP["other"]) for level in family_levels}
    figure, axes = plt.subplots(
        1,
        len(sensors),
        figsize=(5.3 * len(sensors), 5.1),
        constrained_layout=False,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
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
        sns.scatterplot(
            data=sensor_df,
            x=expected_mode,
            y="observed",
            hue="sponge_family_size",
            palette=palette,
            s=92,
            edgecolor="black",
            linewidth=0.45,
            ax=axis,
        )
        axis.plot(
            [combined_limits[0], combined_limits[1]],
            [combined_limits[0], combined_limits[1]],
            color="#777777",
            linestyle=":",
            linewidth=1.0,
        )
        annotate_points_smart(
            ax=axis,
            points=[(float(row[expected_mode]), float(row["observed"])) for _, row in sensor_df.iterrows()],
            labels=[str(row["sponge"]) for _, row in sensor_df.iterrows()],
            text_kwargs={"fontsize": 8},
        )
        axis.set_xlim(combined_limits)
        axis.set_ylim(combined_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(str(sensor), pad=8, fontweight="normal")
        axis.set_xlabel("")
        axis.set_ylabel("")
    handles, labels = axes[0][0].get_legend_handles_labels()
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
    for axis in axes[0]:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
        with suppress(Exception):
            axis.set_box_aspect(1.0)
    figure.supxlabel(_expected_axis_label(expected_mode, score_metric=score_metric), y=0.09)
    figure.supylabel(f"Observed multifunction score ({score_metric})", x=0.02)
    figure.subplots_adjust(bottom=0.18, left=0.10, right=0.84, top=0.88, wspace=0.24)
    return figure


def _build_fingerprint_figure(
    *,
    fingerprint_df: pd.DataFrame,
    score_metric: str,
) -> Any | None:
    if fingerprint_df.empty:
        return None
    selected_sponge = str(fingerprint_df["selected_sponge"].dropna().astype(str).iloc[0])
    sensor_levels = sorted(fingerprint_df["sensor"].dropna().astype(str).unique().tolist())
    comparison_order = [
        group for group in ("tetO reference", "Selected sponge") if group in set(fingerprint_df["comparison_group"])
    ]
    stats = (
        fingerprint_df.groupby(["sensor", "comparison_group"], dropna=False)["value"]
        .agg(mean="mean", sd="std", n="size")
        .reset_index()
    )
    y_limits = shared_numeric_limits(
        fingerprint_df["value"].to_numpy(dtype=float, copy=False),
        center=0.0,
        pad_fraction=0.12,
        min_span=0.10,
    )
    comparison_colors = {
        "tetO reference": "#f3ebe7",
        "Selected sponge": "#4c72b0",
    }
    edge_colors = {
        "tetO reference": "#9b7d72",
        "Selected sponge": "#1f3552",
    }
    point_facecolors = {
        "tetO reference": "#ffffff",
        "Selected sponge": "#4c72b0",
    }
    source_counts = (
        fingerprint_df.groupby(["sensor", "comparison_group"], dropna=False)["value"]
        .size()
        .reset_index(name="n_sources")
    )
    max_sources = int(source_counts["n_sources"].max()) if not source_counts.empty else 0
    width = 0.34
    offsets = {group: ((idx - (len(comparison_order) - 1) / 2.0) * width) for idx, group in enumerate(comparison_order)}
    figure, axis = plt.subplots(
        figsize=(max(6.0, 1.8 + 1.45 * len(sensor_levels)), 4.9),
        constrained_layout=False,
    )
    x_positions = {sensor: float(idx) for idx, sensor in enumerate(sensor_levels)}
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
    axis.axhline(0.0, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
    axis.set_xlim(-0.55, max(0.55, len(sensor_levels) - 0.45))
    axis.set_ylim(y_limits)
    axis.set_xticks([x_positions[sensor] for sensor in sensor_levels])
    axis.set_xticklabels(sensor_levels)
    axis.set_xlabel("Relevant sensor arm", fontsize=11)
    axis.set_ylabel(_metric_axis_label(score_metric), fontsize=11)
    axis.tick_params(axis="x", labelsize=9)
    axis.tick_params(axis="y", labelsize=8)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.50)
    axis.set_title(selected_sponge, pad=10, fontweight="normal", fontsize=11)
    axis.legend(frameon=False, title=None, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    figure.suptitle("Sponge fingerprint", y=0.97, x=0.5, ha="center", fontweight="normal", fontsize=13)
    figure.text(
        0.5,
        0.92,
        (
            "Bars show means across source experiments; points show source-level replicates against the matched tetO reference."
            if max_sources > 1
            else "Bars show the single available source experiment per arm (n=1 in this view); points mark that source-level estimate against the matched tetO reference."
        ),
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
    )
    figure.subplots_adjust(bottom=0.16, left=0.11, right=0.80, top=0.84)
    return figure


def _metric_axis_label(metric: str) -> str:
    labels = {
        "S_AUC": "Scaled on-target effect (S_AUC)",
        "O_AUC": "Sign-corrected effect (O_AUC)",
        "C": "Matched-control-normalized ratio response (C)",
        "D": "IPTG-state effect (D)",
        "M": "Stress modulation (M)",
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
    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("retron_review: manifest 'sources' must be a list")
    resolved: list[RetronReviewSource] = []
    for idx, raw_source in enumerate(raw_sources, start=1):
        if not isinstance(raw_source, dict):
            raise ValueError(f"retron_review: sources[{idx}] must be a mapping")
        label = str(raw_source.get("label") or raw_source.get("family") or f"source_{idx}").strip()
        experiment_raw = raw_source.get("experiment")
        config_raw = raw_source.get("config")
        summary_raw = raw_source.get("summary")
        trace_raw = raw_source.get("trace")

        experiment_root: Path | None = None
        config_path: Path | None = None
        if experiment_raw is not None:
            experiment_root = _resolve_manifest_path(
                manifest_path,
                str(experiment_raw),
                relative_to=source_root,
            )
            config_path = experiment_root / "config.yaml"
        elif config_raw is not None:
            config_path = _resolve_manifest_path(
                manifest_path,
                str(config_raw),
                relative_to=source_root,
            )
            experiment_root = config_path.parent

        if summary_raw is not None:
            summary_path = _resolve_manifest_path(manifest_path, str(summary_raw))
        else:
            if experiment_root is None:
                raise ValueError(
                    "retron_review: each source must declare either 'experiment' or explicit 'summary'/'trace' paths"
                )
            summary_path = _resolve_default_semantic_table(
                experiment_root,
                record_id="semantic_metrics/summary",
                export_name="semantic_summary.csv",
            )
        if trace_raw is not None:
            trace_path = _resolve_manifest_path(manifest_path, str(trace_raw))
        else:
            if experiment_root is None:
                raise ValueError(
                    "retron_review: each source must declare either 'experiment' or explicit 'summary'/'trace' paths"
                )
            trace_path = _resolve_default_semantic_table(
                experiment_root,
                record_id="semantic_metrics/trace",
                export_name="semantic_trace.csv",
            )

        if not summary_path.exists() or not trace_path.exists():
            command = ""
            if config_path is not None:
                command = f" Run 'uv run reader run {config_path}' and 'uv run reader export {config_path}'."
            missing = [str(path) for path in (summary_path, trace_path) if not path.exists()]
            raise FileNotFoundError(f"retron_review: source exports are missing for {label}: {missing}.{command}")
        experiment_id = str(
            raw_source.get("experiment_id") or (experiment_root.name if experiment_root is not None else label)
        )
        resolved.append(
            RetronReviewSource(
                label=label,
                experiment_id=experiment_id,
                experiment_root=experiment_root,
                config_path=config_path,
                summary_path=summary_path.resolve(),
                trace_path=trace_path.resolve(),
            )
        )
    return resolved


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
