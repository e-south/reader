from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

PLOT_STAGE_ORDER = ("1. QC", "2. Assay kinetics", "3. Ranking and overview")
DEFAULT_RETRON_BURDEN_METRIC = "D_growth_AUC"
RETRON_QC_PLOT_IDS = frozenset({"raw_kinetics", "support_kinetics"})


@dataclass(frozen=True)
class ExperimentPlotGuideMetadata:
    title: str
    stage: str
    question: str
    math: str
    record: str
    meaning: str
    display_order: int = 999
    selector_title: str | None = None
    selector_tag: str | None = None


@dataclass(frozen=True)
class SourceSelectorPresentation:
    display_label: str

    def selector_label(self, *, experiment_id: str, duplicate_count: int) -> str:
        if duplicate_count <= 1:
            return self.display_label
        return f"{self.display_label} • {experiment_id}"


@dataclass(frozen=True)
class AggregatePlotGuideMetadata:
    title: str
    question: str
    math: str
    meaning: str
    display_order: int = 999


@dataclass(frozen=True)
class AggregatePlotCatalogEntry:
    guide: AggregatePlotGuideMetadata
    supporting_table_title: str


_RETRON_SOURCE_SELECTOR_PRESENTATIONS = {
    "mono": SourceSelectorPresentation(display_label="mono"),
    "bi": SourceSelectorPresentation(display_label="bi"),
    "tri": SourceSelectorPresentation(display_label="tri"),
    "quad": SourceSelectorPresentation(display_label="tetra"),
    "tetra": SourceSelectorPresentation(display_label="tetra"),
    "control": SourceSelectorPresentation(display_label="control"),
    "lexA_cpxr_baer_bi": SourceSelectorPresentation(display_label="bi · LexA/CpxR/BaeR"),
    "sox_bi": SourceSelectorPresentation(display_label="bi · Sox family"),
}

_RETRON_EXPERIMENT_PLOT_GUIDES = {
    "raw_kinetics": ExperimentPlotGuideMetadata(
        title="QC raw channels",
        stage="1. QC",
        question="Are the raw channels clean enough to trust the screen?",
        math="OD600(t), YFP(t), CFP(t), and R(t)=log2(YFP/CFP)",
        record="overflow/df for raw OD600/YFP/CFP; ratio_yfp_od600/df for support ratios",
        meaning=(
            "QC screen for sheet shifts, saturated channels, and failed wells before any derived score is trusted. "
            "Support ratios stay available, but they do not lead the review."
        ),
        display_order=10,
    ),
    "support_kinetics": ExperimentPlotGuideMetadata(
        title="QC raw channels",
        stage="1. QC",
        question=("Do the fluorescence channels track biomass cleanly, or is one reporter behaving strangely?"),
        math="YFP/OD600, CFP/OD600, R(t)=log2(YFP/CFP), with OD600(t), YFP(t), and CFP(t) for context",
        record="ratio_yfp_od600/df for support ratios; overflow/df for raw OD600/YFP/CFP context",
        meaning=(
            "Separates broad growth or brightness shifts from a genuine reporter imbalance. Overflow can "
            "flatten one channel without implying a ratio bug."
        ),
        display_order=21,
    ),
    "control_burden_panel": ExperimentPlotGuideMetadata(
        title="Advanced QC / burden",
        stage="1. QC",
        question="How much movement comes from IPTG-driven retron expression alone in tetO wells?",
        math="tetO-only traces over R(t)=log2(YFP/CFP) and mu(t)=d ln(OD600) / dt.",
        record="semantic_metrics/trace",
        meaning="Secondary control check for tetO-only burden and ratio movement under the same assay timing.",
        display_order=60,
        selector_title="Advanced QC / burden",
    ),
    "baseline_shifted_kinetics": ExperimentPlotGuideMetadata(
        title="Advanced: shift from pre-stress state",
        stage="2. Assay kinetics",
        question="After shifting each well to its own pre-stress state, how do the trajectories compare?",
        math="B(t)=R(t)-R_pre, where R_pre is the mean of the last three pre-stress reads.",
        record="semantic_metrics/trace",
        meaning="Mechanism view only. It hides absolute preload, so it should not lead the assay review.",
        display_order=70,
        selector_title="Advanced: shift from pre-stress state",
    ),
    "matched_control_kinetics": ExperimentPlotGuideMetadata(
        title="Advanced: matched-control deviation",
        stage="2. Assay kinetics",
        question=("After same-sensor tetO subtraction, where does the sponge still depart from the matched control?"),
        math="C(t)=B(t)-mean(B matched tetO at same sensor, plate, stress, IPTG, and time).",
        record="semantic_metrics/trace",
        meaning=(
            "Debug and mechanism view. It makes pre-stress departures visible, but it is too abstract for first-pass review."
        ),
        display_order=80,
        selector_title="Advanced: matched-control deviation",
    ),
    "induced_effect_kinetics": ExperimentPlotGuideMetadata(
        title="Post-stress increment over time",
        stage="2. Assay kinetics",
        question="After removing preload, how much new movement appears after stress in +IPTG relative to -IPTG?",
        math="D(t)=mean(C +IPTG)-mean(C -IPTG) within each sensor and stress state.",
        record="semantic_metrics/trace",
        meaning=(
            "Mechanistic view of the post-stress increment after preload removal, not the full IPTG-dependent effect."
        ),
        display_order=40,
        selector_title="Post-stress increment over time",
    ),
    "absolute_effect_kinetics": ExperimentPlotGuideMetadata(
        title="Total effect beyond matched tetO over time",
        stage="2. Assay kinetics",
        question="Across the run, does +IPTG move the sponge beyond what matched tetO induction already does?",
        math="D_abs(t)=mean(R-R_tetO,matched)(+IPTG)-mean(R-R_tetO,matched)(-IPTG) within each sensor and stress state.",
        record="semantic_metrics/trace",
        meaning=(
            "Main kinetic evidence after QC. It keeps strict matched-tetO comparison while preserving preload that D(t) intentionally removes."
        ),
        display_order=30,
        selector_title="Total effect beyond matched tetO over time",
    ),
    "control_anchored_decomposition": ExperimentPlotGuideMetadata(
        title="Reporter-ratio shifts by IPTG state against matched tetO",
        stage="2. Assay kinetics",
        question=(
            "Does +IPTG shift the reporter ratio away from the matched tetO control, and is that shift already "
            "present before stress or does it emerge after stress?"
        ),
        math=(
            "R(t)=log2(YFP/CFP)\n"
            "P_pre=delta_IPTG[R_pre-R_pre,tetO,matched]\n"
            "D_abs_AUC=AUC_window[D_abs(t)]\n"
            "D_AUC=AUC_window[D(t)]"
        ),
        record="semantic_metrics/trace",
        meaning=(
            "Primary assay readout. Each row keeps the raw reporter-ratio traces beside preload, total effect, "
            "and post-stress summaries so the matched tetO comparison stays tied to the signal the wells produced."
        ),
        display_order=20,
        selector_title="Sponge vs matched tetO",
    ),
    "interaction_summary": ExperimentPlotGuideMetadata(
        title="IPTG and stress state summary",
        stage="3. Ranking and overview",
        question="Across the four assay states, is the signal dominated by IPTG state, stress state, or their combination?",
        math="C_AUC or C_END across the four IPTG/stress states: H2O/-IPTG, H2O/+IPTG, stress/-IPTG, stress/+IPTG.",
        record="semantic_metrics/trace + semantic_metrics/summary",
        meaning="Compact state summary after matched-control normalization. Use it after the trace views, not before.",
        display_order=90,
        selector_title="Advanced: IPTG x stress summary",
    ),
    "library_heatmaps": ExperimentPlotGuideMetadata(
        title="Library heatmaps",
        stage="3. Ranking and overview",
        question=(
            "Across the library, which designs carry their signal in total matched-tetO effect, in the post-stress "
            "increment, or in preload before stress?"
        ),
        math=(
            "Total effect: S_abs_AUC = O_abs_AUC / |G_sensor|\n"
            "Post-stress: S_AUC = O_AUC / |G_sensor|\n"
            "Preload: P_pre = delta_IPTG[R_pre - R_pre,tetO,matched]"
        ),
        record="semantic_metrics/summary",
        meaning=(
            "These panels summarize the relevant-stress rows. The AUC panels come from the primary post-stress "
            "window, while preload shows the matched-control offset that was already present before the stress pulse."
        ),
        display_order=50,
        selector_title="Library heatmaps",
    ),
    "stress_modulation_scores": ExperimentPlotGuideMetadata(
        title="Stress modulation scores",
        stage="3. Ranking and overview",
        question="Which on-target effects become materially stronger once the relevant pathway is stressed?",
        math="M_AUC=AUC(D(relevant stress)-D(H2O)).",
        record="semantic_metrics/summary",
        meaning="Ranks how strongly stress unmasks a sponge effect.",
        display_order=110,
        selector_title="Stress-gated score",
    ),
    "pareto_ranking": ExperimentPlotGuideMetadata(
        title="Pareto ranking",
        stage="3. Ranking and overview",
        question="Which candidates balance strong absolute on-target effect with low burden and low leakiness?",
        math=(
            "Absolute on-target score S_abs_AUC versus construct-specific burden (D_growth_AUC by default), "
            "with |L_pre| encoded as point size."
        ),
        record="semantic_metrics/summary",
        meaning="Balances total effect against burden and leakiness when you need to rank candidates.",
        display_order=120,
        selector_title="Pareto ranking",
    ),
}

_RETRON_ASSAY_CONTEXT = (
    {
        "Topic": "Design unit",
        "Details": (
            "Each genotype is one sensor plasmid plus one sponge plasmid measured in a 2x2 H2O or stress "
            "by -IPTG or +IPTG design."
        ),
    },
    {
        "Topic": "Timing",
        "Details": (
            "IPTG is present in the starting media and sets the retron-expression state from the start of the "
            "run. The t=0 boundary in kinetics plots marks stress addition and the plate-sheet junction, not "
            "IPTG addition or sponge induction."
        ),
    },
    {
        "Topic": "Matched control",
        "Details": (
            "Every real sponge row is normalized to the same-sensor tetO control on the same plate, stress "
            "state, IPTG state, and timepoint."
        ),
    },
    {
        "Topic": "Primary score",
        "Details": (
            "Read the assay in this order: observed reporter ratio, preload shift, total effect beyond matched tetO, "
            "post-stress increment, burden, then cross-sensor ranking."
        ),
    },
    {
        "Topic": "Decision logic",
        "Details": (
            "Start with the matched-tetO comparison. Then ask whether the difference was already present before "
            "stress, whether it grows after stress, and what it costs in burden."
        ),
    },
)

_RETRON_TRANSFORM_LADDER = (
    {
        "Step": "Raw channels",
        "Formula": "OD600(t), CFP(t), YFP(t)",
        "Output": "raw QC only",
        "Meaning": "Check growth, saturation, drift, and failed wells before reading any score.",
    },
    {
        "Step": "Support channels",
        "Formula": "YFP/OD600 and CFP/OD600",
        "Output": "support QC only",
        "Meaning": "Separate growth-linked channel shifts from a genuine reporter-ratio change.",
    },
    {
        "Step": "Primary ratio",
        "Formula": "R(t)=log2(YFP/CFP)",
        "Output": "trace metric R",
        "Meaning": "Observed reporter ratio for each well.",
    },
    {
        "Step": "Pre-stress baseline",
        "Formula": "R_pre=mean(last 3 pre-stress reads of R)",
        "Output": "summary metric R_pre",
        "Meaning": "Defines each well's level just before the stress pulse.",
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
        "Meaning": "Moves each well to its own pre-stress baseline while preserving later shape.",
    },
    {
        "Step": "Matched tetO normalization",
        "Formula": "C(t)=B(t)-mean(B matched tetO at same sensor, plate, stress, IPTG, and time)",
        "Output": "trace metric C",
        "Meaning": "Subtracts the same-sensor tetO control on the same plate and assay state.",
    },
    {
        "Step": "IPTG-state effect",
        "Formula": "D(t)=mean(C +IPTG)-mean(C -IPTG)",
        "Output": "trace metric D; summary D_AUC and D_END",
        "Meaning": (
            "Compares +IPTG and -IPTG after matched-control normalization. Because IPTG is present from the start, "
            "this is the new post-stress increment after preload removal, not the full IPTG effect."
        ),
    },
    {
        "Step": "Absolute matched-control effect",
        "Formula": "D_abs(t)=mean(R-R_tetO,matched)(+IPTG)-mean(R-R_tetO,matched)(-IPTG)",
        "Output": "trace metric D_abs; summary D_abs_AUC and D_abs_END",
        "Meaning": (
            "Keeps same-sensor tetO subtraction but preserves preload that D(t) intentionally removes. This is the "
            "main within-sensor evidence for whether a sponge works at all."
        ),
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
        "Meaning": (
            "Scales total and incremental effects by native sensor range so cross-sensor comparisons stay interpretable."
        ),
    },
    {
        "Step": "Leakiness and burden",
        "Formula": "L_pre, L_post_AUC, D_growth_AUC, T_ratio_AUC, T_growth_AUC, T_finalOD",
        "Output": "summary metrics",
        "Meaning": (
            "Separates strong hits from leaky constructs and distinguishes sponge-specific burden from tetO burden."
        ),
    },
)

_RETRON_AGGREGATE_FIGURES = (
    {
        "Figure": "Target activity matrix",
        "Math": "Cross-run pivot over relevant-stress S_AUC or O_AUC for tested on-target sensor/sponge pairs.",
        "Why": (
            "Shows how mono, bi, tri, and quad sponges distribute activity across the sensor arms they were "
            "actually tested against."
        ),
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
        "Math": (
            "Selected sponge versus matched tetO reference across sensors over relevant-stress S_AUC or O_AUC, "
            "with source-level points when available."
        ),
        "Why": (
            "Shows whether a multi-functional sponge is balanced across its intended sensor arms and whether "
            "that signal sits above the matched tetO baseline."
        ),
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
        "Coverage": "Advanced QC / burden follow-on",
        "Math": "tetO-only R(t) and mu(t) traces in a compact control-only follow-on view.",
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
        "Figure": "Figure 9 — Sponge vs matched tetO",
        "Scope": "Per experiment",
        "Surface": "control_anchored_decomposition",
        "Coverage": "Exact compiled plot",
        "Math": (
            "Relevant-stress and H2O R(t) traces for the selected sponge and matched tetO under +/-IPTG, with "
            "replicate-preserving preload, total-effect, and post-stress summaries beneath the trace grid."
        ),
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
        "Math": "Relevant-stress total-effect, post-stress, and preload heatmaps.",
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
        "Math": "Relevant-stress O_abs_AUC or S_abs_AUC pivoted over sponge x sensor.",
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
        "Math": (
            "Selected multifunction sponge versus matched tetO reference across relevant sensors over O_abs_AUC, S_abs_AUC, O_AUC, or S_AUC."
        ),
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

RETRON_AGGREGATE_PLOT_CATALOG = {
    "specificity_matrix": AggregatePlotCatalogEntry(
        guide=AggregatePlotGuideMetadata(
            title="Target activity matrix",
            question="Across the tested on-target sensor arms, which designs have the strongest total effect in the expected direction?",
            math="score(sensor,sponge)=mean_source[O_abs_AUC or S_abs_AUC] at the sensor-matched stress",
            meaning=(
                "Shows how mono, bi, tri, and quad sponges distribute activity across the sensor arms they were "
                "actually tested against, without implying exhaustive off-target specificity coverage."
            ),
            display_order=10,
        ),
        supporting_table_title="Relevant-stress on-target matrix behind the heatmap",
    ),
    "pareto_ranking": AggregatePlotCatalogEntry(
        guide=AggregatePlotGuideMetadata(
            title="Pareto ranking",
            question="Which sponge designs stay strong after burden and leakiness are considered across the review set?",
            math=("x=mean[O_abs_AUC or S_abs_AUC]; y=mean[-D_growth_AUC]; size=mean[abs(L_pre)]"),
            meaning="Ranks candidates across the full review set while keeping total effect, burden, and leakiness in view.",
            display_order=20,
        ),
        supporting_table_title="Aggregate on-target, burden, and leakiness table for candidate ranking",
    ),
    "architecture_plot": AggregatePlotCatalogEntry(
        guide=AggregatePlotGuideMetadata(
            title="Architecture plot",
            question="Does adding extra motifs preserve, dilute, or redistribute the intended sponge effect?",
            math="x=motif_count or irrelevant_motif_count; y=O_abs_AUC, S_abs_AUC, O_AUC, or S_AUC at the sensor-matched stress",
            meaning="Tests whether extra motifs preserve, dilute, or redistribute the relevant sponge arm.",
            display_order=30,
        ),
        supporting_table_title="Architecture score table behind the sensor-faceted scatter plots",
    ),
    "expected_vs_observed": AggregatePlotCatalogEntry(
        guide=AggregatePlotGuideMetadata(
            title="Expected vs observed multifunction performance",
            question="Do multifunctional designs behave additively, better than additive, or worse than additive?",
            math="x=expected_sum or expected_best_single from mono arms; y=observed multifunction score",
            meaning="Separates additive multifunction behavior from dilution or synergy.",
            display_order=40,
        ),
        supporting_table_title="Observed and mono-derived expected scores for multifunctional sponges",
    ),
    "sponge_fingerprint": AggregatePlotCatalogEntry(
        guide=AggregatePlotGuideMetadata(
            title="Relevant sensor arms by sponge",
            question="Across the multifunctional sponges, which intended sensor arms are strong and which are weak?",
            math=(
                "score(sensor,sponge)=O_abs_AUC, S_abs_AUC, O_AUC, or S_AUC at the relevant stress, "
                "shown beside matched tetO references"
            ),
            meaning=(
                "Shows whether each multifunctional sponge is balanced across its intended sensor arms and how far "
                "each arm moves away from the matched tetO baseline."
            ),
            display_order=50,
        ),
        supporting_table_title="Relevant-sensor score table for multifunctional sponges and their source-matched tetO references",
    ),
}


def summary_supporting_metrics(
    *,
    view: str,
    metric: str,
    burden_metric: str,
) -> list[str]:
    view_id = str(view)
    metric_names = _SUMMARY_SUPPORTING_METRIC_BUILDERS.get(view_id, _default_summary_supporting_metrics)(
        metric=metric,
        burden_metric=burden_metric,
    )
    return list(dict.fromkeys(name for name in metric_names if name))


def _interaction_summary_supporting_metrics(*, metric: str, burden_metric: str) -> tuple[str, ...]:
    del burden_metric
    return (metric or "C_AUC",)


def _stress_modulation_supporting_metrics(*, metric: str, burden_metric: str) -> tuple[str, ...]:
    del burden_metric
    return (metric or "M_AUC",)


def _pareto_summary_supporting_metrics(*, metric: str, burden_metric: str) -> tuple[str, ...]:
    return (metric or "S_abs_AUC", burden_metric or DEFAULT_RETRON_BURDEN_METRIC, "L_pre", "P_pre")


def _heatmap_summary_supporting_metrics(*, metric: str, burden_metric: str) -> tuple[str, ...]:
    del metric, burden_metric
    return ("S_abs_AUC", "S_AUC", "P_pre")


def _default_summary_supporting_metrics(*, metric: str, burden_metric: str) -> tuple[str, ...]:
    del burden_metric
    return (metric,) if metric else ()


_SUMMARY_SUPPORTING_METRIC_BUILDERS: dict[str, Callable[..., tuple[str, ...]]] = {
    "interaction": _interaction_summary_supporting_metrics,
    "stress_modulation": _stress_modulation_supporting_metrics,
    "pareto": _pareto_summary_supporting_metrics,
    "heatmap": _heatmap_summary_supporting_metrics,
}


def default_experiment_plot_guide(plot_id: str) -> ExperimentPlotGuideMetadata:
    return ExperimentPlotGuideMetadata(
        title=plot_id.replace("_", " ").title(),
        stage=PLOT_STAGE_ORDER[-1],
        question="What does this plot contribute to the retron sponge decision path?",
        math="Protocol-specific transform guide not registered.",
        record="see compiled plot spec",
        meaning="Interpret in the context of the compiled assay semantics.",
    )


def experiment_plot_guide(plot_id: str) -> ExperimentPlotGuideMetadata:
    return _RETRON_EXPERIMENT_PLOT_GUIDES.get(str(plot_id), default_experiment_plot_guide(str(plot_id)))


def experiment_plot_display_order(plot_id: str) -> int:
    return int(experiment_plot_guide(plot_id).display_order)


def aggregate_plot_display_order(plot_id: str) -> int:
    entry = RETRON_AGGREGATE_PLOT_CATALOG.get(str(plot_id))
    if entry is None:
        return 999
    return int(entry.guide.display_order)


def source_selector_presentation(*, label: str, experiment_id: str) -> SourceSelectorPresentation:
    normalized = str(label).strip()
    if normalized in _RETRON_SOURCE_SELECTOR_PRESENTATIONS:
        return _RETRON_SOURCE_SELECTOR_PRESENTATIONS[normalized]
    humanized = normalized.replace("_", " ")
    return SourceSelectorPresentation(display_label=humanized or str(experiment_id))


def _copy_rows(rows: tuple[dict[str, str], ...]) -> list[dict[str, str]]:
    return [dict(row) for row in rows]


def retron_transform_ladder_rows() -> list[dict[str, str]]:
    return _copy_rows(_RETRON_TRANSFORM_LADDER)


def retron_aggregate_figure_rows() -> list[dict[str, str]]:
    return _copy_rows(_RETRON_AGGREGATE_FIGURES)


def retron_figure_coverage_rows() -> list[dict[str, str]]:
    return _copy_rows(_RETRON_FIGURE_COVERAGE)


def retron_assay_context_rows() -> list[dict[str, str]]:
    return _copy_rows(_RETRON_ASSAY_CONTEXT)
