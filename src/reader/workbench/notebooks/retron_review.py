from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from reader.workbench.notebooks import context as notebook_context

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "relevant", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "irrelevant", "off"}
_PLOT_STAGE_ORDER = ("1. QC", "2. Semantic kinetics", "3. Ranking and overview")
_FAMILY_ORDER = {"mono": 0, "bi": 1, "tri": 2, "quad": 3, "control": 4}

load_notebook_workbench_context = notebook_context.load_notebook_workbench_context

_RETRON_PLOT_GUIDE = {
    "raw_kinetics": {
        "stage": "1. QC",
        "math": "Raw OD600(t), CFP(t), and YFP(t) after ingest plus blank/overflow preprocessing only.",
        "record": "ratio_yfp_od600/df",
        "meaning": "Find failed wells, saturation, mixing artifacts, or growth collapse before normalization.",
    },
    "support_kinetics": {
        "stage": "1. QC",
        "math": "Support ratios YFP/OD600 and CFP/OD600 from the tidy dataframe.",
        "record": "ratio_yfp_od600/df",
        "meaning": "Separate broad physiology shifts from reporter-specific movement; support only, not the primary score.",
    },
    "control_burden_panel": {
        "stage": "1. QC",
        "math": "tetO-only traces over R(t)=log2(YFP/CFP) and mu(t)=d ln(OD600) / dt.",
        "record": "semantic_metrics/trace",
        "meaning": "Measures IPTG and retron burden without a cognate sponge site.",
    },
    "baseline_shifted_kinetics": {
        "stage": "2. Semantic kinetics",
        "math": "B(t)=R(t)-R_pre, where R_pre is the mean of the last three pre-stress reads.",
        "record": "semantic_metrics/trace",
        "meaning": "Removes pre-stress offsets so post-stress motion is comparable well to well.",
    },
    "matched_control_kinetics": {
        "stage": "2. Semantic kinetics",
        "math": "C(t)=B(t)-mean(B matched tetO at same sensor, plate, stress, IPTG, and time).",
        "record": "semantic_metrics/trace",
        "meaning": "Shows sponge-specific deviation after same-sensor tetO control subtraction.",
    },
    "induced_effect_kinetics": {
        "stage": "2. Semantic kinetics",
        "math": "D(t)=mean(C +IPTG)-mean(C -IPTG) within each sensor and stress state.",
        "record": "semantic_metrics/trace",
        "meaning": "Isolates induction-dependent sponge activity after matched-control normalization.",
    },
    "interaction_summary": {
        "stage": "3. Ranking and overview",
        "math": "C_AUC or C_END across the four 2x2 states: H2O/-IPTG, H2O/+IPTG, stress/-IPTG, stress/+IPTG.",
        "record": "semantic_metrics/summary",
        "meaning": "Shows whether activity is inducible, stress-gated, leaky, or burden-dominated.",
    },
    "library_heatmaps": {
        "stage": "3. Ranking and overview",
        "math": "Heatmaps over D_AUC(H2O), D_AUC(relevant stress), M_AUC, and S_AUC.",
        "record": "semantic_metrics/summary",
        "meaning": "Summarizes library-wide on-target, stress-gated, and cross-sensor scaled effects.",
    },
    "stress_modulation_scores": {
        "stage": "3. Ranking and overview",
        "math": "M_AUC=AUC(D(relevant stress)-D(H2O)).",
        "record": "semantic_metrics/summary",
        "meaning": "Ranks how strongly stress unmasks a sponge effect.",
    },
    "pareto_ranking": {
        "stage": "3. Ranking and overview",
        "math": "On-target score S_AUC versus burden (T_growth_AUC by default), with |L_pre| encoded as point size.",
        "record": "semantic_metrics/summary",
        "meaning": "Balances efficacy against burden and leakiness for candidate selection.",
    },
}

_RETRON_TRANSFORM_LADDER = (
    {
        "Step": "Raw channels",
        "Formula": "OD600(t), CFP(t), YFP(t)",
        "Output": "raw QC only",
        "Meaning": "Check growth, saturation, drift, and failed wells before semantic scoring.",
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
        "Step": "Induced effect",
        "Formula": "D(t)=mean(C +IPTG)-mean(C -IPTG)",
        "Output": "trace metric D; summary D_AUC and D_END",
        "Meaning": "Clean sequence-specific sponge effect after matched-control subtraction.",
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
        "Formula": "L_pre, L_post_AUC, T_ratio_AUC, T_growth_AUC, T_finalOD",
        "Output": "summary metrics",
        "Meaning": "Separates strong hits from leaky or burdensome constructs.",
    },
)

_RETRON_AGGREGATE_FIGURES = (
    {
        "Figure": "Specificity matrix",
        "Math": "Cross-run pivot over relevant-stress S_AUC or O_AUC for tested sensor/sponge pairs.",
        "Why": "Shows whether mono, bi, tri, and quad sponges distribute activity across intended sensors.",
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
        "Math": "Selected sponge across sensors over relevant-stress S_AUC or O_AUC.",
        "Why": "Shows whether a multi-functional sponge is balanced across its intended sensor arms.",
    },
)

_RETRON_FIGURE_COVERAGE = (
    {
        "Figure": "Figure 1 — Raw kinetics QC",
        "Scope": "Per experiment",
        "Surface": "raw_kinetics",
        "Coverage": "Exact compiled plot",
        "Math": "Raw OD600(t), CFP(t), and YFP(t).",
    },
    {
        "Figure": "Figure 2 — Support ratios per OD",
        "Scope": "Per experiment",
        "Surface": "support_kinetics",
        "Coverage": "Exact compiled plot",
        "Math": "YFP/OD600 and CFP/OD600 support channels.",
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
        "Surface": "semantic summary review",
        "Coverage": "Derived from semantic tables",
        "Math": "R_pre and L_pre from semantic_metrics/summary.",
    },
    {
        "Figure": "Figure 5 — Raw ratio kinetics by sensor",
        "Scope": "Per experiment",
        "Surface": "semantic trace review",
        "Coverage": "Derived from semantic tables",
        "Math": "R(t)=log2(YFP/CFP) from semantic_metrics/trace.",
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
        "Figure": "Figure 8 — Induced sponge effect over time",
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
        "Math": "C_AUC or C_END across the four induction/stress states.",
    },
    {
        "Figure": "Figure 10 — Library heatmaps",
        "Scope": "Per experiment",
        "Surface": "library_heatmaps",
        "Coverage": "Exact compiled plot",
        "Math": "D_AUC, M_AUC, and scaled ranking heatmaps.",
    },
    {
        "Figure": "Figure 11 — Stress modulation forest plot",
        "Scope": "Per experiment",
        "Surface": "stress_modulation_scores",
        "Coverage": "Exact compiled plot",
        "Math": "M_AUC summary over relevant sensor/sponge pairs.",
    },
    {
        "Figure": "Figure 12 — Leakiness panel",
        "Scope": "Per experiment",
        "Surface": "semantic summary review",
        "Coverage": "Derived from semantic tables",
        "Math": "L_pre and L_post_AUC from semantic_metrics/summary.",
    },
    {
        "Figure": "Figure 13 — Specificity matrix",
        "Scope": "Cross run",
        "Surface": "notebook/retron_sponge_aggregate",
        "Coverage": "Exact aggregate notebook figure",
        "Math": "Relevant-stress O_AUC or S_AUC pivoted over sponge x sensor.",
    },
    {
        "Figure": "Figure 14 — Pareto ranking",
        "Scope": "Per experiment",
        "Surface": "pareto_ranking",
        "Coverage": "Exact compiled plot",
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
        "Math": "Selected multifunction sponge across relevant sensors over O_AUC or S_AUC.",
    },
    {
        "Figure": "Figure 18 — Growth impact summary",
        "Scope": "Per experiment",
        "Surface": "control_burden_panel plus semantic summary review",
        "Coverage": "Derived from semantic tables",
        "Math": "mu(t), T_growth_AUC, and T_finalOD burden summaries.",
    },
    {
        "Figure": "Figure 19 — Plate-position heatmaps",
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


def retron_transform_ladder_rows() -> list[dict[str, str]]:
    return [dict(row) for row in _RETRON_TRANSFORM_LADDER]


def retron_aggregate_figure_rows() -> list[dict[str, str]]:
    return [dict(row) for row in _RETRON_AGGREGATE_FIGURES]


def retron_figure_coverage_rows() -> list[dict[str, str]]:
    return [dict(row) for row in _RETRON_FIGURE_COVERAGE]


def retron_plot_guide_rows(plot_ids: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for plot_id in plot_ids:
        guide = _RETRON_PLOT_GUIDE.get(str(plot_id))
        rows.append(
            {
                "Stage": (guide or {}).get("stage", _PLOT_STAGE_ORDER[-1]),
                "Plot id": str(plot_id),
                "Math / transform": (guide or {}).get("math", "Protocol-specific transform guide not registered."),
                "Source record": (guide or {}).get("record", "see compiled plot spec"),
                "How to read": (guide or {}).get(
                    "meaning", "Interpret in the context of the compiled assay semantics."
                ),
            }
        )
    rows.sort(key=lambda row: (_plot_stage_rank(row["Stage"]), row["Plot id"]))
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
        summary_frame = _read_semantic_csv(source.summary_path, kind="summary")
        trace_frame = _read_semantic_csv(source.trace_path, kind="trace")
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


def build_specificity_matrix(
    summary_df: pd.DataFrame,
    *,
    score_metric: str = "S_AUC",
) -> pd.DataFrame:
    scores = aggregate_on_target_scores(summary_df, score_metric=score_metric)
    if scores.empty:
        return pd.DataFrame()
    pivot = scores.pivot_table(index="sponge", columns="sensor", values="value", aggfunc="mean")
    if pivot.empty:
        return pivot
    row_order = sorted(pivot.index.tolist(), key=_sponge_sort_key)
    col_order = sorted(pivot.columns.tolist())
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
    sponge: str,
    score_metric: str = "S_AUC",
) -> pd.DataFrame:
    scores = aggregate_on_target_scores(summary_df, score_metric=score_metric)
    if scores.empty:
        return pd.DataFrame()
    frame = scores[scores["sponge"].astype(str) == str(sponge)].copy()
    return frame.sort_values("sensor", kind="stable").reset_index(drop=True)


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
            summary_path = experiment_root / "outputs" / "exports" / "retron" / "semantic_summary.csv"
        if trace_raw is not None:
            trace_path = _resolve_manifest_path(manifest_path, str(trace_raw))
        else:
            if experiment_root is None:
                raise ValueError(
                    "retron_review: each source must declare either 'experiment' or explicit 'summary'/'trace' paths"
                )
            trace_path = experiment_root / "outputs" / "exports" / "retron" / "semantic_trace.csv"

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


def _read_semantic_csv(path: Path, *, kind: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
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
