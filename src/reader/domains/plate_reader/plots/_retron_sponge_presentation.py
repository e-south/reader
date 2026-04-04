from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRACE_PRIMARY_WINDOW_METRICS = frozenset({"C", "D", "D_abs", "D_growth", "M", "O", "O_abs"})
TRACE_INSET_METRICS = frozenset()


@dataclass(frozen=True)
class SummaryTextSpec:
    lead: str | None = None
    include_primary_window_note: bool = False
    include_endpoint_window_note: bool = False


@dataclass(frozen=True)
class TraceTextSpec:
    formula: str
    subtitle: str | None = None

    def figure_subtitle(self) -> str:
        return str(self.subtitle or self.formula)


@dataclass(frozen=True)
class DecisionCardMetricSpec:
    metric: str
    label: str
    color: str
    units: str
    axis_label: str
    display_multiplier: float = 1.0


@dataclass(frozen=True)
class MetricSemantics:
    metric_id: str
    user_label: str
    role: str
    scope: str
    units: str
    sign_mode: str
    default_summary_kind: str
    show_by_default: bool


_METRIC_AXIS_LABELS = {
    "B": "Shift from the pre-stress state",
    "C": "Matched tetO deviation",
    "D": "D(t) = mean[C(+IPTG)] - mean[C(-IPTG)]",
    "D_abs": "D_abs(t) = ΔIPTG[R(t) - R_tetO,matched(t)]",
    "D_growth": "Construct-specific growth burden",
    "M": "Stress-gated effect",
    "O": "Expected-direction-aligned post-stress increment",
    "O_abs": "Expected-direction-aligned matched tetO separation",
    "R": "Observed reporter ratio log2(YFP/CFP)",
    "mu": "Instantaneous growth rate d ln(OD600) / dt",
}

_COMPACT_METRIC_AXIS_LABELS = {
    "D_abs": "ΔIPTG ΔR vs matched tetO",
}

_SUMMARY_METRIC_LABELS = {
    "R_pre": "Pre-stress baseline",
    "P_pre": "Preload shift",
    "C_AUC": "tetO-subtracted AUC",
    "C_END": "tetO-subtracted endpoint",
    "D_AUC": "Signed post-stress increment AUC",
    "D_abs_AUC": "Signed total effect beyond matched tetO AUC",
    "D_growth_AUC": "Growth burden AUC",
    "L_pre": "Pre-stress leakiness",
    "L_post_AUC": "Post-stress leakiness AUC",
    "M_AUC": "Stress-gated AUC",
    "O_AUC": "Expected-direction post-stress area",
    "O_abs_AUC": "Expected-direction total area",
    "S_AUC": "Scaled expected-direction post-stress area",
    "S_abs_AUC": "Scaled expected-direction total area",
    "T_ratio_AUC": "tetO reporter burden AUC",
    "T_growth_AUC": "tetO growth burden AUC",
    "T_finalOD": "tetO endpoint burden",
}

_TRACE_TEXT_SPECS = {
    "R": TraceTextSpec(
        formula="R(t)=log2(YFP/CFP)",
        subtitle="Observed reporter ratio over time",
    ),
    "B": TraceTextSpec(
        formula="B(t)=R(t)-R_pre",
        subtitle="Each trajectory shifted to its own pre-stress state",
    ),
    "C": TraceTextSpec(
        formula="C(t)=B(t)-B_tetO,matched(t)",
        subtitle="Deviation from the matched tetO control over time",
    ),
    "D": TraceTextSpec(
        formula="D(t)=mean C(+IPTG)-mean C(-IPTG)",
        subtitle="D(t) = mean C(+IPTG) - mean C(-IPTG)",
    ),
    "D_abs": TraceTextSpec(
        formula="D_abs(t)=delta_IPTG[R-R_tetO,matched]",
        subtitle="D_abs(t) = delta_IPTG[R - R_tetO,matched]",
    ),
    "D_growth": TraceTextSpec(
        formula="D_growth(t)=delta_IPTG[mu-mu_tetO,matched]",
        subtitle="Construct-specific growth burden over time",
    ),
    "M": TraceTextSpec(
        formula="M(t)=D(sensor-matched stress)-D(H2O)",
        subtitle="Stress-specific gain beyond the H2O condition",
    ),
    "O": TraceTextSpec(
        formula="O(t)=expected_sign x D(t)",
        subtitle="Post-stress increment aligned to the expected sensor direction",
    ),
    "O_abs": TraceTextSpec(
        formula="O_abs(t)=expected_sign x D_abs(t)",
        subtitle="Total effect aligned to the expected sensor direction",
    ),
    "mu": TraceTextSpec(
        formula="mu(t)=d ln(OD600) / dt",
        subtitle="Instantaneous growth rate over time",
    ),
}

_SUMMARY_METRIC_TEXT_SPECS = {
    "P_pre": SummaryTextSpec(
        lead="Preload shift before stress after matched tetO subtraction",
        include_primary_window_note=True,
    ),
    "C_AUC": SummaryTextSpec(lead="AUC of the matched tetO deviation trace", include_primary_window_note=True),
    "C_END": SummaryTextSpec(
        lead="Endpoint of the matched tetO deviation trace",
        include_primary_window_note=True,
        include_endpoint_window_note=True,
    ),
    "D_AUC": SummaryTextSpec(
        lead="Signed post-stress integral after preload removal in log2-ratio space",
        include_primary_window_note=True,
    ),
    "D_abs_AUC": SummaryTextSpec(
        lead="Signed total post-stress integral beyond matched tetO in log2-ratio space",
        include_primary_window_note=True,
    ),
    "D_growth_AUC": SummaryTextSpec(
        lead="AUC of the construct-specific growth burden trace",
        include_primary_window_note=True,
    ),
    "M_AUC": SummaryTextSpec(lead="AUC of the stress-specific gain", include_primary_window_note=True),
    "O_AUC": SummaryTextSpec(
        lead="Positive-area integral of the expected-direction post-stress ΔR after preload removal",
        include_primary_window_note=True,
    ),
    "O_abs_AUC": SummaryTextSpec(
        lead="Positive-area integral of the expected-direction matched-tetO ΔR over the post-stress window",
        include_primary_window_note=True,
    ),
    "S_AUC": SummaryTextSpec(
        lead="Expected-direction post-stress area scaled by the native sensor range",
        include_primary_window_note=True,
    ),
    "S_abs_AUC": SummaryTextSpec(
        lead="Expected-direction total area scaled by the native sensor range",
        include_primary_window_note=True,
    ),
}

LIBRARY_HEATMAP_TEXT_SPEC = SummaryTextSpec(
    lead="Relevant-stress rows only; preload uses the pre-stress baseline",
    include_primary_window_note=True,
)

DECOMPOSITION_TEXT_SPEC = SummaryTextSpec(
    lead=None,
    include_primary_window_note=False,
)

_DECISION_CARD_METRIC_SPECS = (
    DecisionCardMetricSpec(
        metric="P_pre",
        label="Pre-stress baseline ΔR",
        color="#6f6f6f",
        units="log2 ratio",
        axis_label="Baseline ΔR vs matched tetO",
    ),
    DecisionCardMetricSpec(
        metric="O_AUC",
        label="Expected-direction state area",
        color="#56B4E9",
        units="log2 ratio x h",
        axis_label="Positive ∫ΔR dt vs matched tetO",
    ),
)

_METRIC_SEMANTICS = {
    "R": MetricSemantics(
        metric_id="R",
        user_label="Observed reporter ratio",
        role="qc",
        scope="within_sensor",
        units="log2 ratio",
        sign_mode="raw",
        default_summary_kind="trace",
        show_by_default=True,
    ),
    "P_pre": MetricSemantics(
        metric_id="P_pre",
        user_label="Preload shift before stress",
        role="mechanistic",
        scope="within_sensor",
        units="log2 ratio",
        sign_mode="raw",
        default_summary_kind="window_mean",
        show_by_default=True,
    ),
    "D_abs_AUC": MetricSemantics(
        metric_id="D_abs_AUC",
        user_label="Total effect beyond matched tetO",
        role="primary_evidence",
        scope="within_sensor",
        units="log2 ratio x h",
        sign_mode="raw",
        default_summary_kind="AUC",
        show_by_default=True,
    ),
    "D_AUC": MetricSemantics(
        metric_id="D_AUC",
        user_label="Post-stress increment",
        role="mechanistic",
        scope="within_sensor",
        units="log2 ratio x h",
        sign_mode="raw",
        default_summary_kind="AUC",
        show_by_default=True,
    ),
    "D_growth_AUC": MetricSemantics(
        metric_id="D_growth_AUC",
        user_label="Construct-specific burden",
        role="mechanistic",
        scope="within_sensor",
        units="growth delta x h",
        sign_mode="raw",
        default_summary_kind="AUC",
        show_by_default=True,
    ),
    "O_abs_AUC": MetricSemantics(
        metric_id="O_abs_AUC",
        user_label="Expected-direction total area",
        role="ranking",
        scope="cross_sensor",
        units="log2 ratio x h",
        sign_mode="expected_direction_positive_area",
        default_summary_kind="AUC",
        show_by_default=True,
    ),
    "S_abs_AUC": MetricSemantics(
        metric_id="S_abs_AUC",
        user_label="Scaled expected-direction total area",
        role="ranking",
        scope="cross_sensor",
        units="scaled effect",
        sign_mode="expected_direction_positive_area",
        default_summary_kind="AUC",
        show_by_default=True,
    ),
    "O_AUC": MetricSemantics(
        metric_id="O_AUC",
        user_label="Expected-direction post-stress area",
        role="ranking",
        scope="cross_sensor",
        units="log2 ratio x h",
        sign_mode="expected_direction_positive_area",
        default_summary_kind="AUC",
        show_by_default=True,
    ),
    "S_AUC": MetricSemantics(
        metric_id="S_AUC",
        user_label="Scaled expected-direction post-stress area",
        role="ranking",
        scope="cross_sensor",
        units="scaled effect",
        sign_mode="expected_direction_positive_area",
        default_summary_kind="AUC",
        show_by_default=True,
    ),
}


def should_annotate_primary_window(metric: str) -> bool:
    return str(metric) in TRACE_PRIMARY_WINDOW_METRICS


def has_trace_summary_inset(metric: str) -> bool:
    return str(metric) in TRACE_INSET_METRICS


def metric_axis_label(metric: str, *, metric_label_map: Mapping[str, str] | None = None) -> str:
    if metric_label_map and str(metric) in metric_label_map:
        return str(metric_label_map[str(metric)])
    return _METRIC_AXIS_LABELS.get(str(metric), f"Retron sponge metric ({metric})")


def compact_metric_axis_label(metric: str, *, metric_label_map: Mapping[str, str] | None = None) -> str:
    if metric_label_map and str(metric) in metric_label_map:
        return str(metric_label_map[str(metric)])
    return _COMPACT_METRIC_AXIS_LABELS.get(str(metric), metric_axis_label(metric, metric_label_map=metric_label_map))


def summary_metric_label(metric: str) -> str:
    return _SUMMARY_METRIC_LABELS.get(str(metric), f"Retron sponge summary metric ({metric})")


def trace_text_spec(metric: str) -> TraceTextSpec:
    metric_id = str(metric)
    if metric_id in _TRACE_TEXT_SPECS:
        return _TRACE_TEXT_SPECS[metric_id]
    return TraceTextSpec(formula=metric_id)


def summary_metric_text_spec(metric: str) -> SummaryTextSpec:
    metric_id = str(metric)
    if metric_id in _SUMMARY_METRIC_TEXT_SPECS:
        return _SUMMARY_METRIC_TEXT_SPECS[metric_id]
    return SummaryTextSpec(
        include_primary_window_note=True,
        include_endpoint_window_note=metric_id.endswith("_END"),
    )


def decision_card_metric_specs() -> tuple[DecisionCardMetricSpec, ...]:
    return _DECISION_CARD_METRIC_SPECS


def metric_semantics(metric: str) -> MetricSemantics | None:
    return _METRIC_SEMANTICS.get(str(metric))


def render_summary_text(spec: SummaryTextSpec, *, trace: pd.DataFrame | None = None) -> str:
    notes: list[str] = []
    lead = str(spec.lead or "").strip()
    if lead:
        notes.append(lead)
    if spec.include_primary_window_note:
        summary_note = primary_window_compact_note_from_trace(trace)
        if summary_note:
            notes.append(summary_note)
    if spec.include_endpoint_window_note:
        endpoint_note = endpoint_window_note_from_trace(trace)
        if endpoint_note:
            notes.append(endpoint_note)
    return "; ".join(notes)


def primary_window_compact_note_from_trace(trace: pd.DataFrame | None) -> str:
    if trace is None or trace.empty:
        return ""
    explicit_span = _explicit_primary_window_span(trace)
    if explicit_span is not None:
        start_h, end_h = explicit_span
        if np.isclose(start_h, end_h):
            return f"Window uses the flagged read at {end_h:.1f} h"
        return f"Window {start_h:.1f} to {end_h:.1f} h after stress addition"
    configured = _configured_primary_window_hours(trace)
    if configured is not None:
        return f"Window first {configured:.1f} h after stress addition"
    required = {"time_from_stress", "in_primary_post_stress"}
    if not required.issubset(trace.columns):
        return ""
    post = trace[trace["in_primary_post_stress"].fillna(False)].copy()
    if post.empty:
        return ""
    maxima = _window_group_maxima(post)
    finite = maxima[np.isfinite(maxima)]
    if finite.size == 0:
        return ""
    return f"Window first {float(finite.max()):.1f} h after stress addition"


def window_span_text(start_h: float, end_h: float) -> str:
    return f"{float(start_h):.1f} to {float(end_h):.1f} h"


def primary_window_span_bounds(
    trace: pd.DataFrame | None,
    *,
    stress_condition: str | None,
) -> tuple[float, float] | None:
    if trace is None or trace.empty:
        return None
    required = {"stress_condition", "time_from_stress", "in_primary_post_stress"}
    if not required.issubset(trace.columns):
        return None
    post = trace[trace["in_primary_post_stress"].fillna(False)].copy()
    if stress_condition is not None:
        post = post[post["stress_condition"].astype(str) == str(stress_condition)].copy()
    if post.empty:
        return None
    explicit_span = _explicit_primary_window_span(post)
    if explicit_span is not None:
        return explicit_span
    configured = _configured_primary_window_hours(post)
    if configured is not None and configured > 0.0:
        end = float(configured)
    else:
        maxima = _window_group_maxima(post)
        finite = maxima[np.isfinite(maxima)]
        if finite.size == 0:
            return None
        end = float(finite.max())
    if end <= 0.0:
        return None
    return 0.0, end


def burden_axis_label(metric: str) -> str:
    if str(metric) == "D_growth_AUC":
        return "Mean burden penalty (-D_growth_AUC)"
    if str(metric) == "T_growth_AUC":
        return "Mean tetO growth burden (T_growth_AUC)"
    if str(metric) == "T_finalOD":
        return "Mean tetO endpoint burden (T_finalOD)"
    return f"Burden summary ({metric})"


def pre_window_span_bounds(trace: pd.DataFrame | None) -> tuple[float, float] | None:
    if trace is None or trace.empty:
        return None
    required = {"time_from_stress", "in_pre_window"}
    if not required.issubset(trace.columns):
        return None
    pre = trace[trace["in_pre_window"].fillna(False)].copy()
    if pre.empty:
        return None
    values = pd.to_numeric(pre["time_from_stress"], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    return float(values.min()), float(values.max())


def decision_card_metric_title(
    metric: str,
    *,
    trace: pd.DataFrame | None = None,
    summary_window_start_h: float | None = None,
    summary_window_end_h: float | None = None,
) -> str:
    metric_id = str(metric)
    if metric_id == "P_pre":
        pre_span = pre_window_span_bounds(trace)
        if pre_span is not None and not np.isclose(pre_span[0], pre_span[1]):
            return _multiline_metric_title("Pre-stress baseline ΔR", window_span_text(*pre_span))
        return _multiline_metric_title("Pre-stress baseline ΔR", "R_pre; last pre-stress reads")
    if metric_id in {"D_abs_AUC", "D_AUC", "O_AUC", "O_abs_AUC"}:
        start_h = pd.to_numeric(pd.Series([summary_window_start_h]), errors="coerce").iloc[0]
        end_h = pd.to_numeric(pd.Series([summary_window_end_h]), errors="coerce").iloc[0]
        if np.isfinite(start_h) and np.isfinite(end_h):
            label_map = {
                "D_abs_AUC": "Total integrated contrast",
                "D_AUC": "Post-stress integrated contrast",
                "O_abs_AUC": "Expected-direction total area",
                "O_AUC": "Expected-direction state area",
            }
            label = label_map[metric_id]
            return _multiline_metric_title(label, window_span_text(float(start_h), float(end_h)))
    return {
        "P_pre": "Pre-stress baseline ΔR",
        "D_abs_AUC": "Total integrated contrast",
        "D_AUC": "Post-stress integrated contrast",
        "O_abs_AUC": "Expected-direction total area",
        "O_AUC": "Expected-direction state area",
    }.get(metric_id, summary_metric_label(metric_id))


def library_heatmap_subtitle(trace: pd.DataFrame | None) -> str:
    span = primary_window_span_bounds(trace, stress_condition=None)
    if span is None:
        window_text = "AUC summaries use the primary post-stress window."
    else:
        window_text = f"AUC summaries use {window_span_text(*span)} after stress addition."
    return f"Relevant-stress rows only; {window_text} Preload uses the pre-stress baseline."


def endpoint_window_note_from_trace(trace: pd.DataFrame | None) -> str:
    if trace is None or trace.empty:
        return ""
    required = {"in_endpoint_window", "time"}
    if not required.issubset(trace.columns):
        return ""
    endpoint = trace[trace["in_endpoint_window"].fillna(False)].copy()
    if endpoint.empty:
        return ""
    count = _endpoint_time_count(endpoint)
    if count is None:
        return "Endpoint uses the last flagged reads inside the summary window"
    noun = "read" if count == 1 else "reads"
    return f"Endpoint uses the last {count} flagged {noun} inside the summary window"


def _configured_primary_window_hours(trace: pd.DataFrame | None) -> float | None:
    if trace is None or trace.empty or "configured_max_post_stress_hours" not in trace.columns:
        return None
    values = pd.to_numeric(trace["configured_max_post_stress_hours"], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    return float(values[0])


def _explicit_primary_window_span(trace: pd.DataFrame | None) -> tuple[float, float] | None:
    if trace is None or trace.empty:
        return None
    required = {"summary_window_start_h", "summary_window_end_h"}
    if not required.issubset(trace.columns):
        return None
    start_values = pd.to_numeric(trace["summary_window_start_h"], errors="coerce").dropna().to_numpy(dtype=float)
    end_values = pd.to_numeric(trace["summary_window_end_h"], errors="coerce").dropna().to_numpy(dtype=float)
    if start_values.size == 0 or end_values.size == 0:
        return None
    start_h = float(start_values[0])
    end_h = float(end_values[0])
    if not np.isfinite(start_h) or not np.isfinite(end_h):
        return None
    return start_h, end_h


def _window_group_maxima(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["time_from_stress"], errors="coerce")
    group_columns = [column for column in ("plate_id", "sensor", "stress_condition") if column in frame.columns]
    if not group_columns:
        finite = values[np.isfinite(values)]
        return finite.to_numpy(dtype=float, copy=False)
    grouped = frame.assign(__time_from_stress=values).groupby(group_columns, dropna=False)["__time_from_stress"].max()
    finite = pd.to_numeric(grouped, errors="coerce")
    finite = finite[np.isfinite(finite)]
    return finite.to_numpy(dtype=float, copy=False)


def _endpoint_time_count(frame: pd.DataFrame) -> int | None:
    group_columns = [column for column in ("plate_id", "sensor", "stress_condition") if column in frame.columns]
    if not group_columns:
        count = int(pd.to_numeric(frame["time"], errors="coerce").dropna().nunique())
        return count or None
    counts = (
        frame.assign(__time=pd.to_numeric(frame["time"], errors="coerce"))
        .groupby(group_columns, dropna=False)["__time"]
        .nunique()
    )
    counts = counts[counts > 0]
    if counts.empty:
        return None
    modes = counts.mode(dropna=True)
    if modes.empty:
        return int(counts.iloc[0])
    return int(modes.iloc[0])


def _multiline_metric_title(label: str, detail: str) -> str:
    return f"{str(label).strip()}\n({str(detail).strip()})"


def _validate_presentation_catalog() -> None:
    missing_trace_labels = sorted(set(_TRACE_TEXT_SPECS) - set(_METRIC_AXIS_LABELS))
    if missing_trace_labels:
        joined = ", ".join(missing_trace_labels)
        raise RuntimeError(f"retron_sponge presentation: missing axis labels for trace metrics: {joined}")
    missing_summary_labels = sorted(set(_SUMMARY_METRIC_TEXT_SPECS) - set(_SUMMARY_METRIC_LABELS))
    if missing_summary_labels:
        joined = ", ".join(missing_summary_labels)
        raise RuntimeError(f"retron_sponge presentation: missing labels for summary metrics: {joined}")
    endpoint_contract_gaps = sorted(
        metric
        for metric, spec in _SUMMARY_METRIC_TEXT_SPECS.items()
        if metric.endswith("_END") and not spec.include_endpoint_window_note
    )
    if endpoint_contract_gaps:
        joined = ", ".join(endpoint_contract_gaps)
        raise RuntimeError(f"retron_sponge presentation: endpoint metrics missing endpoint note contract: {joined}")
    if not TRACE_INSET_METRICS.issubset(TRACE_PRIMARY_WINDOW_METRICS):
        joined = ", ".join(sorted(TRACE_INSET_METRICS - TRACE_PRIMARY_WINDOW_METRICS))
        raise RuntimeError(f"retron_sponge presentation: inset metrics outside primary-window metrics: {joined}")


_validate_presentation_catalog()
