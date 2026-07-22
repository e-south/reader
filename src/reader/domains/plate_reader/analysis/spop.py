from __future__ import annotations

import math
import operator
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

SPOP_ACRONYM = "sponging_percent_of_positive"
SPOP_METRIC_ID = "reader_spop_endpoint_dose_mean_v1"
SPOP_NUMERIC_SCOPE = "reader_experiment_normalized_tf_sponging"
SPOP_NORMALIZATION_BASIS = "rfp_od600_derepression_fraction_relative_to_atc_positive_control"
SPOP_REPORTER_READOUT = "RFP/OD600"
SPOP_VIABILITY_READOUT = "OD600"
SPOP_DEFAULT_LAMBDA = 0.5
SPOP_EPS_POSITIVE = 1.0e-8


@dataclass(frozen=True, slots=True)
class SpopDoseValue:
    """One nonzero IPTG endpoint dose used by the SPOP scalar contract."""

    iptg_uM: float
    rfp_over_od600: float
    od600: float
    replicate_count: int = 1


@dataclass(frozen=True, slots=True)
class SpopEndpointScore:
    """Reader-owned SPOP endpoint dose-mean scalar and its QC support vectors."""

    metric_id: str
    numeric_scope: str
    normalization_basis: str
    iptg_doses_uM: tuple[float, ...]
    y_derepression_by_dose: tuple[float, ...]
    viability_by_dose: tuple[float, ...]
    replicate_count_min: int
    spop_potency: float
    spop_viability: float
    spop_score: float
    spop_score_raw: float
    raw_value: float
    normalized_value: float
    qc_flags: tuple[str, ...]


class SpopScoringError(ValueError):
    """Raised when endpoint measurements cannot satisfy the Reader SPOP contract."""


def score_spop_endpoint(
    *,
    baseline_rfp_over_od600: float,
    positive_control_rfp_over_od600: float,
    baseline_od600: float,
    dose_values: Iterable[SpopDoseValue],
    lambda_viability: float = SPOP_DEFAULT_LAMBDA,
) -> SpopEndpointScore:
    """Score a single Reader SPOP endpoint dose ladder.

    The score is the positive-clipped endpoint derepression mean across nonzero
    IPTG doses, multiplied by a one-sided viability factor. It is intentionally
    not a time AUC.
    """

    lambda_value = _finite_float(lambda_viability, field="lambda_viability")
    if not 0.0 <= lambda_value <= 1.0:
        raise SpopScoringError("lambda_viability must be finite and in [0, 1].")
    baseline_z = _finite_float(baseline_rfp_over_od600, field="baseline_rfp_over_od600")
    positive_z = _finite_float(positive_control_rfp_over_od600, field="positive_control_rfp_over_od600")
    baseline_od = _finite_float(baseline_od600, field="baseline_od600")
    if baseline_od <= 0.0:
        raise SpopScoringError("baseline_od600 must be positive.")
    positive_denominator = positive_z - baseline_z
    if positive_denominator <= SPOP_EPS_POSITIVE:
        raise SpopScoringError("positive_control_rfp_over_od600 must be above baseline_rfp_over_od600.")

    sorted_doses = sorted(dose_values, key=lambda row: row.iptg_uM)
    if not sorted_doses:
        raise SpopScoringError("SPOP endpoint scoring requires at least one nonzero IPTG dose.")

    doses: list[float] = []
    y_values: list[float] = []
    viability_values: list[float] = []
    replicate_counts: list[int] = []
    qc_flags: set[str] = set()
    for row in sorted_doses:
        dose = _finite_float(row.iptg_uM, field="dose.iptg_uM")
        if dose <= 0.0:
            raise SpopScoringError("SPOP endpoint dose_values must contain only nonzero IPTG doses.")
        dose_z = _finite_float(row.rfp_over_od600, field="dose.rfp_over_od600")
        dose_od = _finite_float(row.od600, field="dose.od600")
        if dose_od < 0.0:
            raise SpopScoringError("dose.od600 must be non-negative.")
        replicate_count = _positive_integer(row.replicate_count, field="dose.replicate_count")
        y = (dose_z - baseline_z) / positive_denominator
        viability = min(1.0, dose_od / baseline_od)
        if y > 1.0:
            qc_flags.add("derepression_exceeds_atc_positive")
        if y < 0.0:
            qc_flags.add("derepression_below_zero_inducer")
        if viability < 0.8:
            qc_flags.add("induction_growth_penalty")
        doses.append(dose)
        y_values.append(y)
        viability_values.append(viability)
        replicate_counts.append(replicate_count)

    if len(doses) == 1:
        qc_flags.add("single_dose_endpoint")

    raw_potency = statistics.fmean(y_values)
    potency = statistics.fmean(max(0.0, value) for value in y_values)
    viability_mean = statistics.fmean(viability_values)
    raw_score = raw_potency * ((1.0 - lambda_value) + (lambda_value * viability_mean))
    score = potency * ((1.0 - lambda_value) + (lambda_value * viability_mean))
    return SpopEndpointScore(
        metric_id=SPOP_METRIC_ID,
        numeric_scope=SPOP_NUMERIC_SCOPE,
        normalization_basis=SPOP_NORMALIZATION_BASIS,
        iptg_doses_uM=tuple(doses),
        y_derepression_by_dose=tuple(y_values),
        viability_by_dose=tuple(viability_values),
        replicate_count_min=min(replicate_counts),
        spop_potency=potency,
        spop_viability=viability_mean,
        spop_score=score,
        spop_score_raw=raw_score,
        raw_value=raw_score,
        normalized_value=score,
        qc_flags=tuple(sorted(qc_flags)),
    )


def _finite_float(value: object, *, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SpopScoringError(f"{field} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise SpopScoringError(f"{field} must be finite.")
    return numeric


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise SpopScoringError(f"{field} must be a positive integer.")
    try:
        numeric = operator.index(value)
    except TypeError as exc:
        raise SpopScoringError(f"{field} must be a positive integer.") from exc
    if numeric <= 0:
        raise SpopScoringError(f"{field} must be a positive integer.")
    return numeric


__all__ = [
    "SPOP_ACRONYM",
    "SPOP_DEFAULT_LAMBDA",
    "SPOP_METRIC_ID",
    "SPOP_NORMALIZATION_BASIS",
    "SPOP_NUMERIC_SCOPE",
    "SPOP_REPORTER_READOUT",
    "SPOP_VIABILITY_READOUT",
    "SpopDoseValue",
    "SpopEndpointScore",
    "SpopScoringError",
    "score_spop_endpoint",
]
