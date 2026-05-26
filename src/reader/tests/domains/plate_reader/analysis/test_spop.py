from __future__ import annotations

from pathlib import Path

import pytest

from reader.domains.plate_reader.analysis.spop import (
    SPOP_ACRONYM,
    SPOP_METRIC_ID,
    SPOP_NORMALIZATION_BASIS,
    SPOP_NUMERIC_SCOPE,
    SpopDoseValue,
    SpopScoringError,
    score_spop_endpoint,
)


def test_spop_endpoint_metric_name_matches_endpoint_dose_mean_semantics() -> None:
    assert SPOP_ACRONYM == "sponging_percent_of_positive"
    assert SPOP_METRIC_ID == "reader_spop_endpoint_dose_mean_v1"
    assert "auc" not in SPOP_METRIC_ID
    assert SPOP_NUMERIC_SCOPE == "reader_experiment_normalized_tf_sponging"


def test_spop_source_of_truth_doc_matches_public_api_contract() -> None:
    doc_path = _repo_root() / "docs" / "lib" / "spop_endpoint_in_reader.md"
    doc = doc_path.read_text(encoding="utf-8")
    compact_doc = " ".join(doc.split())

    assert SPOP_METRIC_ID in doc
    assert SPOP_NUMERIC_SCOPE in doc
    assert SPOP_NORMALIZATION_BASIS in doc
    assert "score_spop_endpoint" in doc
    assert "endpoint dose-ladder mean, not an AUC" in compact_doc
    assert "does not integrate over time" in compact_doc
    assert "not guaranteed to stay in" in doc
    assert "Reader-owned endpoint scalar" in doc
    assert "Bridge Provenance Contract" in doc
    assert "reader_artifact_record_id" in doc
    assert "positive integer" in doc
    assert r"\mathrm{SPOP}" in doc
    assert r"\(" not in doc
    assert r"\)" not in doc
    assert "Relationship to SFXI" not in doc
    assert "SFXI" not in doc
    assert "This page" not in doc


def test_score_spop_endpoint_scores_positive_clipped_dose_ladder() -> None:
    score = score_spop_endpoint(
        baseline_rfp_over_od600=100.0,
        positive_control_rfp_over_od600=500.0,
        baseline_od600=1.0,
        dose_values=[
            SpopDoseValue(iptg_uM=5.0, rfp_over_od600=160.0, od600=1.0, replicate_count=3),
            SpopDoseValue(iptg_uM=50.0, rfp_over_od600=300.0, od600=1.0, replicate_count=3),
            SpopDoseValue(iptg_uM=500.0, rfp_over_od600=460.0, od600=1.0, replicate_count=3),
        ],
    )

    assert score.metric_id == SPOP_METRIC_ID
    assert score.iptg_doses_uM == (5.0, 50.0, 500.0)
    assert score.y_derepression_by_dose == pytest.approx((0.15, 0.5, 0.9))
    assert score.viability_by_dose == (1.0, 1.0, 1.0)
    assert score.replicate_count_min == 3
    assert score.spop_potency == pytest.approx((0.15 + 0.5 + 0.9) / 3.0)
    assert score.spop_score == pytest.approx(score.spop_potency)
    assert score.raw_value == pytest.approx(score.normalized_value)
    assert score.qc_flags == ()


def test_score_spop_endpoint_keeps_raw_score_and_qc_when_derepression_is_negative() -> None:
    score = score_spop_endpoint(
        baseline_rfp_over_od600=100.0,
        positive_control_rfp_over_od600=500.0,
        baseline_od600=1.0,
        dose_values=[
            SpopDoseValue(iptg_uM=5.0, rfp_over_od600=80.0, od600=0.9),
            SpopDoseValue(iptg_uM=500.0, rfp_over_od600=300.0, od600=0.7),
        ],
    )

    assert score.spop_score > score.spop_score_raw
    assert "derepression_below_zero_inducer" in score.qc_flags
    assert "induction_growth_penalty" in score.qc_flags


def test_score_spop_endpoint_rejects_non_endpoint_or_unanchored_inputs() -> None:
    with pytest.raises(SpopScoringError, match="positive_control"):
        score_spop_endpoint(
            baseline_rfp_over_od600=100.0,
            positive_control_rfp_over_od600=100.0,
            baseline_od600=1.0,
            dose_values=[SpopDoseValue(iptg_uM=500.0, rfp_over_od600=300.0, od600=1.0)],
        )
    with pytest.raises(SpopScoringError, match="nonzero IPTG"):
        score_spop_endpoint(
            baseline_rfp_over_od600=100.0,
            positive_control_rfp_over_od600=500.0,
            baseline_od600=1.0,
            dose_values=[SpopDoseValue(iptg_uM=0.0, rfp_over_od600=300.0, od600=1.0)],
        )


@pytest.mark.parametrize("replicate_count", [0, -1, 1.9, float("nan"), True])
def test_score_spop_endpoint_rejects_invalid_replicate_counts(replicate_count: object) -> None:
    with pytest.raises(SpopScoringError, match="replicate_count"):
        score_spop_endpoint(
            baseline_rfp_over_od600=100.0,
            positive_control_rfp_over_od600=500.0,
            baseline_od600=1.0,
            dose_values=[
                SpopDoseValue(
                    iptg_uM=500.0,
                    rfp_over_od600=300.0,
                    od600=1.0,
                    replicate_count=replicate_count,  # type: ignore[arg-type]
                )
            ],
        )


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise AssertionError("Could not resolve reader repository root.")
