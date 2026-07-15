from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[5] / "domains" / "plate_reader" / "analysis" / "response_window"


def test_response_window_modules_stay_bounded() -> None:
    limits = {
        "aggregation.py": 220,
        "bundle.py": 280,
        "contracts.py": 340,
        "display.py": 190,
        "materialize.py": 340,
        "plot_style.py": 80,
        "preflight.py": 150,
        "promoter_evidence_bundle.py": 280,
        "promoter_evidence_bundle_contract.py": 80,
        "promoter_evidence_cards.py": 180,
        "promoter_evidence_components.py": 220,
        "promoter_evidence_figure.py": 220,
        "promoter_evidence_overlay.py": 240,
        "promoter_evidence_overlay_verification.py": 100,
        "promoter_evidence_selected_binding.py": 120,
        "promoter_evidence_verification.py": 260,
        "reporting.py": 190,
        "reporting_plots.py": 280,
        "reporting_quality_plots.py": 180,
        "review_collection.py": 220,
        "review_cross_experiment.py": 170,
        "review_cross_experiment_contract.py": 120,
        "review_cross_experiment_summaries.py": 180,
        "review_cross_experiment_trajectories.py": 180,
        "review_experiment_labels.py": 70,
        "review_reduction_options.py": 100,
        "review.py": 280,
        "review_endpoint_plots.py": 260,
        "review_replicates.py": 220,
        "review_time_series.py": 180,
        "review_time_series_components.py": 180,
        "review_views.py": 160,
        "sources.py": 340,
        "uncertainty.py": 120,
        "verification.py": 220,
        "verification_invariants.py": 200,
        "verification_manifest_contract.py": 100,
        "verification_request.py": 100,
        "verification_request_payload.py": 180,
        "verification_source_catalog.py": 80,
        "visual_labels.py": 150,
    }
    observed = {name: len((PACKAGE / name).read_text(encoding="utf-8").splitlines()) for name in limits}
    assert {name: lines for name, lines in observed.items() if lines > limits[name]} == {}


def test_response_window_domain_has_no_sfxi_or_workbench_imports() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.glob("*.py"))
    assert "reader.domains.logic.sfxi" not in source
    assert "reader.workbench" not in source
    assert "dnadesign.studies" not in source
    assert "dnadesign.usr" not in source


def test_candidate_binding_contract_is_shared_outside_metric_packages() -> None:
    reader_root = PACKAGE.parents[3]
    shared = reader_root / "domains" / "promoter" / "candidate_bindings"

    assert (shared / "loader.py").is_file()
    assert (shared / "contract.py").is_file()
    assert not list(PACKAGE.glob("promoter_evidence_binding*.py"))

    limits = {"annotations.py": 200, "contract.py": 130, "rows.py": 180, "loader.py": 280}
    observed = {name: len((shared / name).read_text(encoding="utf-8").splitlines()) for name in limits}
    assert {name: lines for name, lines in observed.items() if lines > limits[name]} == {}

    sequence_panel = shared.parent / "sequence_panel.py"
    sequence_source = sequence_panel.read_text(encoding="utf-8")
    assert len(sequence_source.splitlines()) <= 120
    assert "dnadesign.baserender" in sequence_source
    assert "dnadesign.usr" not in sequence_source
    assert "response_window" not in sequence_source
    assert "sfxi" not in sequence_source.lower()


def test_sfxi_triptych_does_not_resolve_candidates_through_dnadesign_usr() -> None:
    reader_root = PACKAGE.parents[3]
    triptych = reader_root / "domains" / "logic" / "sfxi" / "triptych_sequence.py"
    source = triptych.read_text(encoding="utf-8")

    assert "dnadesign.usr" not in source
    assert "load_promoter_candidate_bindings" in source


def test_response_window_never_calls_reference_relative_fluorescence_brightness() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.glob("*.py"))

    assert "BRIGHTNESS" not in source
    assert "brightness" not in source.lower()
