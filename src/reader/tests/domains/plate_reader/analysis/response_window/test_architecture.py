from __future__ import annotations

from pathlib import Path

DOMAIN_ROOT = Path(__file__).resolve().parents[5] / "domains" / "plate_reader"
ANALYSIS_PACKAGE = DOMAIN_ROOT / "analysis" / "response_window"
EVIDENCE_PACKAGE = DOMAIN_ROOT / "evidence" / "response_window"
PLOT_PACKAGE = DOMAIN_ROOT / "plots" / "response_window"


def test_response_window_modules_stay_bounded() -> None:
    limits = {
        ANALYSIS_PACKAGE / "aggregation.py": 220,
        ANALYSIS_PACKAGE / "contracts.py": 360,
        ANALYSIS_PACKAGE / "display.py": 190,
        ANALYSIS_PACKAGE / "materialize.py": 280,
        ANALYSIS_PACKAGE / "provenance.py": 60,
        ANALYSIS_PACKAGE / "reduction.py": 260,
        ANALYSIS_PACKAGE / "sources.py": 340,
        ANALYSIS_PACKAGE / "uncertainty.py": 130,
        EVIDENCE_PACKAGE / "bundle.py": 260,
        EVIDENCE_PACKAGE / "preflight.py": 150,
        EVIDENCE_PACKAGE / "publication.py": 180,
        EVIDENCE_PACKAGE / "verification.py": 240,
        EVIDENCE_PACKAGE / "verification_invariants.py": 200,
        EVIDENCE_PACKAGE / "verification_manifest_contract.py": 100,
        EVIDENCE_PACKAGE / "verification_request.py": 80,
        EVIDENCE_PACKAGE / "verification_request_payload.py": 180,
        EVIDENCE_PACKAGE / "verification_source_catalog.py": 80,
        EVIDENCE_PACKAGE / "verification_trace_support.py": 220,
        EVIDENCE_PACKAGE / "verification_value_provenance.py": 130,
        PLOT_PACKAGE / "censor_display.py": 60,
        PLOT_PACKAGE / "plot_style.py": 80,
        PLOT_PACKAGE / "reporting.py": 200,
        PLOT_PACKAGE / "reporting_plots.py": 280,
        PLOT_PACKAGE / "reporting_quality_plots.py": 180,
        PLOT_PACKAGE / "review.py": 240,
        PLOT_PACKAGE / "review_collection.py": 220,
        PLOT_PACKAGE / "review_cross_experiment.py": 170,
        PLOT_PACKAGE / "review_cross_experiment_contract.py": 120,
        PLOT_PACKAGE / "review_cross_experiment_summaries.py": 180,
        PLOT_PACKAGE / "review_cross_experiment_trajectories.py": 180,
        PLOT_PACKAGE / "review_endpoint_plots.py": 260,
        PLOT_PACKAGE / "review_experiment_labels.py": 70,
        PLOT_PACKAGE / "review_reduction_options.py": 100,
        PLOT_PACKAGE / "review_replicates.py": 220,
        PLOT_PACKAGE / "review_time_series.py": 180,
        PLOT_PACKAGE / "review_time_series_components.py": 180,
        PLOT_PACKAGE / "review_time_series_handoff.py": 220,
        PLOT_PACKAGE / "review_views.py": 160,
        PLOT_PACKAGE / "visual_labels.py": 150,
    }
    observed = {path: len(path.read_text(encoding="utf-8").splitlines()) for path in limits}
    violations = {
        path.relative_to(DOMAIN_ROOT).as_posix(): lines for path, lines in observed.items() if lines > limits[path]
    }
    assert violations == {}


def test_response_window_roles_have_distinct_packages() -> None:
    analysis_modules = {path.name for path in ANALYSIS_PACKAGE.glob("*.py")}

    assert analysis_modules == {
        "__init__.py",
        "aggregation.py",
        "contracts.py",
        "display.py",
        "materialize.py",
        "provenance.py",
        "reduction.py",
        "sources.py",
        "uncertainty.py",
    }
    assert (EVIDENCE_PACKAGE / "__init__.py").is_file()
    assert (PLOT_PACKAGE / "__init__.py").is_file()


def test_response_window_dependencies_point_outward_from_core() -> None:
    core_source = "\n".join(path.read_text(encoding="utf-8") for path in ANALYSIS_PACKAGE.glob("*.py"))
    plot_source = "\n".join(path.read_text(encoding="utf-8") for path in PLOT_PACKAGE.rglob("*.py"))

    assert "reader.domains.plate_reader.evidence" not in core_source
    assert "reader.domains.plate_reader.plots" not in core_source
    assert "reader.domains.plate_reader.evidence" not in plot_source


def test_response_window_domain_has_no_sfxi_or_workbench_imports() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for package in (ANALYSIS_PACKAGE, EVIDENCE_PACKAGE, PLOT_PACKAGE)
        for path in package.rglob("*.py")
    )
    assert "reader.domains.logic.sfxi" not in source
    assert "reader.workbench" not in source


def test_response_window_never_calls_reference_relative_fluorescence_brightness() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for package in (ANALYSIS_PACKAGE, EVIDENCE_PACKAGE, PLOT_PACKAGE)
        for path in package.rglob("*.py")
    )

    assert "BRIGHTNESS" not in source
    assert "brightness" not in source.lower()
