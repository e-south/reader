from __future__ import annotations

import importlib
import json
from pathlib import Path

from typer.testing import CliRunner

from reader.response_window import (
    ExperimentPreflight,
    ResponseWindowBundle,
    ResponseWindowPreflight,
)
from reader.response_window_review import PromoterEvidenceBundle
from reader.workbench import cli

response_window_cli = importlib.import_module("reader.workbench.cli.response_window")


def test_response_window_cli_exposes_explicit_lifecycle() -> None:
    result = CliRunner().invoke(cli.app, ["response-window", "--help"])

    assert result.exit_code == 0
    for command in ("preflight", "build", "verify", "review", "promoter-evidence", "promoter-evidence-verify"):
        assert command in result.output


def test_response_window_preflight_emits_machine_readable_readiness(monkeypatch, tmp_path: Path) -> None:
    result = _preflight(tmp_path)
    monkeypatch.setattr(response_window_cli, "preflight_response_window_request", lambda **_kwargs: result)

    invocation = CliRunner().invoke(
        cli.app,
        ["response-window", "preflight", "request.yaml", "--format", "json"],
    )

    assert invocation.exit_code == 0
    payload = json.loads(invocation.output)
    assert payload["ready"] is True
    assert payload["study_id"] == "stress_ethanol_cipro_growth"
    assert payload["experiments"][0]["event_time_uncertainty_h"] == 0.25
    assert payload["primary_reduction_id"] == "post_6_12h"


def test_response_window_verify_emits_bundle_contract(monkeypatch, tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(response_window_cli, "verify_response_window_bundle", lambda _path: bundle)

    invocation = CliRunner().invoke(
        cli.app,
        ["response-window", "verify", str(tmp_path), "--format", "json"],
    )

    assert invocation.exit_code == 0
    payload = json.loads(invocation.output)
    assert payload["schema_version"] == "reader.response_window.bundle.v5"
    assert payload["study_id"] == "stress_ethanol_cipro_growth"
    assert payload["counts"] == {"experiments": 1, "plots": 5}


def test_response_window_review_verifies_before_launch(monkeypatch, tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    reader_root = tmp_path / "reader-project"
    calls: list[tuple[str, Path, bool, int | None, Path | None]] = []
    monkeypatch.setattr(response_window_cli, "verify_response_window_bundle", lambda _path: bundle)
    monkeypatch.setattr(
        response_window_cli,
        "_launch_marimo",
        lambda mode, target, *, has_fcs, headless, port, repo_root: calls.append(
            (mode, target, headless, port, repo_root)
        ),
    )

    invocation = CliRunner().invoke(
        cli.app,
        [
            "response-window",
            "review",
            str(tmp_path),
            "--reader-root",
            str(reader_root),
            "--headless",
            "--port",
            "9123",
        ],
    )

    assert invocation.exit_code == 0
    assert calls == [("run", bundle.notebook_path, True, 9123, reader_root)]


def test_promoter_evidence_cli_emits_selection_and_artifact_paths(monkeypatch, tmp_path: Path) -> None:
    bundle = _promoter_bundle(tmp_path)
    monkeypatch.setattr(response_window_cli, "build_promoter_evidence_bundle", lambda **_kwargs: bundle)

    invocation = CliRunner().invoke(
        cli.app,
        [
            "response-window",
            "promoter-evidence",
            "response-bundle",
            "candidate-bindings",
            "--out-dir",
            "evidence",
            "--experiment-id",
            "experiment",
            "--design-id",
            "design",
            "--reduction-id",
            "primary",
            "--format",
            "json",
        ],
    )

    assert invocation.exit_code == 0
    payload = json.loads(invocation.output)
    assert payload["schema_version"] == "reader.response_window.promoter_evidence_bundle.v3"
    assert payload["selection"]["candidate_id"] == "candidate"
    assert payload["png"] == str(bundle.png_path)
    assert payload["pdf"] == str(bundle.pdf_path)


def _preflight(tmp_path: Path) -> ResponseWindowPreflight:
    experiment = ExperimentPreflight(
        experiment_id="20260117_example",
        response_designs=35,
        magnitude_designs=36,
        trajectory_designs=36,
        response_rows=560,
        magnitude_rows=576,
        trajectory_rows=576,
        event_interval_start_assay_h=7.5,
        event_interval_end_assay_h=8.0,
        event_time_estimate_assay_h=7.75,
        event_time_uncertainty_h=0.25,
        post_event_coverage_h=14.0,
        record_digests={"record": "sha256:" + "1" * 64},
    )
    return ResponseWindowPreflight(
        ready=True,
        study_id="stress_ethanol_cipro_growth",
        request_id="stress-response-window-v1",
        request_path=tmp_path / "request.yaml",
        request_sha256="sha256:" + "2" * 64,
        schema_version="reader.response_window.request.v3",
        state_order=("00", "10", "01", "11"),
        primary_reduction_id="post_6_12h",
        reduction_ids=("post_6_12h", "post_8_14h"),
        experiments=(experiment,),
        observed_design_ids=("pDual-10", "spyp"),
        missing_display_examples=(),
    )


def _bundle(tmp_path: Path) -> ResponseWindowBundle:
    return ResponseWindowBundle(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        notebook_path=tmp_path / "review.py",
        manifest={
            "schema_version": "reader.response_window.bundle.v5",
            "study_id": "stress_ethanol_cipro_growth",
            "request_id": "stress-response-window-v1",
        },
        counts={"experiments": 1, "plots": 5},
    )


def _promoter_bundle(tmp_path: Path) -> PromoterEvidenceBundle:
    root = tmp_path / "evidence"
    return PromoterEvidenceBundle(
        root=root,
        manifest_path=root / "manifest.json",
        png_path=root / "promoter_evidence.png",
        pdf_path=root / "promoter_evidence.pdf",
        manifest={
            "schema_version": "reader.response_window.promoter_evidence_bundle.v3",
            "claim_status": "objective_neutral",
            "selection": {
                "experiment_id": "experiment",
                "design_id": "design",
                "candidate_id": "candidate",
                "reduction_id": "primary",
            },
        },
    )
