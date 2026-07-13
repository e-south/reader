from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from reader.domains.plate_reader.analysis.response_window import bundle as bundle_module
from reader.domains.plate_reader.analysis.response_window.provenance import sha256_file
from reader.errors import ContractError
from reader.response_window import (
    BUNDLE_SCHEMA_VERSION,
    RECORD_ARTIFACTS,
    RECORD_CONTRACTS,
    verify_response_window_bundle,
)
from reader.runtime import builtin_runtime


def test_verify_bundle_enforces_record_and_artifact_contracts(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)

    bundle = verify_response_window_bundle(root)

    assert bundle.root == root.resolve()
    assert bundle.counts["experiments"] == 1


def test_verify_bundle_rejects_artifact_drift(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    path = root / RECORD_ARTIFACTS["designs"]
    content = path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 0x01]) + content[1:])

    with pytest.raises(RuntimeError, match="digest mismatch"):
        verify_response_window_bundle(root)


def test_verify_bundle_rejects_schema_drift_with_matching_digest(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    path = root / RECORD_ARTIFACTS["designs"]
    pd.DataFrame({"experiment_id": ["experiment"]}).to_parquet(path, index=False)
    _refresh_artifact_manifest(root, RECORD_ARTIFACTS["designs"])

    with pytest.raises(ContractError, match="missing required column"):
        verify_response_window_bundle(root)


def test_verify_bundle_rejects_cross_table_state_drift(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    path = root / RECORD_ARTIFACTS["wells"]
    wells = pd.read_parquet(path)
    wells = wells.loc[~wells["state"].eq("11")]
    wells.to_parquet(path, index=False)
    _refresh_artifact_manifest(root, RECORD_ARTIFACTS["wells"])
    _set_manifest_count(root, "well_rows", len(wells))

    with pytest.raises(ValueError, match="all four conditions"):
        verify_response_window_bundle(root)


def test_verify_bundle_rejects_missing_display_contract(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["display"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="bundle.display"):
        verify_response_window_bundle(root)


def test_bundle_build_removes_staging_directory_on_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request_path = tmp_path / "request.yaml"
    request_path.write_text("request", encoding="utf-8")
    monkeypatch.setattr(bundle_module, "load_response_window_request", lambda _path: object())

    def interrupt(**_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(bundle_module, "_build_staged_bundle", interrupt)

    with pytest.raises(KeyboardInterrupt):
        bundle_module.build_response_window_bundle(
            request_path=request_path,
            out_dir=tmp_path / "latest",
            contracts=builtin_runtime().contracts,
            source_loader=lambda *_args: pytest.fail("source loader should not run"),
        )

    assert list(tmp_path.glob(".latest.staging-*")) == []


def _bundle_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    (root / "tables").mkdir(parents=True)
    (root / "plots").mkdir()
    (root / "sources" / "experiment").mkdir(parents=True)
    frames = _record_frames()
    for record_id, artifact_id in RECORD_ARTIFACTS.items():
        frames[record_id].to_parquet(root / artifact_id, index=False)
    (root / "request.yaml").write_text("schema_version: reader.response_window.request.v2\n", encoding="utf-8")
    (root / "review.py").write_text("import marimo\n", encoding="utf-8")
    (root / "sources" / "experiment" / "config.yaml").write_text("schema: reader/v7\n", encoding="utf-8")
    (root / "sources" / "experiment" / "records.json").write_text("{}\n", encoding="utf-8")
    (root / "plots" / "test.png").write_bytes(b"png")
    pd.DataFrame(
        [
            {
                "plot_id": "test",
                "tier": "assay_contract",
                "title": "The fixture preserves one review premise",
                "premise": "The fixture is internally consistent.",
                "decision_value": "Exercises verifier invariants.",
                "rationale": "A valid fixture isolates adversarial mutations.",
                "alt_text": "A placeholder fixture plot.",
                "non_claim_boundary": "This is not assay evidence.",
                "data_table": "tables/designs.parquet",
                "path": "plots/test.png",
            }
        ]
    ).to_csv(root / "tables" / "plot_manifest.csv", index=False)
    artifacts = _artifact_manifest(root)
    request_digest = artifacts["request.yaml"]["sha256"]
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "request_id": "test-request",
        "request": {"artifact_id": "request.yaml", "sha256": request_digest},
        "state_order": ["00", "10", "01", "11"],
        "display": _display(),
        "primary_reduction_id": "primary",
        "contracts": RECORD_CONTRACTS,
        "records": {
            record_id: {"contract_id": RECORD_CONTRACTS[record_id], "artifact_id": artifact_id}
            for record_id, artifact_id in RECORD_ARTIFACTS.items()
        },
        "counts": {
            "experiments": 1,
            "well_rows": 8,
            "design_rows": 2,
            "bootstrap_draw_rows": 2,
            "trace_rows": 24,
            "unique_design_ids": 2,
            "repeated_design_ids": 0,
            "reductions": 1,
            "plots": 1,
        },
        "source_records": [
            {
                "experiment_id": "experiment",
                "config_artifact": "sources/experiment/config.yaml",
                "records_artifact": "sources/experiment/records.json",
                "records": {"record": "sha256:" + "1" * 64},
            }
        ],
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _record_frames() -> dict[str, pd.DataFrame]:
    wells: list[dict[str, object]] = []
    designs: list[dict[str, object]] = []
    draws: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for design_id, is_reference in (("reference", True), ("design", False)):
        designs.append(_design_row(design_id, is_reference=is_reference))
        draws.append(_draw_row(design_id, is_reference=is_reference))
        for state in ("00", "10", "01", "11"):
            wells.append(_well_row(design_id, state=state, is_reference=is_reference))
            for signal_kind in ("response", "magnitude", "growth"):
                traces.append(_trace_row(design_id, state=state, signal_kind=signal_kind, is_reference=is_reference))
    return {
        "wells": pd.DataFrame(wells),
        "designs": pd.DataFrame(designs),
        "bootstrap_draws": pd.DataFrame(draws),
        "traces": pd.DataFrame(traces),
        "events": pd.DataFrame(
            [
                {
                    "experiment_id": "experiment",
                    "event_id": "event",
                    "event_kind": "perturbation_addition",
                    "event_interval_start_assay_h": 1.0,
                    "event_interval_end_assay_h": 1.5,
                    "event_time_estimate_assay_h": 1.25,
                    "event_time_estimate_method": "segment_gap_midpoint",
                    "event_time_uncertainty_h": 0.25,
                    "post_event_coverage_h": 12.0,
                    "declaration": "declared event",
                }
            ]
        ),
    }


def _well_row(design_id: str, *, state: str, is_reference: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": "experiment",
        "design_id": design_id,
        "state": state,
        "position": "A1",
        "response_well": 0.1,
        "magnitude_well": 0.2,
        "reduction_id": "primary",
        "reduction_method": "geometric_time_mean",
        "response_basis": "post_window",
        "reduction_role": "primary",
        "event_time_estimate_assay_h": 1.25,
        "window_start_event_h": 1.0,
        "window_end_event_h": 2.0,
        "window_start_assay_h": 2.25,
        "window_end_assay_h": 3.25,
        "is_reference": is_reference,
    }
    for family in ("response", "magnitude"):
        for period in ("", "pre_"):
            row[f"{family}_{period}observed_point_count"] = 3 if not period else 0
            row[f"{family}_{period}integration_point_count"] = 5 if not period else 0
            row[f"{family}_{period}max_interior_gap_h"] = 0.5 if not period else 0.0
    return row


def _design_row(design_id: str, *, is_reference: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": "experiment",
        "design_id": design_id,
        "reference_design_id": "reference",
        "reduction_id": "primary",
        "reduction_method": "geometric_time_mean",
        "response_basis": "post_window",
        "reduction_role": "primary",
        "replicate_stat": "median",
        "bootstrap_samples": 1,
        "confidence_level": 0.95,
        "event_id": "event",
        "event_time_estimate_assay_h": 1.25,
        "event_time_uncertainty_h": 0.25,
        "window_start_event_h": 1.0,
        "window_end_event_h": 2.0,
        "is_reference": is_reference,
        "min_replicates_per_state": 1,
        "min_observed_points_per_trace": 3,
        "max_interior_gap_h": 0.5,
        "min_pre_observed_points_per_trace": 0,
        "max_pre_interior_gap_h": 0.0,
    }
    for index, state in enumerate(("00", "10", "01", "11")):
        row[f"r{state}"] = float(index)
        row[f"b{state}"] = 0.0 if is_reference else float(index) / 2.0
        for prefix in ("r", "b"):
            row[f"{prefix}{state}_bootstrap_sd"] = 0.1
            row[f"{prefix}{state}_ci_low"] = -0.1
            row[f"{prefix}{state}_ci_high"] = 0.1
            row[f"{prefix}{state}_event_half_range"] = 0.05
        row[f"n{state}"] = 1
    return row


def _draw_row(design_id: str, *, is_reference: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": "experiment",
        "design_id": design_id,
        "reduction_id": "primary",
        "draw_index": 0,
        "is_reference": is_reference,
    }
    for index, state in enumerate(("00", "10", "01", "11")):
        row[f"r{state}"] = float(index)
        row[f"b{state}"] = 0.0 if is_reference else float(index) / 2.0
    return row


def _trace_row(design_id: str, *, state: str, signal_kind: str, is_reference: bool) -> dict[str, object]:
    return {
        "experiment_id": "experiment",
        "design_id": design_id,
        "position": "A1",
        "state": state,
        "time": 1.25,
        "time_from_event_h": 0.0,
        "value": 1.0,
        "signal_kind": signal_kind,
        "is_reference": is_reference,
    }


def _display() -> dict[str, object]:
    return {
        "schema_version": "reader.response_window.display.v1",
        "study_label": "Example response study",
        "event_label": "Stress addition",
        "state_labels": {
            "00": "No stress",
            "10": "Ethanol",
            "01": "Ciprofloxacin",
            "11": "Ethanol + ciprofloxacin",
        },
        "channels": {
            "response_ratio": "YFP/CFP",
            "magnitude_ratio": "YFP/OD600",
            "growth": "OD600",
            "reference_design_id": "reference",
        },
        "examples": [
            {"design_id": "reference", "label": "Reference anchor", "role": "reference_anchor"},
            {"design_id": "design", "label": "Response example", "role": "response_example"},
        ],
    }


def _artifact_manifest(root: Path) -> dict[str, dict[str, object]]:
    artifacts: dict[str, dict[str, object]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        artifacts[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return artifacts


def _refresh_artifact_manifest(root: Path, artifact_id: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = root / artifact_id
    manifest["artifacts"][artifact_id] = {
        "path": artifact_id,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _set_manifest_count(root: Path, key: str, value: int) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"][key] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
