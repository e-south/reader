from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

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
from reader.tests.domains.plate_reader.analysis.response_window.test_response_window_contracts import _payload


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


def test_verify_bundle_rejects_one_conflicting_well_reduction_row(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    path = root / RECORD_ARTIFACTS["wells"]
    wells = pd.read_parquet(path)
    wells.loc[wells.index[0], "reduction_method"] = "integrated_linear_mean"
    wells.to_parquet(path, index=False)
    _refresh_artifact_manifest(root, RECORD_ARTIFACTS["wells"])

    with pytest.raises(ValueError, match="conflicting semantics"):
        verify_response_window_bundle(root)


def test_verify_bundle_rejects_missing_display_contract(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["display"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="bundle.display"):
        verify_response_window_bundle(root)


def test_verify_bundle_rejects_study_identity_drift(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["study_id"] = "another_promoter_study"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="study identity"):
        verify_response_window_bundle(root)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("manifest", "manifest fields"),
        ("request", "request provenance fields"),
        ("artifact", "artifact metadata fields"),
        ("source_record", "source-record fields"),
        ("record_digest", "source record identities"),
    ],
)
def test_verify_bundle_rejects_unknown_contract_fields(tmp_path: Path, target: str, message: str) -> None:
    root = _bundle_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if target == "manifest":
        manifest["unexpected_field"] = "value"
    elif target == "request":
        manifest["request"]["unexpected_digest"] = "sha256:" + "a" * 64
    elif target == "artifact":
        manifest["artifacts"]["request.yaml"]["unexpected_digest"] = "sha256:" + "a" * 64
    elif target == "source_record":
        manifest["source_records"][0]["unexpected_alias"] = "design"
    else:
        digest = "sha256:" + "a" * 64
        manifest["source_records"][0]["records"]["unexpected/df"] = digest
        catalog_path = root / "sources" / "experiment" / "records.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog["latest"]["unexpected/df"] = {
            "record_id": "unexpected/df",
            "contract_id": "plate_reader.annotated.v1",
            "content_digest": digest,
        }
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        _refresh_artifact_manifest(root, "sources/experiment/records.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) | {
            "source_records": manifest["source_records"]
        }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_response_window_bundle(root)


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("experiment_ids", "experiment identities"),
        ("source_record_ids", "source record identities"),
        ("event_id", "event identity"),
        ("event_kind", "event kind"),
        ("event_declaration", "event declaration"),
        ("reduction_set", "reduction semantics"),
        ("reduction_method", "reduction semantics"),
        ("reduction_window", "reduction semantics"),
        ("replicate_stat", "aggregation semantics"),
        ("bootstrap_samples", "aggregation semantics"),
        ("confidence_level", "aggregation semantics"),
        ("positive_floor", "positive floor"),
        ("max_interior_gap_h", "interior-gap limit"),
        ("min_replicates_per_state", "replicate minimum"),
    ],
)
def test_verify_bundle_rejects_request_payload_drift(tmp_path: Path, drift: str, message: str) -> None:
    root = _bundle_fixture(tmp_path)
    request_path = root / "request.yaml"
    request = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    if drift == "experiment_ids":
        request["experiment_ids"] = ["different-experiment"]
    elif drift == "source_record_ids":
        request["source"]["response_record_id"] = "different/response"
    elif drift == "event_id":
        request["event"]["event_id"] = "different-event"
    elif drift == "event_kind":
        request["event"]["event_kind"] = "different-kind"
    elif drift == "event_declaration":
        request["event"]["declaration"] = "A different event declaration."
    elif drift == "reduction_set":
        request["reductions"].append(
            {
                "id": "sensitivity",
                "window_start_event_h": 1.0,
                "window_end_event_h": 2.0,
                "method": "integrated_linear_mean",
                "response_basis": "post_window",
                "role": "sensitivity",
            }
        )
    elif drift == "reduction_method":
        request["reductions"][0]["method"] = "integrated_linear_mean"
    elif drift == "reduction_window":
        request["reductions"][0]["window_start_event_h"] = 1.25
    elif drift == "replicate_stat":
        request["aggregation"]["replicate_stat"] = "mean"
    elif drift == "bootstrap_samples":
        request["aggregation"]["bootstrap_samples"] = 101
    elif drift == "confidence_level":
        request["aggregation"]["confidence_level"] = 0.9
    elif drift == "positive_floor":
        request["quality"]["positive_floor"] = 2.0
    elif drift == "max_interior_gap_h":
        request["quality"]["max_interior_gap_h"] = 0.25
    else:
        request["quality"]["min_replicates_per_state"] = 3
    request_path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")
    _refresh_artifact_manifest(root, "request.yaml")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["request"]["sha256"] = manifest["artifacts"]["request.yaml"]["sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_response_window_bundle(root)


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("selected_record_id", "record identity"),
        ("selected_contract", "record contract"),
        ("selected_digest", "record digest"),
    ],
)
def test_verify_bundle_rejects_source_catalog_drift(tmp_path: Path, drift: str, message: str) -> None:
    root = _bundle_fixture(tmp_path)
    catalog_path = root / "sources" / "experiment" / "records.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    selected = catalog["latest"]["ratio_response/df"]
    if drift == "selected_record_id":
        selected["record_id"] = "different/response"
    elif drift == "selected_contract":
        selected["contract_id"] = "tidy.v1"
    else:
        selected["content_digest"] = "sha256:" + "f" * 64
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    _refresh_artifact_manifest(root, "sources/experiment/records.json")

    with pytest.raises(ValueError, match=message):
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


def test_bundle_build_does_not_replace_destination_created_while_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.yaml"
    request_path.write_text("request", encoding="utf-8")
    destination = tmp_path / "latest"
    monkeypatch.setattr(bundle_module, "load_response_window_request", lambda _path: object())

    def build_staged_bundle(**kwargs: object) -> None:
        staging = kwargs["staging"]
        assert isinstance(staging, Path)
        (staging / "staged.txt").write_text("staged output", encoding="utf-8")
        destination.mkdir()
        (destination / "owner.txt").write_text("concurrent output", encoding="utf-8")

    monkeypatch.setattr(bundle_module, "_build_staged_bundle", build_staged_bundle)
    monkeypatch.setattr(
        bundle_module,
        "verify_response_window_bundle",
        lambda *_args, **_kwargs: pytest.fail("concurrent output must not be replaced"),
    )

    with pytest.raises(FileExistsError, match="output already exists"):
        bundle_module.build_response_window_bundle(
            request_path=request_path,
            out_dir=destination,
            contracts=builtin_runtime().contracts,
            source_loader=lambda *_args: pytest.fail("source loader should not run"),
        )

    assert (destination / "owner.txt").read_text(encoding="utf-8") == "concurrent output"
    assert list(tmp_path.glob(".latest.staging-*")) == []
    assert list(tmp_path.glob(".latest.backup-*")) == []


def test_bundle_build_does_not_delete_destination_when_publish_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.yaml"
    request_path.write_text("request", encoding="utf-8")
    destination = tmp_path / "latest"
    monkeypatch.setattr(bundle_module, "load_response_window_request", lambda _path: object())

    def build_staged_bundle(**kwargs: object) -> None:
        staging = kwargs["staging"]
        assert isinstance(staging, Path)
        (staging / "staged.txt").write_text("staged output", encoding="utf-8")

    original_rename = Path.rename

    def fail_publish_rename(path: Path, target: Path) -> Path:
        if path.name.startswith(".latest.staging-") and target == destination:
            destination.mkdir()
            (destination / "owner.txt").write_text("concurrent output", encoding="utf-8")
            raise OSError("injected publish race")
        return original_rename(path, target)

    monkeypatch.setattr(bundle_module, "_build_staged_bundle", build_staged_bundle)
    monkeypatch.setattr(Path, "rename", fail_publish_rename)

    with pytest.raises(OSError, match="injected publish race"):
        bundle_module.build_response_window_bundle(
            request_path=request_path,
            out_dir=destination,
            contracts=builtin_runtime().contracts,
            source_loader=lambda *_args: pytest.fail("source loader should not run"),
        )

    assert (destination / "owner.txt").read_text(encoding="utf-8") == "concurrent output"
    assert list(tmp_path.glob(".latest.staging-*")) == []
    assert list(tmp_path.glob(".latest.backup-*")) == []


@pytest.mark.parametrize("destination_kind", ["file", "directory_symlink", "dangling_symlink"])
def test_bundle_build_rejects_non_directory_or_symlink_destination_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_kind: str,
) -> None:
    request_path = tmp_path / "request.yaml"
    request_path.write_text("request", encoding="utf-8")
    monkeypatch.setattr(bundle_module, "load_response_window_request", lambda _path: object())
    monkeypatch.setattr(
        bundle_module,
        "_build_staged_bundle",
        lambda **_kwargs: pytest.fail("invalid output must be rejected before staging"),
    )
    destination = tmp_path / "latest"
    if destination_kind == "file":
        destination.write_text("keep this file", encoding="utf-8")
    elif destination_kind == "directory_symlink":
        target = tmp_path / "linked-target"
        target.mkdir()
        (target / "sentinel.txt").write_text("keep this directory", encoding="utf-8")
        destination.symlink_to(target, target_is_directory=True)
    else:
        destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(ValueError, match="output must be a real directory path"):
        bundle_module.build_response_window_bundle(
            request_path=request_path,
            out_dir=destination,
            contracts=builtin_runtime().contracts,
            source_loader=lambda *_args: pytest.fail("source loader should not run"),
            overwrite=True,
        )

    if destination_kind == "file":
        assert destination.read_text(encoding="utf-8") == "keep this file"
    elif destination_kind == "directory_symlink":
        assert destination.is_symlink()
        assert (target / "sentinel.txt").read_text(encoding="utf-8") == "keep this directory"
    else:
        assert destination.is_symlink()
        assert not destination.exists()
    assert list(tmp_path.glob(".latest.staging-*")) == []
    assert list(tmp_path.glob(".latest.backup-*")) == []


def _bundle_fixture(tmp_path: Path, *, study_id: str = "stress_ethanol_cipro_growth") -> Path:
    root = tmp_path / "bundle"
    (root / "tables").mkdir(parents=True)
    (root / "plots").mkdir()
    (root / "sources" / "experiment").mkdir(parents=True)
    frames = _record_frames()
    for record_id, artifact_id in RECORD_ARTIFACTS.items():
        frames[record_id].to_parquet(root / artifact_id, index=False)
    request = _payload()
    request["study_id"] = study_id
    request["request_id"] = "test-request"
    request["experiment_ids"] = ["experiment"]
    request["display"] = {
        "study_label": "Example response study",
        "event_label": "Stress addition",
        "state_labels": {
            "00": "No stress",
            "10": "Ethanol",
            "01": "Ciprofloxacin",
            "11": "Ethanol + ciprofloxacin",
        },
        "examples": [
            {"design_id": "reference", "label": "Reference anchor", "role": "reference_anchor"},
            {"design_id": "design", "label": "Response example", "role": "response_example"},
        ],
    }
    request["source"]["response_channel"] = "YFP/CFP"
    request["source"]["magnitude_channel"] = "YFP/OD600"
    request["source"]["growth_channel"] = "OD600"
    request["event"]["event_id"] = "event"
    request["event"]["declaration"] = "declared event"
    request["reductions"] = [
        {
            "id": "primary",
            "window_start_event_h": 1.0,
            "window_end_event_h": 2.0,
            "method": "geometric_time_mean",
            "response_basis": "post_window",
            "role": "primary",
        }
    ]
    request["aggregation"] = {
        "replicate_stat": "median",
        "bootstrap_samples": 100,
        "confidence_level": 0.95,
        "random_seed": 17,
    }
    (root / "request.yaml").write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")
    (root / "review.py").write_text("import marimo\n", encoding="utf-8")
    (root / "sources" / "experiment" / "config.yaml").write_text(
        "schema: reader/v7\nexperiment:\n  id: experiment\n  title: Example promoter experiment\n",
        encoding="utf-8",
    )
    source_digests = {
        "annotated/df": "sha256:" + "1" * 64,
        "ratio_magnitude/df": "sha256:" + "2" * 64,
        "ratio_response/df": "sha256:" + "3" * 64,
    }
    source_catalog = {
        "schema_version": 3,
        "history": {},
        "latest": {
            record_id: {
                "record_id": record_id,
                "contract_id": "plate_reader.annotated.v1",
                "content_digest": digest,
            }
            for record_id, digest in source_digests.items()
        },
    }
    (root / "sources" / "experiment" / "records.json").write_text(json.dumps(source_catalog), encoding="utf-8")
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
        "study_id": study_id,
        "request_id": "test-request",
        "created_at": "2026-07-14T00:00:00+00:00",
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
            "well_rows": 16,
            "design_rows": 2,
            "bootstrap_draw_rows": 200,
            "trace_rows": 48,
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
                "records": source_digests,
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
        draws.extend(_draw_row(design_id, draw_index=index, is_reference=is_reference) for index in range(100))
        for state in ("00", "10", "01", "11"):
            for position in ("A1", "A2"):
                wells.append(_well_row(design_id, state=state, position=position, is_reference=is_reference))
                for signal_kind in ("response", "magnitude", "growth"):
                    traces.append(
                        _trace_row(
                            design_id,
                            state=state,
                            position=position,
                            signal_kind=signal_kind,
                            is_reference=is_reference,
                        )
                    )
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


def _well_row(design_id: str, *, state: str, position: str, is_reference: bool) -> dict[str, object]:
    state_index = ("00", "10", "01", "11").index(state)
    replicate_offset = -0.1 if position == "A1" else 0.1
    row: dict[str, object] = {
        "experiment_id": "experiment",
        "design_id": design_id,
        "state": state,
        "position": position,
        "response_well": float(state_index) + replicate_offset,
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
        "bootstrap_samples": 100,
        "confidence_level": 0.95,
        "event_id": "event",
        "event_time_estimate_assay_h": 1.25,
        "event_time_uncertainty_h": 0.25,
        "window_start_event_h": 1.0,
        "window_end_event_h": 2.0,
        "is_reference": is_reference,
        "min_replicates_per_state": 2,
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
        row[f"n{state}"] = 2
    return row


def _draw_row(design_id: str, *, draw_index: int, is_reference: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": "experiment",
        "design_id": design_id,
        "reduction_id": "primary",
        "draw_index": draw_index,
        "is_reference": is_reference,
    }
    for index, state in enumerate(("00", "10", "01", "11")):
        row[f"r{state}"] = float(index)
        row[f"b{state}"] = 0.0 if is_reference else float(index) / 2.0
    return row


def _trace_row(
    design_id: str,
    *,
    state: str,
    position: str,
    signal_kind: str,
    is_reference: bool,
) -> dict[str, object]:
    return {
        "experiment_id": "experiment",
        "design_id": design_id,
        "position": position,
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
