from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd
import pytest

from reader.domains.plate_reader.analysis.response_window.provenance import sha256_file
from reader.domains.promoter import sequence_panel as sequence_panel_module
from reader.response_window_review import (
    PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    build_promoter_evidence_bundle,
    verify_promoter_evidence_bundle,
)
from reader.tests.domains.plate_reader.analysis.response_window.test_bundle import _bundle_fixture
from reader.tests.domains.plate_reader.analysis.response_window.test_promoter_evidence_bindings import (
    _configure_genbank,
    _rewrite_binding_table,
    _write_binding_fixture,
)
from reader.tests.domains.plate_reader.analysis.response_window.test_promoter_evidence_figure import (
    _FakeBaseRender,
    _FakeGenBankBaseRender,
)
from reader.tests.domains.plate_reader.analysis.response_window.test_promoter_evidence_overlay import (
    _write_overlay,
)


def test_build_promoter_evidence_publishes_white_png_pdf_and_digest_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    response_root = _bundle_fixture(tmp_path)
    binding_root = _write_binding_fixture(tmp_path, reader_design_id="design")
    out_dir = tmp_path / "promoter-evidence"
    monkeypatch.setattr(sequence_panel_module, "require_baserender_api", lambda: _FakeBaseRender)

    bundle = build_promoter_evidence_bundle(
        response_bundle_root=response_root,
        bindings_root=binding_root,
        out_dir=out_dir,
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
    )
    verified = verify_promoter_evidence_bundle(out_dir)

    assert bundle.root == out_dir.resolve()
    assert verified.manifest == bundle.manifest
    assert bundle.png_path.is_file()
    assert bundle.pdf_path.read_bytes().startswith(b"%PDF")
    image = mpimg.imread(bundle.png_path)
    assert image.shape[1] > image.shape[0]
    assert image[0, 0, :3].tolist() == [1.0, 1.0, 1.0]
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert manifest["claim_status"] == "objective_neutral"
    assert manifest["selection"] == {
        "experiment_id": "experiment",
        "design_id": "design",
        "candidate_id": "candidate-spyp",
        "reduction_id": "primary",
    }
    assert manifest["sources"]["response_window"]["schema_version"] == "reader.response_window.bundle.v3"
    assert manifest["sources"]["candidate_bindings"]["schema_id"] == ("dnadesign.study.promoter_candidate_bindings.v1")
    assert manifest["sources"]["candidate_bindings"]["study_id"] == "stress_ethanol_cipro_growth"
    assert manifest["selected_binding"] == {
        "sequence_sha256": "sha256:" + hashlib.sha256(b"ACGTACGT").hexdigest(),
        "sequence_authority_dataset_id": "source-dataset",
        "sequence_authority_id": "source-row-1",
        "sequence_authority_sha256": "sha256:" + "a" * 64,
        "source_class": "measured_reference",
        "design_family": "stress_promoter",
        "binding_status": "resolved",
        "binding_method": "exact_alias",
        "densegen_plan": "plan-v1",
        "densegen_run_id": "run-v1",
        "densegen_sampling_library_hash": "library-v1",
    }
    assert set(manifest["artifacts"]) == {"promoter_evidence.pdf", "promoter_evidence.png"}
    for artifact_id, record in manifest["artifacts"].items():
        assert record["path"] == artifact_id
        assert record["sha256"] == sha256_file(out_dir / artifact_id)


def test_build_promoter_evidence_records_a_study_supplied_screen_only_overlay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    response_root = _bundle_fixture(tmp_path)
    binding_root = _write_binding_fixture(tmp_path, reader_design_id="design")
    overlay_path = _write_overlay(tmp_path)
    monkeypatch.setattr(sequence_panel_module, "require_baserender_api", lambda: _FakeBaseRender)

    bundle = build_promoter_evidence_bundle(
        response_bundle_root=response_root,
        bindings_root=binding_root,
        objective_overlay_path=overlay_path,
        out_dir=tmp_path / "screen-only-evidence",
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
    )

    overlay = bundle.manifest["objective_overlay"]
    assert bundle.manifest["claim_status"] == "screen_only"
    assert overlay["objective_id"] == "response_magnitude_feasibility_v1"
    assert overlay["manifest_sha256"] == sha256_file(overlay_path)
    assert set(overlay) == {
        "schema_version",
        "objective_id",
        "claim_status",
        "manifest_sha256",
        "components",
    }


def test_genbank_evidence_records_explicit_null_densegen_provenance(tmp_path: Path, monkeypatch) -> None:
    response_root = _bundle_fixture(tmp_path)
    binding_root = _write_binding_fixture(tmp_path, reader_design_id="design")
    frame = pd.read_parquet(binding_root / "bindings.parquet")
    _configure_genbank(frame)
    _rewrite_binding_table(binding_root, frame)
    monkeypatch.setattr(sequence_panel_module, "require_baserender_api", lambda: _FakeGenBankBaseRender)

    bundle = build_promoter_evidence_bundle(
        response_bundle_root=response_root,
        bindings_root=binding_root,
        out_dir=tmp_path / "genbank-evidence",
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
    )

    selected_binding = bundle.manifest["selected_binding"]
    assert selected_binding["densegen_plan"] is None
    assert selected_binding["densegen_run_id"] is None
    assert selected_binding["densegen_sampling_library_hash"] is None
    assert bundle.manifest["sources"]["baserender"]["adapter_kind"] == "usr_genbank_annotations_v1"


def test_promoter_evidence_verifier_rejects_artifact_and_claim_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    response_root = _bundle_fixture(tmp_path)
    binding_root = _write_binding_fixture(tmp_path, reader_design_id="design")
    overlay_path = _write_overlay(tmp_path)
    out_dir = tmp_path / "verified-evidence"
    monkeypatch.setattr(sequence_panel_module, "require_baserender_api", lambda: _FakeBaseRender)
    bundle = build_promoter_evidence_bundle(
        response_bundle_root=response_root,
        bindings_root=binding_root,
        objective_overlay_path=overlay_path,
        out_dir=out_dir,
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
    )
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    manifest["objective_overlay"]["calibrated_score"] = 0.5
    bundle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="objective overlay fields must be exactly"):
        verify_promoter_evidence_bundle(out_dir)

    del manifest["objective_overlay"]["calibrated_score"]
    bundle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle.png_path.write_bytes(bundle.png_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        verify_promoter_evidence_bundle(out_dir)


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("pdf_signature", "PDF signature"),
        ("negative_diagnostics", "BaseRender diagnostics"),
        ("production_claim", "unsupported claim status"),
        ("selected_extra", "selected-binding fields"),
        ("selected_sequence_digest", "selected-binding provenance"),
        ("selected_binding_method", "selected-binding provenance"),
        ("selected_densegen_missing", "requires selected-binding"),
        ("selected_adapter_mismatch", "null DenseGen"),
        ("overlay_too_many", "between one and six"),
        ("overlay_empty_objective", "identity or claim status"),
        ("overlay_empty_label", "component is malformed"),
    ],
)
def test_promoter_evidence_verifier_rejects_semantic_manifest_drift(
    tmp_path: Path,
    monkeypatch,
    drift: str,
    message: str,
) -> None:
    response_root = _bundle_fixture(tmp_path)
    binding_root = _write_binding_fixture(tmp_path, reader_design_id="design")
    overlay_path = _write_overlay(tmp_path)
    out_dir = tmp_path / f"evidence-{drift}"
    monkeypatch.setattr(sequence_panel_module, "require_baserender_api", lambda: _FakeBaseRender)
    bundle = build_promoter_evidence_bundle(
        response_bundle_root=response_root,
        bindings_root=binding_root,
        objective_overlay_path=overlay_path,
        out_dir=out_dir,
        experiment_id="experiment",
        design_id="design",
        reduction_id="primary",
    )
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    if drift == "pdf_signature":
        bundle.pdf_path.write_bytes(b"not-a-pdf")
        manifest["artifacts"]["promoter_evidence.pdf"] = {
            "path": "promoter_evidence.pdf",
            "bytes": bundle.pdf_path.stat().st_size,
            "sha256": sha256_file(bundle.pdf_path),
        }
    elif drift == "negative_diagnostics":
        manifest["sources"]["baserender"]["image_width_px"] = -1
    elif drift == "production_claim":
        manifest["claim_status"] = "production"
        manifest["objective_overlay"]["claim_status"] = "production"
    elif drift == "selected_extra":
        manifest["selected_binding"]["legacy_sequence"] = "ACGTACGT"
    elif drift == "selected_sequence_digest":
        manifest["selected_binding"]["sequence_sha256"] = "not-a-digest"
    elif drift == "selected_binding_method":
        manifest["selected_binding"]["binding_method"] = "prefix_alias"
    elif drift == "selected_densegen_missing":
        manifest["selected_binding"]["densegen_plan"] = None
    elif drift == "selected_adapter_mismatch":
        manifest["sources"]["baserender"]["adapter_kind"] = "usr_genbank_annotations_v1"
    elif drift == "overlay_too_many":
        manifest["objective_overlay"]["components"] *= 7
    elif drift == "overlay_empty_objective":
        manifest["objective_overlay"]["objective_id"] = ""
    else:
        manifest["objective_overlay"]["components"][0]["label"] = ""
    bundle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_promoter_evidence_bundle(out_dir)
