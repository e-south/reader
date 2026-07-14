"""Atomic publication for one objective-neutral promoter evidence figure."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from reader.domains.promoter.candidate_bindings import (
    PromoterCandidateBinding,
    PromoterCandidateBindings,
    load_promoter_candidate_bindings,
)

from .bundle import ResponseWindowBundle
from .promoter_evidence_bundle_contract import (
    PROMOTER_EVIDENCE_ARTIFACT_IDS,
    PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    PROMOTER_EVIDENCE_NON_CLAIM,
    PromoterEvidenceBundle,
)
from .promoter_evidence_figure import promoter_evidence_figure
from .promoter_evidence_overlay import (
    OBJECTIVE_OVERLAY_SCHEMA_VERSION,
    ObjectiveDisplayOverlay,
    load_objective_display_overlay,
)
from .promoter_evidence_selected_binding import selected_binding_record
from .promoter_evidence_verification import verify_promoter_evidence_bundle
from .provenance import sha256_file
from .review import load_review_tables, selected_handoff_row


def build_promoter_evidence_bundle(
    *,
    response_bundle: ResponseWindowBundle,
    bindings_root: Path,
    out_dir: Path,
    experiment_id: str,
    design_id: str,
    reduction_id: str,
    objective_overlay_path: Path | None = None,
    overwrite: bool = False,
) -> PromoterEvidenceBundle:
    """Build and atomically publish one verified promoter-evidence bundle."""

    destination = Path(out_dir).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"promoter-evidence output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _build_staged_bundle(
            response_bundle=response_bundle,
            bindings_root=bindings_root,
            staging=staging,
            experiment_id=experiment_id,
            design_id=design_id,
            reduction_id=reduction_id,
            objective_overlay_path=objective_overlay_path,
        )
        verify_promoter_evidence_bundle(staging)
        if destination.exists():
            destination.rename(backup)
        try:
            staging.rename(destination)
            published = verify_promoter_evidence_bundle(destination)
        except BaseException:
            if destination.exists():
                shutil.rmtree(destination)
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return published
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _build_staged_bundle(
    *,
    response_bundle: ResponseWindowBundle,
    bindings_root: Path,
    staging: Path,
    experiment_id: str,
    design_id: str,
    reduction_id: str,
    objective_overlay_path: Path | None,
) -> None:
    bindings = load_promoter_candidate_bindings(bindings_root)
    response_study_id = response_bundle.manifest.get("study_id")
    if bindings.study_id != response_study_id:
        raise ValueError(
            "promoter-evidence study identity mismatch: "
            f"response_window={response_study_id!r}, candidate_bindings={bindings.study_id!r}."
        )
    binding = bindings.resolve(design_id)
    designs, _wells, traces, events = load_review_tables(response_bundle.root)
    selected = selected_handoff_row(
        designs,
        experiment_id=experiment_id,
        design_id=design_id,
        reduction_id=reduction_id,
    ).iloc[0]
    display = response_bundle.manifest.get("display")
    if not isinstance(display, dict):
        raise ValueError("verified response-window bundle lacks its display contract.")
    overlay = None if objective_overlay_path is None else load_objective_display_overlay(objective_overlay_path)
    figure, diagnostics = promoter_evidence_figure(
        experiment_id=experiment_id,
        design_id=design_id,
        reduction_id=reduction_id,
        selected=selected,
        traces=traces,
        events=events,
        display=display,
        binding=binding,
        objective_overlay=overlay,
    )
    try:
        figure.savefig(
            staging / "promoter_evidence.png",
            dpi=300,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
        )
        figure.savefig(
            staging / "promoter_evidence.pdf",
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
        )
    finally:
        plt.close(figure)
    manifest = _manifest(
        staging=staging,
        response_bundle=response_bundle,
        bindings=bindings,
        binding=binding,
        diagnostics=diagnostics,
        overlay=overlay,
        experiment_id=experiment_id,
        design_id=design_id,
        reduction_id=reduction_id,
    )
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest(
    *,
    staging: Path,
    response_bundle: ResponseWindowBundle,
    bindings: PromoterCandidateBindings,
    binding: PromoterCandidateBinding,
    diagnostics: object,
    overlay: ObjectiveDisplayOverlay | None,
    experiment_id: str,
    design_id: str,
    reduction_id: str,
) -> dict[str, object]:
    return {
        "schema_version": PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "claim_status": "objective_neutral" if overlay is None else overlay.claim_status,
        "non_claim_boundary": PROMOTER_EVIDENCE_NON_CLAIM,
        "selection": {
            "experiment_id": experiment_id,
            "design_id": design_id,
            "candidate_id": binding.candidate_id,
            "reduction_id": reduction_id,
        },
        "selected_binding": selected_binding_record(binding),
        "sources": {
            "response_window": {
                "schema_version": response_bundle.manifest["schema_version"],
                "study_id": response_bundle.manifest["study_id"],
                "request_id": response_bundle.manifest["request_id"],
                "experiment_id": experiment_id,
                "reduction_id": reduction_id,
                "manifest_sha256": sha256_file(response_bundle.manifest_path),
            },
            "candidate_bindings": {
                "schema_id": bindings.schema_id,
                "schema_version": bindings.schema_version,
                "study_id": bindings.study_id,
                "manifest_sha256": bindings.manifest_sha256,
                "records_sha256": bindings.records_sha256,
                "candidate_table_id": bindings.candidate_table_id,
                "candidate_selection_sha256": "sha256:" + bindings.candidate_selection_sha256,
            },
            "baserender": _diagnostics_record(diagnostics),
        },
        "objective_overlay": None if overlay is None else _overlay_record(overlay),
        "artifacts": {
            artifact_id: _artifact_record(staging, artifact_id) for artifact_id in PROMOTER_EVIDENCE_ARTIFACT_IDS
        },
    }


def _artifact_record(root: Path, artifact_id: str) -> dict[str, object]:
    path = root / artifact_id
    return {"path": artifact_id, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _diagnostics_record(diagnostics: object) -> dict[str, object]:
    fields = (
        "contract_id",
        "contract_version",
        "style_profile",
        "renderer_name",
        "adapter_kind",
        "sequence_length_bp",
        "feature_count",
        "strand_count",
        "legend_entries",
        "image_width_px",
        "image_height_px",
    )
    record = {field: getattr(diagnostics, field) for field in fields}
    record["legend_entries"] = list(record["legend_entries"])
    return record


def _overlay_record(overlay: ObjectiveDisplayOverlay) -> dict[str, object]:
    return {
        "schema_version": OBJECTIVE_OVERLAY_SCHEMA_VERSION,
        "objective_id": overlay.objective_id,
        "claim_status": overlay.claim_status,
        "experiment_id": overlay.experiment_id,
        "reader_design_id": overlay.reader_design_id,
        "reduction_id": overlay.reduction_id,
        "manifest_sha256": overlay.manifest_sha256,
        "components": [
            {
                "component_id": component.component_id,
                "label": component.label,
                "value": component.value,
                "unit": component.unit,
            }
            for component in overlay.components
        ],
    }


__all__ = [
    "PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "PromoterEvidenceBundle",
    "build_promoter_evidence_bundle",
    "verify_promoter_evidence_bundle",
]
