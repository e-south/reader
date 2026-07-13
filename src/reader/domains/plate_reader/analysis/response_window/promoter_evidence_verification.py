"""Independent verification for published promoter-evidence bundles."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

from .promoter_evidence_bundle_contract import (
    PROMOTER_EVIDENCE_ARTIFACT_IDS,
    PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    PROMOTER_EVIDENCE_NON_CLAIM,
    PromoterEvidenceBundle,
)
from .promoter_evidence_overlay import OBJECTIVE_OVERLAY_SCHEMA_VERSION
from .promoter_evidence_selected_binding import verify_selected_binding
from .provenance import sha256_file

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def verify_promoter_evidence_bundle(path: Path) -> PromoterEvidenceBundle:
    """Verify one published evidence bundle and its root-confined artifacts."""

    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"promoter-evidence manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "created_at",
        "claim_status",
        "non_claim_boundary",
        "selection",
        "selected_binding",
        "sources",
        "objective_overlay",
        "artifacts",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError(f"promoter-evidence manifest fields must be exactly {sorted(fields)}.")
    if manifest["schema_version"] != PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"promoter-evidence manifest must use {PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION!r}.")
    _created_at(manifest["created_at"])
    claim_status = str(manifest["claim_status"])
    if claim_status not in {"objective_neutral", "screen_only"}:
        raise ValueError("promoter-evidence bundle has an unsupported claim status.")
    if manifest["non_claim_boundary"] != PROMOTER_EVIDENCE_NON_CLAIM:
        raise ValueError("promoter-evidence bundle must preserve its objective-neutral claim boundary.")
    _verify_selection(manifest["selection"])
    _verify_sources(manifest["sources"])
    verify_selected_binding(
        manifest["selected_binding"],
        baserender_adapter_kind=manifest["sources"]["baserender"]["adapter_kind"],
    )
    _verify_overlay_record(manifest["objective_overlay"], claim_status=claim_status)
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(PROMOTER_EVIDENCE_ARTIFACT_IDS):
        raise ValueError(f"promoter-evidence artifacts must be exactly {sorted(PROMOTER_EVIDENCE_ARTIFACT_IDS)}.")
    for artifact_id in PROMOTER_EVIDENCE_ARTIFACT_IDS:
        _verify_artifact(root, artifact_id, artifacts[artifact_id])
    return PromoterEvidenceBundle(
        root=root,
        manifest_path=manifest_path,
        png_path=root / "promoter_evidence.png",
        pdf_path=root / "promoter_evidence.pdf",
        manifest=manifest,
    )


def _verify_artifact(root: Path, artifact_id: str, value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"promoter-evidence artifact {artifact_id!r} metadata is malformed.")
    if value["path"] != artifact_id:
        raise ValueError(f"promoter-evidence artifact {artifact_id!r} path disagrees with its identity.")
    path = (root / str(value["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"promoter-evidence artifact {artifact_id!r} escapes the bundle root.") from exc
    size = value["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or not path.is_file():
        raise ValueError(f"promoter-evidence artifact {artifact_id!r} is missing or has an invalid size.")
    if path.stat().st_size != size or sha256_file(path) != value["sha256"]:
        raise ValueError(f"promoter-evidence artifact {artifact_id!r} digest or size mismatch.")
    with path.open("rb") as stream:
        signature = stream.read(8)
    if artifact_id.endswith(".pdf") and not signature.startswith(b"%PDF"):
        raise ValueError("promoter-evidence PDF signature is invalid.")
    if artifact_id.endswith(".png") and signature != b"\x89PNG\r\n\x1a\n":
        raise ValueError("promoter-evidence PNG signature is invalid.")


def _verify_selection(value: object) -> None:
    fields = {"experiment_id", "design_id", "candidate_id", "reduction_id"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"promoter-evidence selection fields must be exactly {sorted(fields)}.")
    if any(not isinstance(item, str) or not item.strip() for item in value.values()):
        raise ValueError("promoter-evidence selection values must be non-empty strings.")


def _verify_sources(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"response_window", "candidate_bindings", "baserender"}:
        raise ValueError("promoter-evidence sources must name response_window, candidate_bindings, and baserender.")
    response = value["response_window"]
    binding = value["candidate_bindings"]
    baserender = value["baserender"]
    if not isinstance(response, dict) or set(response) != {"schema_version", "request_id", "manifest_sha256"}:
        raise ValueError("promoter-evidence response-window source metadata is malformed.")
    if response["schema_version"] != "reader.response_window.bundle.v3" or not _is_sha256(response["manifest_sha256"]):
        raise ValueError("promoter-evidence source must be a verified response-window bundle v3.")
    binding_fields = {
        "schema_id",
        "schema_version",
        "study_id",
        "manifest_sha256",
        "records_sha256",
        "candidate_table_id",
        "candidate_selection_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != binding_fields:
        raise ValueError("promoter-evidence candidate-binding source metadata is malformed.")
    if (
        binding["schema_id"] != "dnadesign.study.promoter_candidate_bindings.v1"
        or binding["schema_version"] != "1"
        or binding["study_id"] != "stress_ethanol_cipro_growth"
        or any(not _is_sha256(binding[key]) for key in binding if key.endswith("sha256"))
    ):
        raise ValueError("promoter-evidence source must be the supported candidate-binding contract.")
    diagnostic_fields = {
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
    }
    if not isinstance(baserender, dict) or set(baserender) != diagnostic_fields:
        raise ValueError("promoter-evidence BaseRender diagnostics are malformed.")
    if (
        baserender["contract_id"] != "dnadesign.baserender.sequence_panel.v1"
        or str(baserender["contract_version"]) != "1"
        or baserender["adapter_kind"] not in {"densegen_tfbs", "usr_genbank_annotations_v1"}
    ):
        raise ValueError("promoter-evidence source must use the supported BaseRender contract.")
    positive_diagnostics = ("sequence_length_bp", "strand_count", "image_width_px", "image_height_px")
    if (
        any(not _is_int_at_least(baserender[key], 1) for key in positive_diagnostics)
        or not _is_int_at_least(baserender["feature_count"], 0)
        or not isinstance(baserender["style_profile"], str)
        or not baserender["style_profile"].strip()
        or not isinstance(baserender["renderer_name"], str)
        or not baserender["renderer_name"].strip()
        or not isinstance(baserender["legend_entries"], list)
        or any(not isinstance(item, str) or not item.strip() for item in baserender["legend_entries"])
    ):
        raise ValueError("promoter-evidence BaseRender diagnostics contain invalid values.")


def _verify_overlay_record(value: object, *, claim_status: str) -> None:
    if value is None:
        if claim_status != "objective_neutral":
            raise ValueError("non-neutral promoter evidence requires an objective overlay record.")
        return
    fields = {
        "schema_version",
        "objective_id",
        "claim_status",
        "manifest_sha256",
        "components",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"promoter-evidence objective overlay fields must be exactly {sorted(fields)}.")
    if (
        value["schema_version"] != OBJECTIVE_OVERLAY_SCHEMA_VERSION
        or not _is_nonempty_text(value["objective_id"])
        or value["claim_status"] != claim_status
        or claim_status == "objective_neutral"
        or not _is_sha256(value["manifest_sha256"])
    ):
        raise ValueError("promoter-evidence objective overlay identity or claim status is invalid.")
    _verify_components(value["components"])


def _verify_components(value: object) -> None:
    if not isinstance(value, list) or not 1 <= len(value) <= 6:
        raise ValueError(
            "promoter-evidence objective overlay components must contain between one and six raw components."
        )
    fields = {"component_id", "label", "value", "unit"}
    ids: list[str] = []
    for component in value:
        if (
            not isinstance(component, dict)
            or set(component) != fields
            or not _is_nonempty_text(component["component_id"])
            or not _is_nonempty_text(component["label"])
            or not _is_nonempty_text(component["unit"])
            or not _is_finite(component["value"])
        ):
            raise ValueError("promoter-evidence objective component is malformed.")
        ids.append(component["component_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("promoter-evidence objective component identities must be unique.")


def _created_at(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("promoter-evidence created_at must be an ISO-8601 timestamp.")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("promoter-evidence created_at must be an ISO-8601 timestamp.") from exc
    if timestamp.tzinfo is None:
        raise ValueError("promoter-evidence created_at must include a timezone.")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_int_at_least(value: object, minimum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


__all__ = ["verify_promoter_evidence_bundle"]
