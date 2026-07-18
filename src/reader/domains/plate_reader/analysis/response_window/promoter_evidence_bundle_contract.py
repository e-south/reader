"""Contract values for published promoter-evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION = "reader.response_window.promoter_evidence_bundle.v4"
PROMOTER_EVIDENCE_ARTIFACT_IDS = ("promoter_evidence.pdf", "promoter_evidence.png")
PROMOTER_EVIDENCE_NON_CLAIM = (
    "Reader presents response-window evidence and sequence context; downstream objective scoring, "
    "normalization or calibration, and promotion remain outside Reader."
)


@dataclass(frozen=True)
class PromoterEvidenceBundle:
    root: Path
    manifest_path: Path
    png_path: Path
    pdf_path: Path
    manifest: dict[str, object]


__all__ = [
    "PROMOTER_EVIDENCE_ARTIFACT_IDS",
    "PROMOTER_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "PROMOTER_EVIDENCE_NON_CLAIM",
    "PromoterEvidenceBundle",
]
