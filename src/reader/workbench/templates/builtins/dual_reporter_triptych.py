from __future__ import annotations

from reader.workbench.assets.types import AssetCapabilities, AssetRequirement
from reader.workbench.templates.model import NotebookTemplateDescriptor

DESCRIPTOR = NotebookTemplateDescriptor(
    template="notebook/dual_reporter_triptych",
    domain="plate_reader",
    family="screen_review",
    summary="Dual-reporter OD600 + ratio kinetics + snapshot triptych.",
    tags=("dual_reporter", "triptych", "plate_reader"),
    source_package=__package__,
    source_name="dual_reporter_triptych.marimo.py.txt",
    capabilities=AssetCapabilities(
        requires_any=(
            AssetRequirement(domain="plate_reader"),
            AssetRequirement(record_contract="plate_reader.annotated.v1"),
        )
    ),
)
