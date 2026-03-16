from __future__ import annotations

from reader.workbench.assets.types import AssetCapabilities, AssetRequirement
from reader.workbench.templates.model import NotebookTemplateDescriptor

DESCRIPTOR = NotebookTemplateDescriptor(
    template="notebook/sfxi_eda",
    domain="logic",
    family="logic_summary",
    summary="SFXI vec8 explorer (EDA scaffold + time slice → corners → vec8).",
    tags=("eda", "sfxi", "logic"),
    source_package=__package__,
    source_name="sfxi_eda.marimo.py.txt",
    capabilities=AssetCapabilities(
        requires_any=(
            AssetRequirement(tag="sfxi"),
            AssetRequirement(record_contract="plate_reader.annotated.v1"),
            AssetRequirement(record_contract_prefix="sfxi.vec8."),
        )
    ),
)
