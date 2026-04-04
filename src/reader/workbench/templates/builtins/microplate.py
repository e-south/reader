from __future__ import annotations

from reader.workbench.assets.types import AssetCapabilities
from reader.workbench.templates.model import NotebookTemplateDescriptor

DESCRIPTOR = NotebookTemplateDescriptor(
    template="notebook/microplate",
    domain="plate_reader",
    family="record_explorer",
    summary="Minimal dataframe-record explorer (same scaffold as notebook/basic).",
    tags=("eda", "microplate"),
    source_package=__package__,
    source_name="microplate.marimo.py.txt",
    capabilities=AssetCapabilities(),
)
