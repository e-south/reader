from __future__ import annotations

from reader.workbench.assets.types import AssetCapabilities
from reader.workbench.templates.model import NotebookTemplateDescriptor

DESCRIPTOR = NotebookTemplateDescriptor(
    template="notebook/cytometry",
    domain="cytometry",
    family="cytometry_eda",
    summary="Cytometry EDA scaffold (FSC/SSC scatter + fluorophore histograms).",
    tags=("eda", "cytometry"),
    source_package=__package__,
    source_name="cytometry.marimo.py.txt",
    capabilities=AssetCapabilities(),
)
