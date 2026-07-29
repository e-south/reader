from __future__ import annotations

from reader.workbench.assets.types import AssetCapabilities
from reader.workbench.templates.model import NotebookTemplateDescriptor

DESCRIPTOR = NotebookTemplateDescriptor(
    template="notebook/basic",
    domain="generic",
    family="record_explorer",
    summary="Minimal dataframe-record explorer with design/treatment table and parquet preview.",
    tags=("eda", "records"),
    source_package=__package__,
    source_name="basic.marimo.py.txt",
    capabilities=AssetCapabilities(),
)
