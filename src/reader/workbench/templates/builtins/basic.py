from __future__ import annotations

from reader.workbench.assets.types import AssetCapabilities
from reader.workbench.templates.model import NotebookTemplateDescriptor

DESCRIPTOR = NotebookTemplateDescriptor(
    template="notebook/basic",
    domain="generic",
    family="record_explorer",
    summary="Minimal verified dataframe-record explorer with a design/treatment table.",
    tags=("eda", "records"),
    source_package=__package__,
    source_name="basic.marimo.py.txt",
    capabilities=AssetCapabilities(),
)
