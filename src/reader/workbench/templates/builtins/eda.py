from __future__ import annotations

from reader.workbench.assets.types import AssetCapabilities
from reader.workbench.templates.model import NotebookTemplateDescriptor

DESCRIPTOR = NotebookTemplateDescriptor(
    template="notebook/eda",
    domain="generic",
    family="record_explorer",
    summary="Minimal dataframe-record explorer.",
    tags=("eda", "records", "microplate"),
    source_package=__package__,
    source_name="eda.marimo.py.txt",
    capabilities=AssetCapabilities(
        supports_plot_filters=True,
        inject_plot_specs=True,
    ),
)
