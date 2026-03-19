from __future__ import annotations

from reader.workbench.templates.model import NotebookTemplateDescriptor

DESCRIPTOR = NotebookTemplateDescriptor(
    template="notebook/retron_sponge_aggregate",
    domain="generic",
    family="screen_review",
    summary="Cross-run retron sponge library review notebook over exported semantic tables.",
    source_package="reader.workbench.templates.builtins",
    source_name="retron_sponge_aggregate.marimo.py.txt",
)
