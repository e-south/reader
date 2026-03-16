from reader.workbench.assets import AssetCatalog as NotebookTemplateCatalog
from reader.workbench.assets import AssetDescriptor as NotebookTemplateDescriptor
from reader.workbench.assets import (
    list_notebook_template_assets,
    notebook_template_asset_catalog,
    resolve_notebook_template_asset,
)


def notebook_template_catalog() -> NotebookTemplateCatalog:
    return notebook_template_asset_catalog()


def list_notebook_templates() -> list[tuple[str, str]]:
    return list_notebook_template_assets()


def resolve_notebook_template_descriptor(name: str) -> NotebookTemplateDescriptor:
    return resolve_notebook_template_asset(name)


__all__ = [
    "NotebookTemplateCatalog",
    "NotebookTemplateDescriptor",
    "list_notebook_templates",
    "notebook_template_catalog",
    "resolve_notebook_template_descriptor",
]
