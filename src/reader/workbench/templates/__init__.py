from __future__ import annotations

import importlib

_EXPORTS = {
    "catalog": {
        "builtin_notebook_template_catalog",
        "compatible_notebook_templates",
        "list_notebook_templates",
        "require_notebook_template_for_protocol",
        "resolve_notebook_template_descriptor",
        "select_default_notebook_template",
    },
    "model": {
        "NotebookTemplateCatalog",
        "NotebookTemplateDescriptor",
    },
}

__all__ = [
    "NotebookTemplateCatalog",
    "NotebookTemplateDescriptor",
    "builtin_notebook_template_catalog",
    "compatible_notebook_templates",
    "list_notebook_templates",
    "require_notebook_template_for_protocol",
    "resolve_notebook_template_descriptor",
    "select_default_notebook_template",
]


def __getattr__(name: str):
    for module_name, names in _EXPORTS.items():
        if name in names:
            module = importlib.import_module(f"reader.workbench.templates.{module_name}")
            return getattr(module, name)
    raise AttributeError(name)
