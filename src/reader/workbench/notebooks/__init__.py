import importlib

__all__ = [
    "NotebookTemplateCatalog",
    "NotebookTemplateDescriptor",
    "list_notebook_templates",
    "notebook_template_catalog",
    "resolve_notebook_template_descriptor",
    "write_experiment_notebook",
]


def __getattr__(name: str):
    if name in {
        "NotebookTemplateCatalog",
        "NotebookTemplateDescriptor",
        "list_notebook_templates",
        "notebook_template_catalog",
        "resolve_notebook_template_descriptor",
    }:
        _catalog = importlib.import_module("reader.workbench.notebooks.catalog")
        return getattr(_catalog, name)
    if name == "write_experiment_notebook":
        return importlib.import_module("reader.workbench.notebooks.scaffold").write_experiment_notebook
    raise AttributeError(name)
