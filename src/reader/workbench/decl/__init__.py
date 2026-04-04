from __future__ import annotations

import importlib

_EXPORTS = {
    "build": {"bind_recipe_steps", "build_workbench_decl", "load_workbench_decl"},
    "model": {
        "ExperimentDecl",
        "FileInputDecl",
        "InputBindingDecl",
        "NotebookDecl",
        "NotebookTemplateCallDecl",
        "PipelineDecl",
        "PluginStepDecl",
        "RecipeSourceDecl",
        "RecordInputDecl",
        "RecordOutputDecl",
        "ResourceInputDecl",
        "SurfaceDecl",
        "WorkbenchDecl",
    },
}

__all__ = tuple(sorted({name for names in _EXPORTS.values() for name in names}))


def __getattr__(name: str):
    for module_name, names in _EXPORTS.items():
        if name in names:
            module = importlib.import_module(f"reader.workbench.decl.{module_name}")
            return getattr(module, name)
    raise AttributeError(name)
