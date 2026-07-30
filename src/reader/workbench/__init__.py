from __future__ import annotations

import importlib

_EXPORTS = {
    "graph": {
        "FileRef",
        "InputRef",
        "OutputRef",
        "PluginStep",
        "ProvenanceInput",
        "RecipeSource",
        "RecordRef",
        "ResourceRef",
        "Workbench",
        "ensure_unique_workbench_ids",
        "input_ref_display",
        "input_ref_from_dict",
        "input_ref_to_dict",
        "materialize_workbench",
        "output_ref_display",
        "output_ref_from_dict",
        "output_ref_to_dict",
        "provenance_input_from_dict",
        "provenance_input_to_dict",
        "resolve_workbench",
        "select_workbench_specs",
    },
    "ontology": {
        "PluginCategory",
        "PluginSemantics",
        "WorkbenchItemKind",
        "WorkbenchPluginStepKind",
        "WorkbenchProducerKind",
        "WorkbenchRecordKind",
        "WorkbenchSurfaceSemantics",
        "get_workbench_surface_semantics",
    },
    "records": {
        "DataFrameArtifactRecord",
        "FileBundleRecord",
        "RecordProducer",
        "RecordRecipeSource",
        "RecordStore",
        "WorkbenchRecord",
        "record_from_dict",
        "record_paths",
        "record_to_dict",
    },
}

__all__ = tuple(sorted({name for names in _EXPORTS.values() for name in names}))


def __getattr__(name: str):
    for module_name, names in _EXPORTS.items():
        if name in names:
            module = importlib.import_module(f"reader.workbench.{module_name}")
            return getattr(module, name)
    raise AttributeError(name)
