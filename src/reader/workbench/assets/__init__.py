from __future__ import annotations

import importlib

_EXPORTS = {
    "builtins": {"build_workbench_asset_catalog", "static_asset_catalog"},
    "types": {
        "AssetCapabilities",
        "AssetCatalog",
        "AssetDescriptor",
        "AssetKind",
        "AssetRequirement",
        "build_plugin_asset",
        "plugin_category_from_id",
    },
}

__all__ = tuple(sorted({name for names in _EXPORTS.values() for name in names}))


def __getattr__(name: str):
    for module_name, names in _EXPORTS.items():
        if name in names:
            module = importlib.import_module(f"reader.workbench.assets.{module_name}")
            return getattr(module, name)
    raise AttributeError(name)
