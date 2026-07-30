from __future__ import annotations

import importlib

_EXPORTS = {
    "types": {
        "AssetCatalog",
        "AssetDescriptor",
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
