"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/registry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Type

from pydantic import BaseModel

from reader.core.errors import RegistryError


class PluginConfig(BaseModel):
    """Base class for per-plugin configs (pydantic v2)."""


class Plugin(ABC):
    """Contract-driven plugin interface."""
    key: str                # short unique key within category
    category: str           # ingest|merge|transform|plot|validator

    ConfigModel = PluginConfig

    @classmethod
    @abstractmethod
    def input_contracts(cls) -> Mapping[str, str]:
        """Label -> contract id. Use 'none' for file inputs or no inputs."""

    @classmethod
    @abstractmethod
    def output_contracts(cls) -> Mapping[str, str]:
        """Label -> contract id."""

    @abstractmethod
    def run(self, ctx, inputs: Dict[str, Any], cfg: PluginConfig) -> Dict[str, Any]:
        """Execute and return dict of outputs by label."""


class Registry:
    """Entry-point based registry; no module scanning fallbacks."""
    def __init__(self) -> None:
        self._by_category: Dict[str, Dict[str, Type[Plugin]]] = {
            "ingest": {}, "merge": {}, "transform": {}, "plot": {}, "validator": {}
        }

    def register(self, category: str, key: str, cls: Type[Plugin]) -> None:
        if key in self._by_category.get(category, {}):
            raise RegistryError(f"Duplicate plugin key '{category}/{key}'")
        self._by_category[category][key] = cls

    def categories(self) -> Mapping[str, Mapping[str, Type[Plugin]]]:
        return self._by_category

    def resolve(self, uses: str) -> Type[Plugin]:
        try:
            category, key = uses.split("/", 1)
        except ValueError as e:
            raise RegistryError(f"'uses' must be 'category/key', got: {uses!r}") from e
        try:
            return self._by_category[category][key]
        except KeyError:
            available = ", ".join(f"{cat}/{k}" for cat, m in self._by_category.items() for k in m.keys())
            raise RegistryError(f"Unknown plugin '{uses}'. Installed: {available}")


def load_entry_points() -> Registry:
    """Register built-in plugins by scanning the package, then load external ones via entry points."""
    import importlib.metadata as md
    reg = Registry()

    # 1) Built-ins: scan reader.plugins.*
    try:
        import reader.plugins as pkg
    except Exception as e:
        raise RegistryError(
            "Failed to import built‑in plugins package 'reader.plugins'. "
            "Ensure the package is installed and includes the 'plugins' subpackage."
        ) from e

    discovered = 0
    for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        module = importlib.import_module(modinfo.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Plugin) and obj is not Plugin:
                reg.register(getattr(obj, "category"), getattr(obj, "key"), obj)
                discovered += 1
    if discovered == 0:
        raise RegistryError(
            "No built‑in plugins were discovered under 'reader.plugins'. "
            "This typically means your distribution excludes that subpackage. "
            "Fix your packaging (include reader* from src/) and add __init__.py files."
        )

    # 2) External plugins via entry points (third-party)
    def _load(group: str, category: str):
        for ep in md.entry_points(group=group):
            cls = ep.load()
            if not issubclass(cls, Plugin):
                raise RegistryError(f"Entry point {ep.name} in {group} is not a Plugin subclass")
            reg.register(category, getattr(cls, "key"), cls)

    for category in ("ingest", "merge", "transform", "plot"):
        _load(f"reader.{category}", category)

    return reg
