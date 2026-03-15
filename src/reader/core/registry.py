"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/registry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import pandas as pd
from pydantic import BaseModel

import reader.plugins as pkg
from reader.core.contracts import BUILTIN, OutputContractSurface, contract_satisfies, validate_df
from reader.core.errors import ContractError, RegistryError
from reader.core.mpl import ensure_mpl_cache_dir
from reader.core.workbench import PluginCatalog, PluginDescriptor, PluginSemantics


class PluginConfig(BaseModel):
    """Base class for per-plugin configs (pydantic v2)."""

    model_config = {"extra": "forbid"}


class Plugin(ABC):
    """Contract-driven plugin interface."""

    key: str  # short unique key within category
    category: str  # ingest|merge|transform|plot|export|validator
    semantics: PluginSemantics

    ConfigModel = PluginConfig

    @classmethod
    @abstractmethod
    def input_contracts(cls) -> Mapping[str, str]:
        """Label -> contract id. Use 'none' for file inputs or no inputs."""

    @classmethod
    @abstractmethod
    def output_contracts(cls) -> Mapping[str, str]:
        """Label -> contract id."""

    def resolve_output_contracts(
        self,
        *,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        cfg: PluginConfig,
        where: str,
    ) -> Mapping[str, str]:
        """Resolve runtime output contracts; default is the declared minimum."""
        del inputs, outputs, cfg, where
        return dict(self.output_contracts())

    @classmethod
    def output_contract_surfaces(cls) -> Mapping[str, OutputContractSurface]:
        return {name: OutputContractSurface(minimum=contract) for name, contract in cls.output_contracts().items()}

    @classmethod
    def passthrough_output_contract_surfaces(
        cls,
        *,
        passthrough: Mapping[str, str],
        promoted_examples: Mapping[str, tuple[str, ...]] | None = None,
        note: str | None = None,
    ) -> dict[str, OutputContractSurface]:
        surfaces = {name: OutputContractSurface(minimum=contract) for name, contract in cls.output_contracts().items()}
        promoted_examples = promoted_examples or {}
        for out_name in passthrough:
            minimum = cls.output_contracts().get(out_name)
            if minimum is None or minimum == "none":
                continue
            surfaces[out_name] = OutputContractSurface(
                minimum=minimum,
                runtime_mode="passthrough",
                promoted=tuple(promoted_examples.get(out_name, ())),
                note=note,
            )
        return surfaces

    @classmethod
    def promoted_output_contract_surfaces(
        cls,
        *,
        promotions: Mapping[str, tuple[str, ...]],
        note: str | None = None,
    ) -> dict[str, OutputContractSurface]:
        surfaces = {name: OutputContractSurface(minimum=contract) for name, contract in cls.output_contracts().items()}
        for out_name, promoted in promotions.items():
            minimum = cls.output_contracts().get(out_name)
            if minimum is None or minimum == "none":
                continue
            surfaces[out_name] = OutputContractSurface(
                minimum=minimum,
                runtime_mode="promoted",
                promoted=tuple(promoted),
                note=note,
            )
        return surfaces

    @classmethod
    def inherit_dataframe_output_contracts(
        cls,
        *,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        passthrough: Mapping[str, str],
        where: str,
    ) -> dict[str, str]:
        """
        Preserve stricter dataframe contracts across pass-through transforms when
        the emitted dataframe still validates against the input contract.
        """
        resolved = dict(cls.output_contracts())
        for out_name, in_name in passthrough.items():
            if out_name not in resolved or in_name not in inputs or out_name not in outputs:
                continue
            minimum = resolved[out_name]
            actual = getattr(inputs[in_name], "contract_id", None)
            if not contract_satisfies(actual=actual, expected=minimum):
                continue
            if not isinstance(outputs[out_name], pd.DataFrame):
                continue
            contract = BUILTIN.get(actual)
            if contract is None:
                continue
            try:
                validate_df(outputs[out_name], contract, where=f"{where}:{out_name}")
            except ContractError:
                continue
            resolved[out_name] = actual
        return resolved

    @abstractmethod
    def run(self, ctx, inputs: dict[str, Any], cfg: PluginConfig) -> dict[str, Any]:
        """Execute and return dict of outputs by label."""

    @classmethod
    def descriptor(cls) -> PluginDescriptor:
        semantics = getattr(cls, "semantics", None)
        if semantics is None:
            raise RegistryError(f"Plugin {cls.__module__}.{cls.__name__} must declare 'semantics'")
        if semantics.category != cls.category:
            raise RegistryError(
                f"Plugin {cls.__module__}.{cls.__name__} declares category={cls.category!r} "
                f"but semantics.category={semantics.category!r}"
            )
        if not str(cls.key).strip():
            raise RegistryError(f"Plugin {cls.__module__}.{cls.__name__} must declare a non-empty key")
        return PluginDescriptor(
            uses=f"{cls.category}/{cls.key}",
            category=cls.category,
            key=cls.key,
            cls=cls,
            semantics=semantics,
        )


class Registry:
    """Entry-point based registry; no module scanning fallbacks."""

    def __init__(self) -> None:
        self._by_category: dict[str, dict[str, type[Plugin]]] = {
            "ingest": {},
            "merge": {},
            "transform": {},
            "plot": {},
            "export": {},
            "validator": {},
        }
        self._descriptors: dict[str, PluginDescriptor] = {}

    def register(self, category: str, key: str, cls: type[Plugin]) -> None:
        if category != cls.category:
            raise RegistryError(
                f"Registry category mismatch for {cls.__module__}.{cls.__name__}: "
                f"register(category={category!r}) but plugin declares {cls.category!r}"
            )
        descriptor = cls.descriptor()
        if descriptor.key != key:
            raise RegistryError(
                f"Registry key mismatch for {cls.__module__}.{cls.__name__}: "
                f"register(key={key!r}) but plugin declares {descriptor.key!r}"
            )
        if key in self._by_category.get(category, {}):
            raise RegistryError(f"Duplicate plugin key '{category}/{key}'")
        self._by_category[category][key] = cls
        self._descriptors[descriptor.uses] = descriptor

    def categories(self) -> Mapping[str, Mapping[str, type[Plugin]]]:
        return self._by_category

    def catalog(self) -> PluginCatalog:
        return PluginCatalog(list(self._descriptors.values()))

    def resolve_descriptor(self, uses: str) -> PluginDescriptor:
        try:
            return self._descriptors[uses]
        except KeyError:
            available = ", ".join(sorted(self._descriptors))
            raise RegistryError(f"Unknown plugin '{uses}'. Installed: {available}") from None

    def resolve(self, uses: str) -> type[Plugin]:
        return self.resolve_descriptor(uses).cls


def load_entry_points(categories: set[str] | None = None) -> Registry:
    """Register built-in plugins by scanning the package, then load external ones via entry points."""
    reg = Registry()
    wanted = set(categories) if categories else None

    if wanted is None or "plot" in wanted:
        ensure_mpl_cache_dir()

    discovered = 0
    for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if wanted is not None:
            parts = modinfo.name.split(".")
            if len(parts) < 3:
                continue
            category = parts[2]
            if category not in wanted:
                continue
        module = importlib.import_module(modinfo.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Plugin) and obj is not Plugin:
                reg.register(obj.category, obj.key, obj)
                discovered += 1
    if discovered == 0:
        raise RegistryError(
            "No built-in plugins were discovered under 'reader.plugins'. "
            "This typically means your distribution excludes that subpackage. "
            "Fix your packaging (include reader* from src/) and add __init__.py files."
        )

    # 2) External plugins via entry points (third-party)
    def _load(group: str, category: str):
        for ep in md.entry_points(group=group):
            cls = ep.load()
            if not issubclass(cls, Plugin):
                raise RegistryError(f"Entry point {ep.name} in {group} is not a Plugin subclass")
            reg.register(category, cls.key, cls)

    for category in ("ingest", "merge", "transform", "plot", "export"):
        if wanted is not None and category not in wanted:
            continue
        _load(f"reader.{category}", category)

    return reg
