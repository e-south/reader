"""
--------------------------------------------------------------------------------
<reader project>
src/reader/workbench/registry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from reader.contracts import ContractCatalog, ContractId, OutputContractSurface
from reader.errors import ContractError, RegistryError
from reader.plotting.mpl import ensure_mpl_cache_dir
from reader.workbench.assets import AssetCatalog, AssetDescriptor, plugin_category_from_id
from reader.workbench.ports import (
    InputPortSpec,
    OutputPortSpec,
    validate_input_ports,
    validate_output_ports,
)


class PluginConfig(BaseModel):
    """Base class for per-plugin configs (pydantic v2)."""

    model_config = {"extra": "forbid"}


@dataclass(frozen=True)
class PreflightIssue:
    kind: str
    message: str


class Plugin(ABC):
    """Contract-driven plugin interface."""

    ConfigModel = PluginConfig

    def __init__(self) -> None:
        self._descriptor: AssetDescriptor | None = None
        self._contracts: ContractCatalog | None = None

    @classmethod
    @abstractmethod
    def input_ports(cls) -> Mapping[str, InputPortSpec]:
        """Typed input port declarations."""

    @classmethod
    @abstractmethod
    def output_ports(cls) -> Mapping[str, OutputPortSpec]:
        """Typed output port declarations."""

    def resolve_output_ports(
        self,
        *,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        cfg: PluginConfig,
        where: str,
    ) -> Mapping[str, OutputPortSpec]:
        """Resolve runtime output ports; default is the declared minimum."""
        del inputs, outputs, cfg, where
        return dict(type(self).output_ports())

    @classmethod
    def output_port_surfaces(cls) -> Mapping[str, OutputContractSurface]:
        surfaces: dict[str, OutputContractSurface] = {}
        for name, port in cls.output_ports().items():
            surface = port.contract_surface
            if surface is not None:
                surfaces[name] = surface
        return surfaces

    @classmethod
    def preflight_readiness(
        cls,
        *,
        exp_dir: Path,
        cfg: PluginConfig,
        reads: Mapping[str, Any],
    ) -> tuple[PreflightIssue, ...]:
        del exp_dir, cfg, reads
        return ()

    @classmethod
    def resolve_missing_file_inputs(
        cls,
        *,
        exp_dir: Path,
        cfg: PluginConfig,
        inputs: Mapping[str, Any],
    ) -> Mapping[str, Path]:
        """Resolve optional file inputs that were not bound in the graph."""
        del exp_dir, cfg, inputs
        return {}

    @classmethod
    def passthrough_output_ports(
        cls,
        *,
        outputs: Mapping[str, OutputPortSpec],
        passthrough: Mapping[str, str],
        promoted_examples: Mapping[str, tuple[ContractId, ...]] | None = None,
        note: str | None = None,
    ) -> dict[str, OutputPortSpec]:
        ports = dict(outputs)
        promoted_examples = promoted_examples or {}
        for out_name in passthrough:
            port = ports.get(out_name)
            if port is None or port.kind != "dataframe" or port.contract is None:
                continue
            ports[out_name] = replace(
                port,
                surface=OutputContractSurface(
                    minimum=port.contract,
                    runtime_mode="passthrough",
                    promoted=tuple(promoted_examples.get(out_name, ())),
                    note=note,
                ),
            )
        return ports

    @classmethod
    def promoted_output_ports(
        cls,
        *,
        outputs: Mapping[str, OutputPortSpec],
        promotions: Mapping[str, tuple[ContractId, ...]],
        note: str | None = None,
    ) -> dict[str, OutputPortSpec]:
        ports = dict(outputs)
        for out_name, promoted in promotions.items():
            port = ports.get(out_name)
            if port is None or port.kind != "dataframe" or port.contract is None:
                continue
            ports[out_name] = replace(
                port,
                surface=OutputContractSurface(
                    minimum=port.contract,
                    runtime_mode="promoted",
                    promoted=tuple(promoted),
                    note=note,
                ),
            )
        return ports

    def inherit_dataframe_output_ports(
        self,
        *,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        passthrough: Mapping[str, str],
        where: str,
    ) -> dict[str, OutputPortSpec]:
        """
        Preserve stricter dataframe contracts across pass-through transforms when
        the emitted dataframe still validates against the input contract.
        """
        resolved = dict(type(self).output_ports())
        for out_name, in_name in passthrough.items():
            if out_name not in resolved or in_name not in inputs or out_name not in outputs:
                continue
            port = resolved[out_name]
            if port.kind != "dataframe" or port.contract is None:
                continue
            actual = getattr(inputs[in_name], "contract_id", None)
            if actual in (None, "none"):
                continue
            if not self.contracts.satisfies(actual=actual, expected=port.contract):
                continue
            if not isinstance(outputs[out_name], pd.DataFrame):
                continue
            try:
                self.contracts.validate(outputs[out_name], contract_id=actual, where=f"{where}:{out_name}")
            except ContractError:
                continue
            resolved[out_name] = replace(port, contract=actual)
        return resolved

    @abstractmethod
    def run(self, ctx, inputs: dict[str, Any], cfg: PluginConfig) -> dict[str, Any]:
        """Execute and return dict of outputs by label."""

    def bind_runtime(self, *, descriptor: AssetDescriptor, contracts: ContractCatalog) -> None:
        if descriptor.kind != "plugin":
            raise RegistryError(f"Cannot bind non-plugin descriptor {descriptor.name!r} to plugin instance")
        if descriptor.cls is not type(self):
            raise RegistryError(
                f"Descriptor {descriptor.plugin_id!r} points to {descriptor.cls.__module__}.{descriptor.cls.__name__}, "
                f"not {type(self).__module__}.{type(self).__name__}"
            )
        self._descriptor = descriptor
        self._contracts = contracts

    @property
    def descriptor(self) -> AssetDescriptor:
        if self._descriptor is None:
            raise RegistryError(
                f"Plugin instance {type(self).__module__}.{type(self).__name__} is missing a bound descriptor"
            )
        return self._descriptor

    @property
    def contracts(self) -> ContractCatalog:
        if self._contracts is None:
            raise RegistryError(
                f"Plugin instance {type(self).__module__}.{type(self).__name__} is missing a bound contract catalog"
            )
        return self._contracts

    @property
    def plugin_id(self) -> str:
        return self.descriptor.plugin_id

    @property
    def plugin_key(self) -> str:
        return self.descriptor.key

    @property
    def plugin_category(self) -> str:
        category = self.descriptor.category
        if category is None:
            raise RegistryError(f"Plugin descriptor {self.descriptor.name!r} is missing a category")
        return category


class Registry:
    """Descriptor-driven plugin registry with explicit built-ins and explicit entry points."""

    def __init__(self, *, contracts: ContractCatalog) -> None:
        self._descriptors: dict[str, AssetDescriptor] = {}
        self.contracts = contracts

    def register(self, descriptor: AssetDescriptor) -> None:
        if descriptor.kind != "plugin":
            raise RegistryError(f"Registry can only register plugin descriptors, got {descriptor.kind!r}")
        if not issubclass(descriptor.cls, Plugin):
            raise RegistryError(
                f"Plugin descriptor {descriptor.plugin_id!r} must point to a Plugin subclass, "
                f"got {descriptor.cls.__module__}.{descriptor.cls.__name__}"
            )
        if descriptor.plugin in self._descriptors:
            raise RegistryError(f"Duplicate plugin {descriptor.plugin!r}")
        validate_input_ports(descriptor.cls.input_ports(), where=descriptor.plugin)
        outputs = validate_output_ports(descriptor.cls.output_ports(), where=descriptor.plugin)
        surfaces = descriptor.cls.output_port_surfaces()
        for name, surface in surfaces.items():
            port = outputs.get(name)
            if port is None:
                raise RegistryError(f"{descriptor.plugin}: output port surface {name!r} does not match a declared port")
            if port.kind != "dataframe":
                raise RegistryError(f"{descriptor.plugin}: output port surface {name!r} requires a dataframe port")
            if surface.minimum != port.contract:
                raise RegistryError(
                    f"{descriptor.plugin}: output port surface {name!r} minimum {surface.minimum!r} "
                    f"must match declared contract {port.contract!r}"
                )
        self._descriptors[descriptor.plugin] = descriptor

    def categories(self) -> Mapping[str, Mapping[str, type[Plugin]]]:
        grouped: dict[str, dict[str, type[Plugin]]] = {
            "ingest": {},
            "transform": {},
            "plot": {},
            "export": {},
            "validator": {},
        }
        for descriptor in self._descriptors.values():
            category = descriptor.category
            if category is None:
                continue
            grouped[category][descriptor.key] = descriptor.cls
        return grouped

    def catalog(self) -> AssetCatalog:
        return AssetCatalog(list(self._descriptors.values()))

    def resolve_descriptor(self, plugin: str) -> AssetDescriptor:
        try:
            return self._descriptors[plugin]
        except KeyError:
            available = ", ".join(sorted(self._descriptors))
            raise RegistryError(f"Unknown plugin '{plugin}'. Installed: {available}") from None

    def resolve(self, plugin: str) -> type[Plugin]:
        return self.resolve_descriptor(plugin).cls


def _coerce_external_descriptor(loaded: Any, *, ep_name: str) -> AssetDescriptor:
    if isinstance(loaded, AssetDescriptor):
        descriptor = loaded
    elif isinstance(loaded, type):
        raise RegistryError(
            f"Entry point {ep_name!r} must expose a plugin descriptor or descriptor factory, not a plugin class"
        )
    elif isinstance(loaded, Callable):
        descriptor = loaded()
    else:
        descriptor = loaded
    if not isinstance(descriptor, AssetDescriptor):
        raise RegistryError(
            f"Entry point {ep_name!r} must load an AssetDescriptor or a zero-arg callable returning one"
        )
    if descriptor.kind != "plugin":
        raise RegistryError(f"Entry point {ep_name!r} must resolve to a plugin descriptor")
    if ep_name != descriptor.plugin_id:
        raise RegistryError(f"Entry point {ep_name!r} must match descriptor plugin id {descriptor.plugin_id!r}")
    return descriptor


def load_plugin_catalog(*, contracts: ContractCatalog, categories: set[str] | None = None) -> Registry:
    """Register built-in plugins from an explicit manifest, then load external descriptors via entry points."""
    reg = Registry(contracts=contracts)
    wanted = set(categories) if categories else None

    if wanted is None or "plot" in wanted:
        ensure_mpl_cache_dir()

    builtin_descriptors = importlib.import_module("reader.workbench.assets.plugin_manifest").builtin_plugin_descriptors(
        categories=wanted
    )
    if not builtin_descriptors:
        raise RegistryError("No built-in plugin descriptors were declared in the built-in plugin manifest.")
    for descriptor in builtin_descriptors:
        reg.register(descriptor)

    for ep in md.entry_points(group="reader.plugins"):
        entry_point_category = plugin_category_from_id(ep.name)
        if wanted is not None and entry_point_category not in wanted:
            continue
        descriptor = _coerce_external_descriptor(ep.load(), ep_name=ep.name)
        reg.register(descriptor)

    return reg
