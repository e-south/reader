from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from reader.errors import ConfigError
from reader.workbench.ontology import (
    PluginCategory,
    PluginDomain,
    PluginSemantics,
    validate_plugin_domain,
)


@dataclass(frozen=True)
class AssetDescriptor:
    name: str
    domain: PluginDomain
    family: str
    summary: str
    plugin_cls: type[Any]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", validate_plugin_domain(self.domain))
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("AssetDescriptor.name must be a non-empty string.")
        plugin_category_from_id(self.name)
        if not isinstance(self.plugin_cls, type):
            raise ValueError(f"Plugin asset {self.name!r} must declare a plugin class.")

    @property
    def plugin(self) -> str:
        return self.name

    @property
    def plugin_id(self) -> str:
        return self.plugin

    @property
    def key(self) -> str:
        return self.name.split("/", 1)[1] if "/" in self.name else self.name

    @property
    def category(self) -> PluginCategory:
        return plugin_category_from_id(self.name)

    @property
    def cls(self) -> type[Any]:
        return self.plugin_cls


class AssetCatalog:
    def __init__(self, descriptors: list[AssetDescriptor]):
        self._descriptors = tuple(sorted(descriptors, key=lambda item: (item.domain, item.family, item.name)))
        by_name: dict[str, AssetDescriptor] = {}
        for item in self._descriptors:
            if item.name in by_name:
                raise ConfigError(f"Duplicate plugin {item.name!r}.")
            by_name[item.name] = item
        self._by_name = by_name

    def all(self) -> tuple[AssetDescriptor, ...]:
        return self._descriptors

    def filter(
        self,
        *,
        category: str | None = None,
        domain: str | None = None,
        family: str | None = None,
        tag: str | None = None,
    ) -> tuple[AssetDescriptor, ...]:
        return tuple(
            descriptor
            for descriptor in self._descriptors
            if (category is None or descriptor.category == category)
            and (domain is None or descriptor.domain == domain)
            and (family is None or descriptor.family == family)
            and (tag is None or tag in descriptor.tags)
        )

    def resolve(self, name: str) -> AssetDescriptor:
        try:
            return self._by_name[name]
        except KeyError:
            opts = ", ".join(sorted(item.name for item in self._descriptors))
            raise ConfigError(f"Unknown plugin {name!r}. Available plugins: {opts}") from None

    def list(
        self,
        *,
        domain: str | None = None,
        family: str | None = None,
    ) -> list[tuple[str, str]]:
        items = self.filter(domain=domain, family=family)
        return [(item.name, item.summary) for item in items]


def plugin_category_from_id(plugin_id: str) -> PluginCategory:
    if not isinstance(plugin_id, str) or "/" not in plugin_id:
        raise ValueError(f"Plugin id must be 'category/key', got {plugin_id!r}")
    category = plugin_id.split("/", 1)[0]
    if category not in {"ingest", "transform", "plot", "export", "validator"}:
        raise ValueError(f"Unknown plugin category {category!r} in plugin id {plugin_id!r}")
    return cast(PluginCategory, category)


def build_plugin_asset(*, plugin_id: str, semantics: PluginSemantics, plugin_cls: type[Any]) -> AssetDescriptor:
    plugin_category_from_id(plugin_id)
    return AssetDescriptor(
        name=plugin_id,
        domain=semantics.domain,
        family=semantics.family,
        summary=semantics.summary,
        plugin_cls=plugin_cls,
        tags=tuple(semantics.tags),
    )
