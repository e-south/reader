from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from reader.errors import ConfigError
from reader.workbench.ontology import PluginCategory, PluginDomain, PluginSemantics, validate_plugin_domain

AssetKind = Literal["plugin", "template"]


@dataclass(frozen=True)
class AssetRequirement:
    plugin: str | None = None
    domain: PluginDomain | None = None
    tag: str | None = None
    record_contract: str | None = None
    record_contract_prefix: str | None = None

    def __post_init__(self) -> None:
        if self.domain is not None:
            object.__setattr__(self, "domain", validate_plugin_domain(self.domain))
        if (
            self.plugin is None
            and self.domain is None
            and self.tag is None
            and self.record_contract is None
            and self.record_contract_prefix is None
        ):
            raise ValueError("AssetRequirement must declare at least one selector.")


@dataclass(frozen=True)
class AssetCapabilities:
    supports_plot_filters: bool = False
    inject_plot_specs: bool = False
    requires_any: tuple[AssetRequirement, ...] = ()


@dataclass(frozen=True)
class AssetDescriptor:
    kind: AssetKind
    name: str
    domain: PluginDomain
    family: str
    summary: str
    tags: tuple[str, ...] = ()
    plugin_cls: type[Any] | None = None
    body: str | None = None
    capabilities: AssetCapabilities = field(default_factory=AssetCapabilities)

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", validate_plugin_domain(self.domain))
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("AssetDescriptor.name must be a non-empty string.")
        if self.kind == "plugin":
            plugin_category_from_id(self.name)
            if self.plugin_cls is None:
                raise ValueError(f"Plugin asset {self.name!r} must declare a plugin class.")
        elif self.kind == "template":
            if self.body is None:
                raise ValueError(f"Template asset {self.name!r} must declare a body.")
            if self.plugin_cls is not None:
                raise ValueError(f"Template asset {self.name!r} cannot carry plugin fields.")

    @property
    def plugin(self) -> str:
        if self.kind != "plugin":
            raise AttributeError("Only plugin assets expose .plugin")
        return self.name

    @property
    def plugin_id(self) -> str:
        return self.plugin

    @property
    def template(self) -> str:
        if self.kind != "template":
            raise AttributeError("Only template assets expose .template")
        return self.name

    @property
    def key(self) -> str:
        if self.kind != "plugin":
            raise AttributeError("Only plugin assets expose .key")
        return self.name.split("/", 1)[1] if "/" in self.name else self.name

    @property
    def category(self) -> PluginCategory | None:
        if self.kind != "plugin":
            return None
        return plugin_category_from_id(self.name)

    @property
    def cls(self) -> type[Any]:
        if self.kind != "plugin" or self.plugin_cls is None:
            raise AttributeError("Only plugin assets expose .cls")
        return self.plugin_cls


class AssetCatalog:
    def __init__(self, descriptors: list[AssetDescriptor]):
        self._descriptors = tuple(
            sorted(descriptors, key=lambda item: (item.kind, item.domain, item.family, item.name))
        )
        by_key: dict[tuple[AssetKind, str], AssetDescriptor] = {}
        for item in self._descriptors:
            key = (item.kind, item.name)
            if key in by_key:
                raise ConfigError(f"Duplicate {item.kind} {item.name!r}.")
            by_key[key] = item
        self._by_key = by_key

    def all(self) -> tuple[AssetDescriptor, ...]:
        return self._descriptors

    def filter(
        self,
        *,
        kind: AssetKind | None = None,
        category: str | None = None,
        domain: str | None = None,
        family: str | None = None,
        tag: str | None = None,
    ) -> tuple[AssetDescriptor, ...]:
        return tuple(
            descriptor
            for descriptor in self._descriptors
            if (kind is None or descriptor.kind == kind)
            and (category is None or descriptor.category == category)
            and (domain is None or descriptor.domain == domain)
            and (family is None or descriptor.family == family)
            and (tag is None or tag in descriptor.tags)
        )

    def resolve(self, name: str, *, kind: AssetKind) -> AssetDescriptor:
        try:
            return self._by_key[(kind, name)]
        except KeyError:
            opts = ", ".join(sorted(item.name for item in self.filter(kind=kind)))
            raise ConfigError(f"Unknown {kind} {name!r}. Available {kind}s: {opts}") from None

    def list(
        self,
        *,
        kind: AssetKind,
        domain: str | None = None,
        family: str | None = None,
    ) -> list[tuple[str, str]]:
        items = self.filter(kind=kind, domain=domain, family=family)
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
        kind="plugin",
        name=plugin_id,
        domain=semantics.domain,
        family=semantics.family,
        summary=semantics.summary,
        tags=tuple(semantics.tags),
        plugin_cls=plugin_cls,
    )
