from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ontology import PluginCategory, PluginSemantics


@dataclass(frozen=True)
class PluginDescriptor:
    uses: str
    category: PluginCategory
    key: str
    cls: type[Any]
    semantics: PluginSemantics

    @property
    def domain(self) -> str:
        return self.semantics.domain

    @property
    def family(self) -> str:
        return self.semantics.family

    @property
    def summary(self) -> str:
        return self.semantics.summary

    @property
    def tags(self) -> tuple[str, ...]:
        return self.semantics.tags


class PluginCatalog:
    def __init__(self, descriptors: list[PluginDescriptor]):
        self._descriptors = tuple(sorted(descriptors, key=lambda item: (item.category, item.family, item.key)))

    def all(self) -> tuple[PluginDescriptor, ...]:
        return self._descriptors

    def filter(
        self,
        *,
        category: str | None = None,
        domain: str | None = None,
        family: str | None = None,
    ) -> tuple[PluginDescriptor, ...]:
        return tuple(
            descriptor
            for descriptor in self._descriptors
            if (category is None or descriptor.category == category)
            and (domain is None or descriptor.domain == domain)
            and (family is None or descriptor.family == family)
        )

    def categories(self) -> dict[str, tuple[PluginDescriptor, ...]]:
        out: dict[str, list[PluginDescriptor]] = {}
        for descriptor in self._descriptors:
            out.setdefault(descriptor.category, []).append(descriptor)
        return {category: tuple(items) for category, items in out.items()}

    def domains(self) -> dict[str, tuple[PluginDescriptor, ...]]:
        out: dict[str, list[PluginDescriptor]] = {}
        for descriptor in self._descriptors:
            out.setdefault(descriptor.domain, []).append(descriptor)
        return {domain: tuple(items) for domain, items in out.items()}

    def families(self) -> dict[str, tuple[PluginDescriptor, ...]]:
        out: dict[str, list[PluginDescriptor]] = {}
        for descriptor in self._descriptors:
            out.setdefault(descriptor.family, []).append(descriptor)
        return {family: tuple(items) for family, items in out.items()}
