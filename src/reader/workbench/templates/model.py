from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files

from reader.domains.semantics import PluginDomain, validate_plugin_domain
from reader.errors import ConfigError
from reader.workbench.assets.types import AssetCapabilities, AssetDescriptor


@dataclass(frozen=True)
class NotebookTemplateDescriptor:
    template: str
    domain: PluginDomain
    family: str
    summary: str
    source_package: str
    source_name: str
    tags: tuple[str, ...] = ()
    capabilities: AssetCapabilities = field(default_factory=AssetCapabilities)

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", validate_plugin_domain(self.domain))
        if not isinstance(self.template, str) or not self.template.strip():
            raise ValueError("NotebookTemplateDescriptor.template must be a non-empty string.")
        if not isinstance(self.source_package, str) or not self.source_package.strip():
            raise ValueError("NotebookTemplateDescriptor.source_package must be a non-empty string.")
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise ValueError("NotebookTemplateDescriptor.source_name must be a non-empty string.")

    def load_body(self) -> str:
        try:
            return files(self.source_package).joinpath(self.source_name).read_text(encoding="utf-8")
        except Exception as exc:
            raise ConfigError(
                f"Notebook template {self.template!r} could not load source {self.source_package}:{self.source_name}."
            ) from exc

    def as_asset(self) -> AssetDescriptor:
        return AssetDescriptor(
            kind="template",
            name=self.template,
            domain=self.domain,
            family=self.family,
            summary=self.summary,
            tags=self.tags,
            body=self.load_body(),
            capabilities=self.capabilities,
        )


class NotebookTemplateCatalog:
    def __init__(self, descriptors: list[NotebookTemplateDescriptor]):
        self._descriptors = tuple(sorted(descriptors, key=lambda item: (item.domain, item.family, item.template)))
        by_template: dict[str, NotebookTemplateDescriptor] = {}
        for item in self._descriptors:
            if item.template in by_template:
                raise ConfigError(f"Duplicate template {item.template!r}.")
            by_template[item.template] = item
        self._by_template = by_template

    def all(self) -> tuple[NotebookTemplateDescriptor, ...]:
        return self._descriptors

    def resolve(self, name: str) -> NotebookTemplateDescriptor:
        try:
            return self._by_template[name]
        except KeyError:
            options = ", ".join(sorted(item.template for item in self._descriptors))
            raise ConfigError(f"Unknown template {name!r}. Available templates: {options}") from None

    def list(self, *, domain: str | None = None, family: str | None = None) -> list[tuple[str, str]]:
        return [
            (item.template, item.summary)
            for item in self._descriptors
            if (domain is None or item.domain == domain) and (family is None or item.family == family)
        ]
