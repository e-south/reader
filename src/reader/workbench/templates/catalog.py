from __future__ import annotations

from reader.errors import ConfigError
from reader.protocols.model import BoundProtocol

from .builtins import BUILTIN_NOTEBOOK_TEMPLATES
from .model import NotebookTemplateCatalog, NotebookTemplateDescriptor

_BUILTIN_TEMPLATE_CATALOG = NotebookTemplateCatalog(list(BUILTIN_NOTEBOOK_TEMPLATES))


def builtin_notebook_template_catalog() -> NotebookTemplateCatalog:
    return _BUILTIN_TEMPLATE_CATALOG


def compatible_notebook_templates(
    *,
    protocol: BoundProtocol,
    family: str | None = None,
    domain: str | None = None,
) -> tuple[NotebookTemplateDescriptor, ...]:
    descriptors: list[NotebookTemplateDescriptor] = []
    for template in protocol.allowed_notebook_templates:
        descriptor = resolve_notebook_template_descriptor(template)
        if family is not None and descriptor.family != family:
            continue
        if domain is not None and descriptor.domain != domain:
            continue
        descriptors.append(descriptor)
    return tuple(descriptors)


def list_notebook_templates(
    *,
    protocol: BoundProtocol | None = None,
    family: str | None = None,
    domain: str | None = None,
) -> list[tuple[str, str]]:
    if protocol is None:
        return builtin_notebook_template_catalog().list(family=family, domain=domain)
    return [
        (item.template, item.summary)
        for item in compatible_notebook_templates(protocol=protocol, family=family, domain=domain)
    ]


def resolve_notebook_template_descriptor(name: str) -> NotebookTemplateDescriptor:
    return builtin_notebook_template_catalog().resolve(name)


def require_notebook_template_for_protocol(name: str, *, protocol: BoundProtocol) -> NotebookTemplateDescriptor:
    descriptor = resolve_notebook_template_descriptor(name)
    if not protocol.allows_notebook_template(descriptor.template):
        options = ", ".join(protocol.allowed_notebook_templates) or "—"
        raise ConfigError(
            f"Protocol {protocol.id!r} does not allow notebook template {descriptor.template!r}. "
            f"Allowed templates: {options}"
        )
    return descriptor


def select_default_notebook_template(*, protocol: BoundProtocol) -> NotebookTemplateDescriptor:
    return require_notebook_template_for_protocol(protocol.default_notebook_template, protocol=protocol)
