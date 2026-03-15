from __future__ import annotations

from dataclasses import dataclass

from reader.core.errors import ConfigError

from . import templates


@dataclass(frozen=True)
class NotebookTemplateDescriptor:
    uses: str
    family: str
    summary: str
    template: str
    tags: tuple[str, ...] = ()


class NotebookTemplateCatalog:
    def __init__(self, descriptors: list[NotebookTemplateDescriptor], aliases: dict[str, str] | None = None):
        self._descriptors = tuple(sorted(descriptors, key=lambda item: (item.family, item.uses)))
        self._aliases = dict(aliases or {})
        self._by_uses = {item.uses: item for item in self._descriptors}

    def all(self) -> tuple[NotebookTemplateDescriptor, ...]:
        return self._descriptors

    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def normalize(self, uses: str) -> str:
        return self._aliases.get(uses, uses)

    def resolve(self, uses: str) -> NotebookTemplateDescriptor:
        canonical = self.normalize(uses)
        try:
            return self._by_uses[canonical]
        except KeyError:
            opts = ", ".join(sorted(self._by_uses))
            raise ConfigError(f"Unknown notebook template {uses!r}. Available templates: {opts}") from None

    def list(self) -> list[tuple[str, str]]:
        return [(item.uses, item.summary) for item in self._descriptors]


_CATALOG = NotebookTemplateCatalog(
    descriptors=[
        NotebookTemplateDescriptor(
            uses="notebook/eda",
            family="record_explorer",
            summary="Minimal dataframe-record explorer.",
            template=templates.EXPERIMENT_NOTEBOOK_EDA_TEMPLATE,
            tags=("eda", "records", "microplate"),
        ),
        NotebookTemplateDescriptor(
            uses="notebook/basic",
            family="record_explorer",
            summary="Minimal dataframe-record explorer with design/treatment table and parquet preview.",
            template=templates.EXPERIMENT_EDA_BASIC_TEMPLATE,
            tags=("eda", "records"),
        ),
        NotebookTemplateDescriptor(
            uses="notebook/microplate",
            family="record_explorer",
            summary="Minimal dataframe-record explorer (same scaffold as notebook/basic).",
            template=templates.EXPERIMENT_EDA_MICROPLATE_TEMPLATE,
            tags=("eda", "microplate"),
        ),
        NotebookTemplateDescriptor(
            uses="notebook/cytometry",
            family="cytometry_eda",
            summary="Cytometry EDA scaffold (FSC/SSC scatter + fluorophore histograms).",
            template=templates.EXPERIMENT_EDA_CYTOMETRY_TEMPLATE,
            tags=("eda", "cytometry"),
        ),
        NotebookTemplateDescriptor(
            uses="notebook/sfxi_eda",
            family="logic_summary",
            summary="SFXI vec8 explorer (EDA scaffold + time slice → corners → vec8).",
            template=templates.EXPERIMENT_SFXI_EDA_TEMPLATE,
            tags=("eda", "sfxi", "logic"),
        ),
    ],
)


def notebook_template_catalog() -> NotebookTemplateCatalog:
    return _CATALOG


def list_notebook_presets() -> list[tuple[str, str]]:
    return notebook_template_catalog().list()


def normalize_notebook_preset(name: str) -> str:
    return notebook_template_catalog().normalize(name)


def resolve_notebook_template_descriptor(name: str) -> NotebookTemplateDescriptor:
    return notebook_template_catalog().resolve(name)


def resolve_notebook_preset(name: str) -> str:
    return resolve_notebook_template_descriptor(name).template
