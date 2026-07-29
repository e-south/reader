from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from reader.runtime import ReaderRuntime
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return deepcopy(value)


class ApiResult:
    """Base for typed public results with a JSON-ready projection."""

    def to_dict(self) -> dict[str, object]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class ExperimentEvidence(ApiResult):
    data_class: str
    data_class_reason: str
    replicate_kind: str
    replicate_identity_field: str | None = None


@dataclass(frozen=True)
class ExperimentIdentity(ApiResult):
    id: str
    title: str
    lifecycle: str
    protocol: str
    config: str
    root: str
    evidence: ExperimentEvidence | None = None


@dataclass(frozen=True)
class Experiment:
    """A loaded, protocol-bound experiment handle for public API operations."""

    config_path: Path
    _spec: ReaderSpec = field(repr=False)
    _declaration: WorkbenchDecl = field(repr=False)
    _runtime: ReaderRuntime = field(repr=False)

    @property
    def identity(self) -> ExperimentIdentity:
        decl = self._declaration
        semantic_evidence = decl.experiment_semantics.evidence
        return ExperimentIdentity(
            id=decl.experiment.id,
            title=decl.experiment.title,
            lifecycle=decl.experiment.lifecycle,
            protocol=decl.experiment_semantics.protocol.id,
            config=str(self.config_path),
            root=str(decl.experiment.root),
            evidence=(
                ExperimentEvidence(
                    data_class=semantic_evidence.data_class,
                    data_class_reason=semantic_evidence.data_class_reason,
                    replicate_kind=semantic_evidence.replicate_kind,
                    replicate_identity_field=semantic_evidence.replicate_identity_field,
                )
                if semantic_evidence is not None
                else None
            ),
        )


@dataclass(frozen=True)
class InspectionResult(ApiResult):
    experiment: ExperimentIdentity
    authoring: Mapping[str, object]
    semantics: Mapping[str, object]
    implementation: Mapping[str, object]


@dataclass(frozen=True)
class ValidationResult(ApiResult):
    experiment: ExperimentIdentity
    check_files: bool
    status: str
    summary: Mapping[str, object]
    validation: Mapping[str, object]


@dataclass(frozen=True)
class PlanResult(ApiResult):
    experiment: ExperimentIdentity
    plan: Mapping[str, object]
    compiled: Mapping[str, object]
    semantics: Mapping[str, object]


@dataclass(frozen=True)
class NotebookResult(ApiResult):
    experiment: ExperimentIdentity
    path: str
    template: str
    created: bool


@dataclass(frozen=True)
class SurfaceCatalogResult(ApiResult):
    experiment: ExperimentIdentity
    kind: str
    protocol: str
    selection: Mapping[str, object]
    summary: Mapping[str, object]
    entries: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class RecordCatalogResult(ApiResult):
    experiment: ExperimentIdentity
    catalog_exists: bool
    catalog: Mapping[str, object]
    include_history: bool
    summary: Mapping[str, object]
    entries: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class VerificationResult(ApiResult):
    experiment: ExperimentIdentity
    status: str
    summary: Mapping[str, object]
    issues: tuple[Mapping[str, object], ...]
    records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class SelectedSteps(ApiResult):
    pipeline: tuple[str, ...]
    plots: tuple[str, ...]
    exports: tuple[str, ...]


@dataclass(frozen=True)
class RecordRevision(ApiResult):
    record_id: str
    revision: int
    revision_digest: str


@dataclass(frozen=True)
class DataFrameRecordResult(ApiResult):
    """One content-digest-verified, contract-validated dataframe record."""

    experiment: ExperimentIdentity
    record: RecordRevision
    contract_id: str
    content_digest: str
    dataframe: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataframe", self.dataframe.copy(deep=True))

    def to_dict(self) -> dict[str, object]:
        """Project JSON-ready metadata without serializing dataframe values."""

        return {
            "experiment": self.experiment.to_dict(),
            "record": self.record.to_dict(),
            "contract_id": self.contract_id,
            "content_digest": self.content_digest,
            "dataframe": {
                "rows": int(self.dataframe.shape[0]),
                "columns": [str(column) for column in self.dataframe.columns],
            },
        }


@dataclass(frozen=True)
class RunResult(ApiResult):
    experiment: ExperimentIdentity
    invocation_id: str | None
    operation: Literal["run", "plot", "export", "mixed"]
    status: Literal["planned", "succeeded"]
    dry_run: bool
    selected_steps: SelectedSteps
    produced_record_revisions: tuple[RecordRevision, ...]
    ledger_path: str | None


@dataclass(frozen=True)
class PluginSummary(ApiResult):
    plugin: str
    key: str
    category: str
    domain: str
    family: str
    summary: str
    implementation: str


@dataclass(frozen=True)
class PluginCatalogResult(ApiResult):
    selection: Mapping[str, object]
    summary: Mapping[str, object]
    plugins: tuple[PluginSummary, ...]


@dataclass(frozen=True)
class PluginPort(ApiResult):
    name: str
    kind: str
    optional: bool
    contract: str | None
    contract_surface: Mapping[str, object] | None = None


@dataclass(frozen=True)
class PluginDescriptorResult(ApiResult):
    plugin: PluginSummary
    config_schema: Mapping[str, object]
    input_ports: tuple[PluginPort, ...]
    output_ports: tuple[PluginPort, ...]


__all__ = [
    "ApiResult",
    "DataFrameRecordResult",
    "Experiment",
    "ExperimentEvidence",
    "ExperimentIdentity",
    "InspectionResult",
    "NotebookResult",
    "PlanResult",
    "PluginCatalogResult",
    "PluginDescriptorResult",
    "PluginPort",
    "PluginSummary",
    "RecordCatalogResult",
    "RecordRevision",
    "RunResult",
    "SelectedSteps",
    "SurfaceCatalogResult",
    "ValidationResult",
    "VerificationResult",
]
