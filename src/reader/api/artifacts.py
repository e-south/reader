from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reader.workbench.engine.artifacts import ArtifactWrite
from reader.workbench.engine.artifacts import publish_artifact_bundle as _publish
from reader.workbench.engine.invocations import InvocationLedger
from reader.workbench.records import current_build_identity
from reader.workbench.templates import require_notebook_template_for_protocol

from .models import ApiResult, Experiment, ExperimentIdentity, RecordRevision


@dataclass(frozen=True)
class ArtifactSpec:
    """One confined file to materialize inside an experiment-owned bundle."""

    relative_path: str | Path
    description: str
    writer: Callable[[Path], None] = field(repr=False, compare=False)


@dataclass(frozen=True)
class ArtifactBundleResult(ApiResult):
    """A ledger-backed, manifest-backed artifact-bundle publication."""

    experiment: ExperimentIdentity
    invocation_id: str
    provenance_epoch_id: str
    record: RecordRevision
    paths: tuple[str, ...]
    ledger_path: str


def publish_artifact_bundle(
    experiment: Experiment,
    *,
    record_id: str,
    producer_id: str,
    template: str,
    upstream_records: Mapping[str, str],
    producer_config: Mapping[str, Any],
    description: str,
    artifacts: Sequence[ArtifactSpec],
) -> ArtifactBundleResult:
    """Publish interactive deliverables through Reader's canonical provenance path."""

    decl = experiment._declaration
    runtime = experiment._runtime
    bound_protocol = runtime.bind_protocol(decl.experiment_semantics.protocol)
    descriptor = require_notebook_template_for_protocol(template, protocol=bound_protocol)
    writes = tuple(
        ArtifactWrite(
            relative_path=Path(item.relative_path),
            description=item.description,
            writer=item.writer,
        )
        for item in artifacts
    )
    layout = decl.experiment_semantics.layout
    store = runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=decl.experiment.root,
        create=False,
    )
    publication = _publish(
        store=store,
        ledger=InvocationLedger.for_store(store=store),
        config_digest=decl.config_digest,
        build_identity=current_build_identity(),
        producer_id=producer_id,
        producer_template=descriptor.template,
        record_id=record_id,
        upstream_records=upstream_records,
        producer_config=producer_config,
        description=description,
        artifacts=writes,
    )
    return ArtifactBundleResult(
        experiment=experiment.identity,
        invocation_id=publication.invocation_id,
        provenance_epoch_id=publication.provenance_epoch_id,
        record=RecordRevision(
            record_id=publication.record.record_id,
            revision=publication.revision,
            revision_digest=publication.revision_digest,
        ),
        paths=tuple(str(path) for path in publication.record.files),
        ledger_path=str(publication.ledger_path),
    )


__all__ = ["ArtifactBundleResult", "ArtifactSpec", "publish_artifact_bundle"]
