from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from reader.errors import InvocationFinalizationError, RecordError
from reader.workbench.graph import ProvenanceInput, RecordRef
from reader.workbench.paths import resolve_path_within_root
from reader.workbench.records import (
    BuildIdentity,
    FileBundleRecord,
    PathDescription,
    RecordInputEvidence,
    RecordStore,
    digest_json,
)

from .invocations import (
    InvocationLedger,
    capture_revision_snapshot,
    produced_record_revisions,
)

ArtifactWriter = Callable[[Path], None]
_SAFE_PRODUCER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ArtifactWrite:
    """Normalized engine input for one file in a published artifact bundle."""

    relative_path: Path
    description: str
    writer: ArtifactWriter

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or path == Path(".") or any(part == ".." for part in path.parts):
            raise RecordError("Artifact paths must identify relative, confined files")
        description = str(self.description).strip()
        if not description or "\n" in description or "\r" in description:
            raise RecordError("Artifact descriptions must be non-empty, single-line strings")
        if not callable(self.writer):
            raise RecordError("Artifact writers must be callable")
        object.__setattr__(self, "relative_path", path)
        object.__setattr__(self, "description", description)


@dataclass(frozen=True)
class ArtifactPublication:
    record: FileBundleRecord
    invocation_id: str
    ledger_path: Path
    revision: int
    revision_digest: str


def publish_artifact_bundle(
    *,
    store: RecordStore,
    ledger: InvocationLedger,
    config_digest: str,
    build_identity: BuildIdentity,
    producer_id: str,
    producer_template: str,
    record_id: str,
    upstream_records: Mapping[str, str],
    producer_config: Mapping[str, Any],
    description: str,
    artifacts: Sequence[ArtifactWrite],
) -> ArtifactPublication:
    """Publish a confined bundle through RecordStore and the invocation ledger."""

    normalized_producer_id = str(producer_id).strip()
    if not _SAFE_PRODUCER_ID.fullmatch(normalized_producer_id) or normalized_producer_id in {".", ".."}:
        raise RecordError("Artifact producer_id must be one safe path segment")
    normalized_record_id = str(record_id).strip()
    if not normalized_record_id:
        raise RecordError("Artifact record_id must be a non-empty string")
    normalized_template = str(producer_template).strip()
    if not normalized_template:
        raise RecordError("Artifact producer template must be a non-empty string")
    normalized_description = str(description).strip()
    if not normalized_description or "\n" in normalized_description or "\r" in normalized_description:
        raise RecordError("Artifact bundle description must be a non-empty, single-line string")
    writes = tuple(artifacts)
    if not writes or any(not isinstance(item, ArtifactWrite) for item in writes):
        raise RecordError("Artifact bundles must contain ArtifactWrite values")
    relative_paths = tuple(item.relative_path for item in writes)
    if len(set(relative_paths)) != len(relative_paths):
        raise RecordError("Artifact bundle paths must be unique")
    if not upstream_records:
        raise RecordError("Artifact bundles require at least one upstream record")
    if not store.catalog_exists():
        raise RecordError("Artifact publication requires an existing canonical record catalog")

    normalized_upstream_records = _normalized_upstream_records(upstream_records)
    if normalized_record_id in normalized_upstream_records.values():
        raise RecordError(f"Artifact record {normalized_record_id!r} must not reference itself as an upstream record")
    existing_record = store.latest_record(normalized_record_id)
    if existing_record is not None and existing_record.kind != "file_bundle":
        raise RecordError(
            f"Record id {normalized_record_id!r} is already used by a dataframe record; choose a unique id."
        )

    inputs = _capture_upstream_inputs(store=store, upstream_records=normalized_upstream_records)
    declared_inputs = [
        {
            "phase": "exports",
            "step_id": normalized_producer_id,
            "port": label,
            "ref": {"record": upstream_record_id},
        }
        for label, upstream_record_id in sorted(normalized_upstream_records.items())
    ]
    attempt = ledger.append_attempt(
        config_digest=config_digest,
        build_identity=build_identity,
        operation="export",
        selected_step_ids={"pipeline": [], "plots": [], "exports": [normalized_producer_id]},
        declared_inputs=declared_inputs,
    )
    before = capture_revision_snapshot(store)
    promoted_dir: Path | None = None
    staging_dir: Path | None = None
    try:
        staging_parent = _confined_staging_parent(outputs_dir=store.root)
        staging_dir = Path(mkdtemp(prefix=f"{normalized_producer_id}__", dir=staging_parent))
        staged_paths: list[Path] = []
        for spec in writes:
            staged_path = resolve_path_within_root(spec.relative_path, root=staging_dir)
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            spec.writer(staged_path)
            staged_paths.append(staged_path)
        _validate_written_artifacts(staged_paths, staging_dir=staging_dir)

        store.exports_dir.mkdir(parents=True, exist_ok=True)
        final_dir = _next_revision_dir(store.exports_dir / normalized_producer_id)
        staging_dir.replace(final_dir)
        staging_dir = None
        promoted_dir = final_dir
        final_paths = [final_dir / path for path in relative_paths]
        record = store.append_notebook_file_bundle(
            producer_id=normalized_producer_id,
            producer_template=normalized_template,
            record_id=normalized_record_id,
            inputs=inputs,
            config_digest=config_digest,
            producer_config_digest=digest_json(dict(producer_config)),
            files=final_paths,
            description=normalized_description,
            path_descriptions=tuple(
                PathDescription(path=path, description=spec.description)
                for path, spec in zip(final_paths, writes, strict=True)
            ),
            build_identity=build_identity,
        )
    except BaseException as exc:
        if promoted_dir is not None:
            shutil.rmtree(promoted_dir, ignore_errors=True)
        try:
            ledger.append_result(attempt, exit_status=1, produced_record_revisions=[], failure=exc)
        except BaseException as ledger_failure:
            exc.add_note(
                f"Reader also could not persist the failed invocation result ({type(ledger_failure).__name__})."
            )
        raise
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)

    # RecordStore now owns the promoted revision. Ledger finalization is a
    # separate durable write: if it cannot be confirmed, preserve the record
    # and report an indeterminate terminal event instead of writing a false
    # failure or deleting catalog-owned evidence.
    revisions: list[dict[str, Any]] = []
    try:
        revisions = produced_record_revisions(before=before, after=capture_revision_snapshot(store))
        produced = next(item for item in revisions if item["record_id"] == normalized_record_id)
        ledger.append_result(attempt, exit_status=0, produced_record_revisions=revisions)
    except BaseException as exc:
        raise InvocationFinalizationError(
            "Artifact records were committed, but Reader could not confirm the invocation result. "
            "Keep the committed evidence and run reader verify before handoff.",
            invocation_id=attempt.invocation_id,
            produced_record_revisions=tuple(dict(item) for item in revisions),
        ) from exc
    return ArtifactPublication(
        record=record,
        invocation_id=attempt.invocation_id,
        ledger_path=ledger.path,
        revision=int(produced["revision"]),
        revision_digest=str(produced["revision_digest"]),
    )


def _normalized_upstream_records(upstream_records: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_label, raw_record_id in upstream_records.items():
        label = str(raw_label).strip()
        record_id = str(raw_record_id).strip()
        if not label or not record_id:
            raise RecordError("Artifact upstream record labels and ids must be non-empty strings")
        if label in normalized:
            raise RecordError(f"Artifact upstream record label {label!r} is duplicated")
        normalized[label] = record_id
    return normalized


def _capture_upstream_inputs(
    *,
    store: RecordStore,
    upstream_records: Mapping[str, str],
) -> tuple[RecordInputEvidence, ...]:
    inputs: list[ProvenanceInput] = []
    resolved_inputs: dict[str, Any] = {}
    for label, upstream_record_id in sorted(_normalized_upstream_records(upstream_records).items()):
        upstream = store.latest_record(upstream_record_id)
        if upstream is None:
            raise RecordError(
                f"Input record {upstream_record_id!r} is missing; produce it before publishing artifacts."
            )
        inputs.append(ProvenanceInput(label=label, ref=RecordRef(record_id=upstream_record_id)))
        resolved_inputs[label] = upstream
    return store.capture_inputs(inputs, resolved_inputs=resolved_inputs)


def _confined_staging_parent(*, outputs_dir: Path) -> Path:
    raw = outputs_dir / ".staging"
    message = "Artifact staging directory must stay within the experiment outputs directory"
    if raw.is_symlink():
        raise RecordError(message)
    try:
        staging_parent = resolve_path_within_root(".staging", root=outputs_dir)
        staging_parent.mkdir(parents=True, exist_ok=True)
        resolved = staging_parent.resolve(strict=True)
        resolved.relative_to(outputs_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RecordError(message) from exc
    if staging_parent.is_symlink() or not resolved.is_dir():
        raise RecordError(message)
    return resolved


def _validate_written_artifacts(paths: Sequence[Path], *, staging_dir: Path) -> None:
    staging_root = staging_dir.resolve(strict=True)
    expected: set[Path] = set()
    for path in paths:
        if path.is_symlink():
            raise RecordError(f"Artifact {path} must be a confined regular file")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(staging_root)
        except (FileNotFoundError, ValueError) as exc:
            raise RecordError(f"Artifact {path} must be a confined regular file") from exc
        if not resolved.is_file() or resolved.stat().st_size == 0:
            raise RecordError(f"Artifact {path} must be a non-empty confined regular file")
        expected.add(resolved.relative_to(staging_root))

    discovered: set[Path] = set()
    for path in staging_root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise RecordError("Artifact writers must create exactly the declared non-empty files")
        if path.is_file():
            if path.stat().st_size == 0:
                raise RecordError("Artifact writers must create exactly the declared non-empty files")
            discovered.add(path.relative_to(staging_root))
    if discovered != expected:
        raise RecordError("Artifact writers must create exactly the declared non-empty files")


def _next_revision_dir(base: Path) -> Path:
    revision = 1
    while True:
        candidate = base if revision == 1 else base.with_name(f"{base.name}__r{revision}")
        if not candidate.exists():
            return candidate
        revision += 1


__all__ = ["ArtifactPublication", "ArtifactWrite", "publish_artifact_bundle"]
