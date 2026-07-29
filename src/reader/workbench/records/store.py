from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from reader.contracts import ContractCatalog, ContractId
from reader.errors import ProvenanceEpochChangedError, RecordError
from reader.workbench.graph import FileRef, ProvenanceInput, RecipeSource, RecordRef, ResourceRef, SourceRecordRef
from reader.workbench.ontology import WorkbenchProducerKind, WorkbenchRecordKind
from reader.workbench.paths import resolve_confined_sink_root
from reader.workbench.records.model import (
    DataFrameArtifactRecord,
    FileBundleRecord,
    PathDescription,
    RecordProducer,
    RecordRecipeSource,
    normalize_file_bundle_metadata,
    record_from_dict,
    record_to_dict,
    sha256_file,
    verify_record_artifact_integrity,
)

from .evidence import RecordInputEvidence, capture_artifact_evidence
from .identity import BuildIdentity, current_build_identity, digest_json
from .locking import ProvenanceFileLock, provenance_lock_scope
from .model import record_revision_digest

RECORD_CATALOG_SCHEMA_VERSION = 4
_RECORD_CATALOG_FIELDS = frozenset({"schema_version", "provenance_epoch_id", "latest", "history"})


def _is_canonical_uuid4(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


def _empty_catalog() -> dict[str, Any]:
    return {
        "schema_version": RECORD_CATALOG_SCHEMA_VERSION,
        "provenance_epoch_id": str(uuid4()),
        "latest": {},
        "history": {},
    }


@dataclass(frozen=True)
class RecordCatalogSnapshot:
    schema_version: int
    provenance_epoch_id: str
    active_invocation_ledger: Path
    latest_records: tuple[DataFrameArtifactRecord | FileBundleRecord, ...]
    revision_counts: dict[str, int]


class RecordStore:
    def __init__(
        self,
        outputs_dir: Path,
        *,
        contracts: ContractCatalog,
        plots_subdir: str | None = "plots",
        exports_subdir: str | None = "exports",
        experiment_root: Path | None = None,
        create: bool = True,
    ) -> None:
        self.root = Path(outputs_dir).expanduser().absolute()
        self.contracts = contracts
        self.artifacts_dir = self.root / "artifacts"
        self.manifests_dir = self.root / "manifests"
        self.records_path = self.manifests_dir / "records.json"
        self._catalog_lock_path = self.manifests_dir / ".records.lock"
        self._catalog_lock = ProvenanceFileLock(self._catalog_lock_path, timeout=30)
        self._bound_provenance_epoch_id: str | None = None
        self.experiment_root = Path(experiment_root or self.root.parent).expanduser().absolute()
        if plots_subdir in (None, "", ".", "./"):
            self.plots_dir = self.root
        else:
            self.plots_dir = self.root / plots_subdir
        if exports_subdir in (None, "", ".", "./"):
            self.exports_dir = self.root
        else:
            self.exports_dir = self.root / exports_subdir
        self._validate_sink_roots()
        if create:
            self.ensure_layout()

    def _validate_sink_roots(self) -> None:
        try:
            resolve_confined_sink_root(self.root, root=self.experiment_root, label="outputs")
            for label, path in (
                ("artifacts", self.artifacts_dir),
                ("manifests", self.manifests_dir),
                ("plots", self.plots_dir),
                ("exports", self.exports_dir),
            ):
                resolve_confined_sink_root(path, root=self.root, label=label)
        except ValueError as exc:
            raise RecordError(str(exc)) from exc

    def _validate_catalog_roots(self) -> None:
        try:
            resolve_confined_sink_root(self.root, root=self.experiment_root, label="outputs")
            resolve_confined_sink_root(self.manifests_dir, root=self.root, label="manifests")
        except ValueError as exc:
            raise RecordError(str(exc)) from exc

    def ensure_layout(self) -> None:
        self._validate_sink_roots()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        if self.plots_dir != self.root:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        if self.exports_dir != self.root:
            self.exports_dir.mkdir(parents=True, exist_ok=True)
        if not self.records_path.exists():
            self._write_catalog(_empty_catalog(), create_only=True)

    def catalog_exists(self) -> bool:
        return self.records_path.exists() or self.records_path.is_symlink()

    def provenance_lock_exists(self) -> bool:
        """Return whether the non-followed writer coordination entry already exists."""

        return self._catalog_lock_path.exists() or self._catalog_lock_path.is_symlink()

    def reset_catalog(self) -> str:
        """Atomically begin a fresh catalog and invocation provenance epoch."""

        if self._bound_provenance_epoch_id is not None:
            raise RecordError("A RecordStore bound to a provenance epoch cannot reset the catalog")
        self.ensure_layout()
        catalog = _empty_catalog()
        self._write_catalog(catalog)
        return str(catalog["provenance_epoch_id"])

    def provenance_epoch_id(self) -> str:
        """Return the catalog-owned identity for the active provenance epoch."""

        return str(self._read_catalog()["provenance_epoch_id"])

    def invocation_ledger_path(self) -> Path:
        """Return the active epoch's deterministic invocation-ledger path."""

        epoch_id = self.provenance_epoch_id()
        return self.manifests_dir / "invocations" / f"{epoch_id}.jsonl"

    def catalog_snapshot(self) -> RecordCatalogSnapshot:
        """Read one internally consistent catalog projection for inspection surfaces."""

        with self._catalog_lock_scope(operation="inspect the record catalog"):
            catalog = self._read_catalog()
            epoch_id = str(catalog["provenance_epoch_id"])
            records = tuple(
                self._materialize(record_id, payload) for record_id, payload in sorted(catalog["latest"].items())
            )
            revision_counts = {record_id: len(revisions) for record_id, revisions in sorted(catalog["history"].items())}
        return RecordCatalogSnapshot(
            schema_version=RECORD_CATALOG_SCHEMA_VERSION,
            provenance_epoch_id=epoch_id,
            active_invocation_ledger=self.manifests_dir / "invocations" / f"{epoch_id}.jsonl",
            latest_records=records,
            revision_counts=revision_counts,
        )

    def assert_provenance_epoch(self, expected_epoch_id: str) -> None:
        """Fail if another catalog epoch has replaced the caller's active epoch."""

        active_epoch_id = self.provenance_epoch_id()
        if active_epoch_id != expected_epoch_id:
            raise ProvenanceEpochChangedError(
                "The record catalog provenance epoch changed during this operation; "
                "discard the stale result and start a new Reader invocation."
            )

    def bind_provenance_epoch(self, expected_epoch_id: str) -> None:
        """Bind subsequent reads and writes to one invocation's catalog epoch."""

        self.assert_provenance_epoch(expected_epoch_id)
        self._bound_provenance_epoch_id = expected_epoch_id

    @property
    def provenance_lock(self) -> ProvenanceFileLock:
        """Reentrant writer lease shared by operations, catalog commits, and ledger appends."""

        return self._catalog_lock

    @contextmanager
    def _catalog_lock_scope(self, *, operation: str) -> Iterator[None]:
        """Normalize lock failures without swallowing errors from the protected operation."""

        with provenance_lock_scope(
            self._catalog_lock,
            acquire_error=RecordError(f"Could not {operation}: the record catalog provenance lock is unavailable"),
            release_error=RecordError(f"Could not {operation}: the record catalog provenance lock could not release"),
            release_note="Reader also could not release the record catalog lock",
        ):
            yield

    def assert_catalog_snapshot(self, *, provenance_epoch_id: str, catalog_digest: str) -> None:
        """Fail unless the catalog still matches a previously read snapshot."""

        with self._catalog_lock_scope(operation="validate the record catalog snapshot"):
            self.assert_provenance_epoch(provenance_epoch_id)
            if digest_json(self._read_catalog()) != catalog_digest:
                raise RecordError("The record catalog changed concurrently; retry from the current catalog state.")

    def _read_catalog(self) -> dict[str, Any]:
        self._validate_catalog_roots()
        if self.records_path.is_symlink():
            raise RecordError("records.json must not be a symlink")
        if not self.records_path.exists():
            raise RecordError("records.json is missing")
        descriptor = -1
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(self.records_path, flags)
            catalog_stat = os.fstat(descriptor)
            if not stat.S_ISREG(catalog_stat.st_mode) or catalog_stat.st_nlink != 1:
                raise RecordError("records.json must be a regular file with a single link")
            with os.fdopen(descriptor, mode="r", encoding="utf-8") as stream:
                descriptor = -1
                content = stream.read()
        except UnicodeError as exc:
            raise RecordError(f"records.json is not valid UTF-8: {exc}") from exc
        except RecordError:
            raise
        except OSError as exc:
            raise RecordError(f"Could not read records.json: {exc}") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RecordError(f"records.json is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise RecordError("records.json must be a JSON object")
        schema_version = data.get("schema_version")
        if schema_version != RECORD_CATALOG_SCHEMA_VERSION:
            raise RecordError(
                f"records.json schema_version must be {RECORD_CATALOG_SCHEMA_VERSION} (got {schema_version!r})"
            )
        if set(data) != _RECORD_CATALOG_FIELDS:
            raise RecordError(
                "records.json must contain exactly schema_version, provenance_epoch_id, latest, and history"
            )
        if not _is_canonical_uuid4(data.get("provenance_epoch_id")):
            raise RecordError("records.json provenance_epoch_id must be a canonical UUID4")
        if "latest" not in data or "history" not in data:
            raise RecordError("records.json must include 'latest' and 'history' objects")
        if not isinstance(data["latest"], dict) or not isinstance(data["history"], dict):
            raise RecordError("records.json 'latest' and 'history' must be JSON objects")
        self._validate_catalog_lineage(data)
        if (
            self._bound_provenance_epoch_id is not None
            and data["provenance_epoch_id"] != self._bound_provenance_epoch_id
        ):
            raise ProvenanceEpochChangedError(
                "The record catalog provenance epoch changed during this operation; "
                "discard the stale result and start a new Reader invocation."
            )
        return data

    def _validate_catalog_lineage(self, catalog: dict[str, Any]) -> None:
        latest = catalog["latest"]
        history = catalog["history"]
        latest_ids = set(latest)
        history_ids = set(history)
        if latest_ids != history_ids:
            missing_history = sorted(latest_ids - history_ids)
            orphan_history = sorted(history_ids - latest_ids)
            details = []
            if missing_history:
                details.append("latest records missing history: " + ", ".join(missing_history))
            if orphan_history:
                details.append("history records missing latest: " + ", ".join(orphan_history))
            raise RecordError("records.json latest/history lineage is inconsistent: " + "; ".join(details))
        for record_id in sorted(latest_ids):
            revisions = history[record_id]
            if not isinstance(revisions, list):
                raise RecordError(f"records.json history for {record_id!r} must be a list")
            if not revisions:
                raise RecordError(f"records.json history for latest record {record_id!r} must be non-empty")
            for payload in revisions:
                self._materialize(record_id, payload)
            self._materialize(record_id, latest[record_id])
            if revisions[-1] != latest[record_id]:
                raise RecordError(
                    f"records.json history for {record_id!r} must end with the exact latest record payload"
                )

    def _write_catalog(
        self,
        payload: dict[str, Any],
        *,
        expected_provenance_epoch_id: str | None = None,
        expected_catalog_digest: str | None = None,
        create_only: bool = False,
    ) -> None:
        self._validate_catalog_roots()
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        staged_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.manifests_dir,
                prefix=f".{self.records_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as staged:
                staged_path = Path(staged.name)
                json.dump(payload, staged, indent=2, sort_keys=True)
                staged.flush()
                os.fsync(staged.fileno())
            try:
                with self._catalog_lock_scope(operation="commit the record catalog"):
                    self._validate_catalog_roots()
                    if create_only and self.records_path.exists():
                        return
                    if expected_provenance_epoch_id is not None:
                        self.assert_provenance_epoch(expected_provenance_epoch_id)
                    if expected_catalog_digest is not None:
                        current_digest = digest_json(self._read_catalog())
                        if current_digest != expected_catalog_digest:
                            raise RecordError(
                                "The record catalog changed concurrently; retry from the current catalog state."
                            )
                    staged_path.replace(self.records_path)
            except OSError as exc:
                raise RecordError(f"Could not atomically replace records.json: {exc}") from exc
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)

    def _materialize(self, record_id: str, payload: dict[str, Any]) -> DataFrameArtifactRecord | FileBundleRecord:
        record = record_from_dict(payload, outputs_dir=self.root, experiment_root=self.experiment_root)
        if record.record_id != record_id:
            raise RecordError(
                f"records.json entry key {record_id!r} does not match payload record_id {record.record_id!r}"
            )
        return record

    def capture_inputs(
        self,
        inputs: Iterable[ProvenanceInput],
        *,
        resolved_inputs: Mapping[str, Any] | None = None,
    ) -> tuple[RecordInputEvidence, ...]:
        """Bind immutable evidence to the exact inputs resolved for one computation."""

        resolved_by_label = dict(resolved_inputs or {})
        from .sources import ResolvedSourceRecord, SourceRecordCollection  # noqa: PLC0415

        resolved_sources: dict[SourceRecordRef, ResolvedSourceRecord] = {}
        for value in resolved_by_label.values():
            if isinstance(value, ResolvedSourceRecord):
                candidates = (value,)
            elif isinstance(value, SourceRecordCollection):
                candidates = value.records
            else:
                continue
            for candidate in candidates:
                previous = resolved_sources.get(candidate.ref)
                if previous is not None and previous.revision_digest != candidate.revision_digest:
                    raise RecordError(
                        f"Resolved source record {candidate.ref.experiment_id}:{candidate.ref.record_id} "
                        "has conflicting revisions"
                    )
                resolved_sources[candidate.ref] = candidate

        evidence: list[RecordInputEvidence] = []
        for item in inputs:
            if isinstance(item.ref, RecordRef):
                upstream = resolved_by_label.get(item.label)
                if upstream is None:
                    upstream = self.latest_record(item.ref.record_id)
                if upstream is None:
                    raise RecordError(
                        f"Input record {item.ref.record_id!r} is missing; produce it before persisting this record."
                    )
                if not isinstance(upstream, (DataFrameArtifactRecord, FileBundleRecord)):
                    raise RecordError(f"Resolved input {item.label!r} must be a Reader record")
                if upstream.record_id != item.ref.record_id:
                    raise RecordError(
                        f"Resolved input {item.label!r} is record {upstream.record_id!r}, "
                        f"expected {item.ref.record_id!r}"
                    )
                verify_record_artifact_integrity(upstream, outputs_dir=self.root)
                evidence.append(
                    RecordInputEvidence(
                        label=item.label,
                        ref=item.ref,
                        discovery_policy="record",
                        record_revision_digest=record_revision_digest(upstream, outputs_dir=self.root),
                    )
                )
                continue
            if isinstance(item.ref, SourceRecordRef):
                from .sources import resolve_source_record  # noqa: PLC0415

                upstream = resolved_by_label.get(item.label)
                if upstream is None:
                    upstream = resolved_sources.get(item.ref)
                if upstream is None:
                    upstream = resolve_source_record(item.ref, contracts=self.contracts)
                if not isinstance(upstream, ResolvedSourceRecord):
                    raise RecordError(f"Resolved input {item.label!r} must be a Reader source record")
                if upstream.ref != item.ref:
                    raise RecordError(
                        f"Resolved input {item.label!r} is source record "
                        f"{upstream.ref.experiment_id}:{upstream.ref.record_id}, expected "
                        f"{item.ref.experiment_id}:{item.ref.record_id}"
                    )
                upstream.verify_artifact_integrity()
                evidence.append(
                    RecordInputEvidence(
                        label=item.label,
                        ref=item.ref,
                        discovery_policy="source_record",
                        record_revision_digest=upstream.revision_digest,
                    )
                )
                continue
            if isinstance(item.ref, ResourceRef):
                default_policy = "declared_resource"
            elif isinstance(item.ref, FileRef):
                default_policy = "declared_file"
            else:
                raise RecordError(f"Unsupported input reference for {item.label!r}")
            evidence.append(
                RecordInputEvidence(
                    label=item.label,
                    ref=item.ref,
                    discovery_policy=item.discovery_policy or default_policy,
                    artifact=capture_artifact_evidence(item.ref.path, root=self.experiment_root),
                )
            )
        return tuple(evidence)

    def _require_captured_inputs(self, inputs: Iterable[RecordInputEvidence]) -> tuple[RecordInputEvidence, ...]:
        captured = tuple(inputs)
        if any(not isinstance(item, RecordInputEvidence) for item in captured):
            raise RecordError(
                "Record persistence requires pre-captured RecordInputEvidence; "
                "capture inputs before computation with RecordStore.capture_inputs()."
            )
        self._assert_captured_inputs_current(captured)
        return captured

    def _assert_captured_inputs_current(self, captured: Iterable[RecordInputEvidence]) -> None:
        for item in captured:
            if isinstance(item.ref, RecordRef):
                current = self.latest_record(item.ref.record_id)
                if current is None:
                    raise RecordError(
                        f"Input record {item.ref.record_id!r} changed after input evidence was captured: "
                        "the current revision is missing."
                    )
                current_revision = record_revision_digest(current, outputs_dir=self.root)
                if current_revision != item.record_revision_digest:
                    raise RecordError(f"Input record {item.ref.record_id!r} changed after input evidence was captured.")
                try:
                    verify_record_artifact_integrity(current, outputs_dir=self.root)
                except RecordError as exc:
                    raise RecordError(
                        f"Input record {item.ref.record_id!r} changed after input evidence was captured: {exc}"
                    ) from exc
                continue
            if isinstance(item.ref, SourceRecordRef):
                from .sources import resolve_source_record  # noqa: PLC0415

                try:
                    current = resolve_source_record(item.ref, contracts=self.contracts)
                    current.verify_artifact_integrity()
                except RecordError as exc:
                    raise RecordError(
                        f"Source record {item.ref.experiment_id}:{item.ref.record_id} changed after input "
                        f"evidence was captured: {exc}"
                    ) from exc
                if current.revision_digest != item.record_revision_digest:
                    raise RecordError(
                        f"Source record {item.ref.experiment_id}:{item.ref.record_id} changed after input "
                        "evidence was captured."
                    )
                continue
            try:
                current_artifact = capture_artifact_evidence(item.ref.path, root=self.experiment_root)
            except (OSError, RecordError) as exc:
                raise RecordError(
                    f"Input file for {item.label!r} changed after input evidence was captured: {exc}"
                ) from exc
            if current_artifact != item.artifact:
                raise RecordError(f"Input file for {item.label!r} changed after input evidence was captured.")

    def iter_latest_records(
        self,
        *,
        kind: WorkbenchRecordKind | None = None,
        producer_kind: WorkbenchProducerKind | None = None,
    ) -> tuple[DataFrameArtifactRecord | FileBundleRecord, ...]:
        catalog = self._read_catalog()
        out: list[DataFrameArtifactRecord | FileBundleRecord] = []
        for record_id, payload in sorted(catalog["latest"].items()):
            record = self._materialize(record_id, payload)
            if kind is not None and record.kind != kind:
                continue
            if producer_kind is not None and record.producer.kind != producer_kind:
                continue
            out.append(record)
        return tuple(out)

    def record_history(self, record_id: str) -> tuple[DataFrameArtifactRecord | FileBundleRecord, ...]:
        catalog = self._read_catalog()
        history = catalog["history"].get(record_id, [])
        if not isinstance(history, list):
            raise RecordError(f"records.json history for {record_id!r} must be a list")
        return tuple(self._materialize(record_id, payload) for payload in history)

    def revision_counts(self, record_ids: Iterable[str] | None = None) -> dict[str, int]:
        catalog = self._read_catalog()
        requested = None if record_ids is None else {str(record_id) for record_id in record_ids}
        counts: dict[str, int] = {}
        for record_id, history in catalog["history"].items():
            if requested is not None and record_id not in requested:
                continue
            if not isinstance(history, list):
                raise RecordError(f"records.json history for {record_id!r} must be a list")
            counts[record_id] = len(history)
        if requested is not None:
            for record_id in requested:
                counts.setdefault(record_id, 0)
        return counts

    def latest_dataframe(self, record_id: str) -> DataFrameArtifactRecord | None:
        catalog = self._read_catalog()
        payload = catalog["latest"].get(record_id)
        if payload is None:
            return None
        record = self._materialize(record_id, payload)
        if not isinstance(record, DataFrameArtifactRecord):
            raise RecordError(f"Record {record_id!r} exists but is not a dataframe artifact")
        return record

    def latest_record(self, record_id: str) -> DataFrameArtifactRecord | FileBundleRecord | None:
        catalog = self._read_catalog()
        payload = catalog["latest"].get(record_id)
        if payload is None:
            return None
        return self._materialize(record_id, payload)

    def read_dataframe(self, record_id: str) -> DataFrameArtifactRecord:
        record = self.latest_dataframe(record_id)
        if record is None:
            raise RecordError(f"Dataframe record '{record_id}' is missing; produce it in an earlier step.")
        return record

    def read_record(self, record_id: str) -> DataFrameArtifactRecord | FileBundleRecord:
        record = self.latest_record(record_id)
        if record is None:
            raise RecordError(f"Record '{record_id}' is missing; produce it in an earlier step.")
        return record

    def _revision_dir(self, step_dir: Path) -> Path:
        index = 1
        while True:
            candidate = step_dir if index == 1 else step_dir.with_name(f"{step_dir.name}__r{index}")
            if not candidate.exists():
                return candidate
            index += 1

    def persist_dataframe(
        self,
        *,
        producer_id: str,
        producer_plugin: str,
        out_name: str,
        record_id: str,
        df: pd.DataFrame,
        contract_id: ContractId,
        inputs: Iterable[RecordInputEvidence],
        config_digest: str,
        producer_config_digest: str | None = None,
        code_digest: str | None = None,
        build_identity: BuildIdentity | None = None,
        producer_kind: WorkbenchProducerKind = "pipeline",
        source_recipe: RecipeSource | None = None,
    ) -> DataFrameArtifactRecord:
        self.contracts.validate(df, contract_id=contract_id, where=f"{producer_id}:{out_name}")
        producer = RecordProducer(
            kind=producer_kind,
            id=producer_id,
            plugin=producer_plugin,
            source_recipe=(
                RecordRecipeSource(recipe=source_recipe.recipe, with_=dict(source_recipe.with_ or {}))
                if source_recipe is not None
                else None
            ),
        )
        record_inputs = self._require_captured_inputs(inputs)
        effective_build_identity = build_identity or current_build_identity()
        effective_code_digest = code_digest or effective_build_identity.source_digest
        effective_producer_config_digest = producer_config_digest or config_digest
        self.ensure_layout()
        base_name = f"{producer_id}.{producer_plugin.replace('/', '_')}"
        base_step_dir = self.artifacts_dir / base_name
        catalog = self._read_catalog()
        provenance_epoch_id = str(catalog["provenance_epoch_id"])
        catalog_digest = digest_json(catalog)
        previous_payload = catalog["latest"].get(record_id)
        previous_record = None
        if previous_payload is not None:
            previous_record = self._materialize(record_id, previous_payload)
            if not isinstance(previous_record, DataFrameArtifactRecord):
                raise RecordError(
                    f"Record id {record_id!r} is already used by a non-dataframe record; choose a unique id."
                )
        filename = f"{out_name}.parquet"
        staged_path: Path | None = None
        revision_dir: Path | None = None
        data_path: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=self.artifacts_dir,
                prefix=f".{base_name}.",
                suffix=".parquet",
                delete=False,
            ) as staged:
                staged_path = Path(staged.name)
            df.to_parquet(staged_path, index=False)
            content_digest = sha256_file(staged_path)
            if (
                previous_record is not None
                and previous_record.config_digest == config_digest
                and previous_record.content_digest == content_digest
                and previous_record.contract_id == contract_id
                and previous_record.code_digest == effective_code_digest
                and previous_record.producer_config_digest == effective_producer_config_digest
                and previous_record.build_identity == effective_build_identity
                and previous_record.producer == producer
                and previous_record.inputs == record_inputs
                and previous_record.path.name == filename
            ):
                previous_record.verify_content_digest()
                self.assert_catalog_snapshot(
                    provenance_epoch_id=provenance_epoch_id,
                    catalog_digest=catalog_digest,
                )
                return previous_record

            revision_dir = self._revision_dir(base_step_dir)
            revision_dir.mkdir(parents=True)
            data_path = revision_dir / filename
            staged_path.replace(data_path)
            staged_path = None
            record = DataFrameArtifactRecord(
                record_id=record_id,
                kind="dataframe_artifact",
                producer=producer,
                created_at=datetime.now(UTC).isoformat(),
                inputs=record_inputs,
                config_digest=config_digest,
                contract_id=contract_id,
                path=data_path,
                content_digest=content_digest,
                code_digest=effective_code_digest,
                producer_config_digest=effective_producer_config_digest,
                build_identity=effective_build_identity,
                size_bytes=data_path.stat().st_size,
            )
            payload = record_to_dict(record, outputs_dir=self.root)
            self._assert_captured_inputs_current(record_inputs)
            catalog["latest"][record_id] = payload
            catalog["history"].setdefault(record_id, []).append(payload)
            self._write_catalog(
                catalog,
                expected_provenance_epoch_id=provenance_epoch_id,
                expected_catalog_digest=catalog_digest,
            )
            return record
        except Exception:
            if data_path is not None:
                data_path.unlink(missing_ok=True)
            if revision_dir is not None:
                with suppress(OSError):
                    revision_dir.rmdir()
            raise
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)

    def append_file_bundle(
        self,
        *,
        producer_kind: WorkbenchProducerKind,
        producer_id: str,
        producer_plugin: str,
        record_id: str,
        inputs: Iterable[RecordInputEvidence],
        config_digest: str,
        producer_config_digest: str | None = None,
        files: list[Path],
        description: str,
        path_descriptions: tuple[PathDescription, ...] = (),
        source_recipe: RecipeSource | None = None,
        build_identity: BuildIdentity | None = None,
    ) -> FileBundleRecord:
        producer = RecordProducer(
            kind=producer_kind,
            id=producer_id,
            plugin=producer_plugin,
            source_recipe=(
                RecordRecipeSource(recipe=source_recipe.recipe, with_=dict(source_recipe.with_ or {}))
                if source_recipe is not None
                else None
            ),
        )
        return self._append_file_bundle_record(
            producer=producer,
            record_id=record_id,
            inputs=inputs,
            config_digest=config_digest,
            producer_config_digest=producer_config_digest,
            files=files,
            description=description,
            path_descriptions=path_descriptions,
            build_identity=build_identity,
        )

    def append_notebook_file_bundle(
        self,
        *,
        producer_id: str,
        producer_template: str,
        record_id: str,
        inputs: Iterable[RecordInputEvidence],
        config_digest: str,
        producer_config_digest: str,
        files: list[Path],
        description: str,
        path_descriptions: tuple[PathDescription, ...],
        build_identity: BuildIdentity | None = None,
    ) -> FileBundleRecord:
        """Append a typed file bundle produced through canonical artifact publication."""

        return self._append_file_bundle_record(
            producer=RecordProducer(kind="notebook", id=producer_id, template=producer_template),
            record_id=record_id,
            inputs=inputs,
            config_digest=config_digest,
            producer_config_digest=producer_config_digest,
            files=files,
            description=description,
            path_descriptions=path_descriptions,
            build_identity=build_identity,
        )

    def _append_file_bundle_record(
        self,
        *,
        producer: RecordProducer,
        record_id: str,
        inputs: Iterable[RecordInputEvidence],
        config_digest: str,
        producer_config_digest: str | None,
        files: list[Path],
        description: str,
        path_descriptions: tuple[PathDescription, ...],
        build_identity: BuildIdentity | None,
    ) -> FileBundleRecord:
        normalized_files = tuple(sorted(dict.fromkeys(files), key=str))
        # Validate bundle semantics before touching the filesystem so malformed
        # descriptions and path mappings fail independently of artifact state.
        description, path_descriptions = normalize_file_bundle_metadata(
            producer_kind=producer.kind,
            files=normalized_files,
            description=description,
            path_descriptions=path_descriptions,
        )
        materialized_paths = tuple(path if path.is_absolute() else self.root / path for path in normalized_files)
        missing = [path for path in materialized_paths if not path.exists()]
        if missing:
            raise RecordError(
                f"File-bundle record {record_id!r} cannot persist missing files: "
                + ", ".join(str(path) for path in missing)
            )
        non_files = [path for path in materialized_paths if not path.is_file()]
        if non_files:
            raise RecordError(
                f"File-bundle record {record_id!r} cannot persist non-file paths: "
                + ", ".join(str(path) for path in non_files)
            )
        file_evidence = tuple(capture_artifact_evidence(path, root=self.root) for path in materialized_paths)
        effective_build_identity = build_identity or current_build_identity()
        record_inputs = self._require_captured_inputs(inputs)
        self.ensure_layout()
        catalog = self._read_catalog()
        provenance_epoch_id = str(catalog["provenance_epoch_id"])
        catalog_digest = digest_json(catalog)
        previous_payload = catalog["latest"].get(record_id)
        if previous_payload is not None:
            previous_record = self._materialize(record_id, previous_payload)
            if isinstance(previous_record, DataFrameArtifactRecord):
                raise RecordError(f"Record id {record_id!r} is already used by a dataframe record; choose a unique id.")
        record = FileBundleRecord(
            record_id=record_id,
            kind="file_bundle",
            producer=producer,
            created_at=datetime.now(UTC).isoformat(),
            inputs=record_inputs,
            config_digest=config_digest,
            files=normalized_files,
            description=description,
            path_descriptions=path_descriptions,
            file_evidence=file_evidence,
            producer_config_digest=producer_config_digest or config_digest,
            build_identity=effective_build_identity,
        )
        payload = record_to_dict(record, outputs_dir=self.root)
        self._assert_captured_inputs_current(record_inputs)
        catalog["latest"][record_id] = payload
        catalog["history"].setdefault(record_id, []).append(payload)
        self._write_catalog(
            catalog,
            expected_provenance_epoch_id=provenance_epoch_id,
            expected_catalog_digest=catalog_digest,
        )
        return record
