from __future__ import annotations

import json
import os
from collections.abc import Iterable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from reader.contracts import ContractCatalog, ContractId
from reader.errors import RecordError
from reader.workbench.graph import ProvenanceInput, RecipeSource
from reader.workbench.ontology import WorkbenchProducerKind, WorkbenchRecordKind
from reader.workbench.records.model import (
    DataFrameArtifactRecord,
    FileBundleRecord,
    PathDescription,
    RecordProducer,
    RecordRecipeSource,
    record_from_dict,
    record_to_dict,
    sha256_file,
)

RECORD_CATALOG_SCHEMA_VERSION = 3


def _empty_catalog() -> dict[str, Any]:
    return {"schema_version": RECORD_CATALOG_SCHEMA_VERSION, "latest": {}, "history": {}}


class RecordStore:
    def __init__(
        self,
        outputs_dir: Path,
        *,
        contracts: ContractCatalog,
        plots_subdir: str | None = "plots",
        exports_subdir: str | None = "exports",
        create: bool = True,
    ) -> None:
        self.root = outputs_dir
        self.contracts = contracts
        self.artifacts_dir = self.root / "artifacts"
        self.manifests_dir = self.root / "manifests"
        self.records_path = self.manifests_dir / "records.json"
        if plots_subdir in (None, "", ".", "./"):
            self.plots_dir = self.root
        else:
            self.plots_dir = self.root / plots_subdir
        if exports_subdir in (None, "", ".", "./"):
            self.exports_dir = self.root
        else:
            self.exports_dir = self.root / exports_subdir
        if create:
            self.ensure_layout()

    def ensure_layout(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        if self.plots_dir != self.root:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        if self.exports_dir != self.root:
            self.exports_dir.mkdir(parents=True, exist_ok=True)
        if not self.records_path.exists():
            self._write_catalog(_empty_catalog())

    def catalog_exists(self) -> bool:
        return self.records_path.exists()

    def _read_catalog(self) -> dict[str, Any]:
        if not self.records_path.exists():
            raise RecordError("records.json is missing")
        try:
            data = json.loads(self.records_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RecordError(f"records.json is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise RecordError("records.json must be a JSON object")
        schema_version = data.get("schema_version")
        if schema_version != RECORD_CATALOG_SCHEMA_VERSION:
            raise RecordError(
                f"records.json schema_version must be {RECORD_CATALOG_SCHEMA_VERSION} (got {schema_version!r})"
            )
        if "latest" not in data or "history" not in data:
            raise RecordError("records.json must include 'latest' and 'history' objects")
        if not isinstance(data["latest"], dict) or not isinstance(data["history"], dict):
            raise RecordError("records.json 'latest' and 'history' must be JSON objects")
        return data

    def _write_catalog(self, payload: dict[str, Any]) -> None:
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
                staged_path.replace(self.records_path)
            except OSError as exc:
                raise RecordError(f"Could not atomically replace records.json: {exc}") from exc
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)

    def _materialize(self, record_id: str, payload: dict[str, Any]) -> DataFrameArtifactRecord | FileBundleRecord:
        record = record_from_dict(payload, outputs_dir=self.root)
        if record.record_id != record_id:
            raise RecordError(
                f"records.json entry key {record_id!r} does not match payload record_id {record.record_id!r}"
            )
        return record

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
        inputs: list[ProvenanceInput],
        config_digest: str,
        code_digest: str | None = None,
        producer_kind: WorkbenchProducerKind = "pipeline",
        source_recipe: RecipeSource | None = None,
    ) -> DataFrameArtifactRecord:
        self.ensure_layout()
        base_name = f"{producer_id}.{producer_plugin.replace('/', '_')}"
        base_step_dir = self.artifacts_dir / base_name
        catalog = self._read_catalog()
        previous_payload = catalog["latest"].get(record_id)
        previous_record = None
        if previous_payload is not None:
            previous_record = self._materialize(record_id, previous_payload)
            if not isinstance(previous_record, DataFrameArtifactRecord):
                raise RecordError(
                    f"Record id {record_id!r} is already used by a non-dataframe record; choose a unique id."
                )
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
        record_inputs = tuple(inputs)
        effective_code_digest = code_digest or ""
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
                and previous_record.producer == producer
                and previous_record.inputs == record_inputs
                and previous_record.path.name == filename
            ):
                previous_record.verify_content_digest()
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
            )
            payload = record_to_dict(record, outputs_dir=self.root)
            catalog["latest"][record_id] = payload
            catalog["history"].setdefault(record_id, []).append(payload)
            self._write_catalog(catalog)
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
        inputs: list[ProvenanceInput],
        config_digest: str,
        files: list[Path],
        description: str,
        path_descriptions: tuple[PathDescription, ...] = (),
        source_recipe: RecipeSource | None = None,
    ) -> FileBundleRecord:
        record = FileBundleRecord(
            record_id=record_id,
            kind="file_bundle",
            producer=RecordProducer(
                kind=producer_kind,
                id=producer_id,
                plugin=producer_plugin,
                source_recipe=(
                    RecordRecipeSource(recipe=source_recipe.recipe, with_=dict(source_recipe.with_ or {}))
                    if source_recipe is not None
                    else None
                ),
            ),
            created_at=datetime.now(UTC).isoformat(),
            inputs=tuple(inputs),
            config_digest=config_digest,
            files=tuple(sorted(dict.fromkeys(files), key=str)),
            description=description,
            path_descriptions=path_descriptions,
        )
        materialized_paths = tuple(path if path.is_absolute() else self.root / path for path in record.files)
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
        self.ensure_layout()
        catalog = self._read_catalog()
        payload = record_to_dict(record, outputs_dir=self.root)
        catalog["latest"][record_id] = payload
        catalog["history"].setdefault(record_id, []).append(payload)
        self._write_catalog(catalog)
        return record
