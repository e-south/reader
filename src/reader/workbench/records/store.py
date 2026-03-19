from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from reader.contracts import ContractCatalog, ContractId
from reader.errors import RecordError
from reader.workbench.graph import ProvenanceInput, RecipeSource
from reader.workbench.ontology import WorkbenchProducerKind, WorkbenchRecordKind
from reader.workbench.records.model import (
    DataFrameArtifactRecord,
    FileBundleRecord,
    RecordProducer,
    RecordRecipeSource,
    record_from_dict,
    record_to_dict,
)


def _sha256_bytes(blob: bytes) -> str:
    return "sha256:" + hashlib.sha256(blob).hexdigest()


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
        self.records_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

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
        validate_contract: bool = True,
        producer_kind: WorkbenchProducerKind = "pipeline",
        source_recipe: RecipeSource | None = None,
    ) -> DataFrameArtifactRecord:
        self.ensure_layout()
        base_name = f"{producer_id}.{producer_plugin.replace('/', '_')}"
        step_dir = self.artifacts_dir / base_name
        catalog = self._read_catalog()
        previous_payload = catalog["latest"].get(record_id)
        previous_record = None
        if previous_payload is not None:
            previous_record = self._materialize(record_id, previous_payload)
            if not isinstance(previous_record, DataFrameArtifactRecord):
                raise RecordError(
                    f"Record id {record_id!r} is already used by a non-dataframe record; choose a unique id."
                )
            step_dir = (
                previous_record.path.parent
                if previous_record.config_digest == config_digest
                else self._revision_dir(step_dir)
            )
        else:
            step_dir = self._revision_dir(step_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
        data_path = step_dir / f"{out_name}.parquet"
        if validate_contract and contract_id != "none":
            self.contracts.validate(df, contract_id=contract_id, where=f"{producer_id}:{out_name}")
        df.to_parquet(data_path, index=False)
        record = DataFrameArtifactRecord(
            record_id=record_id,
            kind="dataframe_artifact",
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
            contract_id=contract_id,
            path=data_path,
            content_digest=_sha256_bytes(data_path.read_bytes()),
            code_digest=code_digest or "",
        )
        payload = record_to_dict(record, outputs_dir=self.root)
        catalog["latest"][record_id] = payload
        catalog["history"].setdefault(record_id, []).append(payload)
        self._write_catalog(catalog)
        return record

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
        source_recipe: RecipeSource | None = None,
    ) -> FileBundleRecord:
        self.ensure_layout()
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
        )
        catalog = self._read_catalog()
        payload = record_to_dict(record, outputs_dir=self.root)
        catalog["latest"][record_id] = payload
        catalog["history"].setdefault(record_id, []).append(payload)
        self._write_catalog(catalog)
        return record
