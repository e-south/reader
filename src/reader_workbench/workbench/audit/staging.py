from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from reader_workbench.errors import ConfigError, RecordError
from reader_workbench.runtime import ReaderRuntime
from reader_workbench.workbench.experiment import ResourceCatalog
from reader_workbench.workbench.experiments import find_experiments_root
from reader_workbench.workbench.graph import SourceRecordRef
from reader_workbench.workbench.records import (
    record_paths,
    record_to_dict,
    verify_record_artifact_integrity,
)
from reader_workbench.workbench.records.store import RECORD_CATALOG_SCHEMA_VERSION


def stage_experiment(source_dir: Path, target_dir: Path) -> Path:
    shutil.copytree(
        source_dir,
        target_dir,
        ignore=shutil.ignore_patterns("outputs", "__pycache__", ".DS_Store"),
        symlinks=True,
    )
    _retarget_internal_symlinks(source_dir=source_dir, target_dir=target_dir)
    return target_dir / "config.yaml"


def stage_audit_workspace(
    *,
    config_path: Path,
    target_root: Path,
    resources: ResourceCatalog,
    runtime: ReaderRuntime,
) -> Path:
    """Stage one writable experiment and exact read-only snapshots of its record resources."""

    record_resources = [resource for resource in resources.by_id.values() if resource.kind == "record"]
    try:
        source_experiments_root = find_experiments_root(config_path.parent)
    except ConfigError:
        if record_resources:
            raise
        source_experiments_root = None
    staged_experiments_root = target_root / "experiments"
    relative_experiment_dir = (
        config_path.parent.resolve().relative_to(source_experiments_root)
        if source_experiments_root is not None
        else Path(config_path.parent.name)
    )
    staged_config = stage_experiment(
        config_path.parent,
        staged_experiments_root / relative_experiment_dir,
    )

    resources_by_provider: dict[tuple[Path, Path], set[str]] = {}
    for resource in record_resources:
        provider_key = (resource.experiment_root.resolve(), resource.outputs_dir.resolve())
        resources_by_provider.setdefault(provider_key, set()).add(resource.record_id)
    if resources_by_provider and source_experiments_root is None:
        raise ConfigError("Cross-experiment record resources require a canonical experiments/ workspace")
    for (provider_root, provider_outputs), record_ids in sorted(
        resources_by_provider.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1])),
    ):
        _stage_record_provider(
            provider_root=provider_root,
            provider_outputs=provider_outputs,
            record_ids=record_ids,
            source_experiments_root=source_experiments_root,
            staged_experiments_root=staged_experiments_root,
            staged_target_dir=staged_config.parent,
            runtime=runtime,
        )
    return staged_config


def _stage_record_provider(
    *,
    provider_root: Path,
    provider_outputs: Path,
    record_ids: set[str],
    source_experiments_root: Path,
    staged_experiments_root: Path,
    staged_target_dir: Path,
    runtime: ReaderRuntime,
) -> None:
    staged_provider_dir = _stage_catalog_config(
        source_root=provider_root,
        source_experiments_root=source_experiments_root,
        staged_experiments_root=staged_experiments_root,
        staged_target_dir=staged_target_dir,
    )
    relative_outputs = provider_outputs.relative_to(provider_root)
    staged_outputs = staged_provider_dir / relative_outputs
    source_store = runtime.record_store(
        provider_outputs,
        experiment_root=provider_root,
        create=False,
    )
    catalog = {
        "schema_version": RECORD_CATALOG_SCHEMA_VERSION,
        "provenance_epoch_id": source_store.provenance_epoch_id(),
        "latest": {},
        "history": {},
    }
    source_records = {record.record_id: record for record in source_store.iter_latest_records()}
    copied_paths: set[Path] = set()
    for record_id in sorted(record_ids):
        record = source_records.get(record_id)
        if record is None:
            raise RecordError(f"Source record {record_id!r} is missing from {provider_root.name!r}")
        verify_record_artifact_integrity(record, outputs_dir=provider_outputs)
        payload = record_to_dict(record, outputs_dir=provider_outputs)
        catalog["latest"][record_id] = payload
        catalog["history"][record_id] = [payload]
        for source_path in record_paths(record):
            relative_path = source_path.resolve(strict=True).relative_to(provider_outputs)
            if relative_path in copied_paths:
                continue
            copied_paths.add(relative_path)
            staged_path = staged_outputs / relative_path
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, staged_path)
        for evidence in record.inputs:
            if not isinstance(evidence.ref, SourceRecordRef):
                continue
            _stage_catalog_config(
                source_root=evidence.ref.experiment_root,
                source_experiments_root=source_experiments_root,
                staged_experiments_root=staged_experiments_root,
                staged_target_dir=staged_target_dir,
            )

    staged_manifests = staged_outputs / "manifests"
    staged_manifests.mkdir(parents=True, exist_ok=True)
    (staged_manifests / "records.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stage_catalog_config(
    *,
    source_root: Path,
    source_experiments_root: Path,
    staged_experiments_root: Path,
    staged_target_dir: Path,
) -> Path:
    staged_dir = staged_experiments_root / source_root.resolve().relative_to(source_experiments_root)
    if staged_dir == staged_target_dir:
        return staged_dir
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_config = staged_dir / "config.yaml"
    if not staged_config.exists():
        shutil.copy2(source_root / "config.yaml", staged_config)
    return staged_dir


def _retarget_internal_symlinks(*, source_dir: Path, target_dir: Path) -> None:
    """Map confined source symlinks into the staged tree without following links while copying."""

    source_root = source_dir.resolve(strict=True)
    target_root = target_dir.resolve(strict=True)
    for current, dirnames, filenames in os.walk(target_root, followlinks=False):
        current_path = Path(current)
        for name in [*dirnames, *filenames]:
            staged_link = current_path / name
            if not staged_link.is_symlink():
                continue
            relative_link = staged_link.relative_to(target_root)
            source_link = source_root / relative_link
            try:
                source_target = source_link.resolve(strict=True)
                relative_target = source_target.relative_to(source_root)
            except (OSError, RuntimeError, ValueError):
                # External, dangling, or cyclic links remain links. Staged
                # validation can then reject them without copytree traversing them.
                continue
            staged_target = target_root / relative_target
            link_target = os.path.relpath(staged_target, start=staged_link.parent)
            target_is_directory = source_target.is_dir()
            staged_link.unlink()
            staged_link.symlink_to(link_target, target_is_directory=target_is_directory)
