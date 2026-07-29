from __future__ import annotations

from pathlib import Path

from reader.errors import RecordError
from reader.workbench.graph import RecordRef
from reader.workbench.records.model import (
    DataFrameArtifactRecord,
    FileBundleRecord,
    record_revision_digest,
    sha256_file,
)

from .identity import BuildIdentity, current_build_identity

_UNVERIFIABLE_CODES = frozenset({"build.identity_mismatch", "config.digest_mismatch"})


def _issue(*, code: str, field: str, reason: str, remediation: str, retryable: bool = False) -> dict[str, object]:
    return {
        "code": code,
        "field": field,
        "reason": reason,
        "remediation": remediation,
        "retryable": retryable,
    }


def _verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_digest: str,
    field: str,
    code_prefix: str = "artifact",
) -> list[dict[str, object]]:
    if not path.exists():
        return [
            _issue(
                code=f"{code_prefix}.missing",
                field=field,
                reason=f"Recorded artifact is missing: {path}",
                remediation="Regenerate the record from its source inputs.",
            )
        ]
    if not path.is_file():
        return [
            _issue(
                code=f"{code_prefix}.not_file",
                field=field,
                reason=f"Recorded artifact is not a regular file: {path}",
                remediation="Remove the conflicting path and regenerate the record.",
            )
        ]
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return [
            _issue(
                code=f"{code_prefix}.size_mismatch",
                field=field,
                reason=f"Recorded size {expected_size} does not match current size {actual_size}: {path}",
                remediation="Restore the recorded artifact or regenerate it from source inputs.",
            )
        ]
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        return [
            _issue(
                code=f"{code_prefix}.digest_mismatch",
                field=field,
                reason=f"Recorded digest {expected_digest} does not match {actual_digest}: {path}",
                remediation="Restore the recorded artifact or regenerate it from source inputs.",
            )
        ]
    return []


def _confine_input(path: Path, *, experiment_root: Path, field: str) -> tuple[Path | None, list[dict[str, object]]]:
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(experiment_root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return None, [
            _issue(
                code="input.outside_root",
                field=field,
                reason=f"Recorded input resolves outside the experiment root: {path}",
                remediation="Restore the confined source path and rerun the producing surface.",
            )
        ]
    return resolved, []


def verify_record_store(
    store,
    *,
    experiment_root: Path,
    expected_config_digest: str,
    expected_build_identity: BuildIdentity | None = None,
) -> dict[str, object]:
    current_build = expected_build_identity or current_build_identity()
    if not store.catalog_exists():
        return {
            "schema": "reader.verify/v1",
            "status": "failed",
            "summary": {"checked": 0, "failed": 1, "unverifiable": 0},
            "issues": [
                _issue(
                    code="catalog.missing",
                    field="outputs/manifests/records.json",
                    reason="The experiment has no records catalog.",
                    remediation="Run the experiment to produce schema-v5 records, then verify again.",
                )
            ],
            "records": [],
        }
    try:
        records = store.iter_latest_records()
    except RecordError as exc:
        return {
            "schema": "reader.verify/v1",
            "status": "failed",
            "summary": {"checked": 0, "failed": 1, "unverifiable": 0},
            "issues": [
                _issue(
                    code="catalog.invalid",
                    field="outputs/manifests/records.json",
                    reason=str(exc),
                    remediation="Restore a valid records catalog or rerun the experiment from source inputs.",
                )
            ],
            "records": [],
        }
    results: list[dict[str, object]] = []
    failed = 0
    unverifiable = 0
    for record in records:
        issues: list[dict[str, object]] = []
        schema_version = record.schema_version
        record_status = "ok"
        if record.config_digest != expected_config_digest:
            record_status = "unverifiable"
            issues.append(
                _issue(
                    code="config.digest_mismatch",
                    field="config_digest",
                    reason="The current normalized experiment config does not match the record.",
                    remediation="Restore the recorded config or rerun the affected surfaces.",
                )
            )
        if record.build_identity != current_build:
            record_status = "unverifiable"
            issues.append(
                _issue(
                    code="build.identity_mismatch",
                    field="build_identity",
                    reason="The current Reader build does not match the build that produced this record.",
                    remediation="Use the recorded build or rerun the affected surfaces with the current Reader build.",
                )
            )
        if (
            isinstance(record, DataFrameArtifactRecord)
            and record.build_identity is not None
            and record.code_digest != record.build_identity.source_digest
        ):
            issues.append(
                _issue(
                    code="build.code_digest_mismatch",
                    field="code_digest",
                    reason="The dataframe code digest does not match its recorded Reader build identity.",
                    remediation="Restore a valid catalog or rerun the producing surface.",
                )
            )
        for item in record.inputs:
            if isinstance(item.ref, RecordRef):
                upstream = store.latest_record(item.ref.record_id)
                if upstream is None:
                    issues.append(
                        _issue(
                            code="input.record_missing",
                            field=f"inputs.{item.label}",
                            reason=f"Upstream record {item.ref.record_id!r} is missing.",
                            remediation="Reproduce the upstream record, then rerun this producer.",
                        )
                    )
                    continue
                actual_revision = record_revision_digest(upstream, outputs_dir=store.root)
                if actual_revision != item.record_revision_digest:
                    issues.append(
                        _issue(
                            code="input.record_revision_mismatch",
                            field=f"inputs.{item.label}",
                            reason=f"Upstream record {item.ref.record_id!r} no longer matches the consumed revision.",
                            remediation="Rerun this producer against the current upstream record.",
                        )
                    )
                continue
            if item.artifact is None:
                issues.append(
                    _issue(
                        code="input.evidence_missing",
                        field=f"inputs.{item.label}",
                        reason="The file input has no artifact evidence.",
                        remediation="Rerun the producing surface to emit schema-v5 evidence.",
                    )
                )
                continue
            source_path = experiment_root.resolve(strict=False) / item.artifact.relative_path
            confined_source, confinement_issues = _confine_input(
                source_path,
                experiment_root=experiment_root,
                field=f"inputs.{item.label}",
            )
            issues.extend(confinement_issues)
            if confined_source is None:
                continue
            issues.extend(
                _verify_file(
                    confined_source,
                    expected_size=item.artifact.size_bytes,
                    expected_digest=item.artifact.content_digest,
                    field=f"inputs.{item.label}",
                    code_prefix="input",
                )
            )

        if isinstance(record, DataFrameArtifactRecord):
            issues.extend(
                _verify_file(
                    record.path,
                    expected_size=record.size_bytes or -1,
                    expected_digest=record.content_digest,
                    field="path",
                )
            )
        elif isinstance(record, FileBundleRecord):
            evidence_by_path = {item.relative_path: item for item in record.file_evidence}
            for path in record.files:
                relative = path.resolve(strict=False).relative_to(store.root.resolve(strict=False))
                evidence = evidence_by_path.get(relative)
                if evidence is None:
                    issues.append(
                        _issue(
                            code="artifact.evidence_missing",
                            field="file_evidence",
                            reason=f"No file evidence is recorded for {relative}.",
                            remediation="Rerun the producing surface to emit schema-v5 evidence.",
                        )
                    )
                    continue
                issues.extend(
                    _verify_file(
                        path,
                        expected_size=evidence.size_bytes,
                        expected_digest=evidence.content_digest,
                        field=f"files.{relative.as_posix()}",
                    )
                )
        if issues and any(item["code"] not in _UNVERIFIABLE_CODES for item in issues):
            record_status = "failed"
        if record_status == "failed":
            failed += 1
        elif record_status == "unverifiable":
            unverifiable += 1
        results.append(
            {
                "record_id": record.record_id,
                "kind": record.kind,
                "schema_version": schema_version,
                "status": record_status,
                "issues": issues,
            }
        )
    status = "failed" if failed else ("unverifiable" if unverifiable or not records else "ok")
    return {
        "schema": "reader.verify/v1",
        "status": status,
        "summary": {
            "checked": len(records),
            "failed": failed,
            "unverifiable": unverifiable,
        },
        "issues": (
            [
                _issue(
                    code="catalog.empty",
                    field="outputs/manifests/records.json",
                    reason="The records catalog contains no current records.",
                    remediation="Run at least one producing surface, then verify again.",
                )
            ]
            if not records
            else []
        ),
        "records": results,
    }
