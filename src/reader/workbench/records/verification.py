from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from reader.errors import RecordError
from reader.workbench.graph import RecordRef, SourceRecordRef
from reader.workbench.records.model import (
    DataFrameArtifactRecord,
    FileBundleRecord,
    record_revision_digest,
    sha256_file,
    verify_record_artifact_integrity,
)

from .identity import BuildIdentity, current_build_identity

_UNVERIFIABLE_CODES = frozenset({"build.identity_mismatch", "config.digest_mismatch"})


@dataclass(frozen=True)
class RecordVerificationScope:
    """Current record identities owned by one compiled workbench declaration."""

    record_ids: frozenset[str] = field(default_factory=frozenset)
    notebook_templates: frozenset[str] = field(default_factory=frozenset)

    def includes(self, record: DataFrameArtifactRecord | FileBundleRecord) -> bool:
        if record.record_id in self.record_ids:
            return True
        return record.producer.kind == "notebook" and record.producer.template in self.notebook_templates


def _issue(*, code: str, field: str, reason: str, remediation: str, retryable: bool = False) -> dict[str, object]:
    return {
        "code": code,
        "field": field,
        "reason": reason,
        "remediation": remediation,
        "retryable": retryable,
    }


def _file_io_issue(*, path: Path, field: str, code_prefix: str, error: OSError) -> dict[str, object]:
    return _issue(
        code=f"{code_prefix}.io_error",
        field=field,
        reason=f"Recorded artifact could not be read: {path}: {error}",
        remediation="Restore a readable artifact or regenerate it from source inputs.",
    )


def _verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_digest: str,
    field: str,
    code_prefix: str = "artifact",
) -> list[dict[str, object]]:
    try:
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
    except OSError as exc:
        return [_file_io_issue(path=path, field=field, code_prefix=code_prefix, error=exc)]
    if actual_size != expected_size:
        return [
            _issue(
                code=f"{code_prefix}.size_mismatch",
                field=field,
                reason=f"Recorded size {expected_size} does not match current size {actual_size}: {path}",
                remediation="Restore the recorded artifact or regenerate it from source inputs.",
            )
        ]
    try:
        actual_digest = sha256_file(path)
    except OSError as exc:
        return [_file_io_issue(path=path, field=field, code_prefix=code_prefix, error=exc)]
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


_INVOCATION_EVENT_FIELDS = frozenset(
    {
        "schema",
        "invocation_id",
        "timestamp",
        "config_digest",
        "build_identity",
        "operation",
        "selected_step_ids",
        "declared_inputs",
        "event",
        "status",
        "exit_status",
        "produced_record_revisions",
        "failure",
    }
)
_INVOCATION_IDENTITY_FIELDS = (
    "config_digest",
    "build_identity",
    "operation",
    "selected_step_ids",
    "declared_inputs",
)


def _valid_invocation_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


def _valid_invocation_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(parsed)


def _valid_invocation_digest(value: object) -> bool:
    """Match the normalization contract used by InvocationLedger."""

    return isinstance(value, str) and value.startswith("sha256:")


def _valid_selected_steps(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"pipeline", "plots", "exports"}:
        return False
    return all(
        isinstance(value[phase], list) and all(isinstance(step_id, str) and bool(step_id) for step_id in value[phase])
        for phase in ("pipeline", "plots", "exports")
    )


def _valid_declared_inputs(value: object) -> bool:
    if not isinstance(value, list) or len(value) > 4096:
        return False
    allowed_refs = ({"record"}, {"file"}, {"resource", "path"}, {"record_collection"})
    for item in value:
        if not isinstance(item, dict) or set(item) != {"phase", "step_id", "port", "ref"}:
            return False
        if item["phase"] not in {"pipeline", "plots", "exports"}:
            return False
        if not isinstance(item["step_id"], str) or not item["step_id"]:
            return False
        if not isinstance(item["port"], str) or not item["port"]:
            return False
        ref = item["ref"]
        if not isinstance(ref, dict) or set(ref) not in allowed_refs:
            return False
        if any(not isinstance(part, str) or not part for part in ref.values()):
            return False
    return True


def _valid_revision(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"record_id", "revision", "revision_digest"}
        and isinstance(value["record_id"], str)
        and bool(value["record_id"])
        and type(value["revision"]) is int
        and value["revision"] >= 1
        and _valid_invocation_digest(value["revision_digest"])
    )


def _invocation_event_problem(event: dict[str, object]) -> str | None:
    if set(event) != _INVOCATION_EVENT_FIELDS:
        return "The invocation event fields do not match reader.invocation/v1."
    if event["schema"] != "reader.invocation/v1":
        return "The invocation event schema is not reader.invocation/v1."
    if not _valid_invocation_id(event["invocation_id"]):
        return "The invocation id is not a canonical UUID4."
    if not _valid_invocation_timestamp(event["timestamp"]):
        return "The invocation timestamp is not an ISO-8601 UTC timestamp."
    if not _valid_invocation_digest(event["config_digest"]):
        return "The invocation config_digest is not a sha256 digest."
    try:
        BuildIdentity.from_dict(event["build_identity"])
    except RecordError:
        return "The invocation build_identity is invalid."
    if event["operation"] not in {"run", "plot", "export", "mixed"}:
        return "The invocation operation is invalid."
    if not _valid_selected_steps(event["selected_step_ids"]):
        return "The invocation selected_step_ids are invalid."
    if not _valid_declared_inputs(event["declared_inputs"]):
        return "The invocation declared_inputs are invalid."
    if event["event"] not in {"attempt", "result"}:
        return "The invocation event kind is invalid."

    revisions = event["produced_record_revisions"]
    if not isinstance(revisions, list) or any(not _valid_revision(item) for item in revisions):
        return "The invocation produced_record_revisions are invalid."
    if revisions != sorted(revisions, key=lambda item: (item["record_id"], item["revision"])):
        return "The invocation produced_record_revisions are not in canonical order."
    if event["event"] == "attempt":
        if (
            event["status"] != "attempted"
            or event["exit_status"] is not None
            or revisions
            or event["failure"] is not None
        ):
            return "The invocation attempt status, exit_status, revisions, and failure are inconsistent."
        return None

    exit_status = event["exit_status"]
    failure = event["failure"]
    if type(exit_status) is not int or exit_status < 0:
        return "The invocation result exit_status is invalid."
    if exit_status == 0:
        if event["status"] != "succeeded" or failure is not None:
            return "The successful invocation result has inconsistent status or failure details."
        return None
    if event["status"] != "failed":
        return "The failed invocation result has an inconsistent status."
    if (
        not isinstance(failure, dict)
        or set(failure) != {"type", "reason"}
        or any(not isinstance(value, str) or not value for value in failure.values())
        or len(failure["reason"]) > 500
    ):
        return "The failed invocation result has invalid failure details."
    return None


def _verify_invocation_ledger(path: Path, *, store=None) -> tuple[list[dict[str, object]], int]:
    if not path.exists():
        return [], 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [
            _issue(
                code="invocation.ledger_unreadable",
                field="outputs/manifests/invocations.jsonl",
                reason=f"The invocation ledger could not be read: {type(exc).__name__}.",
                remediation="Restore a readable ledger before handing off the experiment.",
            )
        ], 0

    issues: list[dict[str, object]] = []
    events_by_invocation: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            issues.append(
                _issue(
                    code="invocation.ledger_invalid",
                    field=f"outputs/manifests/invocations.jsonl:{line_number}",
                    reason="The invocation ledger contains a blank event line.",
                    remediation="Restore the ledger from a known-good copy before handoff.",
                )
            )
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            issues.append(
                _issue(
                    code="invocation.ledger_invalid",
                    field=f"outputs/manifests/invocations.jsonl:{line_number}",
                    reason="The invocation ledger contains an invalid JSON event.",
                    remediation="Restore the ledger from a known-good copy before handoff.",
                )
            )
            continue
        if not isinstance(event, dict):
            issues.append(
                _issue(
                    code="invocation.event_invalid",
                    field=f"outputs/manifests/invocations.jsonl:{line_number}",
                    reason="The invocation event does not match reader.invocation/v1.",
                    remediation="Restore a valid invocation event before handoff.",
                )
            )
            continue
        problem = _invocation_event_problem(event)
        if problem is not None:
            issues.append(
                _issue(
                    code="invocation.event_invalid",
                    field=f"outputs/manifests/invocations.jsonl:{line_number}",
                    reason=problem,
                    remediation="Restore a valid invocation event before handoff.",
                )
            )
        invocation_id = event.get("invocation_id")
        if not isinstance(invocation_id, str) or not invocation_id:
            continue
        events_by_invocation.setdefault(invocation_id, []).append((line_number, event))

    for invocation_id, positioned_events in sorted(events_by_invocation.items()):
        attempts = [(line, event) for line, event in positioned_events if event.get("event") == "attempt"]
        results = [(line, event) for line, event in positioned_events if event.get("event") == "result"]
        unknown = [event for _, event in positioned_events if event.get("event") not in {"attempt", "result"}]
        if unknown or len(attempts) != 1:
            issues.append(
                _issue(
                    code="invocation.lifecycle_invalid",
                    field=f"invocations.{invocation_id}",
                    reason="The invocation must contain exactly one attempt and only known event kinds.",
                    remediation="Restore the invocation ledger from a known-good copy before handoff.",
                )
            )
        if len(results) == 0 and len(attempts) == 1:
            issues.append(
                _issue(
                    code="invocation.finalization_unconfirmed",
                    field=f"invocations.{invocation_id}",
                    reason="The invocation has no terminal result; its final status is unconfirmed.",
                    remediation="Retain the recorded evidence and resolve the incomplete invocation before handoff.",
                )
            )
        elif len(results) > 1:
            issues.append(
                _issue(
                    code="invocation.terminal_conflict",
                    field=f"invocations.{invocation_id}",
                    reason="The invocation has more than one terminal result.",
                    remediation="Restore an unambiguous invocation ledger before handoff.",
                )
            )
        elif len(results) == 1 and not attempts:
            issues.append(
                _issue(
                    code="invocation.orphan_result",
                    field=f"invocations.{invocation_id}",
                    reason="The invocation result has no matching attempt.",
                    remediation="Restore the missing attempt or a known-good ledger before handoff.",
                )
            )
        if len(attempts) != 1 or len(results) != 1:
            continue
        attempt_line, attempt = attempts[0]
        result_line, result = results[0]
        if result_line < attempt_line:
            issues.append(
                _issue(
                    code="invocation.order_invalid",
                    field=f"invocations.{invocation_id}",
                    reason="The invocation result appears before its attempt.",
                    remediation="Restore the invocation ledger in append order before handoff.",
                )
            )
        changed_fields = [field for field in _INVOCATION_IDENTITY_FIELDS if result.get(field) != attempt.get(field)]
        if changed_fields:
            issues.append(
                _issue(
                    code="invocation.identity_mismatch",
                    field=f"invocations.{invocation_id}",
                    reason="The invocation result changed immutable attempt fields: " + ", ".join(changed_fields) + ".",
                    remediation="Restore a result written from the matching invocation attempt.",
                )
            )
        if store is None or not isinstance(result.get("produced_record_revisions"), list):
            continue
        for revision in result["produced_record_revisions"]:
            if not _valid_revision(revision):
                continue
            try:
                history = store.record_history(revision["record_id"])
            except RecordError:
                continue
            revision_number = revision["revision"]
            if revision_number > len(history):
                issues.append(
                    _issue(
                        code="invocation.revision_missing",
                        field=f"invocations.{invocation_id}.produced_record_revisions",
                        reason=(f"Record {revision['record_id']!r} has no catalog history revision {revision_number}."),
                        remediation="Restore the matching records catalog history or invocation ledger.",
                    )
                )
                continue
            actual_digest = record_revision_digest(history[revision_number - 1], outputs_dir=store.root)
            if actual_digest != revision["revision_digest"]:
                issues.append(
                    _issue(
                        code="invocation.revision_mismatch",
                        field=f"invocations.{invocation_id}.produced_record_revisions",
                        reason=(
                            f"Record {revision['record_id']!r} revision {revision_number} does not match "
                            "the invocation revision digest."
                        ),
                        remediation="Restore the matching records catalog history or invocation ledger.",
                    )
                )
    return issues, len(events_by_invocation)


def verify_record_store(
    store,
    *,
    experiment_root: Path,
    expected_config_digest: str,
    expected_build_identity: BuildIdentity | None = None,
    scope: RecordVerificationScope | None = None,
) -> dict[str, object]:
    current_build = expected_build_identity or current_build_identity()
    invocation_issues, invocations_checked = _verify_invocation_ledger(
        store.root / "manifests" / "invocations.jsonl",
        store=store,
    )
    if not store.catalog_exists():
        return {
            "schema": "reader.verify/v1",
            "status": "failed",
            "summary": {
                "checked": 0,
                "failed": 1,
                "unverifiable": 0,
                "invocations_checked": invocations_checked,
                "invocation_failures": len(invocation_issues),
            },
            "issues": [
                _issue(
                    code="catalog.missing",
                    field="outputs/manifests/records.json",
                    reason="The experiment has no records catalog.",
                    remediation="Run the experiment to produce schema-v5 records, then verify again.",
                ),
                *invocation_issues,
            ],
            "records": [],
        }
    try:
        records = store.iter_latest_records()
        if scope is not None:
            records = tuple(record for record in records if scope.includes(record))
    except RecordError as exc:
        return {
            "schema": "reader.verify/v1",
            "status": "failed",
            "summary": {
                "checked": 0,
                "failed": 1,
                "unverifiable": 0,
                "invocations_checked": invocations_checked,
                "invocation_failures": len(invocation_issues),
            },
            "issues": [
                _issue(
                    code="catalog.invalid",
                    field="outputs/manifests/records.json",
                    reason=str(exc),
                    remediation="Restore a valid records catalog or rerun the experiment from source inputs.",
                ),
                *invocation_issues,
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
                try:
                    verify_record_artifact_integrity(upstream, outputs_dir=store.root)
                except RecordError as exc:
                    issues.append(
                        _issue(
                            code="input.record_artifact_invalid",
                            field=f"inputs.{item.label}",
                            reason=str(exc),
                            remediation=(
                                "Restore the upstream artifact that matches its catalog revision, "
                                "or reproduce the upstream record and rerun this producer."
                            ),
                        )
                    )
                continue
            if isinstance(item.ref, SourceRecordRef):
                from reader.workbench.records.sources import resolve_source_record  # noqa: PLC0415

                try:
                    upstream = resolve_source_record(item.ref, contracts=store.contracts)
                except RecordError as exc:
                    issues.append(
                        _issue(
                            code="input.source_record_missing",
                            field=f"inputs.{item.label}",
                            reason=str(exc),
                            remediation="Restore or reproduce the source experiment record, then rerun this producer.",
                        )
                    )
                    continue
                if upstream.revision_digest != item.record_revision_digest:
                    issues.append(
                        _issue(
                            code="input.source_record_revision_mismatch",
                            field=f"inputs.{item.label}",
                            reason=(
                                f"Source record {item.ref.experiment_id}:{item.ref.record_id} no longer "
                                "matches the consumed revision."
                            ),
                            remediation="Rerun this producer against the current source record revisions.",
                        )
                    )
                try:
                    upstream.verify_artifact_integrity()
                except RecordError as exc:
                    issues.append(
                        _issue(
                            code="input.source_record_artifact_invalid",
                            field=f"inputs.{item.label}",
                            reason=str(exc),
                            remediation=(
                                "Restore the source artifact that matches its catalog revision, "
                                "or reproduce the source record and rerun this producer."
                            ),
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
    status = "failed" if failed or invocation_issues else ("unverifiable" if unverifiable or not records else "ok")
    catalog_issues = (
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
    )
    return {
        "schema": "reader.verify/v1",
        "status": status,
        "summary": {
            "checked": len(records),
            "failed": failed,
            "unverifiable": unverifiable,
            "invocations_checked": invocations_checked,
            "invocation_failures": len(invocation_issues),
        },
        "issues": [*catalog_issues, *invocation_issues],
        "records": results,
    }
