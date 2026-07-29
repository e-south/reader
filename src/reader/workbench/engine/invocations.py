from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from reader.errors import ConfigError, ExecutionError
from reader.workbench.graph import FileRef, RecordRef, ResourceRef
from reader.workbench.records import record_revision_digest
from reader.workbench.records.identity import BuildIdentity

INVOCATION_SCHEMA = "reader.invocation/v1"
_FAILURE_REASON_LIMIT = 500
_DECLARED_INPUT_LIMIT = 4096
_SECRET_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    (?<![a-z0-9])(?P<quote>["']?)
    (?P<key>
        (?:[a-z0-9]+[._-])*
        (?:
            api[._-]?key
            | secret[._-]?access[._-]?key
            | access[._-]?key
            | auth[._-]?token
            | access[._-]?token
            | refresh[._-]?token
            | credentials?
            | signature
            | password
            | secret
            | token
        )
        (?:[._-][a-z0-9]+)*
    )
    (?P=quote)\s*[:=]\s*
    (?:"[^"]*"|'[^']*'|[^\s,;&]+)
    """
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)\bauthorization\s*:\s*(?P<scheme>[a-z][a-z0-9_-]*)\s+"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_URI_USERINFO_RE = re.compile(r"(?i)\b(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@")


@dataclass(frozen=True)
class InvocationAttempt:
    invocation_id: str
    config_digest: str
    build_identity: BuildIdentity
    operation: str
    selected_step_ids: dict[str, list[str]]
    declared_inputs: list[dict[str, Any]]


@dataclass(frozen=True)
class SelectedSteps:
    pipeline: tuple[str, ...]
    plots: tuple[str, ...]
    exports: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, list[str]]) -> SelectedSteps:
        normalized = _normalize_selected_step_ids(value)
        return cls(
            pipeline=tuple(normalized["pipeline"]),
            plots=tuple(normalized["plots"]),
            exports=tuple(normalized["exports"]),
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "pipeline": list(self.pipeline),
            "plots": list(self.plots),
            "exports": list(self.exports),
        }


@dataclass(frozen=True)
class ProducedRecordRevision:
    record_id: str
    revision: int
    revision_digest: str

    @classmethod
    def from_payload(cls, value: dict[str, Any]) -> ProducedRecordRevision:
        normalized = _normalize_revisions([value])[0]
        return cls(
            record_id=normalized["record_id"],
            revision=normalized["revision"],
            revision_digest=normalized["revision_digest"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "revision": self.revision,
            "revision_digest": self.revision_digest,
        }


@dataclass(frozen=True)
class ExecutionResult:
    invocation_id: str | None
    operation: Literal["run", "plot", "export", "mixed"]
    status: Literal["planned", "succeeded"]
    dry_run: bool
    selected_steps: SelectedSteps
    produced_record_revisions: tuple[ProducedRecordRevision, ...]
    ledger_path: Path | None


class InvocationLedger:
    def __init__(self, *, experiment_root: Path, outputs_dir: Path) -> None:
        self.experiment_root = Path(experiment_root).resolve(strict=False)
        self.outputs_dir = Path(outputs_dir).resolve(strict=False)
        if not self.outputs_dir.is_relative_to(self.experiment_root):
            raise ConfigError(f"Invocation outputs directory must stay under the experiment root: {self.outputs_dir}")
        self.path = self.outputs_dir / "manifests" / "invocations.jsonl"

    def append_attempt(
        self,
        *,
        config_digest: str,
        build_identity: BuildIdentity,
        operation: str,
        selected_step_ids: dict[str, list[str]],
        declared_inputs: list[dict[str, Any]],
    ) -> InvocationAttempt:
        selected = _normalize_selected_step_ids(selected_step_ids)
        attempt = InvocationAttempt(
            invocation_id=str(uuid4()),
            config_digest=_require_digest(config_digest),
            build_identity=build_identity,
            operation=_normalize_operation(operation),
            selected_step_ids=selected,
            declared_inputs=_normalize_declared_inputs(declared_inputs),
        )
        self._append(
            self._base_event(attempt)
            | {
                "event": "attempt",
                "status": "attempted",
                "exit_status": None,
                "produced_record_revisions": [],
                "failure": None,
            }
        )
        return attempt

    def append_result(
        self,
        attempt: InvocationAttempt,
        *,
        exit_status: int,
        produced_record_revisions: list[dict[str, Any]],
        failure: BaseException | None = None,
    ) -> None:
        if not isinstance(exit_status, int) or exit_status < 0:
            raise ExecutionError("Invocation exit_status must be a non-negative integer")
        if exit_status == 0 and failure is not None:
            raise ExecutionError("A successful invocation result must not include failure details")
        if exit_status != 0 and failure is None:
            raise ExecutionError("A failed invocation result must include failure details")
        revisions = _normalize_revisions(produced_record_revisions)
        self._append(
            self._base_event(attempt)
            | {
                "event": "result",
                "status": "succeeded" if exit_status == 0 else "failed",
                "exit_status": exit_status,
                "produced_record_revisions": revisions,
                "failure": self._failure_payload(failure) if failure is not None else None,
            }
        )

    def _base_event(self, attempt: InvocationAttempt) -> dict[str, Any]:
        return {
            "schema": INVOCATION_SCHEMA,
            "invocation_id": attempt.invocation_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "config_digest": attempt.config_digest,
            "build_identity": attempt.build_identity.to_dict(),
            "operation": attempt.operation,
            "selected_step_ids": attempt.selected_step_ids,
            "declared_inputs": attempt.declared_inputs,
        }

    def _failure_payload(self, failure: BaseException) -> dict[str, str]:
        reason = " ".join(str(failure).split()) or type(failure).__name__
        for sensitive_path in (self.experiment_root, Path.home().resolve(strict=False)):
            reason = reason.replace(str(sensitive_path), "<redacted-path>")
        reason = _URI_USERINFO_RE.sub(lambda match: f"{match.group('scheme')}<redacted>@", reason)
        reason = _AUTHORIZATION_RE.sub(lambda match: f"Authorization: {match.group('scheme')} <redacted>", reason)
        reason = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group('key')}=<redacted>", reason)
        reason = _BEARER_RE.sub("Bearer <redacted>", reason)
        if len(reason) > _FAILURE_REASON_LIMIT:
            reason = reason[: _FAILURE_REASON_LIMIT - 1] + "…"
        return {"type": type(failure).__name__, "reason": reason}

    def _append(self, event: dict[str, Any]) -> None:
        manifests_dir = self.path.parent
        manifests_dir.mkdir(parents=True, exist_ok=True)
        resolved_parent = manifests_dir.resolve(strict=True)
        if not resolved_parent.is_relative_to(self.experiment_root):
            raise ConfigError(f"Invocation manifest directory must stay under the experiment root: {resolved_parent}")
        if self.path.is_symlink():
            raise ConfigError(f"Invocation ledger must not be a symlink: {self.path}")

        payload = (json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o644)
            try:
                written = os.write(descriptor, payload)
                if written != len(payload):
                    raise ExecutionError(
                        f"Invocation ledger append was incomplete: wrote {written} of {len(payload)} bytes"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise ExecutionError(f"Could not append invocation event under outputs/manifests: {exc}") from exc


def capture_revision_snapshot(store: Any) -> dict[str, dict[str, Any]]:
    if not store.catalog_exists():
        return {}
    records = store.iter_latest_records()
    revision_counts = store.revision_counts(record.record_id for record in records)
    return {
        record.record_id: {
            "record_id": record.record_id,
            "revision": revision_counts[record.record_id],
            "revision_digest": record_revision_digest(record, outputs_dir=store.root),
        }
        for record in records
    }


def declared_input_projection(
    *,
    steps_by_phase: dict[str, list[Any]],
    experiment_root: Path,
) -> list[dict[str, Any]]:
    expected_phases = ("pipeline", "plots", "exports")
    if set(steps_by_phase) != set(expected_phases):
        raise ExecutionError(f"Invocation steps_by_phase must contain exactly {list(expected_phases)}")
    root = Path(experiment_root).resolve(strict=False)
    declared: list[dict[str, Any]] = []
    for phase in expected_phases:
        for step in steps_by_phase[phase]:
            for port, ref in sorted((step.reads or {}).items()):
                declared.append(
                    {
                        "phase": phase,
                        "step_id": step.id,
                        "port": port,
                        "ref": _declared_ref_payload(ref, experiment_root=root),
                    }
                )
    return _normalize_declared_inputs(declared)


def produced_record_revisions(
    *,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    changed = [revision for record_id, revision in after.items() if before.get(record_id) != revision]
    return _normalize_revisions(changed)


def _require_digest(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ExecutionError("Invocation config_digest must be a sha256 digest")
    return value


def _normalize_operation(value: str) -> str:
    allowed = {"run", "plot", "export", "mixed"}
    if value not in allowed:
        raise ExecutionError(f"Invocation operation must be one of {sorted(allowed)}")
    return value


def _normalize_selected_step_ids(value: dict[str, list[str]]) -> dict[str, list[str]]:
    expected = ("pipeline", "plots", "exports")
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ExecutionError(f"Invocation selected_step_ids must contain exactly {list(expected)}")
    normalized: dict[str, list[str]] = {}
    for phase in expected:
        step_ids = value[phase]
        if not isinstance(step_ids, list) or any(not isinstance(step_id, str) or not step_id for step_id in step_ids):
            raise ExecutionError(f"Invocation selected_step_ids.{phase} must be a list of non-empty strings")
        normalized[phase] = list(step_ids)
    return normalized


def _normalize_declared_inputs(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > _DECLARED_INPUT_LIMIT:
        raise ExecutionError(f"Invocation declared_inputs must be a list of at most {_DECLARED_INPUT_LIMIT} items")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"phase", "step_id", "port", "ref"}:
            raise ExecutionError("Invocation declared input must contain exactly phase, step_id, port, and ref")
        phase = item["phase"]
        step_id = item["step_id"]
        port = item["port"]
        if phase not in {"pipeline", "plots", "exports"}:
            raise ExecutionError("Invocation declared input phase is invalid")
        if not isinstance(step_id, str) or not step_id or not isinstance(port, str) or not port:
            raise ExecutionError("Invocation declared input step_id and port must be non-empty strings")
        ref = item["ref"]
        if not isinstance(ref, dict) or set(ref) not in ({"record"}, {"file"}, {"resource", "path"}):
            raise ExecutionError("Invocation declared input ref must be a record, file, or resource reference")
        if any(not isinstance(part, str) or not part for part in ref.values()):
            raise ExecutionError("Invocation declared input reference values must be non-empty strings")
        normalized.append({"phase": phase, "step_id": step_id, "port": port, "ref": dict(ref)})
    return normalized


def _declared_ref_payload(ref: Any, *, experiment_root: Path) -> dict[str, str]:
    if isinstance(ref, RecordRef):
        return {"record": ref.record_id}
    if isinstance(ref, ResourceRef):
        return {
            "resource": ref.resource_id,
            "path": _render_declared_path(ref.path, experiment_root=experiment_root),
        }
    if isinstance(ref, FileRef):
        return {"file": _render_declared_path(ref.path, experiment_root=experiment_root)}
    raise ExecutionError(f"Invocation declared input has unsupported reference type: {type(ref).__name__}")


def _render_declared_path(path: Path, *, experiment_root: Path) -> str:
    candidate = Path(path)
    resolved = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (experiment_root / candidate).resolve(strict=False)
    )
    if resolved.is_relative_to(experiment_root):
        return resolved.relative_to(experiment_root).as_posix()
    raise ExecutionError(f"Invocation declared input must stay under the experiment root: {path}")


def _normalize_revisions(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ExecutionError("Invocation produced_record_revisions must be a list")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"record_id", "revision", "revision_digest"}:
            raise ExecutionError(
                "Invocation record revisions must contain exactly record_id, revision, and revision_digest"
            )
        record_id = item["record_id"]
        revision = item["revision"]
        revision_digest = item["revision_digest"]
        if not isinstance(record_id, str) or not record_id:
            raise ExecutionError("Invocation record revision record_id must be a non-empty string")
        if not isinstance(revision, int) or revision < 1:
            raise ExecutionError("Invocation record revision must be a positive integer")
        if not isinstance(revision_digest, str) or not revision_digest.startswith("sha256:"):
            raise ExecutionError("Invocation record revision_digest must be a sha256 digest")
        normalized.append(
            {
                "record_id": record_id,
                "revision": revision,
                "revision_digest": revision_digest,
            }
        )
    return sorted(normalized, key=lambda item: (item["record_id"], item["revision"]))
