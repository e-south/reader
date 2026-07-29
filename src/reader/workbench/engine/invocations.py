from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from filelock import Timeout

from reader.errors import ConfigError, ExecutionError
from reader.workbench.graph import FileRef, RecordCollectionRef, RecordRef, ResourceRef
from reader.workbench.paths import resolve_confined_sink_root
from reader.workbench.records import record_revision_digest
from reader.workbench.records.identity import BuildIdentity

INVOCATION_SCHEMA = "reader.invocation/v2"
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
    provenance_epoch_id: str
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
    provenance_epoch_id: str | None
    operation: Literal["run", "plot", "export", "mixed"]
    status: Literal["planned", "succeeded"]
    dry_run: bool
    selected_steps: SelectedSteps
    produced_record_revisions: tuple[ProducedRecordRevision, ...]
    ledger_path: Path | None


class InvocationLedger:
    def __init__(
        self,
        *,
        experiment_root: Path,
        outputs_dir: Path,
        provenance_epoch_id: str,
        epoch_guard: Callable[[str], None] | None = None,
        writer_lock: Any | None = None,
    ) -> None:
        self.experiment_root = Path(experiment_root).resolve(strict=False)
        self.outputs_dir = Path(outputs_dir).resolve(strict=False)
        if not self.outputs_dir.is_relative_to(self.experiment_root):
            raise ConfigError(f"Invocation outputs directory must stay under the experiment root: {self.outputs_dir}")
        self.provenance_epoch_id = _require_uuid4(provenance_epoch_id, field="provenance_epoch_id")
        self._epoch_guard = epoch_guard
        self._writer_lock = writer_lock
        self.path = self.outputs_dir / "manifests" / "invocations" / f"{self.provenance_epoch_id}.jsonl"

    @classmethod
    def for_store(cls, *, store: Any) -> InvocationLedger:
        """Bind a ledger to the record catalog's active provenance epoch."""

        epoch_id = store.provenance_epoch_id()
        store.bind_provenance_epoch(epoch_id)
        return cls(
            experiment_root=store.experiment_root,
            outputs_dir=store.root,
            provenance_epoch_id=epoch_id,
            epoch_guard=store.assert_provenance_epoch,
            writer_lock=store.provenance_lock,
        )

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
            provenance_epoch_id=self.provenance_epoch_id,
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
        if attempt.provenance_epoch_id != self.provenance_epoch_id:
            raise ExecutionError("Invocation attempt belongs to a different provenance epoch")
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
            "provenance_epoch_id": attempt.provenance_epoch_id,
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
        if self._writer_lock is None:
            self._append_locked(event)
            return
        try:
            self._writer_lock.acquire()
        except (Timeout, OSError, NotImplementedError) as exc:
            raise ExecutionError("Could not acquire the invocation writer lease") from exc
        try:
            self._append_locked(event)
        finally:
            self._writer_lock.release()

    def _append_locked(self, event: dict[str, Any]) -> None:
        self._assert_active_epoch()
        ledger_dir = self._resolve_ledger_dir(create=True)
        assert ledger_dir is not None
        if self.path.is_symlink():
            raise ConfigError(f"Invocation ledger must not be a symlink: {self.path}")
        ledger_path = ledger_dir / self.path.name

        payload = (json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        try:
            descriptor = os.open(ledger_path, flags, 0o644)
        except OSError as exc:
            raise ExecutionError(f"Could not append invocation event under outputs/manifests: {exc}") from exc
        try:
            self._assert_active_epoch()
            ledger_stat = os.fstat(descriptor)
            if not stat.S_ISREG(ledger_stat.st_mode) or ledger_stat.st_nlink != 1:
                raise ExecutionError("Invocation ledger must be a regular file with a single link")
            initial_size = ledger_stat.st_size
            try:
                written = os.write(descriptor, payload)
                if written != len(payload):
                    raise ExecutionError(
                        f"Invocation ledger append was incomplete: wrote {written} of {len(payload)} bytes"
                    )
                os.fsync(descriptor)
            except BaseException as exc:
                try:
                    os.ftruncate(descriptor, initial_size)
                    os.fsync(descriptor)
                except OSError as rollback_error:
                    raise ExecutionError(
                        "Invocation ledger append failed and Reader could not restore the previous file boundary"
                    ) from rollback_error
                if isinstance(exc, OSError):
                    raise ExecutionError(f"Could not append invocation event under outputs/manifests: {exc}") from exc
                raise
        except OSError as exc:
            raise ExecutionError(f"Could not inspect invocation ledger under outputs/manifests: {exc}") from exc
        finally:
            os.close(descriptor)

    def _assert_active_epoch(self) -> None:
        if self._epoch_guard is not None:
            self._epoch_guard(self.provenance_epoch_id)

    def _resolve_ledger_dir(self, *, create: bool) -> Path | None:
        try:
            outputs_dir = resolve_confined_sink_root(
                self.outputs_dir,
                root=self.experiment_root,
                label="Invocation outputs",
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        manifests_dir = outputs_dir / "manifests"
        ledger_dir = manifests_dir / "invocations"
        if manifests_dir.is_symlink():
            raise ConfigError(f"Invocation manifest directory must not be a symlink: {manifests_dir}")
        if create:
            manifests_dir.mkdir(parents=True, exist_ok=True)
        elif not manifests_dir.exists():
            return None
        if manifests_dir.is_symlink():
            raise ConfigError(f"Invocation manifest directory must not be a symlink: {manifests_dir}")
        if ledger_dir.is_symlink():
            raise ConfigError(f"Invocation ledger directory must not be a symlink: {ledger_dir}")
        if create:
            ledger_dir.mkdir(parents=True, exist_ok=True)
        elif not ledger_dir.exists():
            return None
        if ledger_dir.is_symlink():
            raise ConfigError(f"Invocation ledger directory must not be a symlink: {ledger_dir}")
        resolved_parent = ledger_dir.resolve(strict=True)
        if not resolved_parent.is_relative_to(outputs_dir.resolve(strict=True)):
            raise ConfigError(f"Invocation ledger directory must stay under the outputs directory: {resolved_parent}")
        return resolved_parent


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


def _require_uuid4(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"Invocation {field} must be a canonical UUID4")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ConfigError(f"Invocation {field} must be a canonical UUID4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ConfigError(f"Invocation {field} must be a canonical UUID4")
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
        if not isinstance(ref, dict) or set(ref) not in (
            {"record"},
            {"file"},
            {"resource", "path"},
            {"record_collection"},
        ):
            raise ExecutionError(
                "Invocation declared input ref must be a record, record collection, file, or resource reference"
            )
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
    if isinstance(ref, RecordCollectionRef):
        return {"record_collection": ",".join(item.resource_id for item in ref.records)}
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
