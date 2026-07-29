from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from reader.errors import RecordError
from reader.runtime import builtin_runtime
from reader.workbench.graph import ProvenanceInput, RecordRef
from reader.workbench.paths import resolve_path_within_root
from reader.workbench.records import FileBundleRecord, PathDescription, RecordInputEvidence, digest_json

from .context import NotebookWorkbenchContext

ArtifactWriter = Callable[[Path], None]
_SAFE_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class NotebookArtifactSpec:
    relative_path: str | Path
    description: str
    writer: ArtifactWriter

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or path == Path(".") or any(part == ".." for part in path.parts):
            raise RecordError("Notebook artifact paths must identify relative, confined files")
        if not isinstance(self.description, str) or not self.description.strip():
            raise RecordError("Notebook artifact descriptions must be non-empty strings")
        if "\n" in self.description or "\r" in self.description:
            raise RecordError("Notebook artifact descriptions must be single-line strings")
        if not callable(self.writer):
            raise RecordError("Notebook artifact writers must be callable")
        object.__setattr__(self, "relative_path", path)
        object.__setattr__(self, "description", self.description.strip())


def publish_notebook_artifact_bundle(
    context: NotebookWorkbenchContext,
    *,
    record_id: str,
    producer_id: str,
    template: str,
    upstream_records: Mapping[str, str],
    producer_config: Mapping[str, Any],
    description: str,
    artifacts: Sequence[NotebookArtifactSpec],
) -> FileBundleRecord:
    """Publish one immutable, manifest-backed bundle from a notebook UI."""

    normalized_producer_id = str(producer_id).strip()
    if not _SAFE_BUNDLE_ID.fullmatch(normalized_producer_id) or normalized_producer_id in {".", ".."}:
        raise RecordError("Notebook artifact producer_id must be one safe path segment")
    if not isinstance(record_id, str) or not record_id.strip():
        raise RecordError("Notebook artifact record_id must be a non-empty string")
    if not isinstance(template, str) or not template.strip():
        raise RecordError("Notebook artifact template must be a non-empty string")
    artifact_specs = tuple(artifacts)
    if not artifact_specs:
        raise RecordError("Notebook artifact bundles must contain at least one artifact")
    if any(not isinstance(item, NotebookArtifactSpec) for item in artifact_specs):
        raise RecordError("Notebook artifact bundles must contain NotebookArtifactSpec values")
    if not upstream_records:
        raise RecordError("Notebook artifact bundles require at least one upstream record")

    layout = context.decl.experiment_semantics.layout
    runtime = builtin_runtime()
    store = runtime.record_store(
        context.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=context.experiment_root,
    )
    inputs = _upstream_inputs(store=store, upstream_records=upstream_records)
    exports_dir = _confined_exports_dir(outputs_dir=context.outputs_dir, exports_subdir=layout.exports_subdir)
    target_paths = _validate_artifact_targets(artifact_specs)

    staging_parent = _confined_staging_parent(outputs_dir=context.outputs_dir)
    staging_dir = Path(mkdtemp(prefix=f"{normalized_producer_id}__", dir=staging_parent))
    promoted_dir: Path | None = None
    try:
        staged_paths: list[Path] = []
        for spec, relative_path in zip(artifact_specs, target_paths, strict=True):
            staged_path = resolve_path_within_root(relative_path, root=staging_dir)
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            spec.writer(staged_path)
            staged_paths.append(staged_path)
        _validate_written_artifacts(staged_paths, staging_dir=staging_dir)

        exports_dir.mkdir(parents=True, exist_ok=True)
        final_dir = _next_revision_dir(exports_dir / normalized_producer_id)
        staging_dir.replace(final_dir)
        promoted_dir = final_dir
        final_paths = [promoted_dir / relative_path for relative_path in target_paths]
        return store.append_notebook_file_bundle(
            producer_id=normalized_producer_id,
            producer_template=template.strip(),
            record_id=record_id.strip(),
            inputs=inputs,
            config_digest=context.decl.config_digest,
            producer_config_digest=digest_json(dict(producer_config)),
            files=final_paths,
            description=description,
            path_descriptions=tuple(
                PathDescription(path=path, description=spec.description)
                for path, spec in zip(final_paths, artifact_specs, strict=True)
            ),
        )
    except Exception:
        if promoted_dir is not None:
            shutil.rmtree(promoted_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _upstream_inputs(*, store: Any, upstream_records: Mapping[str, str]) -> tuple[RecordInputEvidence, ...]:
    inputs: list[ProvenanceInput] = []
    resolved_inputs: dict[str, Any] = {}
    for raw_label, raw_record_id in sorted(upstream_records.items()):
        label = str(raw_label).strip()
        upstream_record_id = str(raw_record_id).strip()
        if not label or not upstream_record_id:
            raise RecordError("Notebook artifact upstream record labels and ids must be non-empty strings")
        upstream = store.latest_record(upstream_record_id)
        if upstream is None:
            raise RecordError(
                f"Input record {upstream_record_id!r} is missing; produce it before publishing notebook artifacts."
            )
        inputs.append(ProvenanceInput(label=label, ref=RecordRef(record_id=upstream_record_id)))
        resolved_inputs[label] = upstream
    return store.capture_inputs(inputs, resolved_inputs=resolved_inputs)


def _confined_exports_dir(*, outputs_dir: Path, exports_subdir: str) -> Path:
    raw = Path(".") if exports_subdir in ("", ".", "./") else Path(exports_subdir)
    try:
        return resolve_path_within_root(raw, root=outputs_dir)
    except ValueError as exc:
        raise RecordError(
            "Notebook artifact exports directory must stay within the experiment outputs directory"
        ) from exc


def _confined_staging_parent(*, outputs_dir: Path) -> Path:
    raw = outputs_dir / ".staging"
    message = "Notebook artifact staging directory must stay within the experiment outputs directory"
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


def _validate_artifact_targets(artifacts: Sequence[NotebookArtifactSpec]) -> tuple[Path, ...]:
    paths = tuple(Path(item.relative_path) for item in artifacts)
    if len(set(paths)) != len(paths):
        raise RecordError("Notebook artifact bundle paths must be unique")
    return paths


def _validate_written_artifacts(paths: Sequence[Path], *, staging_dir: Path) -> None:
    staging_root = staging_dir.resolve(strict=True)
    expected: set[Path] = set()
    for path in paths:
        if path.is_symlink():
            raise RecordError(f"Notebook artifact {path} must be a confined regular file")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(staging_root)
        except (FileNotFoundError, ValueError) as exc:
            raise RecordError(f"Notebook artifact {path} must be a confined regular file") from exc
        if not resolved.is_file():
            raise RecordError(f"Notebook artifact {path} must be a confined regular file")
        expected.add(resolved.relative_to(staging_root))

    discovered: set[Path] = set()
    for path in staging_root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise RecordError("Notebook artifact writers must create exactly the declared non-empty files")
        if not path.is_file():
            continue
        if path.stat().st_size == 0:
            raise RecordError("Notebook artifact writers must create exactly the declared non-empty files")
        discovered.add(path.relative_to(staging_root))
    if discovered != expected:
        raise RecordError("Notebook artifact writers must create exactly the declared non-empty files")


def _next_revision_dir(base: Path) -> Path:
    revision = 1
    while True:
        candidate = base if revision == 1 else base.with_name(f"{base.name}__r{revision}")
        if not candidate.exists():
            return candidate
        revision += 1
