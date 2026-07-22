"""
Artifact publication helpers for the SFXI triptych sequence bundle.
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from reader.domains.promoter.candidate_bindings import (
    BASERENDER_CONTRACT_ID,
    BASERENDER_CONTRACT_VERSION,
    PromoterCandidateBindings,
)
from reader.errors import SFXIError

TRIPTYCH_BUNDLE_CONTRACT_VERSION = "reader.sfxi_triptych_sequence_bundle.v3"
_PUBLISH_KEYS = ("poster", "pdf", "index", "frames_dir", "manifest", "movie")


def bundle_paths(*, ctx, bundle_id: str) -> dict[str, Path]:
    plot_dir = ctx.plots_dir / "sfxi_triptych_sequence"
    export_dir = ctx.exports_dir / "sfxi_triptych_sequence"
    paths = {
        "poster": plot_dir / f"{bundle_id}.png",
        "pdf": plot_dir / f"{bundle_id}.pdf",
        "frames_dir": plot_dir / f"{bundle_id}__frames",
        "index": export_dir / f"{bundle_id}_index.csv",
        "manifest": ctx.outputs_dir / "manifests" / f"{bundle_id}_manifest.json",
        "movie": plot_dir / f"{bundle_id}.mp4",
    }
    return paths


def staging_parent(outputs_dir: Path) -> Path:
    path = outputs_dir / ".staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def staging_paths(*, staging_root: Path, bundle_id: str, movie_enabled: bool) -> dict[str, Path]:
    frames = staging_root / f"{bundle_id}__frames"
    frames.mkdir(parents=True, exist_ok=True)
    paths = {
        "poster": staging_root / f"{bundle_id}.png",
        "pdf": staging_root / f"{bundle_id}.pdf",
        "frames_dir": frames,
        "index": staging_root / f"{bundle_id}_index.csv",
        "manifest": staging_root / f"{bundle_id}_manifest.json",
        "frames_txt": staging_root / "_movie_frames.txt",
    }
    if movie_enabled:
        paths["movie"] = staging_root / f"{bundle_id}.mp4"
    return paths


def write_movie(
    *, cfg: Mapping[str, object], records: list[dict[str, object]], staging: Mapping[str, Path], outputs_dir: Path
):
    if not bool(cfg["movie_enabled"]):
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SFXIError("sfxi_triptych_sequence movie output requires ffmpeg on PATH.")
    duration = 1.0 / max(float(cfg["movie_fps"]), 1e-9)
    frame_list = staging["frames_txt"]
    lines: list[str] = []
    for record in records:
        png = outputs_dir / str(record["png_path"])
        png = staging["frames_dir"] / png.name
        lines.append(f"file '{png}'")
        lines.append(f"duration {duration:.6f}")
    if records:
        last = staging["frames_dir"] / Path(str(records[-1]["png_path"])).name
        lines.append(f"file '{last}'")
    frame_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(frame_list),
        "-vf",
        "format=yuv420p",
        "-movflags",
        "+faststart",
        str(staging["movie"]),
    ]
    subprocess.run(cmd, check=True)
    return staging["movie"]


def manifest_payload(
    *,
    ctx,
    cfg: Mapping[str, object],
    records: list[dict[str, object]],
    outputs: Mapping[str, Path],
    movie_path: Path | None,
    scales: Mapping[str, object],
    bindings: PromoterCandidateBindings,
) -> dict[str, object]:
    output_map = {
        "png": relative_to_outputs(outputs["poster"], outputs_dir=ctx.outputs_dir),
        "pdf": relative_to_outputs(outputs["pdf"], outputs_dir=ctx.outputs_dir),
        "index_csv": relative_to_outputs(outputs["index"], outputs_dir=ctx.outputs_dir),
        "manifest_json": relative_to_outputs(outputs["manifest"], outputs_dir=ctx.outputs_dir),
    }
    if movie_path is not None:
        output_map["movie_mp4"] = relative_to_outputs(movie_path, outputs_dir=ctx.outputs_dir)
    return {
        "schema": TRIPTYCH_BUNDLE_CONTRACT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "bundle_id": str(cfg["bundle_id"]),
        "plot_id": "sfxi_triptych_sequence",
        "protocol_id": getattr(getattr(ctx, "protocol", None), "id", None),
        "source_experiment_id": getattr(getattr(ctx, "experiment", None), "id", None),
        "row_count": len(records),
        "row_order": [record["design_id"] for record in records],
        "reference_rows": [record for record in records if record.get("row_kind") == "reference"],
        "snapshot_target_time_h": float(cfg["snapshot_target_time_h"]),
        "treatment_contract": {
            "state_map_ref": str(cfg["state_map_ref"]),
            "column": str(cfg["treatment_column"]),
            "corners": dict(cfg["treatment_map"]),
            "case_sensitive": bool(cfg["treatment_case_sensitive"]),
        },
        "dnadesign_contract_id": BASERENDER_CONTRACT_ID,
        "dnadesign_contract_version": BASERENDER_CONTRACT_VERSION,
        "candidate_bindings": {
            "schema_id": bindings.schema_id,
            "schema_version": bindings.schema_version,
            "record_id": bindings.record_id,
            "manifest_sha256": bindings.manifest_sha256,
            "records_sha256": bindings.records_sha256,
            "candidate_table_id": bindings.candidate_table_id,
            "candidate_selection_sha256": bindings.candidate_selection_sha256,
        },
        "sequence_profile_id": str(cfg["sequence_panel"]["profile"]),
        "axis_scales": scales,
        "outputs": output_map,
        "records": records,
    }


def publish_bundle(*, staging: Mapping[str, Path], final: Mapping[str, Path]) -> None:
    staging_root = Path(staging["poster"]).parent
    try:
        replacements = _validated_bundle_replacements(staging=staging, final=final)
    except Exception as publish_error:
        _raise_after_staging_cleanup(staging_root, publish_error)

    backup_root = staging_root / ".publish-backups"
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for key in _PUBLISH_KEYS:
            target = Path(final[key])
            target.parent.mkdir(parents=True, exist_ok=True)
        backup_root.mkdir()
        for key in _PUBLISH_KEYS:
            target = Path(final[key])
            if target.exists():
                backup = backup_root / f"{key}.backup"
                target.replace(backup)
                backups.append((backup, target))
        for _, source, target in replacements:
            source.replace(target)
            installed.append(target)
    except Exception as publish_error:
        rollback_errors = _rollback_bundle(installed=installed, backups=backups)
        if rollback_errors:
            raise ExceptionGroup(
                "SFXI triptych publication failed and rollback was incomplete",
                [publish_error, *rollback_errors],
            ) from publish_error
        _raise_after_staging_cleanup(staging_root, publish_error)
    else:
        try:
            cleanup_staging_root(staging_root)
        except Exception as cleanup_error:  # publication is committed; provenance must still be recorded
            warnings.warn(
                "SFXI triptych publication committed, but transaction cleanup failed; "
                f"remove {staging_root} after reviewing the retained backups: {cleanup_error}",
                RuntimeWarning,
                stacklevel=2,
            )


def _validated_bundle_replacements(
    *, staging: Mapping[str, Path], final: Mapping[str, Path]
) -> tuple[tuple[str, Path, Path], ...]:
    required = {"poster", "pdf", "index", "manifest", "frames_dir"}
    missing_staging = sorted(required - staging.keys())
    missing_final = sorted(set(_PUBLISH_KEYS) - final.keys())
    if missing_staging or missing_final:
        raise SFXIError(
            "SFXI triptych publication requires matching poster, PDF, index, manifest, frame, and optional movie paths."
        )

    movie_target = Path(final["movie"])
    if "movie" not in staging and movie_target.exists() and movie_target.is_dir():
        raise SFXIError(f"SFXI triptych destination movie must be a file: {movie_target}")

    replacements: list[tuple[str, Path, Path]] = []
    for key in _PUBLISH_KEYS:
        if key not in staging or key not in final:
            continue
        source = Path(staging[key])
        target = Path(final[key])
        if not source.exists():
            raise SFXIError(f"SFXI triptych staged {key} is missing: {source}")
        source_is_dir = source.is_dir()
        if (key == "frames_dir") != source_is_dir:
            expected = "directory" if key == "frames_dir" else "file"
            raise SFXIError(f"SFXI triptych staged {key} must be a {expected}: {source}")
        if target.exists() and target.is_dir() != source_is_dir:
            expected = "directory" if source_is_dir else "file"
            raise SFXIError(f"SFXI triptych destination {key} must be a {expected}: {target}")
        replacements.append((key, source, target))
    return tuple(replacements)


def _rollback_bundle(*, installed: list[Path], backups: list[tuple[Path, Path]]) -> list[Exception]:
    errors: list[Exception] = []
    for target in reversed(installed):
        try:
            _remove_path(target)
        except Exception as exc:  # pragma: no cover - recovery failure is platform-dependent
            errors.append(exc)
    for backup, target in reversed(backups):
        try:
            backup.replace(target)
        except Exception as exc:  # pragma: no cover - recovery failure is platform-dependent
            errors.append(exc)
    return errors


def _raise_after_staging_cleanup(staging_root: Path, publish_error: Exception) -> None:
    try:
        cleanup_staging_root(staging_root)
    except Exception as cleanup_error:
        raise ExceptionGroup(
            "SFXI triptych publication failed and staging cleanup was incomplete",
            [publish_error, cleanup_error],
        ) from publish_error
    raise publish_error


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def cleanup_staging_root(staging_root: Path) -> None:
    if staging_root.exists():
        shutil.rmtree(staging_root)


def relative_to_outputs(path: Path, *, outputs_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(outputs_dir.resolve()))
    except ValueError as exc:
        raise ValueError(f"SFXI artifact is outside the declared outputs directory: {path.resolve()}") from exc


__all__ = [
    "TRIPTYCH_BUNDLE_CONTRACT_VERSION",
    "bundle_paths",
    "cleanup_staging_root",
    "manifest_payload",
    "publish_bundle",
    "relative_to_outputs",
    "staging_parent",
    "staging_paths",
    "write_movie",
]
