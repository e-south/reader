"""
Artifact publication helpers for the SFXI triptych sequence bundle.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from reader.errors import SFXIError

from .triptych_sequence_dnadesign import (
    DNADESIGN_SEQUENCE_PANEL_CONTRACT_ID,
    READER_SUPPORTED_SEQUENCE_PANEL_CONTRACT_VERSION,
)

TRIPTYCH_BUNDLE_CONTRACT_VERSION = "reader.sfxi_triptych_sequence_bundle.v1"


def bundle_paths(*, ctx, bundle_id: str, movie_enabled: bool) -> dict[str, Path]:
    plot_dir = ctx.plots_dir / "sfxi_triptych_sequence"
    export_dir = ctx.exports_dir / "sfxi_triptych_sequence"
    paths = {
        "poster": plot_dir / f"{bundle_id}.png",
        "pdf": plot_dir / f"{bundle_id}.pdf",
        "frames_dir": plot_dir / f"{bundle_id}__frames",
        "index": export_dir / f"{bundle_id}_index.csv",
        "manifest": ctx.outputs_dir / "manifests" / f"{bundle_id}_manifest.json",
    }
    if movie_enabled:
        paths["movie"] = plot_dir / f"{bundle_id}.mp4"
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
        "dnadesign_contract_id": DNADESIGN_SEQUENCE_PANEL_CONTRACT_ID,
        "dnadesign_contract_version": READER_SUPPORTED_SEQUENCE_PANEL_CONTRACT_VERSION,
        "sequence_profile_id": str(cfg["sequence_panel"]["profile"]),
        "axis_scales": scales,
        "outputs": output_map,
        "records": records,
    }


def publish_bundle(*, staging: Mapping[str, Path], final: Mapping[str, Path]) -> None:
    for key in ("poster", "pdf", "index", "manifest"):
        final[key].parent.mkdir(parents=True, exist_ok=True)
    if "movie" in final:
        final["movie"].parent.mkdir(parents=True, exist_ok=True)
    final["frames_dir"].parent.mkdir(parents=True, exist_ok=True)

    frames_dir = final["frames_dir"]
    previous_frames_dir = frames_dir.with_name(f"{frames_dir.name}.__previous")
    if previous_frames_dir.exists():
        shutil.rmtree(previous_frames_dir)

    try:
        if frames_dir.exists():
            frames_dir.rename(previous_frames_dir)
        shutil.move(str(staging["frames_dir"]), str(frames_dir))
        for key in ("poster", "pdf", "index", "manifest", "movie"):
            if key in staging and key in final:
                Path(staging[key]).replace(final[key])
    except Exception:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        if previous_frames_dir.exists():
            previous_frames_dir.rename(frames_dir)
        raise
    finally:
        shutil.rmtree(previous_frames_dir, ignore_errors=True)
        shutil.rmtree(Path(staging["poster"]).parent, ignore_errors=True)


def cleanup_staging_root(staging_root: Path) -> None:
    shutil.rmtree(staging_root, ignore_errors=True)


def relative_to_outputs(path: Path, *, outputs_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(outputs_dir.resolve()))
    except ValueError:
        return str(path.resolve())


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
