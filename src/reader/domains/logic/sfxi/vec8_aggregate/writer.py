from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from reader.errors import SFXIError

from .model import SFXIVec8AggregateArtifacts
from .render import render_sfxi_vec8_heatmap
from .reshape import sfxi_vec8_tidy_rows
from .sources import load_sfxi_vec8_sources


def write_sfxi_vec8_aggregate(
    *,
    sources: list[str | Path] | tuple[str | Path, ...],
    out_dir: Path,
    title: str | None = None,
    filename: str = "sfxi_vec8_heatmap",
    dpi: int = 300,
    overwrite: bool = False,
) -> SFXIVec8AggregateArtifacts:
    dpi_value = _positive_dpi(dpi)
    aggregate = load_sfxi_vec8_sources(sources)
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename_stem(filename)
    heatmap_path = out_dir / f"{stem}.png"
    tidy_path = out_dir / f"{stem}_tidy.csv"
    manifest_path = out_dir / f"{stem}_manifest.json"
    _check_overwrite((heatmap_path, tidy_path, manifest_path), overwrite=overwrite)

    tidy = sfxi_vec8_tidy_rows(aggregate.frame)
    artifacts = SFXIVec8AggregateArtifacts(
        heatmap_path=heatmap_path,
        tidy_path=tidy_path,
        manifest_path=manifest_path,
        aggregate=aggregate,
    )
    try:
        with TemporaryDirectory(prefix=f".{stem}.tmp-", dir=out_dir) as tmp_dir_raw:
            tmp_dir = Path(tmp_dir_raw)
            tmp_heatmap_path = tmp_dir / heatmap_path.name
            tmp_tidy_path = tmp_dir / tidy_path.name
            tmp_manifest_path = tmp_dir / manifest_path.name

            tidy.to_csv(tmp_tidy_path, index=False)
            fig = render_sfxi_vec8_heatmap(aggregate.frame, title=title)
            try:
                fig.savefig(tmp_heatmap_path, dpi=dpi_value)
            finally:
                _close_figure(fig)
            tmp_manifest_path.write_text(
                json.dumps(artifacts.to_payload(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            _commit_artifact_bundle(
                (
                    (tmp_heatmap_path, heatmap_path),
                    (tmp_tidy_path, tidy_path),
                    (tmp_manifest_path, manifest_path),
                ),
                backup_dir=tmp_dir,
                overwrite=overwrite,
            )
    except SFXIError:
        raise
    except Exception as exc:
        raise SFXIError(f"SFXI vec8 aggregate could not write artifact bundle in {out_dir}: {exc}") from exc
    return artifacts


def _safe_filename_stem(filename: str) -> str:
    raw = str(filename).strip()
    if not raw:
        raise SFXIError("SFXI vec8 aggregate filename must be non-empty.")
    path = Path(raw)
    stem = path.stem if path.suffix else path.name
    if path.name != raw or not stem:
        raise SFXIError("SFXI vec8 aggregate filename must be a plain filename, not a path.")
    return stem


def _check_overwrite(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    _check_replaceable_targets(paths)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise SFXIError(
            "SFXI vec8 aggregate output already exists. "
            f"Pass overwrite=True or choose a different --out-dir/--filename: {rendered}"
        )


def _check_replaceable_targets(paths: tuple[Path, ...]) -> None:
    bad_targets = [path for path in paths if path.exists() and not path.is_file()]
    if bad_targets:
        rendered = ", ".join(str(path) for path in bad_targets)
        raise SFXIError(f"SFXI vec8 aggregate output paths must be files when they already exist: {rendered}")


def _commit_artifact_bundle(
    replacements: tuple[tuple[Path, Path], ...],
    *,
    backup_dir: Path,
    overwrite: bool,
) -> None:
    target_paths = tuple(target for _, target in replacements)
    target_parents = {target.parent for target in target_paths}
    if len(target_parents) != 1:
        raise SFXIError("SFXI vec8 aggregate artifacts must share one output directory.")

    with _exclusive_directory_lock(next(iter(target_parents))):
        _check_overwrite(target_paths, overwrite=overwrite)
        _replace_artifact_bundle(replacements, backup_dir=backup_dir)


def _replace_artifact_bundle(replacements: tuple[tuple[Path, Path], ...], *, backup_dir: Path) -> None:
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for _, target in replacements:
            if target.exists():
                backup = backup_dir / f"{target.name}.backup"
                target.replace(backup)
                backups.append((backup, target))
        for tmp_path, target in replacements:
            tmp_path.replace(target)
            committed.append(target)
    except Exception:
        for target in reversed(committed):
            if target.exists():
                target.unlink()
        for backup, target in reversed(backups):
            if backup.exists():
                backup.replace(target)
        raise


@contextmanager
def _exclusive_directory_lock(directory: Path) -> Iterator[None]:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _positive_dpi(dpi: int) -> int:
    try:
        dpi_value = int(dpi)
    except (TypeError, ValueError) as exc:
        raise SFXIError("SFXI vec8 aggregate dpi must be a positive integer.") from exc
    if dpi_value <= 0:
        raise SFXIError("SFXI vec8 aggregate dpi must be a positive integer.")
    return dpi_value


def _close_figure(fig) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close(fig)
