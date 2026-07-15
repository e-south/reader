"""Shared destination validation for response-window publication bundles."""

from __future__ import annotations

from pathlib import Path


def resolve_bundle_destination(
    path: Path,
    *,
    bundle_label: str,
    overwrite: bool,
) -> Path:
    """Resolve a bundle destination without following a final-path symlink."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError(f"{bundle_label} output must be a real directory path, not a symbolic link: {requested}")

    destination = requested.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{bundle_label} output already exists: {destination}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"{bundle_label} output must be a real directory path, not a file: {destination}")
    return destination


__all__ = ["resolve_bundle_destination"]
