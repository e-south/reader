"""
--------------------------------------------------------------------------------
<reader project>
src/reader/core/mpl.py

Matplotlib cache handling.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

from reader.core.errors import ConfigError


def _default_cache_dir(base_dir: Path | None) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser().resolve() / ".cache" / "matplotlib"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "reader" / "matplotlib"
    return Path.home() / ".cache" / "reader" / "matplotlib"


def ensure_mpl_cache_dir(*, base_dir: Path | None = None) -> Path:
    """
    Ensure MPLCONFIGDIR points to a writable directory.

    Priority:
      1) MPLCONFIGDIR (if set)
      2) READER_MPLCONFIGDIR (if set)
      3) base_dir/.cache/matplotlib (if base_dir provided)
      4) $XDG_CACHE_HOME/reader/matplotlib or ~/.cache/reader/matplotlib
    """
    env = os.environ.get("MPLCONFIGDIR") or os.environ.get("READER_MPLCONFIGDIR")
    cache_dir = Path(env).expanduser() if env else _default_cache_dir(base_dir)

    if cache_dir.exists() and not cache_dir.is_dir():
        raise ConfigError(f"Matplotlib cache path is not a directory: {cache_dir}")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise ConfigError(
            "Matplotlib cache directory is not writable. "
            f"Set MPLCONFIGDIR or READER_MPLCONFIGDIR to a writable path (current: {cache_dir})."
        ) from exc

    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    return cache_dir
