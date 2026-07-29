from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "reader-workbench"


def package_version() -> str:
    """Return the installed Reader distribution version."""

    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "0+uninstalled"


__all__ = ["package_version"]
