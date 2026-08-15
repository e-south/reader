"""Verify the repository's supported operator toolchain range."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_uv_runtime_matches_the_shared_phd_workspace_policy() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    assert project["tool"]["uv"]["required-version"] == ">=0.12.3,<0.13"
