from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _project_metadata() -> dict[str, object]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_distribution_namespace_and_cli_entry_point_are_unambiguous() -> None:
    metadata = _project_metadata()
    project = metadata["project"]
    setuptools = metadata["tool"]["setuptools"]

    assert project["name"] == "reader-workbench"
    assert project["scripts"] == {"reader": "reader_workbench.workbench.cli:main"}
    assert setuptools["packages"]["find"]["include"] == ["reader_workbench*"]
    assert setuptools["packages"]["find"]["exclude"] == [
        "reader_workbench.tests",
        "reader_workbench.tests.*",
    ]
    assert not (REPO_ROOT / "src" / "reader").exists()


def test_runtime_declares_direct_scientific_imports_and_no_notebook_extra() -> None:
    project = _project_metadata()["project"]
    dependency_names = {
        dependency.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0] for dependency in project["dependencies"]
    }

    assert {"matplotlib", "numpy"} <= dependency_names
    assert "optional-dependencies" not in project


def test_external_plugin_group_uses_the_distribution_namespace() -> None:
    registry = (REPO_ROOT / "src" / "reader_workbench" / "workbench" / "registry.py").read_text(encoding="utf-8")

    assert 'entry_points(group="reader_workbench.plugins")' in registry
    assert 'entry_points(group="reader.plugins")' not in registry


def test_compile_checks_target_the_existing_import_package() -> None:
    check_surfaces = (
        REPO_ROOT / "QUALITY.md",
        REPO_ROOT / ".agents" / "skills" / "reader-workbench-gardening" / "references" / "verification.md",
        REPO_ROOT / ".github" / "workflows" / "checks.yaml",
    )

    for path in check_surfaces:
        text = path.read_text(encoding="utf-8")
        assert "test -d src/reader_workbench" in text, path
        assert "compileall -q src/reader_workbench" in text, path
        assert "compileall src/reader`" not in text, path
        assert "compileall -q src/reader\n" not in text, path
