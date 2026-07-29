from __future__ import annotations

from pathlib import Path

from reader.workbench.records.identity import _source_digest


def test_source_digest_tracks_only_packaged_runtime_files(tmp_path: Path) -> None:
    package_root = tmp_path / "reader"
    runtime_source = package_root / "runtime.py"
    test_source = package_root / "tests" / "test_runtime.py"
    source_doc = package_root / "domains" / "example" / "docs.md"
    template = package_root / "workbench" / "templates" / "builtins" / "basic.marimo.py.txt"
    for path, content in (
        (runtime_source, "RUNTIME = 1\n"),
        (test_source, "def test_runtime(): ...\n"),
        (source_doc, "# Internal notes\n"),
        (template, "# packaged template\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    baseline = _source_digest(package_root)

    test_source.write_text("def test_runtime_changed(): ...\n", encoding="utf-8")
    source_doc.write_text("# Changed internal notes\n", encoding="utf-8")
    assert _source_digest(package_root) == baseline

    runtime_source.write_text("RUNTIME = 2\n", encoding="utf-8")
    runtime_changed = _source_digest(package_root)
    assert runtime_changed != baseline

    template.write_text("# changed packaged template\n", encoding="utf-8")
    assert _source_digest(package_root) != runtime_changed
