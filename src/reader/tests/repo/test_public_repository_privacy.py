from __future__ import annotations

import csv
import re
import subprocess
import tomllib
from pathlib import Path

from reader.tests.support import REPO_ROOT

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b", re.IGNORECASE)
_ALLOWED_EMAIL_DOMAINS = {"example.com", "users.noreply.github.com"}
_FORBIDDEN_LOCAL_PATH_PARTS = ("/" + "Users/", "\\" + "Users\\", "Drop" + "box/")
_FORBIDDEN_PUBLIC_BINARY_SUFFIXES = {".doc", ".docx", ".xls", ".xlsx", ".xlsm", ".ppt", ".pptx"}
_FORBIDDEN_STUDY_TERMS = (
    "retr" + "on",
    "s" + "pop",
    "o" + "pal",
    "dna" + "design",
    "ln" + "rna",
    "pro" + "moter",
)
_PRIVACY_SENTINEL_PATHS = {
    Path("src/reader/tests/engine/test_invocations.py"),
    Path("src/reader/tests/repo/test_docs_routes.py"),
}


def _tracked_paths() -> tuple[Path, ...]:
    """Return tracked and non-ignored candidate paths in the public tree."""

    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(Path(item.decode()) for item in result.stdout.split(b"\0") if item)


def test_tracked_experiments_are_confined_to_the_synthetic_template() -> None:
    tracked = [path for path in _tracked_paths() if path.parts and path.parts[0] == "experiments"]

    assert tracked
    assert all(len(path.parts) > 1 and path.parts[1] == "template" for path in tracked)
    assert not any(path.suffix.lower() in {".xls", ".xlsx", ".xlsm"} for path in tracked)


def test_tracked_tree_excludes_private_work_products_and_office_documents() -> None:
    forbidden_directories = {"outputs", "raw", "raw_data"}
    findings = [
        str(path)
        for path in _tracked_paths()
        if forbidden_directories.intersection(path.parts) or path.suffix.lower() in _FORBIDDEN_PUBLIC_BINARY_SUFFIXES
    ]

    assert findings == []


def test_synthetic_template_has_only_placeholder_design_ids() -> None:
    path = REPO_ROOT / "experiments/template/inputs/metadata.csv"

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert {row["design_id"] for row in rows} == {"REF", "blank", "g1", "g2", "g3"}


def test_public_package_metadata_does_not_publish_personal_email() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)["project"]

    people = [*metadata.get("authors", []), *metadata.get("maintainers", [])]
    assert people
    assert all("email" not in person for person in people)


def test_tracked_text_does_not_expose_personal_email_or_local_home_path() -> None:
    findings: list[str] = []
    for relative_path in _tracked_paths():
        if relative_path in _PRIVACY_SENTINEL_PATHS:
            continue
        path = REPO_ROOT / relative_path
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if any(marker in content for marker in _FORBIDDEN_LOCAL_PATH_PARTS):
            findings.append(f"{relative_path}: local path")
        for domain in _EMAIL_PATTERN.findall(content):
            if domain.lower() not in _ALLOWED_EMAIL_DOMAINS:
                findings.append(f"{relative_path}: personal email")

    assert findings == []


def test_tracked_text_uses_public_generic_vocabulary() -> None:
    findings: list[str] = []
    for relative_path in _tracked_paths():
        path = REPO_ROOT / relative_path
        try:
            content = path.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeDecodeError):
            continue
        for term in _FORBIDDEN_STUDY_TERMS:
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", content):
                findings.append(f"{relative_path}: private study vocabulary")

    assert findings == []
