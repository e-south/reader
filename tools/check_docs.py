from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}
LINK_RE = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
REQUIRED_LINKS = {
    "README.md": {
        "docs/README.md",
        "docs/guides/getting_started.md",
        "docs/guides/preflight_run_verify.md",
        "docs/guides/automation.md",
        "docs/core/cli.md",
        "docs/core/pipeline.md",
        "docs/repo-maintenance.md",
    },
    "docs/README.md": {
        "guides/getting_started.md",
        "guides/common_routes.md",
        "guides/preflight_run_verify.md",
        "guides/automation.md",
        "guides/experiment_bootstrap.md",
        "guides/demo.md",
        "core/cli.md",
        "core/pipeline.md",
        "repo-change-gate.md",
        "repo-maintenance.md",
        "../QUALITY.md",
        "../RELIABILITY.md",
    },
    "docs/index.md": {
        "README.md",
        "guides/getting_started.md",
        "guides/common_routes.md",
        "guides/preflight_run_verify.md",
        "guides/automation.md",
        "core/cli.md",
        "core/pipeline.md",
    },
}


def iter_markdown_files() -> list[Path]:
    return sorted(path for path in REPO_ROOT.rglob("*.md") if not any(part in SKIP_PARTS for part in path.parts))


def normalize_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split(" ", 1)[0]
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    target = target.split("#", 1)[0]
    if not target:
        return None
    return (source.parent / target).resolve()


def linked_paths(source: Path) -> set[Path]:
    linked: set[Path] = set()
    for raw_target in LINK_RE.findall(source.read_text()):
        normalized = normalize_target(source, raw_target)
        if normalized is not None:
            linked.add(normalized)
    return linked


def check_internal_links(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for file in files:
        for raw_target in LINK_RE.findall(file.read_text()):
            normalized = normalize_target(file, raw_target)
            if normalized is None:
                continue
            if not normalized.exists():
                rel_file = file.relative_to(REPO_ROOT)
                errors.append(f"broken link: {rel_file} -> {raw_target}")
    return errors


def check_required_routes() -> list[str]:
    errors: list[str] = []
    for rel_source, rel_targets in REQUIRED_LINKS.items():
        source = REPO_ROOT / rel_source
        linked = linked_paths(source)
        for rel_target in sorted(rel_targets):
            expected = (source.parent / rel_target).resolve()
            if expected not in linked:
                errors.append(f"missing required route: {rel_source} -> {rel_target}")
    return errors


def main() -> int:
    files = iter_markdown_files()
    errors = check_internal_links(files)
    errors.extend(check_required_routes())
    if errors:
        print("docs integrity check failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"docs integrity ok: {len(files)} markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
