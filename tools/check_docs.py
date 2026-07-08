from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

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
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
HTML_ANCHOR_RE = re.compile(r"""<a\s+(?:[^>]*?\s)?(?:id|name)=["']([^"']+)["']""", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*(```|~~~)")
REQUIRED_LINKS = {
    "README.md": {
        "docs/README.md",
        "docs/guides/getting_started.md",
        "docs/guides/preflight_run_verify.md",
        "docs/guides/automation.md",
        "docs/guides/data_operations_plan.md",
        "docs/core/cli.md",
        "docs/core/pipeline.md",
        "docs/repo-maintenance.md",
    },
    "docs/README.md": {
        "guides/getting_started.md",
        "guides/common_routes.md",
        "guides/preflight_run_verify.md",
        "guides/automation.md",
        "guides/data_operations_plan.md",
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
        "guides/data_operations_plan.md",
        "core/cli.md",
        "core/pipeline.md",
    },
    "docs/guides/experiment_bootstrap.md": {
        "./data_operations_plan.md",
        "./data_operations_plan/data_classes.md",
    },
    "docs/guides/data_operations_plan.md": {
        "../../src/reader/workbench/dop/",
        "../../skills/reader-data-operations-plan/SKILL.md",
        "./data_operations_plan/operating_model.md",
        "./data_operations_plan/data_classes.md",
        "./data_operations_plan/metadata_minimums.md",
        "./data_operations_plan/transfer_and_verification.md",
        "./experiment_bootstrap.md",
        "./preflight_run_verify.md",
    },
    "skills/reader-data-operations-plan/SKILL.md": {
        "../../docs/guides/data_operations_plan.md",
        "../../docs/guides/data_operations_plan/operating_model.md",
        "../../docs/guides/experiment_bootstrap.md",
        "./references/endpoint-contracts.md",
        "./references/external-sources.md",
        "./references/test-matrix.md",
        "./references/workflow.md",
    },
    "docs/core/spec.md": {
        "./pipeline.md",
        "../../src/reader/protocols/",
        "../../src/reader/workbench/dop/",
        "../../src/reader/workbench/experiment/",
        "../../src/reader/workbench/engine/",
        "../../src/reader/plugins/",
        "../../src/reader/contracts/",
        "../repo-maintenance.md",
        "../../QUALITY.md",
        "../../RELIABILITY.md",
    },
    "docs/core/plugins.md": {
        "./pipeline.md",
        "./spec.md",
        "../../ARCHITECTURE.md",
        "../../src/reader/plugins/",
        "../../src/reader/workbench/assets/plugin_manifest.py",
        "../../src/reader/protocols/compiler.py",
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


def normalize_anchor_target(source: Path, raw_target: str) -> tuple[Path, str] | None:
    target = raw_target.strip().split(" ", 1)[0]
    if target.startswith(("http://", "https://", "mailto:")) or "#" not in target:
        return None
    path_raw, anchor_raw = target.split("#", 1)
    anchor = unquote(anchor_raw).strip()
    if not anchor:
        return None
    target_path = source if not path_raw else (source.parent / path_raw).resolve()
    if target_path.suffix.lower() != ".md":
        return None
    return target_path, anchor


def markdown_anchors(source: Path) -> set[str]:
    anchors: set[str] = set()
    slug_counts: dict[str, int] = {}
    in_fence = False
    for line in source.read_text().splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for explicit_anchor in HTML_ANCHOR_RE.findall(line):
            anchors.add(explicit_anchor)
        heading = HEADING_RE.match(line)
        if heading is None:
            continue
        slug = github_heading_slug(heading.group(2))
        count = slug_counts.get(slug, 0)
        anchors.add(slug if count == 0 else f"{slug}-{count}")
        slug_counts[slug] = count + 1
    return anchors


def github_heading_slug(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


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


def check_markdown_anchors(files: list[Path]) -> list[str]:
    anchors_by_file = {file.resolve(): markdown_anchors(file) for file in files}
    errors: list[str] = []
    for file in files:
        for raw_target in LINK_RE.findall(file.read_text()):
            normalized = normalize_anchor_target(file, raw_target)
            if normalized is None:
                continue
            target_file, anchor = normalized
            if not target_file.exists():
                continue
            anchors = anchors_by_file.get(target_file.resolve())
            if anchors is None:
                continue
            if anchor not in anchors:
                rel_file = file.relative_to(REPO_ROOT)
                errors.append(f"broken anchor: {rel_file} -> {raw_target}")
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
    errors.extend(check_markdown_anchors(files))
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
