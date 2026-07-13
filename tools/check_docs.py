from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import yaml

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
DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
REQUIRED_FRONTMATTER_FIELDS = {"doc_id", "surface", "owner", "last_verified", "summary"}
TOP_LEVEL_DOCS = {
    "README.md",
    "ARCHITECTURE.md",
    "DESIGN.md",
    "QUALITY.md",
    "RELIABILITY.md",
    "SECURITY.md",
}
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


def iter_navigable_docs() -> list[Path]:
    top_level = [REPO_ROOT / path for path in sorted(TOP_LEVEL_DOCS)]
    return top_level + sorted((REPO_ROOT / "docs").rglob("*.md"))


def _frontmatter_payload(path: Path) -> tuple[dict[str, object] | None, str | None]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None, "missing opening front matter delimiter"
    try:
        closing = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        return None, "missing closing front matter delimiter"
    try:
        payload = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        return None, f"invalid YAML front matter: {exc}"
    if not isinstance(payload, dict):
        return None, "front matter must be a YAML mapping"
    return payload, None


def check_doc_frontmatter(files: list[Path]) -> list[str]:
    errors: list[str] = []
    seen_ids: dict[str, Path] = {}
    for path in files:
        rel_path = path.relative_to(REPO_ROOT)
        payload, parse_error = _frontmatter_payload(path)
        if parse_error is not None:
            errors.append(f"front matter: {rel_path}: {parse_error}")
            continue
        assert payload is not None
        missing = sorted(REQUIRED_FRONTMATTER_FIELDS - set(payload))
        if missing:
            errors.append(f"front matter: {rel_path}: missing fields {missing}")
            continue
        for field in ("doc_id", "surface", "owner", "summary"):
            value = payload[field]
            if not isinstance(value, str) or not value.strip():
                errors.append(f"front matter: {rel_path}: {field} must be a non-empty string")
        summary = payload["summary"]
        if isinstance(summary, str) and len(summary.strip()) > 200:
            errors.append(f"front matter: {rel_path}: summary must be at most 200 characters")
        doc_id = payload["doc_id"]
        if isinstance(doc_id, str) and doc_id.strip():
            if DOC_ID_RE.fullmatch(doc_id) is None:
                errors.append(f"front matter: {rel_path}: invalid doc_id {doc_id!r}")
            previous = seen_ids.get(doc_id)
            if previous is not None:
                errors.append(
                    f"front matter: {rel_path}: duplicate doc_id {doc_id!r} also used by {previous.relative_to(REPO_ROOT)}"
                )
            else:
                seen_ids[doc_id] = path
        verified_raw = payload["last_verified"]
        if isinstance(verified_raw, date):
            verified = verified_raw
        elif not isinstance(verified_raw, str):
            errors.append(f"front matter: {rel_path}: last_verified must be an ISO date")
            continue
        else:
            try:
                verified = date.fromisoformat(verified_raw)
            except ValueError:
                errors.append(f"front matter: {rel_path}: last_verified must be an ISO date")
                continue
        age_days = (date.today() - verified).days
        if age_days < 0:
            errors.append(f"front matter: {rel_path}: last_verified must not be in the future")
        elif age_days > 365:
            errors.append(f"front matter: {rel_path}: last_verified is stale ({age_days} days old)")
    return errors


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
            rel_file = file.relative_to(REPO_ROOT)
            if not normalized.is_relative_to(REPO_ROOT):
                errors.append(f"link escapes repository: {rel_file} -> {raw_target}")
                continue
            if not normalized.exists():
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
    errors.extend(check_doc_frontmatter(iter_navigable_docs()))
    if errors:
        print("docs integrity check failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"docs integrity ok: {len(files)} markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
