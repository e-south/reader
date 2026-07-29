from __future__ import annotations

import re
from pathlib import Path

from .model import MaintenanceReport

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
SOURCE_ROW_RE = re.compile(r"^\| https?://[^|]+ \| \d{4}-\d{2}-\d{2} \| [^|]+ \|$", re.MULTILINE)


def iter_skill_dirs(skills_dir: Path) -> list[Path]:
    return sorted(path for path in skills_dir.iterdir() if path.is_dir() and not path.name.startswith("."))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def frontmatter_block(text: str, skill_path: Path, repo_root: Path) -> str:
    match = FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError(f"{skill_path.relative_to(repo_root)}: missing frontmatter")
    return match.group(1)


def require_in_block(block: str, needle: str, skill_path: Path, label: str, repo_root: Path) -> list[str]:
    if needle not in block:
        return [f"{skill_path.relative_to(repo_root)}: missing {label}"]
    return []


def audit_skill_dir(skill_dir: Path, repo_root: Path) -> list[str]:
    errors: list[str] = []
    skill_path = skill_dir / "SKILL.md"
    if not skill_path.exists():
        return [f"{skill_dir.relative_to(repo_root)}: missing SKILL.md"]

    text = read_text(skill_path)
    try:
        frontmatter = frontmatter_block(text, skill_path, repo_root)
    except ValueError as exc:
        return [str(exc)]

    errors.extend(
        require_in_block(
            frontmatter,
            f"name: {skill_dir.name}",
            skill_path,
            "frontmatter name matching folder",
            repo_root,
        )
    )
    errors.extend(require_in_block(frontmatter, "description:", skill_path, "frontmatter description", repo_root))
    errors.extend(require_in_block(frontmatter, "metadata:", skill_path, "metadata block", repo_root))
    errors.extend(require_in_block(frontmatter, "version:", skill_path, "metadata.version", repo_root))
    errors.extend(require_in_block(frontmatter, "category:", skill_path, "metadata.category", repo_root))
    errors.extend(require_in_block(frontmatter, "tags:", skill_path, "metadata.tags", repo_root))

    if "Use when" not in frontmatter or "Do not use" not in frontmatter:
        errors.append(
            f"{skill_path.relative_to(repo_root)}: frontmatter description must include "
            "'Use when' and 'Do not use' routing boundaries"
        )

    required_sections = [
        "## Purpose",
        "## Scope",
        "## Required Deliverables",
        "## Output Contract",
        "## Trigger Tests",
    ]
    for section in required_sections:
        if section not in text:
            errors.append(f"{skill_path.relative_to(repo_root)}: missing section {section}")

    external_sources_path = skill_dir / "references" / "external-sources.md"
    if not external_sources_path.exists():
        errors.append(f"{skill_dir.relative_to(repo_root)}: missing references/external-sources.md")
    elif "./references/external-sources.md" not in text:
        errors.append(
            f"{skill_path.relative_to(repo_root)}: top-level skill does not expose references/external-sources.md"
        )
    else:
        external_sources = read_text(external_sources_path)
        if "| URL | Retrieved | Mapped update |" not in external_sources:
            errors.append(
                f"{external_sources_path.relative_to(repo_root)}: missing source table header "
                "'| URL | Retrieved | Mapped update |'"
            )
        if SOURCE_ROW_RE.search(external_sources) is None:
            errors.append(
                f"{external_sources_path.relative_to(repo_root)}: missing at least one source row "
                "with URL, YYYY-MM-DD retrieved date, and mapped update"
            )

    return errors


def check_skills(repo_root: Path) -> MaintenanceReport:
    """Check repo-local skill structure in a Reader source checkout."""

    repo_root = repo_root.resolve()
    skills_dir = repo_root / "skills"
    if not skills_dir.is_dir():
        return MaintenanceReport(
            check="skills",
            repo_root=repo_root,
            checked=0,
            errors=("skills directory missing",),
        )

    errors: list[str] = []
    skill_dirs = iter_skill_dirs(skills_dir)
    for skill_dir in skill_dirs:
        errors.extend(audit_skill_dir(skill_dir, repo_root))

    return MaintenanceReport(
        check="skills",
        repo_root=repo_root,
        checked=len(skill_dirs),
        errors=tuple(errors),
    )
