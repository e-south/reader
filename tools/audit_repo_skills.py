from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "skills"
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
SOURCE_ROW_RE = re.compile(r"^\| https?://[^|]+ \| \d{4}-\d{2}-\d{2} \| [^|]+ \|$", re.MULTILINE)


def iter_skill_dirs() -> list[Path]:
    return sorted(path for path in SKILLS_DIR.iterdir() if path.is_dir() and not path.name.startswith("."))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def frontmatter_block(text: str, skill_path: Path) -> str:
    match = FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError(f"{skill_path.relative_to(REPO_ROOT)}: missing frontmatter")
    return match.group(1)


def require_in_block(block: str, needle: str, skill_path: Path, label: str) -> list[str]:
    if needle not in block:
        return [f"{skill_path.relative_to(REPO_ROOT)}: missing {label}"]
    return []


def audit_skill_dir(skill_dir: Path) -> list[str]:
    errors: list[str] = []
    skill_path = skill_dir / "SKILL.md"
    if not skill_path.exists():
        return [f"{skill_dir.relative_to(REPO_ROOT)}: missing SKILL.md"]

    text = read_text(skill_path)
    try:
        frontmatter = frontmatter_block(text, skill_path)
    except ValueError as exc:
        return [str(exc)]

    errors.extend(
        require_in_block(
            frontmatter,
            f"name: {skill_dir.name}",
            skill_path,
            "frontmatter name matching folder",
        )
    )
    errors.extend(require_in_block(frontmatter, "description:", skill_path, "frontmatter description"))
    errors.extend(require_in_block(frontmatter, "metadata:", skill_path, "metadata block"))
    errors.extend(require_in_block(frontmatter, "version:", skill_path, "metadata.version"))
    errors.extend(require_in_block(frontmatter, "category:", skill_path, "metadata.category"))
    errors.extend(require_in_block(frontmatter, "tags:", skill_path, "metadata.tags"))

    if "Use when" not in frontmatter or "Do not use" not in frontmatter:
        errors.append(
            f"{skill_path.relative_to(REPO_ROOT)}: frontmatter description must include "
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
            errors.append(f"{skill_path.relative_to(REPO_ROOT)}: missing section {section}")

    external_sources_path = skill_dir / "references" / "external-sources.md"
    if not external_sources_path.exists():
        errors.append(f"{skill_dir.relative_to(REPO_ROOT)}: missing references/external-sources.md")
    elif "./references/external-sources.md" not in text:
        errors.append(
            f"{skill_path.relative_to(REPO_ROOT)}: top-level skill does not expose references/external-sources.md"
        )
    else:
        external_sources = read_text(external_sources_path)
        if "| URL | Retrieved | Mapped update |" not in external_sources:
            errors.append(
                f"{external_sources_path.relative_to(REPO_ROOT)}: missing source table header "
                "'| URL | Retrieved | Mapped update |'"
            )
        if SOURCE_ROW_RE.search(external_sources) is None:
            errors.append(
                f"{external_sources_path.relative_to(REPO_ROOT)}: missing at least one source row "
                "with URL, YYYY-MM-DD retrieved date, and mapped update"
            )

    return errors


def main() -> int:
    if not SKILLS_DIR.exists():
        print("skills directory missing", file=sys.stderr)
        return 1

    errors: list[str] = []
    skill_dirs = iter_skill_dirs()
    for skill_dir in skill_dirs:
        errors.extend(audit_skill_dir(skill_dir))

    if errors:
        print("repo skill audit failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"repo skill audit ok: {len(skill_dirs)} skills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
