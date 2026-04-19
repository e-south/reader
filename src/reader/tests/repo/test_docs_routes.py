from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_reader_experiment_bootstrap_skill_routes_to_primary_guide() -> None:
    skill_path = REPO_ROOT / "skills" / "reader-experiment-bootstrap" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    assert "docs/guides/experiment_bootstrap.md" in text
