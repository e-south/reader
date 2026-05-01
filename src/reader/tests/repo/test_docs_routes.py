from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_reader_experiment_bootstrap_skill_routes_to_primary_guide() -> None:
    skill_path = REPO_ROOT / "skills" / "reader-experiment-bootstrap" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    assert "docs/guides/experiment_bootstrap.md" in text
    assert "docs/guides/data_operations_plan.md" in text
    assert "docs/guides/data_operations_plan/data_classes.md" in text


def test_experiment_bootstrap_routes_through_data_operations_plan() -> None:
    guide_path = REPO_ROOT / "docs" / "guides" / "experiment_bootstrap.md"
    text = guide_path.read_text(encoding="utf-8")
    assert "./data_operations_plan.md" in text
    assert "./data_operations_plan/data_classes.md" in text
    assert "Classify the data class" in text


def test_data_operations_plan_uses_progressive_disclosure() -> None:
    guide_path = REPO_ROOT / "docs" / "guides" / "data_operations_plan.md"
    text = guide_path.read_text(encoding="utf-8")
    assert "./data_operations_plan/operating_model.md" in text
    assert "./data_operations_plan/data_classes.md" in text
    assert "./data_operations_plan/metadata_minimums.md" in text
    assert "./data_operations_plan/transfer_and_verification.md" in text


def test_data_operations_plan_routes_to_machine_readable_registry() -> None:
    guide_path = REPO_ROOT / "docs" / "guides" / "data_operations_plan.md"
    text = guide_path.read_text(encoding="utf-8")
    assert "../../src/reader/workbench/dop/" in text
    assert "uv run reader dop classes --format json" in text


def test_data_operations_plan_routes_to_repo_skill() -> None:
    guide_path = REPO_ROOT / "docs" / "guides" / "data_operations_plan.md"
    text = guide_path.read_text(encoding="utf-8")
    assert "../../skills/reader-data-operations-plan/SKILL.md" in text


def test_reader_data_operations_plan_skill_routes_to_owned_surfaces() -> None:
    skill_path = REPO_ROOT / "skills" / "reader-data-operations-plan" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    assert "docs/guides/data_operations_plan.md" in text
    assert "docs/guides/data_operations_plan/operating_model.md" in text
    assert "uv run reader dop classes --format json" in text
    assert "./references/endpoint-contracts.md" in text
    assert "./references/external-sources.md" in text


def test_reader_data_operations_plan_skill_routes_away_from_adjacent_workflows() -> None:
    skill_path = REPO_ROOT / "skills" / "reader-data-operations-plan" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    assert "Do not use for full experiment creation" in text
    assert "reader-experiment-bootstrap" in text
    assert "reader-workbench-gardening" in text
    assert "## Success Criteria" in text


def test_repo_skill_index_lists_data_operations_plan_skill() -> None:
    skill_index_path = REPO_ROOT / "skills" / "README.md"
    text = skill_index_path.read_text(encoding="utf-8")
    assert "./reader-data-operations-plan/SKILL.md" in text
