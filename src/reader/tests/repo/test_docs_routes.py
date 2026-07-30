from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from reader.maintenance import docs as check_docs
from reader.workbench.cli.shared import app

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_root_readme_is_a_human_first_landing_page() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert text.startswith("# ![Reader")
    assert not text.startswith("# reader\n")
    assert "uv tool install ." in text
    assert "python -m pip install ." in text
    assert "not published yet" in text
    assert "\nreader demo\n" in text
    assert "\nreader protocols\n" in text
    assert "uv run reader demo" in text
    assert "docs/guides/getting_started.md" in text
    assert "docs/guides/common_routes.md" in text
    assert "docs/README.md" in text
    assert "doc_id:" not in text


def test_response_window_docs_route_through_public_records_api() -> None:
    text = (REPO_ROOT / "docs" / "lib" / "plate_reader" / "response_window.md").read_text(encoding="utf-8")

    assert "reader.api.records()" in text
    assert "reader.api.read_dataframe()" in text
    assert "RecordStore" not in text


def test_reader_demo_is_a_no_write_command_tour(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["demo"], env={"COLUMNS": "240"})

    assert result.exit_code == 0, result.output
    assert "Reader Demo" in result.output
    assert "Find experiments" in result.output
    assert "Scaffold a new experiment" in result.output
    assert list(tmp_path.iterdir()) == []


def test_template_journal_is_a_neutral_authored_capsule() -> None:
    journal = (REPO_ROOT / "experiments" / "template" / "JOURNAL.md").read_text(encoding="utf-8")
    normalized = " ".join(journal.split())

    assert journal.startswith("# Experiment journal\n")
    assert "human-authored" in journal
    assert "Keep this capsule human-authored" in normalized
    assert "machine invocation events" not in normalized
    assert "### 20" not in journal
    assert "/Users/" not in journal
    assert "uv run reader" not in journal
    assert len(journal.splitlines()) <= 24


def test_docs_inventory_excludes_generated_experiment_outputs(tmp_path: Path, monkeypatch) -> None:
    source_doc = tmp_path / "docs" / "guide.md"
    generated_doc = tmp_path / "experiments" / "2026" / "example" / "outputs" / "report.md"
    scratch_doc = tmp_path / ".tmp" / "package-audit" / "README.md"
    source_doc.parent.mkdir(parents=True)
    generated_doc.parent.mkdir(parents=True)
    scratch_doc.parent.mkdir(parents=True)
    source_doc.write_text("# Guide\n", encoding="utf-8")
    generated_doc.write_text("# Generated report\n", encoding="utf-8")
    scratch_doc.write_text("# Scratch copy\n", encoding="utf-8")

    assert check_docs.iter_markdown_files(tmp_path) == [source_doc]


def test_canonical_public_repository_links_resolve_to_local_docs(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    guide = tmp_path / "docs" / "guide.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("# Guide\n", encoding="utf-8")
    readme.write_text(
        "[Guide](https://github.com/e-south/reader/blob/main/docs/guide.md)\n",
        encoding="utf-8",
    )

    assert check_docs.linked_paths(readme, repo_root=tmp_path) == {guide.resolve()}
    assert check_docs.check_internal_links([readme], tmp_path) == []


def test_repository_python_checks_live_under_reader_source() -> None:
    assert not list((REPO_ROOT / "tools").glob("*.py"))


def test_checks_use_the_portable_default_test_lane() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "checks.yaml").read_text(encoding="utf-8")

    assert "uv run --locked pytest -q" in workflow
    assert "active_experiments" not in workflow


def test_reader_experiment_bootstrap_skill_routes_to_primary_guide() -> None:
    skill_path = REPO_ROOT / ".agents" / "skills" / "reader-experiment-bootstrap" / "SKILL.md"
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
    assert "../../.agents/skills/reader-data-operations-plan/SKILL.md" in text


def test_reader_data_operations_plan_skill_routes_to_owned_surfaces() -> None:
    skill_path = REPO_ROOT / ".agents" / "skills" / "reader-data-operations-plan" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    assert "docs/guides/data_operations_plan.md" in text
    assert "docs/guides/data_operations_plan/operating_model.md" in text
    assert "uv run reader dop classes --format json" in text
    assert "./references/endpoint-contracts.md" in text
    assert "./references/external-sources.md" in text


def test_reader_data_operations_plan_skill_routes_away_from_adjacent_workflows() -> None:
    skill_path = REPO_ROOT / ".agents" / "skills" / "reader-data-operations-plan" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    assert "Do not use for full experiment creation" in text
    assert "reader-experiment-bootstrap" in text
    assert "reader-workbench-gardening" in text
    assert "## Success Criteria" in text


def test_repo_skill_index_lists_data_operations_plan_skill() -> None:
    skill_index_path = REPO_ROOT / ".agents" / "skills" / "README.md"
    text = skill_index_path.read_text(encoding="utf-8")
    assert "./reader-data-operations-plan/SKILL.md" in text
