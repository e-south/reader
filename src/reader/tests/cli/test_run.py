from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from reader.tests.support import base_reader_config, cli_success_data, load_decl, write_config
from reader.workbench.cli import app
from reader.workbench.cli.helpers import resolve_pipeline_step_id


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def _run_config() -> dict:
    return base_reader_config(
        experiment_id="exp_run",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": False},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        annotations={
            "labels": {
                "design_id": {
                    "source": "design_id",
                    "output": "design_id_alias",
                    "values": {},
                }
            }
        },
    )


def test_run_rejects_reversed_range(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg), "--from", "labels", "--until", "ingest", "--dry-run"])
    assert result.exit_code == 1
    assert "--from 'labels' comes after --until 'ingest'" in result.output


def test_run_dry_run_json_surfaces_slice(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg), "--dry-run", "--from", "labels", "--format", "json"])
    assert result.exit_code == 0
    payload = cli_success_data(result.output)
    assert payload["dry_run"] is True
    assert payload["slice"]["from"] == "labels"
    assert payload["implementation"]["plan"]["pipeline_flow"][0] == "labels"
    assert payload["implementation"]["compiled"]["plots"] == []
    assert payload["implementation"]["compiled"]["exports"] == []


def test_run_json_requires_dry_run(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg), "--format", "json"])
    assert result.exit_code != 0
    assert "only supported with --dry-run" in _plain(result.output)


def test_run_dry_run_allows_non_active_lifecycle(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, {**_run_config(), "experiment": {"id": "exp_run", "lifecycle": "draft"}})
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg), "--dry-run"])
    assert result.exit_code == 0
    assert "DRY RUN" in _plain(result.output)


def test_run_dry_run_rejects_invalid_log_level(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())

    result = CliRunner().invoke(app, ["run", str(cfg), "--dry-run", "--log-level", "NOPE"])

    assert result.exit_code != 0
    assert "Invalid log level 'NOPE'" in _plain(result.output)


def test_run_json_dry_run_rejects_invalid_log_level_without_writes(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())

    result = CliRunner().invoke(
        app,
        ["run", str(cfg), "--dry-run", "--log-level", "LOUD", "--format", "json"],
    )

    assert result.exit_code != 0
    assert "Invalid log level 'LOUD'" in result.output
    assert not (tmp_path / "outputs").exists()


def test_run_only_shows_next_steps(tmp_path: Path, monkeypatch) -> None:
    cfg = write_config(tmp_path, _run_config())
    captured: dict[str, object] = {}

    def _fake_run_job(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("reader.workbench.engine.run_job", _fake_run_job)

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg), "--only", "ingest"])

    assert result.exit_code == 0
    kwargs = dict(captured["kwargs"])
    assert kwargs["resume_from"] == "ingest"
    assert kwargs["until"] == "ingest"
    assert kwargs["show_next_steps"] is True


def test_run_reset_records_routes_explicit_full_catalog_rebuild(tmp_path: Path, monkeypatch) -> None:
    cfg = write_config(tmp_path, _run_config())
    captured: dict[str, object] = {}

    def _fake_run_job(*args, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr("reader.workbench.engine.run_job", _fake_run_job)

    result = CliRunner().invoke(app, ["run", str(cfg), "--reset-records"])

    assert result.exit_code == 0
    assert dict(captured["kwargs"])["reset_records"] is True


@pytest.mark.parametrize(
    "extra_args, expected",
    [
        (["--dry-run"], "cannot be combined with --dry-run"),
        (["--only", "ingest"], "requires a complete run"),
        (["--from", "ingest"], "requires a complete run"),
        (["--until", "ingest"], "requires a complete run"),
    ],
)
def test_run_reset_records_rejects_dry_or_partial_execution(
    tmp_path: Path,
    extra_args: list[str],
    expected: str,
) -> None:
    cfg = write_config(tmp_path, _run_config())

    result = CliRunner().invoke(app, ["run", str(cfg), "--reset-records", *extra_args])

    assert result.exit_code != 0
    assert expected in _plain(result.output)


def test_resolve_pipeline_step_id_hint_includes_target_config(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())
    decl = load_decl(cfg)

    with pytest.raises(typer.BadParameter, match="uv run reader steps") as exc_info:
        resolve_pipeline_step_id(decl, "missing_step", job_path=cfg)

    assert str(cfg) in str(exc_info.value)


def test_read_only_commands_do_not_create_journal(tmp_path: Path) -> None:
    cfg_payload = base_reader_config(
        experiment_id="exp_read_only",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [14.0]}},
        protocol_analysis={"crosstalk_pairs": {"enabled": True, "export": True}},
        protocol_outputs={
            "plots": {"profile": "none", "include": ["raw_kinetics"]},
            "exports": {"include": ["crosstalk_pairs_table"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    cfg = write_config(tmp_path, cfg_payload)
    runner = CliRunner()

    commands = [
        ["explain", str(cfg)],
        ["validate", str(cfg), "--no-files"],
        ["run", str(cfg), "--dry-run"],
        ["plot", str(cfg), "--list"],
        ["plot", str(cfg), "--dry-run"],
        ["export", str(cfg), "--list"],
        ["export", str(cfg), "--dry-run"],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, _plain(result.output)

    assert not (tmp_path / "JOURNAL.md").exists()
    assert not (tmp_path / "journal.md").exists()
    assert not (tmp_path / "outputs").exists()


def test_failed_run_records_invocation_without_mutating_journals(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())
    journal = tmp_path / "JOURNAL.md"
    journal.write_text("# Experiment journal\n\nHuman-authored context.\n", encoding="utf-8")
    before = journal.read_bytes()

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(cfg)])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert journal.read_bytes() == before
    assert "journal.md" not in {path.name for path in tmp_path.iterdir()}

    manifests = tmp_path / "outputs" / "manifests"
    catalog = json.loads((manifests / "records.json").read_text(encoding="utf-8"))
    invocation_path = manifests / "invocations" / f"{catalog['provenance_epoch_id']}.jsonl"
    events = [json.loads(line) for line in invocation_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["attempt", "result"]
    assert events[0]["invocation_id"] == events[1]["invocation_id"]
    assert events[0]["config_digest"] == load_decl(cfg).config_digest
    assert events[0]["build_identity"]["reader_version"]
    assert events[0]["build_identity"]["source_digest"].startswith("sha256:")
    assert events[0]["operation"] == "run"
    assert events[0]["selected_step_ids"]["pipeline"]
    assert events[0]["declared_inputs"]
    assert all(str(tmp_path) not in json.dumps(item) for item in events[0]["declared_inputs"])
    assert events[1]["status"] == "failed"
    assert events[1]["exit_status"] == 1
    assert events[1]["produced_record_revisions"] == []
    assert str(tmp_path) not in events[1]["failure"]["reason"]


def test_corrupt_existing_catalog_fails_before_creating_an_orphan_invocation(tmp_path: Path) -> None:
    cfg = write_config(tmp_path, _run_config())
    manifests = tmp_path / "outputs" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "records.json").write_text("{not json", encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(cfg)])

    assert result.exit_code == 1
    assert "records.json is not valid JSON" in result.output
    assert not (manifests / "invocations").exists()
