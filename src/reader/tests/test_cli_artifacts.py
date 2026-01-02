from typer.testing import CliRunner

from reader.core.cli import app


def test_artifacts_requires_manifest(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "experiment:\n  outputs: ./outputs\nsteps: []\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["artifacts", str(config)])
    assert result.exit_code == 1
    assert "No outputs/manifest.json found" in result.output
