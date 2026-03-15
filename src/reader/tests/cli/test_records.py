"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_cli_records.py
--------------------------------------------------------------------------------
"""

import pandas as pd
from typer.testing import CliRunner

from reader.core.cli import app
from reader.core.records import RecordStore


def test_records_requires_catalog(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema: reader/v3\nexperiment:\n  id: exp\npipeline:\n  steps: []\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["records", str(config)])
    assert result.exit_code == 1
    assert "No outputs/manifests/records.json found" in result.output


def test_records_lists_dataframe_and_file_bundle_entries(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema: reader/v3\nexperiment:\n  id: exp\npipeline:\n  steps: []\n",
        encoding="utf-8",
    )
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs)
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    store.persist_dataframe(
        producer_id="ingest",
        producer_uses="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    plot_path = outputs / "plots" / "trace.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("png", encoding="utf-8")
    store.append_file_bundle(
        producer_kind="plot",
        producer_id="qc",
        producer_uses="plot/time_series",
        record_id="plot:qc",
        inputs=["ingest/df"],
        config_digest="sha256:plot",
        files=[plot_path],
    )
    runner = CliRunner()
    result = runner.invoke(app, ["records", str(config)])
    assert result.exit_code == 0
    assert "ingest/df" in result.output
    assert "plot:qc" in result.output
