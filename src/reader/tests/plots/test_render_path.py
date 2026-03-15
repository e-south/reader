"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/plots/test_render_path.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reader.core.config import ReaderSpec
from reader.core.engine import run_spec
from reader.core.engine import runtime as engine_runtime
from reader.core.records import RecordStore
from reader.core.registry import Plugin, PluginConfig, Registry
from reader.core.workbench import PluginSemantics, resolve_workbench
from reader.tests.support import base_reader_config, write_config


class _Cfg(PluginConfig):
    pass


class _DummyPlot(Plugin):
    key = "dummy_plot"
    category = "plot"
    semantics = PluginSemantics(
        category="plot",
        domain="generic",
        family="test_plot",
        summary="Test plot plugin.",
    )
    ConfigModel = _Cfg
    render_called = False

    @classmethod
    def input_contracts(cls):
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls):
        return {"files": "none"}

    def render(self, ctx, inputs, cfg):
        _DummyPlot.render_called = True
        return []

    def run(self, ctx, inputs, cfg):
        self.render(ctx, inputs, cfg)
        return {"files": None}


def test_plot_save_calls_render(monkeypatch, tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, plots_subdir="plots", exports_subdir="exports")
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    store.persist_dataframe(
        producer_id="ingest",
        producer_uses="ingest/dummy",
        out_name="df",
        record_id="raw/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )

    cfg = base_reader_config(
        experiment_id="exp_plot",
        outputs=str(outputs),
        pipeline_steps=[],
        plot_specs=[{"id": "plot_dummy", "uses": "plot/dummy_plot", "reads": {"df": "raw/df"}}],
        plotting={"palette": None},
    )
    cfg_path = write_config(tmp_path, cfg)
    spec = ReaderSpec.load(cfg_path)

    reg = Registry()
    reg.register("plot", "dummy_plot", _DummyPlot)
    monkeypatch.setattr(engine_runtime, "load_entry_points", lambda categories=None: reg)

    plot_specs = resolve_workbench(spec).plots
    run_spec(
        spec,
        include_pipeline=False,
        include_plots=True,
        include_exports=False,
        plot_specs=plot_specs,
        log_level="ERROR",
    )
    assert _DummyPlot.render_called is True
