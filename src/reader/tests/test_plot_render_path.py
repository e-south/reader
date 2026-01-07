"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_plot_render_path.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from reader.core.artifacts import ArtifactStore
from reader.core.config_model import ReaderSpec
from reader.core.engine import run_spec
from reader.core.registry import Plugin, PluginConfig, Registry
from reader.core.specs import resolve_plot_specs


class _Cfg(PluginConfig):
    pass


class _DummyPlot(Plugin):
    key = "dummy_plot"
    category = "plot"
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


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_plot_save_calls_render(monkeypatch, tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    store = ArtifactStore(outputs, plots_subdir="plots", exports_subdir="exports")
    df = pd.DataFrame(
        {"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}
    )
    store.persist_dataframe(
        step_id="ingest",
        plugin_key="dummy",
        out_name="df",
        label="raw/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )

    cfg = {
        "schema": "reader/v2",
        "experiment": {"id": "exp_plot"},
        "paths": {"outputs": str(outputs), "plots": "plots", "exports": "exports"},
        "plotting": {"palette": None},
        "pipeline": {"steps": []},
        "plots": {"specs": [{"id": "plot_dummy", "uses": "plot/dummy_plot", "reads": {"df": "raw/df"}}]},
        "exports": {"specs": []},
    }
    cfg_path = _write_config(tmp_path, cfg)
    spec = ReaderSpec.load(cfg_path)

    reg = Registry()
    reg.register("plot", "dummy_plot", _DummyPlot)
    monkeypatch.setattr("reader.core.engine.load_entry_points", lambda categories=None: reg)

    plot_specs = resolve_plot_specs(spec)
    run_spec(
        spec,
        include_pipeline=False,
        include_plots=True,
        include_exports=False,
        plot_specs=plot_specs,
        log_level="ERROR",
    )
    assert _DummyPlot.render_called is True
