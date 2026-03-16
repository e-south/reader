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

from reader.contracts import builtin_contract_catalog
from reader.runtime import ReaderRuntime
from reader.tests.support import base_reader_config, load_decl, write_config
from reader.workbench import PluginSemantics, resolve_workbench
from reader.workbench.assets import AssetCatalog, build_plugin_asset
from reader.workbench.engine import run_spec
from reader.workbench.ports import dataframe_input, file_bundle_output
from reader.workbench.records import RecordStore
from reader.workbench.registry import Plugin, PluginConfig, Registry


class _Cfg(PluginConfig):
    pass


class _DummyPlot(Plugin):
    ConfigModel = _Cfg
    render_called = False

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return {"artifacts": file_bundle_output("artifacts")}

    def render(self, ctx, inputs, cfg):
        _DummyPlot.render_called = True
        return []

    def run(self, ctx, inputs, cfg):
        self.render(ctx, inputs, cfg)
        return {"artifacts": []}


def test_plot_save_calls_render(monkeypatch, tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(
        outputs,
        contracts=builtin_contract_catalog(),
        plots_subdir="plots",
        exports_subdir="exports",
    )
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/dummy",
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
        plot_specs=[{"id": "plot_dummy", "plugin": "plot/dummy_plot", "reads": {"df": "raw/df"}}],
        plotting={"palette": None},
    )
    cfg_path = write_config(tmp_path, cfg)
    decl = load_decl(cfg_path)

    reg = Registry(contracts=builtin_contract_catalog())
    reg.register(
        build_plugin_asset(
            plugin_id="plot/dummy_plot",
            semantics=PluginSemantics(domain="generic", family="test_plot", summary="Test plot plugin."),
            plugin_cls=_DummyPlot,
        )
    )
    runtime = ReaderRuntime(contracts=builtin_contract_catalog(), plugins=reg, assets=AssetCatalog([]))

    plot_specs = resolve_workbench(decl).plots
    run_spec(
        decl,
        include_pipeline=False,
        include_plots=True,
        include_exports=False,
        plot_specs=plot_specs,
        log_level="ERROR",
        runtime=runtime,
    )
    assert _DummyPlot.render_called is True
