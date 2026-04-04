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
from reader.protocols import ProtocolBinding, builtin_protocol_catalog
from reader.runtime import ReaderRuntime
from reader.workbench import PluginSemantics, resolve_workbench
from reader.workbench.assets import AssetCatalog, build_plugin_asset
from reader.workbench.decl.model import (
    ExperimentDecl,
    NotebookDecl,
    PipelineDecl,
    PluginStepDecl,
    RecordInputDecl,
    SurfaceDecl,
    WorkbenchDecl,
)
from reader.workbench.engine import run_spec
from reader.workbench.experiment import AnnotationSemantics, ExperimentSemantics, OutputLayout, ResourceCatalog
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
        out = ctx.plots_dir / "dummy_plot.pdf"
        out.write_text("plot", encoding="utf-8")
        return [out]

    def run(self, ctx, inputs, cfg):
        self.render(ctx, inputs, cfg)
        return {"artifacts": self.render(ctx, inputs, cfg)}


def test_plot_save_calls_render(tmp_path: Path) -> None:
    _DummyPlot.render_called = False
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

    decl = WorkbenchDecl(
        experiment=ExperimentDecl(id="exp_plot", title="exp_plot", lifecycle="active", root=tmp_path),
        experiment_semantics=ExperimentSemantics(
            protocol=ProtocolBinding(id="workbench/generic"),
            annotations=AnnotationSemantics(),
            resources=ResourceCatalog(),
            layout=OutputLayout(
                outputs_dir=outputs,
                plots_subdir="plots",
                exports_subdir="exports",
                notebooks_subdir="notebooks",
            ),
        ),
        plotting_palette=None,
        pipeline=PipelineDecl(runtime={}, steps=()),
        plots=SurfaceDecl(
            specs=(
                PluginStepDecl(
                    id="plot_dummy",
                    plugin="plot/dummy_plot",
                    reads={"df": RecordInputDecl(record_id="raw/df")},
                ),
            )
        ),
        exports=SurfaceDecl(specs=()),
        notebooks=NotebookDecl(specs=()),
    )

    reg = Registry(contracts=builtin_contract_catalog())
    reg.register(
        build_plugin_asset(
            plugin_id="plot/dummy_plot",
            semantics=PluginSemantics(domain="generic", family="test_plot", summary="Test plot plugin."),
            plugin_cls=_DummyPlot,
        )
    )
    runtime = ReaderRuntime(
        contracts=builtin_contract_catalog(),
        protocols=builtin_protocol_catalog(),
        plugins=reg,
        assets=AssetCatalog([]),
    )

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
    latest_ids = {record.record_id for record in store.iter_latest_records()}
    assert _DummyPlot.render_called is True
    assert "plot:plot_dummy" in latest_ids
    assert (outputs / "plots" / "dummy_plot.pdf").exists()
