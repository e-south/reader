"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/plots/test_render_path.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from reader.contracts import builtin_contract_catalog
from reader.errors import ExecutionError
from reader.plotting.sinks import PlotFigure
from reader.plugins.plot._shared import FigurePlotPlugin, save_rendered_figures
from reader.protocols import ProtocolBinding, ProtocolSemanticProgram, builtin_protocol_catalog
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


@pytest.mark.parametrize("description", ["", "two\nlines"])
def test_plot_figure_rejects_invalid_descriptions(description: str) -> None:
    with pytest.raises(ExecutionError, match="PlotFigure.description"):
        PlotFigure(fig=object(), filename="plot", description=description)


def test_shared_plot_adapter_explains_empty_figure_selections(tmp_path: Path) -> None:
    with pytest.raises(ExecutionError, match="renderer produced no figures.*filters"):
        save_rendered_figures(
            ctx=SimpleNamespace(plots_dir=tmp_path),
            figures=[],
            plot_key="time_series",
        )

    assert list(tmp_path.iterdir()) == []


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


class _DummyExport(Plugin):
    ConfigModel = _Cfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return {"artifacts": file_bundle_output("artifacts")}

    def run(self, ctx, inputs, cfg):
        out = ctx.exports_dir / "dummy_export.csv"
        out.write_text("value\n1\n", encoding="utf-8")
        return {"artifacts": [out]}


class _DummyDescribedPlot(FigurePlotPlugin):
    ConfigModel = _Cfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    def render(self, ctx, inputs, cfg):
        del inputs, cfg
        kinetics, kinetics_ax = plt.subplots()
        kinetics_ax.plot([0.0, 1.0], [0.0, 1.0])
        summary, summary_ax = plt.subplots()
        summary_ax.bar(["control", "treated"], [1.0, 2.0])
        return [
            PlotFigure(
                fig=kinetics,
                filename="custom_kinetics",
                description="Reporter kinetics over assay time.",
            ),
            PlotFigure(
                fig=summary,
                filename="custom_summary",
                description="Endpoint summary by treatment.",
            ),
        ]


def test_plot_files_persist_protocol_figure_descriptions_and_exports_keep_plugin_descriptions(tmp_path: Path) -> None:
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
            protocol_program=ProtocolSemanticProgram(protocol="workbench/generic"),
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
                    id="generic_qc",
                    plugin="plot/dummy_plot",
                    reads={"df": RecordInputDecl(record_id="raw/df")},
                ),
                PluginStepDecl(
                    id="custom_pair",
                    plugin="plot/dummy_described",
                    reads={"df": RecordInputDecl(record_id="raw/df")},
                ),
            )
        ),
        exports=SurfaceDecl(
            specs=(
                PluginStepDecl(
                    id="export_dummy",
                    plugin="export/dummy_export",
                    reads={"df": RecordInputDecl(record_id="raw/df")},
                ),
            )
        ),
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
    reg.register(
        build_plugin_asset(
            plugin_id="plot/dummy_described",
            semantics=PluginSemantics(domain="generic", family="test_plot", summary="Render two test plot files."),
            plugin_cls=_DummyDescribedPlot,
        )
    )
    reg.register(
        build_plugin_asset(
            plugin_id="export/dummy_export",
            semantics=PluginSemantics(domain="generic", family="test_export", summary="Test export plugin."),
            plugin_cls=_DummyExport,
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
        include_exports=True,
        plot_specs=plot_specs,
        export_specs=resolve_workbench(decl).exports,
        log_level="ERROR",
        runtime=runtime,
    )
    latest = {record.record_id: record for record in store.iter_latest_records()}
    assert _DummyPlot.render_called is True
    plot_record = latest["plot:generic_qc"]
    plot_path = outputs / "plots" / "dummy_plot.pdf"
    assert plot_record.description == (
        "Quality-control measurements and diagnostics defined by the selected experiment domain."
    )
    assert plot_record.description_for(plot_path) == (
        "Quality-control measurements and diagnostics defined by the selected experiment domain."
    )
    assert latest["export:export_dummy"].description == "Test export plugin."
    custom_record = latest["plot:custom_pair"]
    assert custom_record.description == "Render two test plot files."
    assert custom_record.description_for(outputs / "plots" / "custom_kinetics.pdf") == (
        "Reporter kinetics over assay time."
    )
    assert custom_record.description_for(outputs / "plots" / "custom_summary.pdf") == ("Endpoint summary by treatment.")
    assert plot_path.exists()
    assert (outputs / "exports" / "dummy_export.csv").exists()
