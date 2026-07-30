from pathlib import Path

from rich.console import Console

from reader_workbench.contracts import OutputContractSurface, builtin_contract_catalog
from reader_workbench.protocols import ProtocolBinding, ProtocolSemanticProgram
from reader_workbench.workbench import PluginSemantics
from reader_workbench.workbench.assets import build_plugin_asset
from reader_workbench.workbench.cli import THEME
from reader_workbench.workbench.decl.model import (
    ExperimentDecl,
    PipelineDecl,
    PluginStepDecl,
    RecordInputDecl,
    RecordOutputDecl,
    SurfaceDecl,
    WorkbenchDecl,
)
from reader_workbench.workbench.engine import explain
from reader_workbench.workbench.experiment import (
    AnnotationSemantics,
    ExperimentSemantics,
    OutputLayout,
    ResourceCatalog,
)
from reader_workbench.workbench.ports import dataframe_input, dataframe_output
from reader_workbench.workbench.registry import Plugin, PluginConfig, Registry


class _Cfg(PluginConfig):
    pass


class _Dummy(Plugin):
    ConfigModel = _Cfg

    @classmethod
    def input_ports(cls):
        return {}

    @classmethod
    def output_ports(cls):
        return {"df": dataframe_output("df", "tidy.v1")}

    def run(self, ctx, inputs, cfg):
        raise AssertionError("not used in explain")


def _workbench_decl(
    *,
    pipeline: tuple[PluginStepDecl, ...] = (),
) -> WorkbenchDecl:
    semantics = ExperimentSemantics(
        protocol=ProtocolBinding(id="workbench/generic"),
        protocol_program=ProtocolSemanticProgram(protocol="workbench/generic"),
        annotations=AnnotationSemantics(),
        resources=ResourceCatalog(),
        layout=OutputLayout(
            outputs_dir=Path("/tmp/reader"),
            plots_subdir="plots",
            exports_subdir="exports",
            notebooks_subdir="notebooks",
        ),
    )
    return WorkbenchDecl(
        experiment=ExperimentDecl(id="exp", title="exp", lifecycle="active", root=Path("/tmp/reader")),
        experiment_semantics=semantics,
        plotting_palette=None,
        pipeline=PipelineDecl(runtime={}, steps=pipeline),
        plots=SurfaceDecl(specs=()),
        exports=SurfaceDecl(specs=()),
    )


def test_explain_renders_without_rich_subtitle_kwargs() -> None:
    registry = Registry(contracts=builtin_contract_catalog())
    registry.register(
        build_plugin_asset(
            plugin_id="transform/dummy",
            semantics=PluginSemantics(domain="generic", family="test_transform", summary="Test transform plugin."),
            plugin_cls=_Dummy,
        )
    )
    decl = _workbench_decl(
        pipeline=(
            PluginStepDecl(
                id="step_one",
                plugin="transform/dummy",
                writes={"df": RecordOutputDecl(record_id="step_one/df")},
            ),
        )
    )
    console = Console(theme=THEME, record=True, width=80)
    explain(decl, console=console, registry=registry)


class _PromotingDummy(Plugin):
    ConfigModel = _Cfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return {
            "df": dataframe_output(
                "df",
                "tidy.v1",
                surface=OutputContractSurface(
                    minimum="tidy.v1",
                    runtime_mode="promoted",
                    promoted=("plate_reader.annotated.v1",),
                ),
            )
        }

    def run(self, ctx, inputs, cfg):
        raise AssertionError("not used in explain")


def test_explain_surfaces_runtime_contract_promotions() -> None:
    registry = Registry(contracts=builtin_contract_catalog())
    registry.register(
        build_plugin_asset(
            plugin_id="transform/promoting_dummy",
            semantics=PluginSemantics(
                domain="plate_reader",
                family="test_transform",
                summary="Test runtime contract promotion plugin.",
            ),
            plugin_cls=_PromotingDummy,
        )
    )
    decl = _workbench_decl(
        pipeline=(
            PluginStepDecl(
                id="step_one",
                plugin="transform/promoting_dummy",
                reads={"df": RecordInputDecl(record_id="input/df")},
                writes={"df": RecordOutputDecl(record_id="step_one/df")},
            ),
        )
    )
    console = Console(theme=THEME, record=True, width=160)
    explain(decl, console=console, registry=registry)
    rendered = console.export_text()
    assert "plate_reader/test_transform" in rendered
    assert "df <- input/df" in rendered
    assert "runtime may promote to" in rendered
    assert "plate_reader.annotated.v1" in rendered


def test_registry_catalog_indexes_semantic_fields() -> None:
    registry = Registry(contracts=builtin_contract_catalog())
    registry.register(
        build_plugin_asset(
            plugin_id="transform/dummy",
            semantics=PluginSemantics(domain="generic", family="test_transform", summary="Test transform plugin."),
            plugin_cls=_Dummy,
        )
    )
    catalog = registry.catalog()

    assert [item.plugin for item in catalog.filter(category="transform")] == ["transform/dummy"]
    assert [item.plugin for item in catalog.filter(domain="generic")] == ["transform/dummy"]
    assert [item.plugin for item in catalog.filter(family="test_transform")] == ["transform/dummy"]
