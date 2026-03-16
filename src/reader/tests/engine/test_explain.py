"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_engine_explain.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from rich.console import Console

from reader.contracts import OutputContractSurface, builtin_contract_catalog
from reader.tests.support import build_decl
from reader.workbench import PluginSemantics
from reader.workbench.assets import build_plugin_asset
from reader.workbench.cli import THEME
from reader.workbench.config import ReaderSpec
from reader.workbench.engine import explain
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig, Registry


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


def test_explain_renders_without_rich_subtitle_kwargs() -> None:
    registry = Registry(contracts=builtin_contract_catalog())
    registry.register(
        build_plugin_asset(
            plugin_id="transform/dummy",
            semantics=PluginSemantics(domain="generic", family="test_transform", summary="Test transform plugin."),
            plugin_cls=_Dummy,
        )
    )
    spec = ReaderSpec.model_validate(
        {
            "schema": "reader/v4",
            "experiment": {"id": "exp"},
            "paths": {"outputs": "/tmp/reader", "plots": "plots", "exports": "exports"},
            "pipeline": {"steps": [{"id": "step_one", "plugin": "transform/dummy"}]},
            "plots": {"specs": []},
            "exports": {"specs": []},
        }
    )
    console = Console(theme=THEME, record=True, width=80)
    explain(build_decl(spec), console=console, registry=registry)


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
    spec = ReaderSpec.model_validate(
        {
            "schema": "reader/v4",
            "experiment": {"id": "exp"},
            "paths": {"outputs": "/tmp/reader", "plots": "plots", "exports": "exports"},
            "pipeline": {
                "steps": [
                    {
                        "id": "step_one",
                        "plugin": "transform/promoting_dummy",
                        "reads": {"df": {"record": "input/df"}},
                    }
                ]
            },
            "plots": {"specs": []},
            "exports": {"specs": []},
        }
    )
    console = Console(theme=THEME, record=True, width=160)
    explain(build_decl(spec), console=console, registry=registry)
    rendered = console.export_text()
    assert "plate_reader/test_transform" in rendered
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


def test_explain_renders_notebook_specs_without_plugin_registry() -> None:
    spec = ReaderSpec.model_validate(
        {
            "schema": "reader/v4",
            "experiment": {"id": "exp"},
            "paths": {"outputs": "/tmp/reader", "plots": "plots", "exports": "exports"},
            "pipeline": {"steps": []},
            "plots": {"specs": []},
            "exports": {"specs": []},
            "notebooks": {"specs": [{"id": "eda", "template": "notebook/eda"}]},
        }
    )
    console = Console(theme=THEME, record=True, width=100)
    explain(build_decl(spec), console=console, registry=None)
    rendered = console.export_text()
    assert "Notebooks" in rendered
    assert "notebook/eda" in rendered
    assert "record_explorer" in rendered
