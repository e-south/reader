"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_engine_explain.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from rich.console import Console

from reader.core.cli import THEME
from reader.core.config import ReaderSpec
from reader.core.contracts import OutputContractSurface
from reader.core.engine import explain
from reader.core.registry import Plugin, PluginConfig, Registry
from reader.core.workbench import PluginSemantics


class _Cfg(PluginConfig):
    pass


class _Dummy(Plugin):
    key = "dummy"
    category = "transform"
    semantics = PluginSemantics(
        category="transform",
        domain="generic",
        family="test_transform",
        summary="Test transform plugin.",
    )
    ConfigModel = _Cfg

    @classmethod
    def input_contracts(cls):
        return {}

    @classmethod
    def output_contracts(cls):
        return {"df": "none"}

    def run(self, ctx, inputs, cfg):
        raise AssertionError("not used in explain")


def test_explain_renders_without_rich_subtitle_kwargs() -> None:
    registry = Registry()
    registry.register("transform", "dummy", _Dummy)
    spec = ReaderSpec.model_validate(
        {
            "schema": "reader/v3",
            "experiment": {"id": "exp"},
            "paths": {"outputs": "/tmp/reader", "plots": "plots", "exports": "exports"},
            "pipeline": {"steps": [{"id": "step_one", "uses": "transform/dummy"}]},
            "plots": {"specs": []},
            "exports": {"specs": []},
        }
    )
    console = Console(theme=THEME, record=True, width=80)
    explain(spec, console=console, registry=registry)


class _PromotingDummy(Plugin):
    key = "promoting_dummy"
    category = "transform"
    semantics = PluginSemantics(
        category="transform",
        domain="plate_reader",
        family="test_transform",
        summary="Test runtime contract promotion plugin.",
    )
    ConfigModel = _Cfg

    @classmethod
    def input_contracts(cls):
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls):
        return {"df": "tidy.v1"}

    @classmethod
    def output_contract_surfaces(cls):
        return {
            "df": OutputContractSurface(
                minimum="tidy.v1",
                runtime_mode="promoted",
                promoted=("plate_reader.annotated.v1",),
            )
        }

    def run(self, ctx, inputs, cfg):
        raise AssertionError("not used in explain")


def test_explain_surfaces_runtime_contract_promotions() -> None:
    registry = Registry()
    registry.register("transform", "promoting_dummy", _PromotingDummy)
    spec = ReaderSpec.model_validate(
        {
            "schema": "reader/v3",
            "experiment": {"id": "exp"},
            "paths": {"outputs": "/tmp/reader", "plots": "plots", "exports": "exports"},
            "pipeline": {
                "steps": [
                    {
                        "id": "step_one",
                        "uses": "transform/promoting_dummy",
                        "reads": {"df": "input/df"},
                    }
                ]
            },
            "plots": {"specs": []},
            "exports": {"specs": []},
        }
    )
    console = Console(theme=THEME, record=True, width=160)
    explain(spec, console=console, registry=registry)
    rendered = console.export_text()
    assert "plate_reader/test_transform" in rendered
    assert "runtime may promote to" in rendered
    assert "plate_reader.annotated.v1" in rendered


def test_registry_catalog_indexes_semantic_fields() -> None:
    registry = Registry()
    registry.register("transform", "dummy", _Dummy)
    catalog = registry.catalog()

    assert [item.uses for item in catalog.filter(category="transform")] == ["transform/dummy"]
    assert [item.uses for item in catalog.filter(domain="generic")] == ["transform/dummy"]
    assert [item.uses for item in catalog.filter(family="test_transform")] == ["transform/dummy"]


def test_explain_renders_notebook_specs_without_plugin_registry() -> None:
    spec = ReaderSpec.model_validate(
        {
            "schema": "reader/v3",
            "experiment": {"id": "exp"},
            "paths": {"outputs": "/tmp/reader", "plots": "plots", "exports": "exports"},
            "pipeline": {"steps": []},
            "plots": {"specs": []},
            "exports": {"specs": []},
            "notebooks": {"specs": [{"id": "eda", "uses": "notebook/eda"}]},
        }
    )
    console = Console(theme=THEME, record=True, width=100)
    explain(spec, console=console, registry=None)
    rendered = console.export_text()
    assert "Notebooks" in rendered
    assert "notebook/eda" in rendered
    assert "record_explorer" in rendered
