from __future__ import annotations

import pytest

from reader.contracts import builtin_contract_catalog
from reader.core.errors import ConfigError
from reader.workbench import PluginSemantics
from reader.workbench.assets import (
    build_plugin_asset,
    build_workbench_asset_catalog,
    resolve_recipe_asset,
    select_default_notebook_template,
)
from reader.workbench.decl import PluginStepDecl
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig, Registry


class _Cfg(PluginConfig):
    pass


class _DummyTransform(Plugin):
    ConfigModel = _Cfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return {"df": dataframe_output("df", "tidy.v1")}

    def run(self, ctx, inputs, cfg):
        raise AssertionError("not used")


def test_build_workbench_asset_catalog_unifies_plugins_recipes_and_templates() -> None:
    registry = Registry(contracts=builtin_contract_catalog())
    registry.register(
        build_plugin_asset(
            plugin_id="transform/dummy_asset",
            semantics=PluginSemantics(
                domain="generic",
                family="test_transform",
                summary="Synthetic transform for asset-kernel tests.",
            ),
            plugin_cls=_DummyTransform,
        )
    )

    catalog = build_workbench_asset_catalog(plugin_registry=registry)

    assert catalog.resolve("transform/dummy_asset", kind="plugin").family == "test_transform"
    assert catalog.resolve("plots/plate_reader_yfp_full", kind="recipe").kind == "recipe"
    assert catalog.resolve("notebook/basic", kind="template").kind == "template"


def test_select_default_notebook_template_uses_declared_default_rules() -> None:
    assert select_default_notebook_template(has_plots=True, has_cytometry=False).template == "notebook/eda"
    assert select_default_notebook_template(has_plots=False, has_cytometry=True).template == "notebook/cytometry"
    assert select_default_notebook_template(has_plots=False, has_cytometry=False).template == "notebook/basic"


def test_recipe_assets_store_typed_step_specs() -> None:
    descriptor = resolve_recipe_asset("plate_reader/dual_reporter_screen_base")

    assert descriptor.steps
    assert all(isinstance(step, PluginStepDecl) for step in descriptor.steps)


def test_build_workbench_asset_catalog_requires_explicit_plugin_registry() -> None:
    with pytest.raises(ConfigError, match="explicit plugin registry"):
        build_workbench_asset_catalog()
