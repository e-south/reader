from __future__ import annotations

import pytest

from reader.contracts import builtin_contract_catalog
from reader.errors import ConfigError
from reader.protocols import ProtocolBinding, builtin_protocol_catalog
from reader.workbench import PluginSemantics
from reader.workbench.assets import (
    build_plugin_asset,
    build_workbench_asset_catalog,
    static_asset_catalog,
)
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig, Registry
from reader.workbench.templates import select_default_notebook_template


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


def test_build_workbench_asset_catalog_unifies_plugins_and_templates() -> None:
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
    assert catalog.resolve("notebook/basic", kind="template").kind == "template"


def test_select_default_notebook_template_uses_protocol_policy() -> None:
    protocols = builtin_protocol_catalog()
    assert (
        select_default_notebook_template(
            protocol=protocols.bind(ProtocolBinding(id="plate_reader/dual_reporter_screen"))
        ).template
        == "notebook/eda"
    )
    assert (
        select_default_notebook_template(protocol=protocols.bind(ProtocolBinding(id="cytometry/flow_panel"))).template
        == "notebook/cytometry"
    )
    assert (
        select_default_notebook_template(protocol=protocols.bind(ProtocolBinding(id="workbench/generic"))).template
        == "notebook/basic"
    )


def test_static_asset_catalog_only_exposes_templates() -> None:
    catalog = static_asset_catalog()
    assert [item.kind for item in catalog.all()] == ["template", "template", "template", "template", "template"]


def test_build_workbench_asset_catalog_requires_explicit_plugin_registry() -> None:
    with pytest.raises(ConfigError, match="explicit plugin registry"):
        build_workbench_asset_catalog()
