from __future__ import annotations

import importlib.metadata as md

import pytest

from reader.contracts import builtin_contract_catalog
from reader.errors import RegistryError
from reader.workbench import PluginSemantics
from reader.workbench.assets import build_plugin_asset
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig, load_plugin_catalog


class _ValidatorCfg(PluginConfig):
    pass


class _ValidatorPlugin(Plugin):
    ConfigModel = _ValidatorCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return {"df": dataframe_output("df", "tidy.v1")}

    def run(self, ctx, inputs, cfg):
        raise AssertionError("not used")


class _EntryPoint:
    name = "validator/external_validator"

    def load(self):
        return build_plugin_asset(
            plugin_id="validator/external_validator",
            semantics=PluginSemantics(
                domain="generic",
                family="contract_guard",
                summary="Synthetic validator plugin for registry tests.",
            ),
            plugin_cls=_ValidatorPlugin,
        )


def test_load_plugin_catalog_registers_validator_plugins(monkeypatch) -> None:
    def _entry_points(*, group: str):
        if group == "reader.plugins":
            return [_EntryPoint()]
        return []

    monkeypatch.setattr(md, "entry_points", _entry_points)

    registry = load_plugin_catalog(contracts=builtin_contract_catalog(), categories={"validator"})

    assert registry.resolve("validator/external_validator") is _ValidatorPlugin


def test_load_plugin_catalog_rejects_bare_plugin_classes(monkeypatch) -> None:
    class _BadEntryPoint:
        name = "validator/external_validator"

        def load(self):
            return _ValidatorPlugin

    def _entry_points(*, group: str):
        if group == "reader.plugins":
            return [_BadEntryPoint()]
        return []

    monkeypatch.setattr(md, "entry_points", _entry_points)

    with pytest.raises(RegistryError, match="descriptor or descriptor factory"):
        load_plugin_catalog(contracts=builtin_contract_catalog(), categories={"validator"})


def test_plugin_semantics_reject_unknown_domain() -> None:
    with pytest.raises(ValueError, match="Unknown plugin domain 'assay'"):
        PluginSemantics(
            domain="assay",
            family="label_enrichment",
            summary="invalid",
        )
