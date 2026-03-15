from __future__ import annotations

import importlib.metadata as md

from reader.core.registry import Plugin, PluginConfig, load_entry_points
from reader.core.workbench import PluginSemantics


class _ValidatorCfg(PluginConfig):
    pass


class _ValidatorPlugin(Plugin):
    key = "external_validator"
    category = "validator"
    semantics = PluginSemantics(
        category="validator",
        domain="generic",
        family="contract_guard",
        summary="Synthetic validator plugin for registry tests.",
    )
    ConfigModel = _ValidatorCfg

    @classmethod
    def input_contracts(cls):
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls):
        return {"df": "tidy.v1"}

    def run(self, ctx, inputs, cfg):
        raise AssertionError("not used")


class _EntryPoint:
    name = "external_validator"

    def load(self):
        return _ValidatorPlugin


def test_load_entry_points_registers_validator_plugins(monkeypatch) -> None:
    def _entry_points(*, group: str):
        if group == "reader.validator":
            return [_EntryPoint()]
        return []

    monkeypatch.setattr(md, "entry_points", _entry_points)

    registry = load_entry_points(categories={"validator"})

    assert registry.resolve("validator/external_validator") is _ValidatorPlugin
