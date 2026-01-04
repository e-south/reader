from rich.console import Console

from reader.core.cli import THEME
from reader.core.config_model import ReaderSpec
from reader.core.engine import explain
from reader.core.registry import Plugin, PluginConfig, Registry


class _Cfg(PluginConfig):
    pass


class _Dummy(Plugin):
    key = "dummy"
    category = "transform"
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
            "schema": "reader/v2",
            "experiment": {"id": "exp"},
            "paths": {"outputs": "/tmp/reader", "plots": "plots", "exports": "exports"},
            "pipeline": {"steps": [{"id": "step_one", "uses": "transform/dummy"}]},
            "plots": {"specs": []},
            "exports": {"specs": []},
        }
    )
    console = Console(theme=THEME, record=True, width=80)
    explain(spec, console=console, registry=registry)
