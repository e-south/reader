from __future__ import annotations

import pytest

from reader.core.errors import ConfigError
from reader.tests.support.configs import base_reader_config, load_models, write_config
from reader.workbench import PluginStep, resolve_workbench
from reader.workbench.engine.setup import resolve_palette_book, slice_pipeline_steps


def _steps(*ids: str) -> list[PluginStep]:
    return [PluginStep(kind="pipeline", id=step_id, plugin="transform/ratio") for step_id in ids]


def test_slice_pipeline_steps_rejects_reversed_range() -> None:
    with pytest.raises(ConfigError, match="comes after"):
        slice_pipeline_steps(_steps("a", "b", "c"), resume_from="c", until="a")


def test_resolve_palette_book_uses_shared_plot_style(tmp_path) -> None:
    cfg_path = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="exp_plot_style",
            pipeline_steps=[],
            plot_specs=[{"id": "plot", "plugin": "plot/time_series", "reads": {"df": "raw/df", "blanks": "raw/df"}}],
            plotting={"palette": "muted"},
        ),
    )
    _, decl = load_models(cfg_path)
    palette_book = resolve_palette_book(decl=decl, steps=resolve_workbench(decl).plots, dry_run=False)
    assert palette_book is not None
    assert palette_book.name == "muted"
