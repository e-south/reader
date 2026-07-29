from __future__ import annotations

from dataclasses import replace

import pytest

from reader.errors import ConfigError
from reader.tests.support.configs import base_reader_config, load_models, write_config
from reader.workbench import PluginStep, resolve_workbench
from reader.workbench.engine.runtime import run_spec
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
            protocol_id="plate_reader/dual_reporter_screen",
            protocol_analysis={"include_fold_change": False},
            protocol_outputs={"plots": {"profile": "none", "include": ["raw_kinetics"]}},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
            plotting={"palette": "muted"},
        ),
    )
    _, decl = load_models(cfg_path)
    palette_book = resolve_palette_book(decl=decl, steps=resolve_workbench(decl).plots, dry_run=False)
    assert palette_book is not None
    assert palette_book.name == "muted"


def test_resolve_palette_book_reports_non_dependency_import_failure(monkeypatch, tmp_path) -> None:
    cfg_path = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="exp_plot_style",
            protocol_id="plate_reader/dual_reporter_screen",
            protocol_analysis={"include_fold_change": False},
            protocol_outputs={"plots": {"profile": "none", "include": ["raw_kinetics"]}},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
            plotting={"palette": "muted"},
        ),
    )
    _, decl = load_models(cfg_path)

    def _boom(name: str):
        raise RuntimeError("broken palette module")

    monkeypatch.setattr("reader.workbench.engine.setup.import_module", _boom)

    with pytest.raises(ConfigError, match="Failed to initialize plot palette support: broken palette module"):
        resolve_palette_book(decl=decl, steps=resolve_workbench(decl).plots, dry_run=False)


def test_run_spec_dry_run_validates_effective_plot_overrides(tmp_path) -> None:
    cfg_path = write_config(
        tmp_path,
        base_reader_config(
            experiment_id="exp_selected_plot",
            protocol_id="plate_reader/single_reporter_screen",
            protocol_inputs={"fold_change": {"report_times": [14.0]}},
            protocol_analysis={"reporter_channel": "RFP", "normalizer_channel": "OD600"},
            protocol_outputs={"plots": {"profile": "none", "include": ["subject_comparison"]}},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        ),
    )
    _, decl = load_models(cfg_path)
    selected = resolve_workbench(decl).plots[0]
    invalid = replace(selected, with_={**selected.with_, "order_hue_ref": "missing_order"})

    with pytest.raises(ConfigError, match="missing_order"):
        run_spec(
            decl,
            dry_run=True,
            include_pipeline=False,
            include_plots=True,
            include_exports=False,
            plot_specs=[invalid],
        )
