from __future__ import annotations

from pathlib import Path

from reader.tests.support import base_reader_config, load_decl, write_config
from reader.workbench import (
    PluginStep,
    RecordRef,
    ResourceRef,
    materialize_workbench,
    resolve_workbench,
)


def _base_config() -> dict:
    return base_reader_config(
        experiment_id="exp_workbench",
        title="Workbench",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_inputs={"fold_change": {"report_times": [14.0]}},
        protocol_analysis={
            "include_fold_change": True,
            "crosstalk_pairs": {"enabled": True, "export": True},
        },
        protocol_outputs={
            "plots": {"profile": "none", "include": ["raw_kinetics"]},
            "exports": {"include": ["crosstalk_pairs_table"]},
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )


def test_resolve_workbench_contains_only_executable_plugin_steps(tmp_path: Path) -> None:
    decl = load_decl(write_config(tmp_path, _base_config()))
    workbench = resolve_workbench(decl)

    assert workbench.counts() == {"pipeline": 10, "plot": 1, "export": 1}
    assert all(isinstance(item, PluginStep) for item in workbench.pipeline + workbench.plots + workbench.exports)
    assert workbench.pipeline[0].plugin_category == "ingest"
    assert workbench.plots[0].id == "raw_kinetics"
    assert workbench.plots[0].plugin_category == "plot"
    assert workbench.exports[0].id == "crosstalk_pairs_table"
    assert workbench.exports[0].plugin_category == "export"


def test_materialize_workbench_emits_only_executable_sections(tmp_path: Path) -> None:
    decl = load_decl(write_config(tmp_path, _base_config()))
    materialized = materialize_workbench(decl)

    assert materialized["pipeline"][0]["id"] == "ingest"
    assert [item["id"] for item in materialized["plots"]] == ["raw_kinetics"]
    assert [item["id"] for item in materialized["exports"]] == ["crosstalk_pairs_table"]
    assert set(materialized) == {"pipeline", "plots", "exports"}


def test_resolve_workbench_preserves_typed_recipe_provenance(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_recipe",
        protocol_id="plate_reader/dual_reporter_screen",
        protocol_analysis={"include_fold_change": False},
        protocol_outputs={"plots": {"profile": "none"}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    decl = load_decl(write_config(tmp_path, cfg))

    workbench = resolve_workbench(decl)

    assert len(workbench.pipeline) >= 1
    assert workbench.pipeline[0].source_recipe is not None
    assert workbench.pipeline[0].source_recipe.recipe == "plate_reader/synergy_h1"


def test_single_reporter_workbench_preserves_base_recipe_provenance(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_single_recipe",
        protocol_id="plate_reader/single_reporter_screen",
        protocol_analysis={
            "reporter_channel": "mCherry",
            "normalizer_channel": "OD700",
            "include_fold_change": False,
        },
        protocol_outputs={"plots": {"profile": "none"}},
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    decl = load_decl(write_config(tmp_path, cfg))

    workbench = resolve_workbench(decl)
    ratio_step = next(step for step in workbench.pipeline if step.id == "ratio_reporter_normalizer")

    assert ratio_step.source_recipe is not None
    assert ratio_step.source_recipe.recipe == "plate_reader/single_reporter_screen_base"
    assert ratio_step.source_recipe.with_ == {
        "reporter_channel": "mCherry",
        "normalizer_channel": "OD700",
    }


def test_resolve_workbench_normalizes_compiled_bindings_to_typed_refs(tmp_path: Path) -> None:
    decl = load_decl(write_config(tmp_path, _base_config()))
    workbench = resolve_workbench(decl)

    merge_map = next(step for step in workbench.pipeline if step.id == "merge_map")
    raw_kinetics = workbench.plots[0]

    assert merge_map.reads["df"] == RecordRef(record_id="ingest/df")
    assert merge_map.reads["sample_map"] == ResourceRef(
        resource_id="sample_map",
        path=(tmp_path / "inputs" / "metadata.xlsx").resolve(),
    )
    assert raw_kinetics.reads["df"] == RecordRef(record_id="ratio_yfp_od600/df")
    assert set(raw_kinetics.reads) == {"df"}


def test_materialize_workbench_serializes_typed_refs_back_to_binding_dicts(tmp_path: Path) -> None:
    decl = load_decl(write_config(tmp_path, _base_config()))

    materialized = materialize_workbench(decl)

    pipeline = {item["id"]: item for item in materialized["pipeline"]}
    plots = {item["id"]: item for item in materialized["plots"]}
    assert pipeline["merge_map"]["reads"]["sample_map"]["resource"] == "sample_map"
    assert pipeline["merge_map"]["reads"]["sample_map"]["path"].endswith("/inputs/metadata.xlsx")
    assert plots["raw_kinetics"]["reads"]["df"] == {"record": "ratio_yfp_od600/df"}
