from __future__ import annotations

from pathlib import Path

from reader.tests.support import base_reader_config, load_decl, write_config
from reader.workbench import (
    FileRef,
    NotebookTemplateCall,
    PluginStep,
    RecordRef,
    materialize_workbench,
    resolve_workbench,
)


def _base_config() -> dict:
    return base_reader_config(
        experiment_id="exp_workbench",
        title="Workbench",
        pipeline_steps=[{"id": "ingest", "plugin": "ingest/synergy_h1", "writes": {"df": "raw/df"}}],
        plot_specs=[
            {"id": "plot_a", "plugin": "plot/time_series", "reads": {"df": "raw/df"}, "with": {"y": ["OD600"]}}
        ],
        export_specs=[{"id": "export_a", "plugin": "export/csv", "reads": {"df": "raw/df"}, "with": {"path": "a.csv"}}],
        notebook_specs=[{"id": "eda", "template": "notebook/eda"}],
    )


def test_resolve_workbench_separates_plugin_steps_from_notebook_templates(tmp_path: Path) -> None:
    decl = load_decl(write_config(tmp_path, _base_config()))
    workbench = resolve_workbench(decl)

    assert workbench.counts() == {"pipeline": 1, "plot": 1, "export": 1, "notebook": 1}
    assert all(isinstance(item, PluginStep) for item in workbench.pipeline + workbench.plots + workbench.exports)
    assert all(isinstance(item, NotebookTemplateCall) for item in workbench.notebooks)
    assert workbench.pipeline[0].plugin_category == "ingest"
    assert workbench.plots[0].plugin_category == "plot"
    assert workbench.exports[0].plugin_category == "export"
    assert workbench.notebooks[0].template == "notebook/eda"


def test_materialize_workbench_emits_pipeline_plot_export_and_notebook_sections(tmp_path: Path) -> None:
    decl = load_decl(write_config(tmp_path, _base_config()))
    materialized = materialize_workbench(decl)

    assert [item["id"] for item in materialized["pipeline"]] == ["ingest"]
    assert [item["id"] for item in materialized["plots"]] == ["plot_a"]
    assert [item["id"] for item in materialized["exports"]] == ["export_a"]
    assert [item["id"] for item in materialized["notebooks"]] == ["eda"]
    assert set(materialized["notebooks"][0]) == {"id", "template"}


def test_resolve_workbench_preserves_typed_recipe_provenance(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_recipe",
        pipeline_steps=[],
        plot_specs=[],
        export_specs=[],
        notebook_specs=[],
    )
    cfg["pipeline"]["steps"] = []
    cfg["pipeline"]["recipes"] = ["plate_reader/synergy_h1"]
    decl = load_decl(write_config(tmp_path, cfg))

    workbench = resolve_workbench(decl)

    assert len(workbench.pipeline) == 1
    assert workbench.pipeline[0].source_recipe is not None
    assert workbench.pipeline[0].source_recipe.recipe == "plate_reader/synergy_h1"


def test_resolve_workbench_normalizes_input_bindings_to_typed_refs(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_graph",
        pipeline_steps=[
            {"id": "ingest", "plugin": "ingest/synergy_h1", "writes": {"df": "raw/df"}},
            {
                "id": "attach",
                "plugin": "transform/sample_map",
                "reads": {
                    "df": {"record": "raw/df"},
                    "sample_map": {"file": "./inputs/metadata.xlsx"},
                },
            },
        ],
        plot_specs=[],
        export_specs=[],
        notebook_specs=[],
    )
    decl = load_decl(write_config(tmp_path, cfg))

    attach = resolve_workbench(decl).pipeline[1]

    assert attach.reads["df"] == RecordRef(record_id="raw/df")
    assert attach.reads["sample_map"] == FileRef(path=(tmp_path / "inputs" / "metadata.xlsx").resolve())


def test_materialize_workbench_serializes_typed_refs_back_to_binding_dicts(tmp_path: Path) -> None:
    cfg = base_reader_config(
        experiment_id="exp_materialize",
        pipeline_steps=[{"id": "ingest", "plugin": "ingest/synergy_h1", "writes": {"df": "raw/df"}}],
        plot_specs=[
            {
                "id": "plot_a",
                "plugin": "plot/time_series",
                "reads": {"df": {"record": "raw/df"}},
                "with": {"y": ["OD600"]},
            }
        ],
        export_specs=[],
        notebook_specs=[],
    )
    decl = load_decl(write_config(tmp_path, cfg))

    materialized = materialize_workbench(decl)

    assert materialized["pipeline"][0]["writes"]["df"] == {"record": "raw/df"}
    assert materialized["plots"][0]["reads"]["df"] == {"record": "raw/df"}
