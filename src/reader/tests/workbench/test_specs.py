from __future__ import annotations

from pathlib import Path

from reader.core.config import ReaderSpec
from reader.core.workbench import WorkbenchSpec, materialize_workbench, resolve_workbench
from reader.tests.support import base_reader_config, write_config


def _base_config() -> dict:
    return base_reader_config(
        experiment_id="exp_workbench",
        title="Workbench",
        pipeline_steps=[{"id": "ingest", "uses": "ingest/synergy_h1", "writes": {"df": "raw/df"}}],
        plot_specs=[{"id": "plot_a", "uses": "plot/time_series", "reads": {"df": "raw/df"}, "with": {"y": ["OD600"]}}],
        export_specs=[{"id": "export_a", "uses": "export/csv", "reads": {"df": "raw/df"}, "with": {"path": "a.csv"}}],
        notebook_specs=[{"id": "eda", "uses": "notebook/eda"}],
    )


def test_resolve_workbench_unifies_all_spec_kinds(tmp_path: Path) -> None:
    spec = ReaderSpec.load(write_config(tmp_path, _base_config()))
    workbench = resolve_workbench(spec)

    assert workbench.counts() == {"pipeline": 1, "plot": 1, "export": 1, "notebook": 1}
    assert all(isinstance(item, WorkbenchSpec) for item in workbench.all_specs())
    assert workbench.pipeline[0].kind == "pipeline"
    assert workbench.plots[0].kind == "plot"
    assert workbench.exports[0].kind == "export"
    assert workbench.notebooks[0].kind == "notebook"
    assert workbench.pipeline[0].uses_category == "ingest"
    assert workbench.plots[0].uses_category == "plot"
    assert workbench.exports[0].uses_category == "export"
    assert workbench.notebooks[0].uses_category == "notebook"


def test_materialize_workbench_emits_pipeline_plot_export_and_notebook_sections(tmp_path: Path) -> None:
    spec = ReaderSpec.load(write_config(tmp_path, _base_config()))
    materialized = materialize_workbench(spec)

    assert [item["id"] for item in materialized["pipeline"]] == ["ingest"]
    assert [item["id"] for item in materialized["plots"]] == ["plot_a"]
    assert [item["id"] for item in materialized["exports"]] == ["export_a"]
    assert [item["id"] for item in materialized["notebooks"]] == ["eda"]
    assert set(materialized["notebooks"][0]) == {"id", "uses", "with"}
