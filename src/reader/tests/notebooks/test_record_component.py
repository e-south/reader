from __future__ import annotations

from reader.workbench.notebooks.components.records import (
    build_dataframe_record_catalog,
    select_default_dataframe_record,
)


def test_dataframe_record_catalog_projects_only_public_dataframe_entries() -> None:
    entries = (
        {
            "kind": "dataframe_artifact",
            "record_id": "ratio/df",
            "created_at": "2026-07-29T12:00:00+00:00",
            "producer": {"kind": "transform", "id": "ratio", "plugin": "transform/ratio"},
            "path": "artifacts/ratio/df.parquet",
        },
        {
            "kind": "file_bundle",
            "record_id": "plot:ratio",
            "producer": {"kind": "plot", "id": "ratio", "plugin": "plot/time_series"},
        },
    )

    info, labels, note = build_dataframe_record_catalog(entries, catalog_exists=True)

    assert labels == ["ratio/df"]
    assert info == {
        "ratio/df": {
            "record_id": "ratio/df",
            "step_id": "ratio",
            "plugin_key": "transform/ratio",
            "created_at": "2026-07-29T12:00:00+00:00",
        }
    }
    assert note == ""
    assert "path" not in info["ratio/df"]


def test_default_dataframe_record_prefers_id_then_pipeline_then_catalog_time() -> None:
    info = {
        "first/df": {"record_id": "first/df", "step_id": "first", "created_at": "2026-07-29T10:00:00Z"},
        "second/df": {"record_id": "second/df", "step_id": "second", "created_at": "2026-07-29T11:00:00Z"},
    }
    labels = sorted(info)

    assert select_default_dataframe_record(info, labels, preferred_record_ids=("first/df",)) == "first/df"
    assert select_default_dataframe_record(info, labels, pipeline_step_ids=("second", "first")) == "first/df"
    assert select_default_dataframe_record(info, labels) == "second/df"
