"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_record_store.py

Tests for the unified workbench record catalog.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from reader.contracts import builtin_contract_catalog
from reader.errors import RecordError
from reader.workbench.graph import ProvenanceInput, RecipeSource, RecordRef
from reader.workbench.records import RecordStore, discover_dataframe_records


def test_records_catalog_invalid_json_raises(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manifests = outputs / "manifests"
    manifests.mkdir()
    (manifests / "records.json").write_text("{not json", encoding="utf-8")
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), create=False)
    with pytest.raises(RecordError):
        store.iter_latest_records()


def test_records_catalog_missing_keys_raises(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manifests = outputs / "manifests"
    manifests.mkdir()
    (manifests / "records.json").write_text(json.dumps({"latest": []}), encoding="utf-8")
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), create=False)
    with pytest.raises(RecordError):
        store.iter_latest_records()


def test_record_store_persists_dataframe_and_file_bundle(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
        source_recipe=RecipeSource(recipe="plate_reader/sample_map", with_={"channel": "OD600"}),
    )
    plot_path = outputs / "plots" / "trace.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("png", encoding="utf-8")
    bundle = store.append_file_bundle(
        producer_kind="plot",
        producer_id="qc_plot",
        producer_plugin="plot/time_series",
        record_id="plot:qc_plot",
        inputs=[ProvenanceInput(label="df", ref=RecordRef(record_id="ingest/df"))],
        config_digest="sha256:plot",
        files=[plot_path],
    )

    latest = store.iter_latest_records()
    assert {item.record_id for item in latest} == {"ingest/df", "plot:qc_plot"}
    assert record.path.exists()
    assert bundle.files == (plot_path,)
    assert len(store.record_history("ingest/df")) == 1
    assert len(store.record_history("plot:qc_plot")) == 1
    assert record.producer.source_recipe is not None
    assert record.producer.source_recipe.recipe == "plate_reader/sample_map"


def test_discover_dataframe_records_is_catalog_only_by_default(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    data_dir = outputs / "artifacts" / "ratio.transform_ratio"
    data_dir.mkdir(parents=True)
    df = pd.DataFrame({"value": [1.0]})
    df.to_parquet(data_dir / "df.parquet", index=False)

    info, labels, note, warning = discover_dataframe_records(outputs)
    assert info == {}
    assert labels == []
    assert "records.json" in note
    assert warning == ""


def test_discover_dataframe_records_can_fall_back_to_scan(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    data_dir = outputs / "artifacts" / "ratio.transform_ratio"
    data_dir.mkdir(parents=True)
    df = pd.DataFrame({"value": [1.0]})
    df.to_parquet(data_dir / "df.parquet", index=False)

    info, labels, note, warning = discover_dataframe_records(outputs, allow_scan=True)
    assert labels
    assert note == ""
    assert "scanning outputs/artifacts" in warning
    first = info[labels[0]]
    assert first["path"].name == "df.parquet"
