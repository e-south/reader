"""Tests for the unified workbench record catalog."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import pandas as pd
import pytest

from reader.contracts import builtin_contract_catalog
from reader.errors import ContractError, RecordError
from reader.workbench.graph import ProvenanceInput, RecipeSource, RecordRef
from reader.workbench.records import (
    DataFrameArtifactRecord,
    PathDescription,
    RecordProducer,
    RecordStore,
)


def test_record_producer_rejects_retired_notebook_publication_kind() -> None:
    with pytest.raises(RecordError, match="pipeline, plot, or export"):
        RecordProducer(kind="notebook", id="review", plugin="notebook/eda")  # type: ignore[arg-type]


def test_records_catalog_invalid_json_raises(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manifests = outputs / "manifests"
    manifests.mkdir()
    (manifests / "records.json").write_text("{not json", encoding="utf-8")
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), create=False)
    with pytest.raises(RecordError):
        store.iter_latest_records()


def test_records_catalog_rejects_a_symlinked_catalog_before_reading_it(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    manifests = outputs / "manifests"
    manifests.mkdir(parents=True)
    outside = tmp_path / "outside-records.json"
    outside.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "provenance_epoch_id": "2c1e4014-6217-4f10-9241-f4efb748bd75",
                "latest": {},
                "history": {},
            }
        ),
        encoding="utf-8",
    )
    try:
        (manifests / "records.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), create=False)

    with pytest.raises(RecordError, match="records.json must not be a symlink"):
        store.provenance_epoch_id()


@pytest.mark.parametrize("sink", ["outputs", "artifacts", "manifests", "plots", "exports"])
def test_record_store_rejects_symlinked_sink_roots_before_writes(tmp_path: Path, sink: str) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outputs = experiment / "outputs"
    if sink == "outputs":
        link = outputs
    else:
        outputs.mkdir()
        link = outputs / sink
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(RecordError, match=rf"{sink}.*symlink|symlink.*{sink}"):
        RecordStore(
            outputs,
            contracts=builtin_contract_catalog(),
            plots_subdir="plots",
            exports_subdir="exports",
            experiment_root=experiment,
        )

    assert list(outside.iterdir()) == []


def test_record_store_creates_normal_confined_sink_roots(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    outputs = experiment / "outputs"

    store = RecordStore(
        outputs,
        contracts=builtin_contract_catalog(),
        plots_subdir="plots",
        exports_subdir="exports",
        experiment_root=experiment,
    )

    assert store.root == outputs
    assert all(path.is_dir() for path in (store.artifacts_dir, store.manifests_dir, store.plots_dir, store.exports_dir))


def test_record_catalog_owns_a_canonical_provenance_epoch(tmp_path: Path) -> None:
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )

    initial_epoch = store.provenance_epoch_id()
    initial_path = store.invocation_ledger_path()
    store.reset_catalog()
    reset_epoch = store.provenance_epoch_id()

    assert UUID(initial_epoch).version == 4
    assert UUID(reset_epoch).version == 4
    assert reset_epoch != initial_epoch
    assert initial_path == store.manifests_dir / "invocations" / f"{initial_epoch}.jsonl"
    assert store.invocation_ledger_path() == store.manifests_dir / "invocations" / f"{reset_epoch}.jsonl"


def test_bound_record_store_rejects_catalog_reset(tmp_path: Path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    store.bind_provenance_epoch(store.provenance_epoch_id())

    with pytest.raises(RecordError, match="bound.*provenance epoch"):
        store.reset_catalog()


def test_catalog_commit_rejects_a_same_epoch_concurrent_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    first = RecordStore(outputs, contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    second = RecordStore(outputs, contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    original_write = first._write_catalog
    injected = False

    def _inject_update(payload, **kwargs):
        nonlocal injected
        if not injected and kwargs.get("expected_provenance_epoch_id") is not None:
            injected = True
            second.persist_dataframe(
                producer_id="second",
                producer_plugin="ingest/synergy_h1",
                out_name="df",
                record_id="second/df",
                df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [2.0]}),
                contract_id="tidy.v1",
                inputs=[],
                config_digest="sha256:second",
            )
        return original_write(payload, **kwargs)

    monkeypatch.setattr(first, "_write_catalog", _inject_update)

    with pytest.raises(RecordError, match="changed concurrently"):
        first.persist_dataframe(
            producer_id="first",
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id="first/df",
            df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
            contract_id="tidy.v1",
            inputs=[],
            config_digest="sha256:first",
        )

    assert second.latest_record("second/df") is not None
    assert second.latest_record("first/df") is None


def test_idempotent_dataframe_write_rejects_a_stale_same_epoch_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    first = RecordStore(outputs, contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    second = RecordStore(outputs, contracts=builtin_contract_catalog(), experiment_root=tmp_path)
    original = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    first.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=original,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    original_verify = DataFrameArtifactRecord.verify_content_digest
    injected = False

    def _inject_update(record: DataFrameArtifactRecord) -> None:
        nonlocal injected
        original_verify(record)
        if not injected:
            injected = True
            second.persist_dataframe(
                producer_id="ingest",
                producer_plugin="ingest/synergy_h1",
                out_name="df",
                record_id="ingest/df",
                df=original.assign(value=[2.0]),
                contract_id="tidy.v1",
                inputs=[],
                config_digest="sha256:test",
            )

    monkeypatch.setattr(DataFrameArtifactRecord, "verify_content_digest", _inject_update)

    with pytest.raises(RecordError, match="changed concurrently"):
        first.persist_dataframe(
            producer_id="ingest",
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id="ingest/df",
            df=original,
            contract_id="tidy.v1",
            inputs=[],
            config_digest="sha256:test",
        )

    assert second.latest_dataframe("ingest/df").load_dataframe()["value"].tolist() == [2.0]


def test_record_store_normalizes_macos_var_alias_consistently() -> None:
    with TemporaryDirectory() as temporary_directory:
        experiment = Path(temporary_directory) / "experiment"
        if experiment.absolute() == experiment.resolve(strict=False):
            pytest.skip("temporary directory does not use a filesystem alias")
        experiment.mkdir()

        store = RecordStore(
            experiment / "outputs",
            contracts=builtin_contract_catalog(),
            experiment_root=experiment,
        )

        assert store.root.is_dir()
        assert store.root.resolve(strict=True).is_relative_to(experiment.resolve(strict=True))


def test_records_catalog_missing_keys_raises(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manifests = outputs / "manifests"
    manifests.mkdir()
    (manifests / "records.json").write_text(json.dumps({"latest": []}), encoding="utf-8")
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), create=False)
    with pytest.raises(RecordError):
        store.iter_latest_records()


def test_records_catalog_unknown_schema_version_raises(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manifests = outputs / "manifests"
    manifests.mkdir()
    (manifests / "records.json").write_text(
        json.dumps({"schema_version": 999, "latest": {}, "history": {}}),
        encoding="utf-8",
    )
    store = RecordStore(outputs, contracts=builtin_contract_catalog(), create=False)
    with pytest.raises(RecordError, match="schema_version must be 4"):
        store.iter_latest_records()


def test_record_payload_schema_v5_requires_full_regeneration(tmp_path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    catalog["latest"]["ingest/df"]["schema_version"] = 5
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(
        RecordError,
        match=r"record payload schema_version must be 6 \(got 5\).*--reset-records",
    ):
        store.read_record("ingest/df")


@pytest.mark.parametrize("schema_version", [None, "5", 3, 4, 5, 7, []])
def test_file_bundle_record_rejects_noncurrent_or_malformed_schema_version(tmp_path, schema_version) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    payload = {
        "schema_version": schema_version,
        "record_id": "plot:qc",
        "kind": "file_bundle",
    }
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    catalog["latest"] = {"plot:qc": payload}
    catalog["history"] = {"plot:qc": [payload]}
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="record payload schema_version must be 6"):
        store.read_record("plot:qc")


def test_record_payload_rejects_ambiguous_provenance_input(tmp_path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    catalog["latest"]["ingest/df"]["inputs"] = [
        {"label": "raw", "record": "upstream/df", "file": "inputs/raw.xlsx", "extra": "ignored"}
    ]
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="invalid provenance input.*unknown kind"):
        store.read_record("ingest/df")


@pytest.mark.parametrize("catalog_section", ["latest", "history"])
@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        ("unknown", "unknown=unexpected"),
        ("missing_recipe", "missing=recipe"),
        ("missing_with", "missing=with"),
    ],
)
def test_schema_v6_record_rejects_non_exact_source_recipe_fields(
    tmp_path,
    catalog_section: str,
    mutation: str,
    expected_detail: str,
) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
        source_recipe=RecipeSource(recipe="plate_reader/synergy_h1", with_={"channels": ["OD600"]}),
    )
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [2.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
        source_recipe=RecipeSource(recipe="plate_reader/synergy_h1", with_={"channels": ["OD600"]}),
    )
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    payloads = [catalog["history"]["ingest/df"][0]]
    if catalog_section == "latest":
        payloads = [catalog["latest"]["ingest/df"], catalog["history"]["ingest/df"][-1]]
    for payload in payloads:
        source_recipe = payload["producer"]["source_recipe"]
        if mutation == "unknown":
            source_recipe["unexpected"] = True
        else:
            source_recipe.pop(mutation.removeprefix("missing_"))
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(
        RecordError,
        match=rf"producer\.source_recipe has unknown or missing fields: {expected_detail}",
    ):
        if catalog_section == "latest":
            store.read_record("ingest/df")
        else:
            store.record_history("ingest/df")


def test_schema_v5_record_payload_rejects_unknown_fields(tmp_path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    catalog["latest"]["ingest/df"]["unexpected"] = True
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="unknown or missing fields.*unexpected"):
        store.read_record("ingest/df")


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
        source_recipe=RecipeSource(recipe="plate_reader/synergy_h1", with_={"channels": ["OD600"]}),
    )
    plot_path = outputs / "plots" / "trace.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("png", encoding="utf-8")
    bundle = store.append_file_bundle(
        producer_kind="plot",
        producer_id="qc_plot",
        producer_plugin="plot/time_series",
        record_id="plot:qc_plot",
        inputs=store.capture_inputs([ProvenanceInput(label="df", ref=RecordRef(record_id="ingest/df"))]),
        config_digest="sha256:plot",
        files=[plot_path],
        description="Render grouped time-series plots from tidy plate-reader traces.",
        path_descriptions=(
            PathDescription(
                path=plot_path,
                description="Grouped time-series traces for the configured channels and treatment groups.",
            ),
        ),
    )

    latest = store.iter_latest_records()
    assert {item.record_id for item in latest} == {"ingest/df", "plot:qc_plot"}
    assert record.path.exists()
    assert bundle.files == (plot_path,)
    assert bundle.description == "Render grouped time-series plots from tidy plate-reader traces."
    assert bundle.description_for(plot_path) == (
        "Grouped time-series traces for the configured channels and treatment groups."
    )
    persisted = json.loads(store.records_path.read_text(encoding="utf-8"))
    assert persisted["latest"]["plot:qc_plot"]["schema_version"] == 6
    assert persisted["latest"]["plot:qc_plot"]["file_evidence"] == [
        {
            "content_digest": "sha256:8f8cbb7dcf46e0bc7d53265749a6c17d116093a6ba95e442764060c76fd4a86c",
            "path": "plots/trace.png",
            "size_bytes": 3,
        }
    ]
    assert persisted["latest"]["plot:qc_plot"]["description"] == bundle.description
    assert persisted["latest"]["plot:qc_plot"]["path_descriptions"] == [
        {
            "description": "Grouped time-series traces for the configured channels and treatment groups.",
            "path": "plots/trace.png",
        }
    ]
    restored_bundle = store.read_record("plot:qc_plot")
    assert restored_bundle.description == bundle.description
    assert restored_bundle.description_for(plot_path) == bundle.description_for(plot_path)
    assert len(store.record_history("ingest/df")) == 1
    assert len(store.record_history("plot:qc_plot")) == 1
    assert store.revision_counts(["ingest/df", "plot:qc_plot"]) == {"ingest/df": 1, "plot:qc_plot": 1}
    assert record.producer.source_recipe is not None
    assert record.producer.source_recipe.recipe == "plate_reader/synergy_h1"


def test_file_bundle_rejects_dataframe_record_id_without_mutating_catalog(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    dataframe_record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="shared/record",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:dataframe",
    )
    export_path = outputs / "exports" / "summary.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text("value\n1\n", encoding="utf-8")
    catalog_before = store.records_path.read_bytes()

    with pytest.raises(RecordError, match="already used by a dataframe record"):
        store.append_file_bundle(
            producer_kind="export",
            producer_id="summary",
            producer_plugin="export/csv",
            record_id="shared/record",
            inputs=[],
            config_digest="sha256:bundle",
            files=[export_path],
            description="Summary export.",
        )

    assert store.records_path.read_bytes() == catalog_before
    assert store.read_record("shared/record") == dataframe_record
    assert store.record_history("shared/record") == (dataframe_record,)


@pytest.mark.parametrize("description", ["   ", None])
def test_file_bundle_description_rejects_missing_or_blank_text(tmp_path, description) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())

    with pytest.raises(RecordError, match="description must be a non-empty string"):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[outputs / "plots" / "trace.png"],
            description=description,
        )


def test_plot_file_bundle_rejects_missing_path_descriptions(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())

    with pytest.raises(RecordError, match="plot file bundles must describe every file"):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[outputs / "plots" / "trace.png"],
            description="Render grouped time-series plots from tidy plate-reader traces.",
        )


def test_file_bundle_rejects_empty_file_list(tmp_path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())

    with pytest.raises(RecordError, match="must contain at least one file"):
        store.append_file_bundle(
            producer_kind="export",
            producer_id="summary",
            producer_plugin="export/xlsx",
            record_id="export:summary",
            inputs=[],
            config_digest="sha256:export",
            files=[],
            description="Workbook export.",
        )


def test_path_description_rejects_non_pathlike_values() -> None:
    with pytest.raises(RecordError, match="path must be path-like"):
        PathDescription(path=None, description="Plot description.")  # type: ignore[arg-type]


def test_file_bundle_rejects_untyped_path_description_entries(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())

    with pytest.raises(RecordError, match="entries must be PathDescription"):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[outputs / "plots" / "trace.png"],
            description="Render grouped time-series plots from tidy plate-reader traces.",
            path_descriptions=(object(),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("path_state", "message"),
    [
        ("missing", "missing files"),
        ("directory", "non-file paths"),
    ],
)
def test_file_bundle_rejects_paths_that_are_not_existing_files(tmp_path, path_state, message) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    plot_path = outputs / "plots" / "trace.png"
    if path_state == "directory":
        plot_path.mkdir(parents=True)

    with pytest.raises(RecordError, match=message):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[plot_path],
            description="Render grouped time-series plots from tidy plate-reader traces.",
            path_descriptions=(
                PathDescription(path=plot_path, description="Grouped time-series traces for the configured channels."),
            ),
        )

    assert store.iter_latest_records() == ()


@pytest.mark.parametrize(
    ("descriptions", "message"),
    [
        (
            (
                ("plots/trace.png", "First description."),
                ("plots/trace.png", "Second description."),
            ),
            "duplicate path descriptions",
        ),
        (
            (("plots/other.png", "Description for a different file."),),
            "unmatched path descriptions",
        ),
        (
            (("plots/trace.png", "Description for only one file."),),
            "missing path descriptions",
        ),
    ],
)
def test_plot_file_bundle_rejects_invalid_path_description_coverage(tmp_path, descriptions, message) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    files = [outputs / "plots" / "trace.png"]
    if "missing" in message:
        files.append(outputs / "plots" / "summary.png")

    with pytest.raises(RecordError, match=message):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=files,
            description="Render grouped time-series plots from tidy plate-reader traces.",
            path_descriptions=tuple(
                PathDescription(path=outputs / path, description=description) for path, description in descriptions
            ),
        )


def test_dataframe_record_load_rejects_content_digest_mismatch(tmp_path) -> None:
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
    )
    df.assign(value=[2.0]).to_parquet(record.path, index=False)

    with pytest.raises(RecordError, match="content digest mismatch"):
        record.load_dataframe()


def test_dataframe_record_parses_the_exact_verified_buffer_when_path_changes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    original = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    replacement = original.assign(value=[9.0])
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=original,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    original_read_parquet = pd.read_parquet

    def _replace_path_then_parse(source, *args, **kwargs):
        replacement.to_parquet(record.path, index=False)
        assert isinstance(source, BytesIO)
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _replace_path_then_parse)

    loaded = record.load_dataframe()

    pd.testing.assert_frame_equal(loaded, original)
    pd.testing.assert_frame_equal(original_read_parquet(record.path), replacement)


def test_same_config_changed_dataframe_uses_immutable_revision_paths(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    first_df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})

    first = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=first_df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:same-config",
    )
    second = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=first_df.assign(value=[2.0]),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:same-config",
    )

    history = store.record_history("ingest/df")
    assert len(history) == 2
    assert first.path != second.path
    assert [record.path for record in history] == [first.path, second.path]
    pd.testing.assert_frame_equal(history[0].load_dataframe(), first_df)
    pd.testing.assert_frame_equal(history[1].load_dataframe(), first_df.assign(value=[2.0]))


def test_identical_dataframe_revision_is_idempotent(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    persist = {
        "producer_id": "ingest",
        "producer_plugin": "ingest/synergy_h1",
        "out_name": "df",
        "record_id": "ingest/df",
        "df": df,
        "contract_id": "tidy.v1",
        "inputs": [],
        "config_digest": "sha256:same-config",
    }

    first = store.persist_dataframe(**persist)
    second = store.persist_dataframe(**persist)

    assert second == first
    assert store.record_history("ingest/df") == (first,)
    assert sorted(path.name for path in store.artifacts_dir.iterdir()) == ["ingest.ingest_synergy_h1"]


def test_dataframe_digesting_does_not_read_entire_artifact(tmp_path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})

    def reject_read_bytes(_path: Path) -> bytes:
        raise AssertionError("artifact hashing must stream file content")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    record = store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )

    record.verify_content_digest()


def test_catalog_replace_failure_preserves_catalog_and_cleans_staging(tmp_path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    original_catalog = store.records_path.read_text(encoding="utf-8")
    plot_path = outputs / "plots" / "trace.png"
    plot_path.write_text("png", encoding="utf-8")
    original_replace = Path.replace

    def fail_catalog_replace(source: Path, target: Path) -> Path:
        if target == store.records_path:
            raise OSError("injected catalog replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_catalog_replace)

    with pytest.raises(RecordError, match="atomically replace records.json"):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[plot_path],
            description="Render grouped time-series plots from tidy plate-reader traces.",
            path_descriptions=(
                PathDescription(path=plot_path, description="Grouped time-series traces for configured channels."),
            ),
        )

    assert store.records_path.read_text(encoding="utf-8") == original_catalog
    assert list(store.manifests_dir.glob(".records.json.*.tmp")) == []


def test_dataframe_catalog_failure_removes_unpublished_revision(tmp_path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    original_replace = Path.replace

    def fail_catalog_replace(source: Path, target: Path) -> Path:
        if target == store.records_path:
            raise OSError("injected catalog replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_catalog_replace)

    with pytest.raises(RecordError, match="atomically replace records.json"):
        store.persist_dataframe(
            producer_id="ingest",
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id="ingest/df",
            df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
            contract_id="tidy.v1",
            inputs=[],
            config_digest="sha256:test",
        )

    assert store.record_history("ingest/df") == ()
    assert list(store.artifacts_dir.iterdir()) == []
    assert list(store.manifests_dir.glob(".records.json.*.tmp")) == []


def test_dataframe_contract_validation_cannot_be_disabled(tmp_path) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())

    with pytest.raises(ContractError):
        store.persist_dataframe(
            producer_id="ingest",
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id="ingest/df",
            df=pd.DataFrame({"value": [1.0]}),
            contract_id="tidy.v1",
            inputs=[],
            config_digest="sha256:test",
        )

    assert store.record_history("ingest/df") == ()
    assert list(store.artifacts_dir.iterdir()) == []


@pytest.mark.parametrize("contract_id", ["none", "files"])
def test_dataframe_record_rejects_reserved_contract_ids(tmp_path, contract_id: str) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())

    with pytest.raises(ContractError, match="unknown contract id"):
        store.persist_dataframe(
            producer_id="ingest",
            producer_plugin="ingest/synergy_h1",
            out_name="df",
            record_id="ingest/df",
            df=pd.DataFrame({"value": [1.0]}),
            contract_id=contract_id,
            inputs=[],
            config_digest="sha256:test",
        )

    assert store.record_history("ingest/df") == ()
    assert list(store.artifacts_dir.iterdir()) == []


@pytest.mark.parametrize("raw_path", ["../outside.parquet", "/tmp/outside.parquet"])
def test_catalog_rejects_dataframe_paths_outside_outputs(tmp_path, raw_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    payload = catalog["latest"]["ingest/df"]
    payload["path"] = raw_path
    catalog["history"]["ingest/df"] = [payload]
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="must resolve within the outputs directory|must be relative"):
        store.read_dataframe("ingest/df")


@pytest.mark.parametrize("raw_path", ["../outside.png", "/tmp/outside.png"])
def test_catalog_rejects_unconfined_file_bundle_paths(tmp_path, raw_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    plot_path = outputs / "plots" / "trace.png"
    plot_path.write_text("png", encoding="utf-8")
    store.append_file_bundle(
        producer_kind="plot",
        producer_id="qc_plot",
        producer_plugin="plot/time_series",
        record_id="plot:qc_plot",
        inputs=[],
        config_digest="sha256:test",
        files=[plot_path],
        description="Time-series plot.",
        path_descriptions=(PathDescription(path=plot_path, description="Time-series plot."),),
    )
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    payload = catalog["latest"]["plot:qc_plot"]
    payload["files"] = [raw_path]
    payload["path_descriptions"] = [{"path": raw_path, "description": "Time-series plot."}]
    catalog["history"]["plot:qc_plot"] = [payload]
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="must resolve within the outputs directory|must be relative"):
        store.read_record("plot:qc_plot")


def test_file_bundle_rejects_path_that_resolves_outside_outputs(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_plot = outside / "trace.png"
    outside_plot.write_text("png", encoding="utf-8")
    plots_link = outputs / "plots" / "external"
    plots_link.symlink_to(outside, target_is_directory=True)
    linked_plot = plots_link / "trace.png"

    with pytest.raises(RecordError, match="must resolve within the outputs directory"):
        store.append_file_bundle(
            producer_kind="plot",
            producer_id="qc_plot",
            producer_plugin="plot/time_series",
            record_id="plot:qc_plot",
            inputs=[],
            config_digest="sha256:plot",
            files=[linked_plot],
            description="Render grouped time-series plots from tidy plate-reader traces.",
            path_descriptions=(
                PathDescription(path=linked_plot, description="Grouped time-series traces for configured channels."),
            ),
        )

    assert store.iter_latest_records() == ()


def test_record_store_revision_counts_bulk_read(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    store = RecordStore(outputs, contracts=builtin_contract_catalog())
    df = pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]})
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df,
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:first",
    )
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=df.assign(value=[2.0]),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:second",
    )

    counts = store.revision_counts(["ingest/df", "missing/df"])

    assert counts == {"ingest/df": 2, "missing/df": 0}


@pytest.mark.parametrize("corruption", ["empty_history", "divergent_final"])
def test_revision_counts_rejects_malformed_latest_history_lineage(tmp_path, corruption) -> None:
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    store.persist_dataframe(
        producer_id="ingest",
        producer_plugin="ingest/synergy_h1",
        out_name="df",
        record_id="ingest/df",
        df=pd.DataFrame({"position": ["A1"], "time": [0.0], "channel": ["OD600"], "value": [1.0]}),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:test",
    )
    catalog = json.loads(store.records_path.read_text(encoding="utf-8"))
    if corruption == "empty_history":
        catalog["history"]["ingest/df"] = []
    else:
        catalog["history"]["ingest/df"][-1]["config_digest"] = "sha256:divergent-history"
    store.records_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(RecordError, match="history"):
        store.revision_counts(["ingest/df"])
