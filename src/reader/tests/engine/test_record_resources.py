from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console

from reader.contracts import builtin_contract_catalog
from reader.errors import ConfigError, RecordError
from reader.protocols.builtins import builtin_protocol_catalog
from reader.tests.support import base_reader_config, write_config
from reader.workbench.decl import load_workbench_decl
from reader.workbench.engine import run_spec
from reader.workbench.engine.invocations import capture_revision_snapshot
from reader.workbench.experiments import ExperimentCatalog
from reader.workbench.graph import ProvenanceInput, SourceRecordRef
from reader.workbench.records import RecordStore, SourceRecordCollection, resolve_source_record, verify_record_store


def _vec8(prefix: str, *, shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "design_id": [f"{prefix}-1", f"{prefix}-2"],
            "sequence": ["ACGT", "TGCA"],
            "id": [f"{prefix}-seq-1", f"{prefix}-seq-2"],
            "time_selected_h": [18.0, 18.0],
            "reference_design_id": ["reference", "reference"],
            "intensity_log2_offset_delta": [0.0, 0.0],
            "r_logic": [4.0, 5.0],
            "v00": [0.0, 0.1],
            "v10": [1.0, 0.9],
            "v01": [0.2, 0.3],
            "v11": [1.0 + shift, 0.5 + shift],
            "y00_star": [0.0, 0.1],
            "y10_star": [1.0, 0.8],
            "y01_star": [0.2, 0.3],
            "y11_star": [1.4, 0.7],
            "flat_logic": [False, False],
        }
    )


def _source(workspace: Path, *, experiment_id: str, prefix: str) -> tuple[Path, RecordStore]:
    root = workspace / "experiments" / "2026" / experiment_id
    root.mkdir(parents=True)
    config = write_config(
        root,
        base_reader_config(experiment_id=experiment_id, protocol_id="workbench/generic"),
    )
    store = RecordStore(root / "outputs", contracts=builtin_contract_catalog(), experiment_root=root)
    store.persist_dataframe(
        producer_id="vec8",
        producer_plugin="transform/sfxi",
        out_name="vec8",
        record_id="vec8/df",
        df=_vec8(prefix),
        contract_id="sfxi.vec8.v3",
        inputs=(),
        config_digest=f"sha256:{experiment_id}",
    )
    return config, store


def _aggregate_config(workspace: Path) -> Path:
    root = workspace / "experiments" / "aggregates" / "vec8-review"
    root.mkdir(parents=True)
    return write_config(
        root,
        base_reader_config(
            experiment_id="vec8-review",
            protocol_id="logic/sfxi_vec8_collection",
            protocol_inputs={"record_resources": ["first", "second"]},
            resources={
                "first": {"kind": "record", "experiment": "source-a", "record": "vec8/df"},
                "second": {"kind": "record", "experiment": "source-b", "record": "vec8/df"},
            },
        ),
    )


def test_record_collection_runs_through_engine_record_store_and_verifier(tmp_path: Path) -> None:
    _, first_store = _source(tmp_path, experiment_id="source-a", prefix="a")
    _source(tmp_path, experiment_id="source-b", prefix="b")
    config = _aggregate_config(tmp_path)
    decl = load_workbench_decl(config, protocols=builtin_protocol_catalog())

    run_spec(decl, verbose=False, console=Console(file=None, quiet=True))

    aggregate_store = RecordStore(
        config.parent / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=config.parent,
        create=False,
    )
    record = aggregate_store.latest_dataframe("collect_vec8/vec8")
    assert record is not None
    assert record.contract_id == "sfxi.vec8_collection.v2"
    collection_frame = record.load_dataframe()
    assert collection_frame[
        ["source_resource_id", "source_experiment_id", "source_record_id"]
    ].drop_duplicates().to_dict(orient="records") == [
        {
            "source_resource_id": "first",
            "source_experiment_id": "source-a",
            "source_record_id": "vec8/df",
        },
        {
            "source_resource_id": "second",
            "source_experiment_id": "source-b",
            "source_record_id": "vec8/df",
        },
    ]
    assert [(item.ref.experiment_id, item.ref.record_id) for item in record.inputs] == [
        ("source-a", "vec8/df"),
        ("source-b", "vec8/df"),
    ]
    assert all(item.discovery_policy == "source_record" for item in record.inputs)
    assert not (config.parent / "outputs" / "sources").exists()
    assert aggregate_store.latest_record("plot:vec8_collection_heatmap") is not None
    assert aggregate_store.latest_record("export:vec8_table") is not None

    verification = verify_record_store(
        aggregate_store,
        experiment_root=config.parent,
        expected_config_digest=decl.config_digest,
    )
    assert verification["status"] == "ok"

    first_store.persist_dataframe(
        producer_id="vec8",
        producer_plugin="transform/sfxi",
        out_name="vec8",
        record_id="vec8/df",
        df=_vec8("a", shift=0.25),
        contract_id="sfxi.vec8.v3",
        inputs=(),
        config_digest="sha256:source-a-updated",
    )
    changed = verify_record_store(
        aggregate_store,
        experiment_root=config.parent,
        expected_config_digest=decl.config_digest,
    )
    assert changed["status"] == "failed"
    issues = [issue for item in changed["records"] for issue in item["issues"]]
    assert any(issue["code"] == "input.source_record_revision_mismatch" for issue in issues)


def test_source_evidence_keeps_the_revision_resolved_for_computation(tmp_path: Path) -> None:
    source_config, source_store = _source(tmp_path, experiment_id="source-a", prefix="a")
    source_ref = SourceRecordRef(
        resource_id="first",
        experiment_id="source-a",
        record_id="vec8/df",
        experiment_root=source_config.parent,
        outputs_dir=source_store.root,
    )
    resolved = resolve_source_record(source_ref, contracts=builtin_contract_catalog())
    collection = SourceRecordCollection((resolved,))

    source_store.persist_dataframe(
        producer_id="vec8",
        producer_plugin="transform/sfxi",
        out_name="vec8",
        record_id="vec8/df",
        df=_vec8("a", shift=0.5),
        contract_id="sfxi.vec8.v3",
        inputs=(),
        config_digest="sha256:source-a-updated",
    )

    aggregate_root = tmp_path / "experiments" / "aggregates" / "aggregate-a"
    aggregate_store = RecordStore(
        aggregate_root / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=aggregate_root,
    )
    captured = aggregate_store.capture_inputs(
        [ProvenanceInput(label="sources[first]", ref=source_ref)],
        resolved_inputs={"sources": collection},
    )

    assert captured[0].record_revision_digest == resolved.revision_digest
    with pytest.raises(RecordError, match="changed after input evidence was captured"):
        aggregate_store.persist_dataframe(
            producer_id="collect",
            producer_plugin="transform/sfxi_vec8_collection",
            out_name="vec8",
            record_id="collect/vec8",
            df=_vec8("aggregate"),
            contract_id="sfxi.vec8.v3",
            inputs=captured,
            config_digest="sha256:aggregate-a",
        )
    assert aggregate_store.latest_dataframe("collect/vec8") is None


def test_catalog_snapshot_builds_one_source_experiment_index_per_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_config, source_store = _source(tmp_path, experiment_id="source-a", prefix="a")
    source_ref = SourceRecordRef(
        resource_id="first",
        experiment_id="source-a",
        record_id="vec8/df",
        experiment_root=source_config.parent,
        outputs_dir=source_store.root,
    )
    resolved = resolve_source_record(source_ref, contracts=builtin_contract_catalog())
    aggregate_root = tmp_path / "experiments" / "aggregates" / "aggregate-a"
    aggregate_store = RecordStore(
        aggregate_root / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=aggregate_root,
    )
    captured = aggregate_store.capture_inputs(
        [ProvenanceInput(label=f"sources[{index}]", ref=source_ref) for index in range(6)],
        resolved_inputs={"source": resolved},
    )
    aggregate_store.persist_dataframe(
        producer_id="collect",
        producer_plugin="transform/sfxi_vec8_collection",
        out_name="vec8",
        record_id="collect/vec8",
        df=_vec8("aggregate"),
        contract_id="sfxi.vec8.v3",
        inputs=captured,
        config_digest="sha256:aggregate-a",
    )
    build_calls = 0
    original_build_index = ExperimentCatalog._build_index

    def counted_build_index(catalog: ExperimentCatalog):
        nonlocal build_calls
        build_calls += 1
        return original_build_index(catalog)

    monkeypatch.setattr(ExperimentCatalog, "_build_index", counted_build_index)

    first = capture_revision_snapshot(aggregate_store)
    assert first["collect/vec8"]["revision"] == 1
    assert build_calls == 1

    duplicate = tmp_path / "experiments" / "duplicate" / "source-a-copy"
    duplicate.mkdir(parents=True)
    write_config(duplicate, base_reader_config(experiment_id="source-a"))

    with pytest.raises(RecordError, match="ambiguous"):
        capture_revision_snapshot(aggregate_store)
    assert build_calls == 2


def test_capture_rejects_corrupt_exact_source_revision_without_output_mutation(tmp_path: Path) -> None:
    source_config, source_store = _source(tmp_path, experiment_id="source-a", prefix="a")
    source_ref = SourceRecordRef(
        resource_id="first",
        experiment_id="source-a",
        record_id="vec8/df",
        experiment_root=source_config.parent,
        outputs_dir=source_store.root,
    )
    resolved = resolve_source_record(source_ref, contracts=builtin_contract_catalog())
    assert resolved.record.path.is_file()
    resolved.record.path.write_bytes(b"corrupt parquet")

    aggregate_root = tmp_path / "experiments" / "aggregates" / "aggregate-a"
    aggregate_store = RecordStore(
        aggregate_root / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=aggregate_root,
        create=False,
    )

    with pytest.raises(RecordError, match="content digest mismatch"):
        aggregate_store.capture_inputs(
            [ProvenanceInput(label="sources[first]", ref=source_ref)],
            resolved_inputs={"sources": SourceRecordCollection((resolved,))},
        )

    assert not aggregate_root.exists()


def test_persistence_rechecks_source_artifact_before_output_mutation(tmp_path: Path) -> None:
    source_config, source_store = _source(tmp_path, experiment_id="source-a", prefix="a")
    source_ref = SourceRecordRef(
        resource_id="first",
        experiment_id="source-a",
        record_id="vec8/df",
        experiment_root=source_config.parent,
        outputs_dir=source_store.root,
    )
    resolved = resolve_source_record(source_ref, contracts=builtin_contract_catalog())
    aggregate_root = tmp_path / "experiments" / "aggregates" / "aggregate-a"
    aggregate_store = RecordStore(
        aggregate_root / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=aggregate_root,
        create=False,
    )
    captured = aggregate_store.capture_inputs(
        [ProvenanceInput(label="sources[first]", ref=source_ref)],
        resolved_inputs={"sources": SourceRecordCollection((resolved,))},
    )
    resolved.record.path.write_bytes(b"corrupt parquet")

    with pytest.raises(RecordError, match="changed after input evidence was captured"):
        aggregate_store.persist_dataframe(
            producer_id="collect",
            producer_plugin="transform/sfxi_vec8_collection",
            out_name="vec8",
            record_id="collect/vec8",
            df=_vec8("aggregate"),
            contract_id="sfxi.vec8.v3",
            inputs=captured,
            config_digest="sha256:aggregate-a",
        )

    assert not aggregate_root.exists()


def test_corrupt_source_artifact_fails_run_before_aggregate_output_mutation(tmp_path: Path) -> None:
    _, source_store = _source(tmp_path, experiment_id="source-a", prefix="a")
    _source(tmp_path, experiment_id="source-b", prefix="b")
    config = _aggregate_config(tmp_path)
    source_store.read_dataframe("vec8/df").path.write_bytes(b"corrupt parquet")
    decl = load_workbench_decl(config, protocols=builtin_protocol_catalog())

    with pytest.raises(ConfigError, match="content digest mismatch"):
        run_spec(decl, verbose=False, console=Console(file=None, quiet=True))

    assert not (config.parent / "outputs").exists()


def test_verify_rejects_corrupt_source_artifact_without_catalog_advancement(tmp_path: Path) -> None:
    source_config, source_store = _source(tmp_path, experiment_id="source-a", prefix="a")
    _source(tmp_path, experiment_id="source-b", prefix="b")
    config = _aggregate_config(tmp_path)
    decl = load_workbench_decl(config, protocols=builtin_protocol_catalog())
    run_spec(decl, verbose=False, console=Console(file=None, quiet=True))
    aggregate_store = RecordStore(
        config.parent / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=config.parent,
        create=False,
    )
    source_ref = SourceRecordRef(
        resource_id="first",
        experiment_id="source-a",
        record_id="vec8/df",
        experiment_root=source_config.parent,
        outputs_dir=source_store.root,
    )
    revision_before_corruption = resolve_source_record(
        source_ref,
        contracts=builtin_contract_catalog(),
    ).revision_digest
    source_store.read_dataframe("vec8/df").path.write_bytes(b"corrupt parquet")

    verification = verify_record_store(
        aggregate_store,
        experiment_root=config.parent,
        expected_config_digest=decl.config_digest,
    )

    assert verification["status"] == "failed"
    issues = [issue for item in verification["records"] for issue in item["issues"]]
    assert any(issue["code"] == "input.source_record_artifact_invalid" for issue in issues)
    assert resolve_source_record(source_ref, contracts=builtin_contract_catalog()).revision_digest == (
        revision_before_corruption
    )


def test_vec8_collection_contract_is_not_a_single_source_vec8() -> None:
    contracts = builtin_contract_catalog()

    assert not contracts.satisfies(actual="sfxi.vec8_collection.v1", expected="sfxi.vec8.v3")


def test_missing_source_record_fails_before_aggregate_output_mutation(tmp_path: Path) -> None:
    source_root = tmp_path / "experiments" / "sources" / "source-a"
    source_root.mkdir(parents=True)
    write_config(source_root, base_reader_config(experiment_id="source-a"))
    aggregate_root = tmp_path / "experiments" / "aggregates" / "aggregate-a"
    aggregate_root.mkdir(parents=True)
    config = write_config(
        aggregate_root,
        base_reader_config(
            experiment_id="aggregate-a",
            protocol_id="logic/sfxi_vec8_collection",
            protocol_inputs={"record_resources": ["source"]},
            resources={"source": {"kind": "record", "experiment": "source-a", "record": "missing/df"}},
        ),
    )
    decl = load_workbench_decl(config, protocols=builtin_protocol_catalog())

    with pytest.raises(ConfigError, match="failed before output mutation"):
        run_spec(decl, verbose=False, console=Console(file=None, quiet=True))

    assert not (aggregate_root / "outputs").exists()


def test_self_source_output_collision_fails_before_record_or_ledger_mutation(tmp_path: Path) -> None:
    aggregate_root = tmp_path / "experiments" / "aggregates" / "aggregate-a"
    aggregate_root.mkdir(parents=True)
    config = write_config(
        aggregate_root,
        base_reader_config(
            experiment_id="aggregate-a",
            protocol_id="logic/sfxi_vec8_collection",
            protocol_inputs={"record_resources": ["self"]},
            resources={
                "self": {
                    "kind": "record",
                    "experiment": "aggregate-a",
                    "record": "collect_vec8/vec8",
                }
            },
        ),
    )
    store = RecordStore(
        aggregate_root / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=aggregate_root,
    )
    store.persist_dataframe(
        producer_id="seed",
        producer_plugin="transform/sfxi",
        out_name="vec8",
        record_id="collect_vec8/vec8",
        df=_vec8("seed"),
        contract_id="sfxi.vec8.v3",
        inputs=(),
        config_digest="sha256:seed",
    )
    catalog_before = store.records_path.read_bytes()
    revision_counts_before = store.revision_counts()
    invocation_path = store.invocation_ledger_path()
    decl = load_workbench_decl(config, protocols=builtin_protocol_catalog())

    with pytest.raises(ConfigError, match="same experiment.*planned output"):
        run_spec(decl, verbose=False, console=Console(file=None, quiet=True))

    assert store.records_path.read_bytes() == catalog_before
    assert store.revision_counts() == revision_counts_before
    assert not invocation_path.exists()


def test_experiment_catalog_uses_config_identity_and_ignores_generated_configs(tmp_path: Path) -> None:
    config, _ = _source(tmp_path, experiment_id="source-a", prefix="a")
    copied = config.parent / "outputs" / "sources" / "copy"
    copied.mkdir(parents=True)
    (copied / "config.yaml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")

    location = ExperimentCatalog(tmp_path / "experiments").resolve("source-a")

    assert location.config_path == config.resolve()


def _annotated_traces() -> pd.DataFrame:
    rows = []
    times = [index * 0.5 for index in range(9)]
    for design_index, design_id in enumerate(("reference", "candidate")):
        for state_index, treatment in enumerate(("none", "a", "b", "a+b")):
            for observation_index in range(2):
                position = f"{design_id}-{state_index}-{observation_index}"
                for channel_index, channel in enumerate(("response", "magnitude", "growth")):
                    for time in times:
                        rows.append(
                            {
                                "position": position,
                                "time": time,
                                "channel": channel,
                                "value": 1.0
                                + design_index * 0.2
                                + state_index * 0.1
                                + observation_index * 0.02
                                + channel_index * 0.3
                                + time * 0.05,
                                "treatment": treatment,
                                "design_id": design_id,
                                "segment": 0 if time <= 1.0 else 1,
                                "value_policy_clipped": False,
                                "value_instrument_overflow": False,
                                "value_bound_kind": "exact",
                            }
                        )
    return pd.DataFrame(rows)


def test_response_window_is_a_normal_protocol_run_with_normal_records(tmp_path: Path) -> None:
    source_root = tmp_path / "experiments" / "2026" / "trace-source"
    source_root.mkdir(parents=True)
    write_config(source_root, base_reader_config(experiment_id="trace-source"))
    source_store = RecordStore(
        source_root / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=source_root,
    )
    frame = _annotated_traces()
    for record_id in ("response/df", "magnitude/df", "trajectory/df"):
        source_store.persist_dataframe(
            producer_id=record_id.split("/")[0],
            producer_plugin="transform/source",
            out_name="df",
            record_id=record_id,
            df=frame,
            contract_id="plate_reader.annotated.v1",
            inputs=(),
            config_digest="sha256:trace-source",
        )

    aggregate_root = tmp_path / "experiments" / "aggregates" / "response-review"
    aggregate_root.mkdir(parents=True)
    config = write_config(
        aggregate_root,
        base_reader_config(
            experiment_id="response-review",
            protocol_id="plate_reader/response_window",
            protocol_inputs={
                "response_records": ["response"],
                "magnitude_records": ["magnitude"],
                "trajectory_records": ["trajectory"],
            },
            protocol_analysis={
                "source": {
                    "response_channel": "response",
                    "magnitude_channel": "magnitude",
                    "growth_channel": "growth",
                    "reference_design_id": "reference",
                    "state_column": "treatment",
                    "state_values": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
                },
                "event": {
                    "event_id": "addition",
                    "event_kind": "intervention",
                    "segment_column": "segment",
                    "pre_segment_index": 0,
                    "post_segment_index": 1,
                    "estimate_method": "segment_gap_midpoint",
                    "declaration": "The event occurred between acquisition segments 0 and 1.",
                },
                "reductions": [
                    {
                        "id": "primary",
                        "window_start_event_h": 0.5,
                        "window_end_event_h": 1.0,
                        "method": "geometric_time_mean",
                        "response_basis": "post_window",
                        "role": "primary",
                    }
                ],
                "aggregation": {
                    "observation_stat": "median",
                    "descriptive_resampling_draws": 100,
                    "descriptive_interval_mass": 0.9,
                    "random_seed": 17,
                },
                "quality": {
                    "positive_floor": 1.0e-12,
                    "max_interior_gap_h": 0.6,
                    "min_observations_per_state": 2,
                },
            },
            protocol_outputs={
                "plots": {
                    "include": ["response_window_diagnostic"],
                    "views": {
                        "response_window_diagnostic": {
                            "source_experiment_id": "trace-source",
                            "design_id": "candidate",
                        }
                    },
                }
            },
            resources={
                "response": {"kind": "record", "experiment": "trace-source", "record": "response/df"},
                "magnitude": {"kind": "record", "experiment": "trace-source", "record": "magnitude/df"},
                "trajectory": {"kind": "record", "experiment": "trace-source", "record": "trajectory/df"},
            },
        ),
    )
    decl = load_workbench_decl(config, protocols=builtin_protocol_catalog())

    run_spec(decl, verbose=False, console=Console(file=None, quiet=True))

    store = RecordStore(
        aggregate_root / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=aggregate_root,
        create=False,
    )
    assert store.latest_dataframe("response_window/designs") is not None
    assert store.latest_dataframe("response_window/events") is not None
    assert store.latest_record("plot:response_window_summary") is not None
    assert store.latest_record("plot:response_window_diagnostic") is not None
    assert store.latest_record("export:designs_table") is not None
    assert store.latest_record("export:events_table") is not None
    assert not (aggregate_root / "outputs" / "manifest.json").exists()
    assert not (aggregate_root / "outputs" / "request.yaml").exists()
    assert not (aggregate_root / "outputs" / "sources").exists()


def test_repository_has_no_aggregate_counter_control_plane() -> None:
    package_root = Path(__file__).resolve().parents[2]

    assert not (package_root / "api" / "response_window").exists()
    assert not (package_root / "runtime" / "response_window.py").exists()
    assert not (package_root / "runtime" / "sfxi_vec8_aggregate.py").exists()
    assert not (package_root / "workbench" / "cli" / "response_window.py").exists()
    assert not (package_root / "domains" / "plate_reader" / "evidence" / "response_window").exists()
