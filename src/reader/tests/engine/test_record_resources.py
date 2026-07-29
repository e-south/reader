from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console

from reader.contracts import builtin_contract_catalog
from reader.errors import ConfigError
from reader.protocols.builtins import builtin_protocol_catalog
from reader.tests.support import base_reader_config, write_config
from reader.workbench.decl import load_workbench_decl
from reader.workbench.engine import run_spec
from reader.workbench.experiments import ExperimentCatalog
from reader.workbench.records import RecordStore, verify_record_store


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
    assert record.contract_id == "sfxi.vec8_collection.v1"
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
            for replicate in range(2):
                position = f"{design_id}-{state_index}-{replicate}"
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
                                + replicate * 0.02
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
                    "replicate_stat": "median",
                    "bootstrap_samples": 100,
                    "confidence_level": 0.9,
                    "random_seed": 17,
                },
                "quality": {
                    "positive_floor": 1.0e-12,
                    "max_interior_gap_h": 0.6,
                    "min_replicates_per_state": 2,
                },
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
