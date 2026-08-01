from __future__ import annotations

from pathlib import Path

import pandas as pd

from reader_workbench.runtime import builtin_runtime
from reader_workbench.tests.support.configs import base_reader_config, load_models, write_config
from reader_workbench.workbench import resolve_workbench
from reader_workbench.workbench.engine.runtime import run_spec
from reader_workbench.workbench.notebooks.components.deliverables import collect_notebook_deliverables
from reader_workbench.workbench.records import record_revision_digest, record_to_dict


def _ratio_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in ("baseline", "induced"):
        for replicate_index, replicate in enumerate(("replicate-1", "replicate-2")):
            for time in (0.0, 1.0, 2.0):
                normalizer = 0.2 + time * 0.1 + replicate_index * 0.01
                reporter = 1.0 + (1.0 if condition == "induced" else 0.0) + time
                for channel, value in (
                    ("OD", normalizer),
                    ("reporter", reporter),
                    ("reporter/OD", reporter / normalizer),
                ):
                    rows.append(
                        {
                            "sample_alias": "sample-a",
                            "condition_alias": condition,
                            "type": "SAMPLE",
                            "treatment": condition,
                            "design_id": "sample-a",
                            "biological_replicate_id": replicate,
                            "position": "A1" if replicate_index == 0 else "A2",
                            "time": time,
                            "channel": channel,
                            "value": value,
                        }
                    )
    return pd.DataFrame.from_records(rows)


def test_engine_persists_single_reporter_diagnostic_for_canonical_notebook_discovery(tmp_path: Path) -> None:
    payload = base_reader_config(
        experiment_id="single-reporter-diagnostic",
        protocol_id="plate_reader/single_reporter_screen",
        protocol_analysis={
            "reporter_channel": "reporter",
            "normalizer_channel": "OD",
            "temporal_reduction": {
                "selection": {
                    "kind": "interval",
                    "time_basis": "absolute",
                    "start_h": 1.0,
                    "end_h": 2.0,
                    "boundary": "inclusive",
                },
                "method": "observed_median",
                "output_space": "linear",
                "support": {
                    "boundary_support": "observed",
                    "minimum_observations": 2,
                    "maximum_interior_gap_h": 1.0,
                    "positive_floor": None,
                    "positive_value_scope": "selected_support",
                    "censored_values": "allow",
                },
            },
            "observation_aggregation": {
                "within_unit_statistic": "median",
                "across_unit_statistic": "median",
            },
        },
        protocol_outputs={
            "plots": {
                "profile": "kinetics_qc",
                "include": ["single_reporter_diagnostic"],
                "exclude": ["raw_kinetics", "value_distributions"],
                "views": {
                    "single_reporter_diagnostic": {
                        "partition": {"by": "sample_alias"},
                        "condition_column": "condition_alias",
                        "condition_order_ref": "conditions",
                        "format": ["png"],
                        "dpi": 72,
                    }
                },
            }
        },
        annotations={
            "orders": {
                "conditions": {
                    "column": "condition_alias",
                    "values": ["baseline", "induced"],
                }
            }
        },
        resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
    )
    payload["evidence"] = {
        "data_class": "plate_reader_screen",
        "data_class_reason": "Synthetic single-reporter engine fixture.",
        "replicate_kind": "biological",
        "replicate_identity_field": "biological_replicate_id",
    }
    config_path = write_config(tmp_path, payload)
    _, declaration = load_models(config_path)
    runtime = builtin_runtime()
    layout = declaration.experiment_semantics.layout
    store = runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=declaration.experiment.root,
    )
    store.persist_dataframe(
        producer_id="sample_measurements",
        producer_plugin="validator/to_tidy_plus_map",
        out_name="df",
        record_id="sample_measurements/df",
        df=_ratio_frame(),
        contract_id="plate_reader.annotated.v1",
        inputs=[],
        config_digest=declaration.config_digest,
        producer_config_digest="sha256:ratio-fixture",
    )

    result = run_spec(
        declaration,
        include_pipeline=False,
        include_plots=True,
        include_exports=False,
        plot_specs=resolve_workbench(declaration).plots,
        log_level="ERROR",
        verbose=False,
        show_next_steps=False,
        runtime=runtime,
    )

    assert result.status == "succeeded"
    records = store.iter_latest_records()
    plot_record = next(record for record in records if record.record_id == "plot:single_reporter_diagnostic")
    assert plot_record.producer.plugin == "plot/single_reporter_diagnostic"
    assert [path.suffix for path in plot_record.files] == [".png"]
    assert all(path.is_file() for path in plot_record.files)

    revision_counts = store.revision_counts(record.record_id for record in records)
    entries = tuple(
        {
            **record_to_dict(record, outputs_dir=layout.outputs_dir),
            "revision": revision_counts[record.record_id],
            "revision_digest": record_revision_digest(record, outputs_dir=layout.outputs_dir),
        }
        for record in records
    )
    deliverables = collect_notebook_deliverables(
        entries,
        outputs_dir=layout.outputs_dir,
        verification_status="ok",
        verification_issues=(),
        verification_records=tuple({"record_id": record.record_id, "status": "ok", "issues": []} for record in records),
    )

    assert len(deliverables.plot_rows) == 1
    assert deliverables.plot_rows[0]["Record ID"] == "plot:single_reporter_diagnostic"
    assert deliverables.plot_rows[0]["Description"] == (
        "One-row normalizer, reporter, ratio, and condition-reduction diagnostic using an explicit endpoint or "
        "interval."
    )
