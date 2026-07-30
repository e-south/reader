from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import ValidationError

from reader.plugins.plot.single_reporter_diagnostic import (
    SingleReporterDiagnosticCfg,
    SingleReporterDiagnosticPlot,
)
from reader.workbench.experiment import (
    AnnotationCollections,
    AnnotationCollectionSpec,
    AnnotationOrders,
    AnnotationOrderSpec,
    AnnotationSemantics,
    ExperimentEvidence,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in ("subject-a", "subject-b"):
        for condition in ("baseline", "induced"):
            for replicate in ("replicate-1", "replicate-2"):
                for time in (0.0, 1.0, 2.0):
                    normalizer = 0.2 + time * 0.1
                    reporter = 1.0 + (1.0 if condition == "induced" else 0.0) + time
                    for channel, value in (
                        ("OD", normalizer),
                        ("reporter", reporter),
                        ("reporter/OD", reporter / normalizer),
                    ):
                        rows.append(
                            {
                                "subject_alias": subject,
                                "condition_alias": condition,
                                "biological_replicate_id": replicate,
                                "position": "A1" if replicate == "replicate-1" else "A2",
                                "time": time,
                                "channel": channel,
                                "value": value,
                            }
                        )
    return pd.DataFrame.from_records(rows)


def _context():
    annotations = AnnotationSemantics(
        orders=AnnotationOrders(
            by_id={"conditions": AnnotationOrderSpec(column="condition_alias", values=["baseline", "induced"])}
        ),
        collections=AnnotationCollections(
            by_id={
                "subjects": AnnotationCollectionSpec(
                    column="subject_alias",
                    items={"A": ["subject-a"], "B": ["subject-b"]},
                )
            }
        ),
    )
    experiment = SimpleNamespace(
        annotations=annotations,
        evidence=ExperimentEvidence(
            data_class="plate_reader_screen",
            data_class_reason="Synthetic plugin fixture.",
            replicate_kind="biological",
            replicate_identity_field="biological_replicate_id",
        ),
    )
    return SimpleNamespace(experiment=experiment, palette_book=None)


def _interval_policy():
    return {
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
    }


def _endpoint_policy():
    return {
        "selection": {
            "kind": "endpoint",
            "time_basis": "absolute",
            "time_h": 1.0,
            "mode": "exact",
            "tolerance_h": 0.0,
        },
        "method": "identity",
        "output_space": "linear",
        "support": {
            "boundary_support": "none",
            "minimum_observations": 1,
            "maximum_interior_gap_h": None,
            "positive_floor": None,
            "positive_value_scope": "selected_support",
            "censored_values": "allow",
        },
    }


def _aggregation_policy():
    return {
        "within_unit_statistic": "median",
        "across_unit_statistic": "median",
    }


def test_single_reporter_diagnostic_plugin_uses_semantic_partition_and_declared_replicates() -> None:
    cfg = SingleReporterDiagnosticCfg(
        partition={"collection_ref": "subjects"},
        condition_column="condition_alias",
        condition_order_ref="conditions",
        temporal_reduction=_interval_policy(),
        observation_aggregation=_aggregation_policy(),
        normalizer_channel="OD",
        reporter_channel="reporter",
        ratio_channel="reporter/OD",
        format=["png", "pdf"],
        dpi=144,
    )

    rendered = SingleReporterDiagnosticPlot().render(_context(), {"df": _frame()}, cfg)

    assert [(item.filename, item.ext, item.dpi) for item in rendered] == [
        ("single_reporter_diagnostic__A", "png", 144),
        ("single_reporter_diagnostic__A", "pdf", 144),
        ("single_reporter_diagnostic__B", "png", 144),
        ("single_reporter_diagnostic__B", "pdf", 144),
    ]
    assert all(item.description and "normalizer QC" in item.description for item in rendered)
    for figure in {item.fig for item in rendered}:
        plt.close(figure)


def test_single_reporter_diagnostic_declares_tidy_record_input() -> None:
    assert SingleReporterDiagnosticPlot.input_ports()["df"].contract == "plate_reader.annotated.v1"


def test_single_reporter_diagnostic_config_requires_compiler_owned_reduction_policy() -> None:
    common = {"normalizer_channel": "OD", "reporter_channel": "reporter", "ratio_channel": "reporter/OD"}

    with pytest.raises(ValidationError, match="temporal_reduction"):
        SingleReporterDiagnosticCfg(**common)
    with pytest.raises(ValidationError, match="observation_aggregation"):
        SingleReporterDiagnosticCfg(temporal_reduction=_endpoint_policy(), **common)


def test_single_reporter_diagnostic_fails_when_declared_replicate_identity_is_missing() -> None:
    cfg = SingleReporterDiagnosticCfg(
        temporal_reduction=_endpoint_policy(),
        observation_aggregation=_aggregation_policy(),
        normalizer_channel="OD",
        reporter_channel="reporter",
        ratio_channel="reporter/OD",
    )

    with pytest.raises(ValueError, match="replicate_identity_field.*absent"):
        SingleReporterDiagnosticPlot().render(
            _context(),
            {"df": _frame().drop(columns="biological_replicate_id")},
            cfg,
        )
