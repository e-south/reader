from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import ValidationError

from reader_workbench.plugins.plot.single_reporter_diagnostic import (
    SingleReporterDiagnosticCfg,
    SingleReporterDiagnosticPlot,
    SingleReporterObservationUnitCfg,
    _reduction_columns,
)
from reader_workbench.workbench.experiment import (
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


def _context(*, replicate_identity_field: str | None = "biological_replicate_id"):
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
            replicate_identity_field=replicate_identity_field,
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
        identity_scope={"entity_columns": ["subject_alias"]},
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


def test_single_reporter_diagnostic_does_not_infer_replicates_from_position() -> None:
    context = _context(replicate_identity_field=None)
    frame = _frame().assign(
        experiment_id="experiment-1",
        plate_id="plate-1",
        sheet_name="read-1",
        well="A1",
    )

    assert context.experiment.evidence.replicate_kind == "biological"
    assert context.experiment.evidence.replicate_identity_field is None
    with pytest.raises(ValueError, match="replicate identity.*observation_unit"):
        _reduction_columns(ctx=context, frame=frame, observation_unit=None)


def test_single_reporter_diagnostic_accepts_explicit_observation_only_units() -> None:
    cfg = SingleReporterDiagnosticCfg(
        temporal_reduction=_endpoint_policy(),
        observation_aggregation=_aggregation_policy(),
        observation_unit={"role": "observation_only", "column": "position"},
        partition={"collection_ref": "subjects"},
        identity_scope={"entity_columns": ["subject_alias"]},
        condition_column="condition_alias",
        normalizer_channel="OD",
        reporter_channel="reporter",
        ratio_channel="reporter/OD",
    )

    rendered = SingleReporterDiagnosticPlot().render(
        _context(replicate_identity_field=None),
        {"df": _frame()},
        cfg,
    )

    assert rendered
    assert all(item.description and "observation-only" in item.description for item in rendered)
    assert all("Observation units (not replicates)" in item.fig.axes[3].get_title() for item in rendered)
    for figure in {item.fig for item in rendered}:
        plt.close(figure)


def test_single_reporter_diagnostic_uses_declared_replicate_without_position_fallback() -> None:
    frame = _frame().drop(columns="position")

    assert _reduction_columns(
        ctx=_context(),
        frame=frame,
        observation_unit=None,
    ) == ("biological_replicate_id", "biological_replicate_id", "declared_replicate")

    assert _reduction_columns(
        ctx=_context(),
        frame=_frame(),
        observation_unit=SingleReporterObservationUnitCfg(role="observation_only", column="position"),
    ) == ("biological_replicate_id", "position", "declared_replicate")


@pytest.mark.parametrize(
    "observation_unit",
    [
        {"column": "position"},
        {"role": "replicate", "column": "position"},
        {"role": "observation_only", "column": "   "},
    ],
)
def test_single_reporter_diagnostic_requires_typed_observation_only_contract(
    observation_unit: dict[str, str],
) -> None:
    with pytest.raises(ValidationError, match="observation_unit|role|column"):
        SingleReporterDiagnosticCfg(
            temporal_reduction=_endpoint_policy(),
            observation_aggregation=_aggregation_policy(),
            identity_scope={"entity_columns": ["subject_alias"]},
            observation_unit=observation_unit,
            normalizer_channel="OD",
            reporter_channel="reporter",
            ratio_channel="reporter/OD",
        )


def test_single_reporter_diagnostic_config_requires_compiler_owned_reduction_policy() -> None:
    common = {
        "identity_scope": {"entity_columns": ["subject_alias"]},
        "normalizer_channel": "OD",
        "reporter_channel": "reporter",
        "ratio_channel": "reporter/OD",
    }

    with pytest.raises(ValidationError, match="temporal_reduction"):
        SingleReporterDiagnosticCfg(**common)
    with pytest.raises(ValidationError, match="observation_aggregation"):
        SingleReporterDiagnosticCfg(temporal_reduction=_endpoint_policy(), **common)


def test_single_reporter_diagnostic_fails_when_declared_replicate_identity_is_missing() -> None:
    cfg = SingleReporterDiagnosticCfg(
        temporal_reduction=_endpoint_policy(),
        observation_aggregation=_aggregation_policy(),
        identity_scope={"entity_columns": ["subject_alias"]},
        condition_column="condition_alias",
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


def test_single_reporter_diagnostic_fails_when_observation_only_column_is_missing() -> None:
    cfg = SingleReporterDiagnosticCfg(
        temporal_reduction=_endpoint_policy(),
        observation_aggregation=_aggregation_policy(),
        identity_scope={"entity_columns": ["subject_alias"]},
        observation_unit={"role": "observation_only", "column": "well_id"},
        condition_column="condition_alias",
        normalizer_channel="OD",
        reporter_channel="reporter",
        ratio_channel="reporter/OD",
    )

    with pytest.raises(ValueError, match="observation_unit.*well_id.*absent"):
        SingleReporterDiagnosticPlot().render(
            _context(replicate_identity_field=None),
            {"df": _frame()},
            cfg,
        )


def test_single_reporter_diagnostic_fails_when_identity_scope_column_is_missing() -> None:
    cfg = SingleReporterDiagnosticCfg(
        temporal_reduction=_endpoint_policy(),
        observation_aggregation=_aggregation_policy(),
        identity_scope={"entity_columns": ["subject_alias"]},
        condition_column="condition_alias",
        normalizer_channel="OD",
        reporter_channel="reporter",
        ratio_channel="reporter/OD",
    )

    with pytest.raises(ValueError, match="missing required columns.*subject_alias"):
        SingleReporterDiagnosticPlot().render(
            _context(),
            {"df": _frame().drop(columns="subject_alias")},
            cfg,
        )


def test_single_reporter_diagnostic_rejects_missing_identity_scope_values() -> None:
    cfg = SingleReporterDiagnosticCfg(
        temporal_reduction=_endpoint_policy(),
        observation_aggregation=_aggregation_policy(),
        identity_scope={"entity_columns": ["subject_alias"]},
        condition_column="condition_alias",
        normalizer_channel="OD",
        reporter_channel="reporter",
        ratio_channel="reporter/OD",
    )
    frame = _frame()
    frame.loc[frame.index[0], "subject_alias"] = None

    with pytest.raises(ValueError, match="subject_alias.*missing identities"):
        SingleReporterDiagnosticPlot().render(_context(), {"df": frame}, cfg)


@pytest.mark.parametrize(
    "identity_scope",
    [
        None,
        {"entity_columns": []},
        {"entity_columns": ["subject_alias", "subject_alias"]},
        {"entity_columns": ["   "]},
    ],
)
def test_single_reporter_diagnostic_requires_explicit_semantic_identity_scope(
    identity_scope: dict[str, list[str]] | None,
) -> None:
    kwargs = {
        "temporal_reduction": _endpoint_policy(),
        "observation_aggregation": _aggregation_policy(),
        "normalizer_channel": "OD",
        "reporter_channel": "reporter",
        "ratio_channel": "reporter/OD",
    }
    if identity_scope is not None:
        kwargs["identity_scope"] = identity_scope

    with pytest.raises(ValidationError, match="identity_scope|entity_columns"):
        SingleReporterDiagnosticCfg(**kwargs)
