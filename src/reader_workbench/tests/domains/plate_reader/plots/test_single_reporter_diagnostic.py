from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from reader_workbench.domains.plate_reader.plots.single_reporter_diagnostic import (
    prepare_single_reporter_diagnostics,
)
from reader_workbench.domains.plate_reader.plots.single_reporter_diagnostic_render import (
    render_single_reporter_diagnostic,
)
from reader_workbench.domains.time_series import (
    EndpointSelection,
    IntervalSelection,
    ObservationAggregationSpec,
    TemporalReductionSpec,
    TemporalSupportPolicy,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in ("subject-a", "subject-b"):
        for condition_index, condition in enumerate(("baseline", "induced")):
            for replicate_index, replicate in enumerate(("replicate-1", "replicate-2")):
                for observation_index, position in enumerate(("A1", "A2")):
                    for time in (0.0, 1.0, 2.0):
                        normalizer = 0.2 + time * 0.1 + replicate_index * 0.02 + observation_index * 0.01
                        reporter = 1.0 + condition_index * 2.0 + time + observation_index * 0.2
                        for channel, value in (
                            ("absorbance", normalizer),
                            ("mScarlet", reporter),
                            ("mScarlet/absorbance", reporter / normalizer),
                        ):
                            rows.append(
                                {
                                    "subject_alias": subject,
                                    "condition_alias": condition,
                                    "biological_replicate_id": replicate,
                                    "position": position,
                                    "time": time,
                                    "channel": channel,
                                    "value": value,
                                }
                            )
    return pd.DataFrame.from_records(rows)


def _prepare(*, endpoint_time_h=None, window_h=None):
    return prepare_single_reporter_diagnostics(
        _frame(),
        group_on="subject_alias",
        collection_items=[{"paired": ["subject-a", "subject-b"]}],
        group_match="exact",
        condition_column="condition_alias",
        condition_order=["baseline", "induced"],
        unit_column="biological_replicate_id",
        observation_column="position",
        time_column="time",
        normalizer_channel="absorbance",
        reporter_channel="mScarlet",
        ratio_channel="mScarlet/absorbance",
        temporal_reduction=_temporal(endpoint_time_h=endpoint_time_h, window_h=window_h),
        observation_aggregation=_aggregation(),
    )


def _temporal(*, endpoint_time_h=None, window_h=None) -> TemporalReductionSpec:
    if endpoint_time_h is not None:
        selection = EndpointSelection(
            time_basis="absolute",
            time_h=endpoint_time_h,
            mode="nearest",
            tolerance_h=0.2,
        )
        method = "identity"
        support = TemporalSupportPolicy(
            boundary_support="none",
            minimum_observations=1,
            maximum_interior_gap_h=None,
            positive_floor=None,
            positive_value_scope="selected_support",
            censored_values="allow",
        )
    else:
        assert window_h is not None
        selection = IntervalSelection(
            time_basis="absolute",
            start_h=window_h[0],
            end_h=window_h[1],
        )
        method = "observed_median"
        support = TemporalSupportPolicy(
            boundary_support="observed",
            minimum_observations=2,
            maximum_interior_gap_h=1.0,
            positive_floor=None,
            positive_value_scope="selected_support",
            censored_values="allow",
        )
    return TemporalReductionSpec(selection=selection, method=method, output_space="linear", support=support)


def _aggregation() -> ObservationAggregationSpec:
    return ObservationAggregationSpec(
        within_unit_statistic="median",
        across_unit_statistic="median",
    )


def test_single_reporter_diagnostic_reduces_observations_within_declared_units() -> None:
    (diagnostic,) = _prepare(window_h=(1.0, 2.0))

    assert diagnostic.group_label == "paired"
    assert diagnostic.condition_order == ("baseline", "induced")
    assert diagnostic.selection.label == "observed median over 1–2 h"
    assert len(diagnostic.reduced_ratio) == 8
    assert len(diagnostic.reduced_normalizer) == 8
    assert set(diagnostic.reduced_ratio["__unit"]) == {
        "subject-a::replicate-1",
        "subject-a::replicate-2",
        "subject-b::replicate-1",
        "subject-b::replicate-2",
    }


def test_single_reporter_diagnostic_endpoint_fails_outside_tolerance() -> None:
    with pytest.raises(ValueError, match="no endpoint observation within"):
        _prepare(endpoint_time_h=3.0)


def test_single_reporter_diagnostic_condition_order_is_closed() -> None:
    with pytest.raises(ValueError, match="exactly match observed conditions"):
        prepare_single_reporter_diagnostics(
            _frame(),
            group_on="subject_alias",
            collection_items=None,
            group_match="exact",
            condition_column="condition_alias",
            condition_order=["baseline"],
            unit_column="biological_replicate_id",
            observation_column="position",
            time_column="time",
            normalizer_channel="absorbance",
            reporter_channel="mScarlet",
            ratio_channel="mScarlet/absorbance",
            temporal_reduction=_temporal(endpoint_time_h=1.0),
            observation_aggregation=_aggregation(),
        )


def test_single_reporter_diagnostic_rejects_nonfinite_ratio_rows() -> None:
    frame = _frame()
    frame.loc[frame["channel"].eq("mScarlet/absorbance").idxmax(), "value"] = float("inf")

    with pytest.raises(ValueError, match="non-finite"):
        prepare_single_reporter_diagnostics(
            frame,
            group_on="subject_alias",
            collection_items=None,
            group_match="exact",
            condition_column="condition_alias",
            condition_order=["baseline", "induced"],
            unit_column="biological_replicate_id",
            observation_column="position",
            time_column="time",
            normalizer_channel="absorbance",
            reporter_channel="mScarlet",
            ratio_channel="mScarlet/absorbance",
            temporal_reduction=_temporal(endpoint_time_h=1.0),
            observation_aggregation=_aggregation(),
        )


def test_single_reporter_diagnostic_requires_channel_pairing_at_each_acquisition_time() -> None:
    frame = _frame()
    missing = (
        frame["subject_alias"].eq("subject-a")
        & frame["condition_alias"].eq("baseline")
        & frame["biological_replicate_id"].eq("replicate-1")
        & frame["time"].eq(1.0)
        & frame["channel"].eq("mScarlet")
    )

    with pytest.raises(ValueError, match="lacks exactly aligned channel times"):
        prepare_single_reporter_diagnostics(
            frame.loc[~missing],
            group_on="subject_alias",
            collection_items=None,
            group_match="exact",
            condition_column="condition_alias",
            condition_order=["baseline", "induced"],
            unit_column="biological_replicate_id",
            observation_column="position",
            time_column="time",
            normalizer_channel="absorbance",
            reporter_channel="mScarlet",
            ratio_channel="mScarlet/absorbance",
            temporal_reduction=_temporal(endpoint_time_h=1.0),
            observation_aggregation=_aggregation(),
        )


def test_single_reporter_diagnostic_renders_four_square_panels_and_visible_normalizer_qc() -> None:
    (diagnostic,) = _prepare(window_h=(1.0, 2.0))

    figure = render_single_reporter_diagnostic(
        diagnostic,
        colors=["#334155", "#2563eb"],
        figsize=(12.0, 3.4),
    )

    main_axes = figure.axes[:4]
    qc_axis = figure.axes[4]
    assert figure.get_gid() == "single-reporter-diagnostic"
    assert [axis.get_title() for axis in main_axes] == [
        "absorbance kinetics",
        "mScarlet kinetics",
        "mScarlet/absorbance kinetics",
        "mScarlet/absorbance by condition",
    ]
    assert len({tuple(axis.get_position().bounds) for axis in figure.axes}) == 4
    assert all(any(patch.get_gid() == "single-reporter-window" for patch in axis.patches) for axis in main_axes[:3])
    assert qc_axis.get_gid() == "single-reporter-normalizer-qc"
    assert qc_axis.get_ylabel() == "Reduced absorbance (QC only)"
    assert any(collection.get_gid() == "single-reporter-normalizer-center" for collection in qc_axis.collections)
    assert [tick.get_text() for tick in main_axes[3].get_xticklabels()] == ["baseline", "induced"]

    plt.close(figure)
