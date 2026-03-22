"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/domains/plate_reader/plots/test_plot_hardening.py

Direct library hardening tests for plate-reader plot semantics.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from rich.console import Console

from reader.domains.plate_reader.plots.panels import time_series as time_series_panel
from reader.domains.plate_reader.plots.retron_sponge import plot_retron_sponge_summary, plot_retron_sponge_trace
from reader.domains.plate_reader.plots.snapshot_barplot import plot_snapshot_barplot
from reader.domains.plate_reader.plots.snapshot_heatmap import plot_snapshot_heatmap, prepare_snapshot_heatmap_inputs
from reader.domains.plate_reader.plots.time_series import plot_time_series
from reader.errors import ConfigError
from reader.plugins.plot.snapshot_barplot import SnapshotBarCfg
from reader.plugins.plot.snapshot_heatmap import HeatmapCfg, SnapshotHeatmapPlot
from reader.protocols import ProtocolBinding
from reader.tests.support import load_decl, write_config
from reader.workbench.engine import validate as validate_job
from reader.workbench.experiment import (
    AnnotationOrders,
    AnnotationOrderSpec,
    AnnotationSemantics,
    ExperimentSemantics,
    OutputLayout,
    ResourceCatalog,
)


def test_snapshot_barplot_returns_empty_list_when_group_filter_removes_all_rows() -> None:
    df = pd.DataFrame(
        {
            "position": ["A1", "A2"],
            "time": [0.0, 0.0],
            "channel": ["YFP", "YFP"],
            "value": [1.0, 2.0],
            "treatment": ["a", "b"],
            "group_id": [None, None],
        }
    )

    figures = plot_snapshot_barplot(
        df=df,
        output_dir=None,
        x="treatment",
        y="YFP",
        hue=None,
        group_on="group_id",
        pool_sets=None,
        time=0.0,
        fig_kwargs={},
        filename=None,
    )

    assert figures == []


def test_snapshot_barplot_rejects_unsupported_file_by_mode() -> None:
    df = pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["YFP"],
            "value": [1.0],
            "treatment": ["a"],
        }
    )

    with pytest.raises(ValueError, match="file_by supports only 'auto' or 'channel'"):
        plot_snapshot_barplot(
            df=df,
            output_dir=None,
            x="treatment",
            y="YFP",
            hue=None,
            group_on=None,
            pool_sets=None,
            time=0.0,
            fig_kwargs={},
            filename=None,
            file_by="group",
        )


def test_snapshot_barplot_uses_unique_filenames_when_multiple_figures_are_emitted() -> None:
    df = pd.DataFrame(
        {
            "position": ["A1", "A2", "B1", "B2"],
            "time": [0.0, 0.0, 0.0, 0.0],
            "channel": ["YFP", "YFP", "YFP", "YFP"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "treatment": ["a", "a", "a", "a"],
            "design_id": ["g1", "g1", "g2", "g2"],
        }
    )

    figures = plot_snapshot_barplot(
        df=df,
        output_dir=None,
        x="treatment",
        y="YFP",
        hue=None,
        group_on="design_id",
        pool_sets=None,
        time=0.0,
        fig_kwargs={},
        filename="fixed",
    )

    filenames = [figure.filename for figure in figures]
    assert filenames == ["fixed__group=g1", "fixed__group=g2"]
    assert len(set(filenames)) == len(filenames)


def test_snapshot_bar_plugin_config_rejects_unsupported_file_by_modes() -> None:
    with pytest.raises(ValidationError):
        SnapshotBarCfg.model_validate(
            {
                "x": "treatment",
                "y": "YFP",
                "time": 0.0,
                "file_by": "group",
            }
        )


def test_snapshot_heatmap_rejects_all_nan_times() -> None:
    df = pd.DataFrame(
        {
            "time": [np.nan],
            "channel": ["YFP"],
            "value": [1.0],
            "treatment": ["a"],
            "design_id": ["d1"],
        }
    )

    with pytest.raises(ValueError, match="snapshot_heatmap: no valid time values"):
        plot_snapshot_heatmap(
            df=df,
            blanks=pd.DataFrame(),
            output_dir=None,
            channel="YFP",
            time=0.0,
            fig_kwargs={},
            filename=None,
        )


def test_snapshot_heatmap_rejects_all_nan_values() -> None:
    df = pd.DataFrame(
        {
            "time": [0.0],
            "channel": ["YFP"],
            "value": [np.nan],
            "treatment": ["a"],
            "design_id": ["d1"],
        }
    )

    with pytest.raises(ValueError, match="selected snapshot contains no finite values"):
        plot_snapshot_heatmap(
            df=df,
            blanks=pd.DataFrame(),
            output_dir=None,
            channel="YFP",
            time=0.0,
            fig_kwargs={},
            filename=None,
        )


def test_snapshot_heatmap_rejects_missing_explicit_order_labels() -> None:
    df = pd.DataFrame(
        {
            "time": [0.0],
            "channel": ["YFP"],
            "value": [1.0],
            "treatment": ["a"],
            "design_id": ["d1"],
        }
    )

    with pytest.raises(ValueError, match="x order includes missing label"):
        plot_snapshot_heatmap(
            df=df,
            blanks=pd.DataFrame(),
            output_dir=None,
            channel="YFP",
            time=0.0,
            order_x=["missing"],
            fig_kwargs={},
            filename=None,
        )


def test_snapshot_heatmap_render_resolves_order_refs_from_semantics() -> None:
    ctx = SimpleNamespace(
        logger=logging.getLogger("reader.tests"),
        experiment=ExperimentSemantics(
            protocol=ProtocolBinding(id="plate_reader/dual_reporter_screen"),
            annotations=AnnotationSemantics(
                orders=AnnotationOrders(
                    by_id={
                        "states_2x2": AnnotationOrderSpec(
                            column="treatment_alias",
                            values=["-IPTG/-stress", "+IPTG/-stress", "-IPTG/+stress", "+IPTG/+stress"],
                        ),
                        "screen_rows": AnnotationOrderSpec(column="design_id_alias", values=["spyP/tetO", "spyP/CpxR"]),
                    }
                )
            ),
            resources=ResourceCatalog(),
            layout=OutputLayout(
                outputs_dir=Path("."), plots_subdir="plots", exports_subdir="exports", notebooks_subdir="notebooks"
            ),
        ),
    )
    df = pd.DataFrame(
        {
            "time": [0.0, 0.0, 0.0, 0.0],
            "channel": ["YFP/CFP", "YFP/CFP", "YFP/CFP", "YFP/CFP"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "treatment_alias": ["+IPTG/-stress", "-IPTG/-stress", "-IPTG/+stress", "+IPTG/+stress"],
            "design_id_alias": ["spyP/CpxR", "spyP/tetO", "spyP/CpxR", "spyP/tetO"],
        }
    )
    cfg = HeatmapCfg.model_validate(
        {
            "channel": "YFP/CFP",
            "time": 0.0,
            "x": "treatment_alias",
            "y": "design_id_alias",
            "order_x_ref": "states_2x2",
            "order_y_ref": "screen_rows",
        }
    )

    figures = SnapshotHeatmapPlot().render(ctx, {"df": df}, cfg)

    assert len(figures) == 1
    assert figures[0].filename.startswith("snapshot_heatmap__YFP/CFP__t0h")


def test_snapshot_heatmap_render_rejects_unknown_order_ref() -> None:
    ctx = SimpleNamespace(
        logger=logging.getLogger("reader.tests"),
        experiment=ExperimentSemantics(
            protocol=ProtocolBinding(id="plate_reader/dual_reporter_screen"),
            annotations=AnnotationSemantics(orders=AnnotationOrders()),
            resources=ResourceCatalog(),
            layout=OutputLayout(
                outputs_dir=Path("."), plots_subdir="plots", exports_subdir="exports", notebooks_subdir="notebooks"
            ),
        ),
    )
    df = pd.DataFrame(
        {
            "time": [0.0],
            "channel": ["YFP/CFP"],
            "value": [1.0],
            "treatment_alias": ["-IPTG/-stress"],
            "design_id_alias": ["spyP/tetO"],
        }
    )
    cfg = HeatmapCfg.model_validate(
        {
            "channel": "YFP/CFP",
            "time": 0.0,
            "x": "treatment_alias",
            "y": "design_id_alias",
            "order_x_ref": "states_2x2",
        }
    )

    with pytest.raises(ValueError, match="Unknown order_x_ref"):
        SnapshotHeatmapPlot().render(ctx, {"df": df}, cfg)


def test_validate_rejects_unknown_heatmap_order_ref(tmp_path) -> None:
    data = {
        "schema": "reader/v7",
        "experiment": {"id": "exp_semantics"},
        "protocol": {
            "id": "plate_reader/dual_reporter_screen",
            "analysis": {"include_fold_change": False},
            "outputs": {
                "plots": {
                    "profile": "none",
                    "include": ["ratio_heatmap"],
                    "views": {
                        "ratio_heatmap": {
                            "channel": "YFP/CFP",
                            "time": 0.0,
                            "x": "treatment_alias",
                            "y": "design_id_alias",
                            "order_x_ref": "missing",
                        }
                    },
                }
            },
        },
        "resources": {"sample_map": {"kind": "file", "path": "./inputs/metadata.xlsx"}},
        "annotations": {"orders": {"rows": {"column": "design_id_alias", "values": ["d1"]}}},
    }
    decl = load_decl(write_config(tmp_path, data))

    with pytest.raises(ConfigError, match="invalid ordering semantic reference"):
        validate_job(decl, console=Console())


def test_prepare_snapshot_heatmap_inputs_for_fc_channel_uses_nearest_time() -> None:
    ctx = SimpleNamespace(logger=logging.getLogger("reader.tests"))
    fc = pd.DataFrame(
        {
            "time": [0.5, 1.0, 1.0],
            "target": ["YFP/CFP", "YFP/CFP", "YFP/CFP"],
            "FC": [1.2, 1.5, 1.6],
            "log2FC": [0.26, 0.58, 0.68],
            "treatment": ["a", "a", "b"],
            "design_id": ["d1", "d2", "d3"],
        }
    )
    cfg = HeatmapCfg.model_validate({"channel": "log2FC_YFP/CFP", "time": 0.9, "time_tolerance": 0.2})

    prepared = prepare_snapshot_heatmap_inputs(ctx=ctx, df_in=None, fc_in=fc, cfg=cfg)

    prepared_df = prepared["df"]
    assert prepared["filename"] == "snapshot_heatmap__log2FC_YFP/CFP__t1h"
    assert prepared["fig_kwargs"]["cbar_label"] == "log2FC (YFP/CFP)"
    assert prepared_df["time"].tolist() == [1.0, 1.0]
    assert prepared_df["channel"].tolist() == ["log2FC_YFP/CFP", "log2FC_YFP/CFP"]
    assert prepared_df["value"].tolist() == [0.58, 0.68]


def test_prepare_snapshot_heatmap_inputs_applies_log_transform_to_matching_channel_only() -> None:
    ctx = SimpleNamespace(logger=logging.getLogger("reader.tests"))
    df = pd.DataFrame(
        {
            "time": [0.0, 0.0, 0.0],
            "channel": ["YFP", "YFP", "CFP"],
            "value": [4.0, -1.0, 8.0],
            "treatment": ["a", "b", "a"],
            "design_id": ["d1", "d2", "d1"],
        }
    )
    cfg = HeatmapCfg.model_validate({"channel": "YFP", "time": 0.0, "value_transform": "log2"})

    prepared = prepare_snapshot_heatmap_inputs(ctx=ctx, df_in=df, fc_in=None, cfg=cfg)

    prepared_df = prepared["df"]
    yfp_values = prepared_df.loc[prepared_df["channel"] == "YFP", "value"].tolist()
    cfp_values = prepared_df.loc[prepared_df["channel"] == "CFP", "value"].tolist()
    assert yfp_values[0] == 2.0
    assert np.isnan(yfp_values[1])
    assert cfp_values == [8.0]
    assert prepared["fig_kwargs"]["cbar_label"] == "log2(YFP)"


def test_snapshot_heatmap_config_rejects_mixed_order_sources() -> None:
    with pytest.raises(ValidationError, match="order_x and order_x_ref"):
        HeatmapCfg.model_validate(
            {
                "channel": "YFP/CFP",
                "time": 0.0,
                "order_x": ["a"],
                "order_x_ref": "states",
            }
        )


def test_time_series_summary_uses_bounded_bootstrap_controls_deterministically() -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "value": [1.0, 1.3, 1.9, 2.0, 2.4, 3.1],
            "treatment": ["a", "a", "a", "a", "a", "a"],
        }
    )

    first = time_series_panel._summarize_time_series_lines(
        data=df,
        x_col="time",
        hue_col="treatment",
        hue_levels=["a"],
        segment_col=None,
        ci=95.0,
        ci_boot=17,
        ci_seed=23,
    )["a"]
    second = time_series_panel._summarize_time_series_lines(
        data=df,
        x_col="time",
        hue_col="treatment",
        hue_levels=["a"],
        segment_col=None,
        ci=95.0,
        ci_boot=17,
        ci_seed=23,
    )["a"]
    changed_seed = time_series_panel._summarize_time_series_lines(
        data=df,
        x_col="time",
        hue_col="treatment",
        hue_levels=["a"],
        segment_col=None,
        ci=95.0,
        ci_boot=17,
        ci_seed=29,
    )["a"]
    no_ci = time_series_panel._summarize_time_series_lines(
        data=df,
        x_col="time",
        hue_col="treatment",
        hue_levels=["a"],
        segment_col=None,
        ci=0.0,
        ci_boot=17,
        ci_seed=23,
    )["a"]

    assert first["lower"].tolist() == second["lower"].tolist()
    assert first["upper"].tolist() == second["upper"].tolist()
    assert first["lower"].tolist() == changed_seed["lower"].tolist()
    assert first["upper"].tolist() == changed_seed["upper"].tolist()
    assert no_ci["lower"].tolist() == no_ci["mean"].tolist()
    assert no_ci["upper"].tolist() == no_ci["mean"].tolist()


def test_time_series_summary_keeps_plate_segments_separate_when_x_values_repeat() -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "value": [1.0, 1.2, 2.0, 2.2, 3.0, 3.2, 4.0, 4.2],
            "treatment": ["a"] * 8,
            "segment": ["plate_1"] * 4 + ["plate_2"] * 4,
        }
    )

    summary = time_series_panel._summarize_time_series_lines(
        data=df,
        x_col="time",
        hue_col="treatment",
        hue_levels=["a"],
        segment_col="segment",
        ci=95.0,
        ci_boot=17,
        ci_seed=23,
    )["a"]

    assert summary["segment"].tolist() == ["plate_1", "plate_1", "plate_2", "plate_2"]
    assert summary["mean"].tolist() == pytest.approx([1.1, 2.1, 3.1, 4.1])


def test_time_series_plot_renders_separate_segment_lines_with_ci() -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0],
            "channel": ["OD600"] * 8,
            "value": [0.10, 0.12, 0.20, 0.22, 0.35, 0.37, 0.50, 0.52],
            "treatment": ["-IPTG/-stress"] * 8,
            "sheet_index": [0, 0, 0, 0, 1, 1, 1, 1],
            "sheet_name": ["Plate 1"] * 4 + ["Plate 2"] * 4,
            "source": ["snapshot"] * 4 + ["kinetic"] * 4,
        }
    )

    figures = plot_time_series(
        df=df,
        blanks=df.iloc[0:0].copy(),
        output_dir=None,
        x="time",
        y=["OD600"],
        hue="treatment",
        channels=None,
        subplots=None,
        group_on=None,
        pool_sets=None,
        pool_match="exact",
        fig_kwargs={},
        add_sheet_line=True,
        sheet_line_kwargs={},
        log_transform=False,
        time_window=None,
        palette_book=None,
        ci=95.0,
        ci_alpha=0.15,
        ci_boot=17,
        ci_seed=23,
        legend_loc="upper left",
        show_replicates=False,
        filename=None,
        xlabel="Time from stress addition (h)",
        ylabel_map={"OD600": "Biomass proxy (OD600)"},
        hue_label_map=None,
        shared_legend=False,
    )

    assert len(figures) == 1
    axis = figures[0].fig.axes[0]
    solid_lines = [line for line in axis.lines if line.get_linestyle() == "-"]
    dashed_lines = [line for line in axis.lines if line.get_linestyle() == "--"]

    assert len(solid_lines) == 2
    assert [list(line.get_xdata()) for line in solid_lines] == [[0.0, 1.0], [3.0, 4.0]]
    assert axis.collections
    assert len(dashed_lines) == 1
    assert list(dashed_lines[0].get_xdata()) == [3.0, 3.0]


def test_time_series_plot_uses_actual_stress_names_in_retron_treatment_legend() -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 0.0, 0.0, 0.0],
            "channel": ["OD600"] * 4,
            "value": [0.10, 0.11, 0.20, 0.22],
            "treatment_alias": ["-IPTG/+stress", "-IPTG/+stress", "+IPTG/+stress", "+IPTG/+stress"],
            "treatment": ["3% EtOH", "3% EtOH", "500 uM IPTG, 3% EtOH", "500 uM IPTG, 3% EtOH"],
        }
    )

    figures = plot_time_series(
        df=df,
        blanks=df.iloc[0:0].copy(),
        output_dir=None,
        x="time",
        y=["OD600"],
        hue="treatment_alias",
        channels=None,
        subplots=None,
        group_on=None,
        pool_sets=None,
        pool_match="exact",
        fig_kwargs={},
        add_sheet_line=False,
        sheet_line_kwargs={},
        log_transform=False,
        time_window=None,
        palette_book=None,
        ci=95.0,
        ci_alpha=0.15,
        ci_boot=17,
        ci_seed=23,
        legend_loc="upper left",
        show_replicates=False,
        filename=None,
        xlabel="Time from stress addition (h)",
        ylabel_map={"OD600": "OD600"},
        hue_label_map={
            "-IPTG/+stress": "Relevant stress, -IPTG",
            "+IPTG/+stress": "Relevant stress, +IPTG",
        },
        shared_legend=False,
    )

    axis = figures[0].fig.axes[0]
    legend = axis.get_legend()

    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {"3% EtOH, -IPTG", "3% EtOH, +IPTG"}


def test_retron_trace_uses_single_shared_xlabel_and_non_overlapping_title_layers() -> None:
    trace = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "tetO",
                "stress_condition": stress,
                "time_from_stress": time_value,
                "metric": metric,
                "value": value,
                "IPTG": iptg,
            }
            for metric, values in {
                "R": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "mu": [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12],
            }.items()
            for stress, iptg, time_value, value in zip(
                ["H2O", "H2O", "H2O", "H2O", "3% EtOH", "3% EtOH", "3% EtOH", "3% EtOH"],
                ["-IPTG", "-IPTG", "+IPTG", "+IPTG", "-IPTG", "-IPTG", "+IPTG", "+IPTG"],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                values,
                strict=True,
            )
        ]
    )

    figures = plot_retron_sponge_trace(
        trace=trace,
        output_dir=None,
        metrics=["R", "mu"],
        title="Control burden panel",
        filename="control_burden_panel",
        palette_book=None,
        only_control=True,
        metric_label_map={"R": "log2(YFP/CFP)", "mu": "d ln(OD600) / dt"},
    )

    assert len(figures) == 1
    figure = figures[0].fig
    assert figure._supxlabel is None
    assert all(axis.get_xlabel() == "Time from stress addition (h)" for axis in figure.axes)
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == "Control burden panel · spyP"
    assert [axis.title.get_text() for axis in figure.axes] == [
        "R · H2O",
        "R · 3% EtOH",
        "mu · H2O",
        "mu · 3% EtOH",
    ]
    assert figure.axes[0].get_ylim() == pytest.approx(figure.axes[1].get_ylim())
    assert figure.axes[2].get_ylim() == pytest.approx(figure.axes[3].get_ylim())
    assert figure.axes[0].get_ylim()[0] <= 0.0 <= figure.axes[0].get_ylim()[1]
    assert any(np.allclose(line.get_ydata(), [0.0, 0.0]) for line in figure.axes[0].lines)
    stress_text = next(text for text in figure.axes[0].texts if text.get_text() == "Stress addition")
    assert stress_text.get_va() == "bottom"
    assert figure.axes[0].get_ylabel() == "log2(YFP/CFP)"
    assert figure.axes[2].get_ylabel() == "d ln(OD600) / dt"
    assert not any("R(t)=log2(YFP/CFP)" in text.get_text() for text in figure.axes[0].texts)
    assert not any("mu(t)=d ln(OD600) / dt" in text.get_text() for text in figure.axes[2].texts)
    assert figure.axes[0].get_box_aspect() == pytest.approx(1.0)
    assert not any("Dashed line = stress addition" in text.get_text() for text in figure.texts)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    suptitle_bbox = figure._suptitle.get_window_extent(renderer)
    for axis in figure.axes[:2]:
        assert not suptitle_bbox.overlaps(axis.title.get_window_extent(renderer))
    legend = figure.legends[0]
    legend_bbox = legend.get_window_extent(renderer)
    assert not any(legend_bbox.overlaps(axis.get_window_extent(renderer)) for axis in figure.axes)
    horizontal_gap = figure.axes[1].get_position().x0 - figure.axes[0].get_position().x1
    vertical_gap = figure.axes[0].get_position().y0 - figure.axes[2].get_position().y1
    assert horizontal_gap < 0.20
    assert vertical_gap < 0.12

    plt.close(figure)


def test_retron_stress_modulation_consolidates_sensors_on_one_axis() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "M_AUC",
                "value": -4.0,
                "relevant_sensor_pair": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "spyP",
                "sponge": "BaeR",
                "metric": "M_AUC",
                "value": 1.5,
                "relevant_sensor_pair": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "soxSp",
                "sponge": "SoxR-SoxS",
                "metric": "M_AUC",
                "value": -20.0,
                "relevant_sensor_pair": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "soxSp",
                "sponge": "BaeR-SoxR",
                "metric": "M_AUC",
                "value": 6.0,
                "relevant_sensor_pair": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "spyP",
                "sponge": "tetO",
                "metric": "M_AUC",
                "stress_condition": "3% EtOH",
                "value": 0.25,
                "relevant_sensor_pair": False,
                "sponge_family_size": "control",
            },
            {
                "sensor": "soxSp",
                "sponge": "tetO",
                "metric": "M_AUC",
                "stress_condition": "3% EtOH",
                "value": -0.10,
                "relevant_sensor_pair": False,
                "sponge_family_size": "control",
            },
        ]
    )
    trace = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": sensor,
                "stress_condition": stress,
                "time_from_stress": time_value,
                "in_primary_post_stress": True,
            }
            for sensor in ("spyP", "soxSp")
            for stress in ("H2O", "3% EtOH")
            for time_value in (0.5, 4.0)
        ]
    )

    figures = plot_retron_sponge_summary(
        summary=summary,
        trace=trace,
        output_dir=None,
        view="stress_modulation",
        title="Stress modulation scores",
        filename="stress_modulation_scores",
        palette_book=None,
        control_name="tetO",
        metric="M_AUC",
        fig_kwargs={},
    )

    figure = figures[0].fig
    axes = [axis for axis in figure.axes if axis.get_visible()]
    assert len(axes) == 1
    legend = axes[0].get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {"tetO reference", "Sample"}
    assert [label.get_text() for label in axes[0].get_yticklabels()] == [
        "soxSp\nBaeR-SoxR",
        "soxSp\nSoxR-SoxS",
        "spyP\nBaeR",
        "spyP\nCpxR",
    ]
    assert len(axes[0].patches) == 8
    assert any("AUC of the stress-specific gain" in text.get_text() for text in figure.texts)
    assert any(
        "Post-stress summary covers the first 4.0 h after stress addition" in text.get_text() for text in figure.texts
    )
    plt.close(figure)


def test_retron_library_heatmaps_render_as_single_shared_row() -> None:
    summary_rows = []
    for sensor, stress in (("spyP", "3% EtOH"), ("soxSp", "15 µM PMS")):
        for sponge, family, base in (("CpxR", "mono", 0.35), ("BaeR-SoxR", "bi", -0.22)):
            summary_rows.extend(
                [
                    {
                        "sensor": sensor,
                        "sponge": sponge,
                        "metric": "S_abs_AUC",
                        "stress_condition": stress,
                        "value": base + 0.22,
                        "relevant_sensor_pair": True,
                        "is_relevant_stress": True,
                        "sponge_family_size": family,
                    },
                    {
                        "sensor": sensor,
                        "sponge": sponge,
                        "metric": "S_AUC",
                        "stress_condition": stress,
                        "value": base + 0.18,
                        "relevant_sensor_pair": True,
                        "is_relevant_stress": True,
                        "sponge_family_size": family,
                    },
                    {
                        "sensor": sensor,
                        "sponge": sponge,
                        "metric": "P_pre",
                        "stress_condition": stress,
                        "value": base - 0.05,
                        "relevant_sensor_pair": True,
                        "is_relevant_stress": True,
                        "sponge_family_size": family,
                    },
                ]
            )
    summary = pd.DataFrame(summary_rows)
    trace = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": sensor,
                "stress_condition": stress,
                "time_from_stress": time_value,
                "in_primary_post_stress": True,
            }
            for sensor, stress in (("spyP", "3% EtOH"), ("soxSp", "15 µM PMS"))
            for time_value in (0.0, 4.0)
        ]
    )

    figures = plot_retron_sponge_summary(
        summary=summary,
        trace=trace,
        output_dir=None,
        view="heatmap",
        title="Library heatmaps",
        filename="library_heatmaps",
        palette_book=None,
        control_name="tetO",
        fig_kwargs={},
    )

    figure = figures[0].fig
    axes = [axis for axis in figure.axes if axis.get_visible()]
    assert len(axes) == 3
    y_offsets = [axis.get_position().y0 for axis in axes]
    assert max(y_offsets) - min(y_offsets) < 0.02
    assert [axis.get_title() for axis in axes] == [
        "Total effect\nS_abs_AUC = O_abs_AUC /\n|G_sensor|",
        "Post-stress\nS_AUC = O_AUC /\n|G_sensor|",
        "Preload\nP_pre = delta_IPTG[R_pre\n- R_pre,tetO]",
    ]
    assert any(label.get_visible() for label in axes[0].get_yticklabels())
    assert all(not label.get_visible() for label in axes[1].get_yticklabels())
    assert all(not label.get_visible() for label in axes[2].get_yticklabels())
    assert any(
        "Relevant-stress summary of total effect, post-stress movement, and preload" in text.get_text()
        for text in figure.texts
    )
    assert any(
        "Post-stress summary covers the first 4.0 h after stress addition" in text.get_text() for text in figure.texts
    )
    x_tick_sizes = {label.get_fontsize() for axis in axes for label in axis.get_xticklabels() if label.get_text()}
    y_tick_sizes = {label.get_fontsize() for label in axes[0].get_yticklabels() if label.get_text()}
    annotation_sizes = {text.get_fontsize() for axis in axes for text in axis.texts if text.get_text()}
    assert x_tick_sizes == {9.0}
    assert y_tick_sizes == {10.0}
    assert annotation_sizes == {8.5}
    plt.close(figure)


def test_retron_library_heatmaps_show_explicit_no_data_panel_for_missing_metric() -> None:
    summary_rows = []
    for sensor, stress in (("spyP", "3% EtOH"), ("soxSp", "15 µM PMS")):
        for sponge, family, base in (("CpxR", "mono", 0.35), ("BaeR-SoxR", "bi", -0.22)):
            summary_rows.extend(
                [
                    {
                        "sensor": sensor,
                        "sponge": sponge,
                        "metric": "S_abs_AUC",
                        "stress_condition": stress,
                        "value": base + 0.22,
                        "relevant_sensor_pair": True,
                        "is_relevant_stress": True,
                        "sponge_family_size": family,
                    },
                    {
                        "sensor": sensor,
                        "sponge": sponge,
                        "metric": "S_AUC",
                        "stress_condition": stress,
                        "value": base + 0.18,
                        "relevant_sensor_pair": True,
                        "is_relevant_stress": True,
                        "sponge_family_size": family,
                    },
                ]
            )
    summary = pd.DataFrame(summary_rows)

    figures = plot_retron_sponge_summary(
        summary=summary,
        trace=None,
        output_dir=None,
        view="heatmap",
        title="Library heatmaps",
        filename="library_heatmaps",
        palette_book=None,
        control_name="tetO",
        fig_kwargs={},
    )

    figure = figures[0].fig
    no_data_axes = [axis for axis in figure.axes if any(text.get_text() == "No data" for text in axis.texts)]
    assert len(no_data_axes) == 1
    assert not no_data_axes[0].axison
    plt.close(figure)


def test_retron_pareto_tick_labels_use_compact_font() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "S_abs_AUC",
                "value": 0.40,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "soxSp",
                "sponge": "SoxR-SoxS",
                "metric": "S_abs_AUC",
                "value": 0.55,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "L_pre",
                "value": 0.02,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "soxSp",
                "sponge": "SoxR-SoxS",
                "metric": "L_pre",
                "value": -0.03,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "D_growth_AUC",
                "value": -0.01,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "soxSp",
                "sponge": "SoxR-SoxS",
                "metric": "D_growth_AUC",
                "value": 0.04,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "bi",
            },
        ]
    )

    figures = plot_retron_sponge_summary(
        summary=summary,
        trace=None,
        output_dir=None,
        view="pareto",
        title="Pareto ranking",
        filename="pareto_ranking",
        palette_book=None,
        control_name="tetO",
        fig_kwargs={},
    )

    axis = figures[0].fig.axes[0]
    x_tick_sizes = {label.get_fontsize() for label in axis.get_xticklabels() if label.get_text()}
    y_tick_sizes = {label.get_fontsize() for label in axis.get_yticklabels() if label.get_text()}
    legend = axis.get_legend()
    assert any(
        "Expected-direction total effect scaled by the native sensor range" in text.get_text()
        for text in figures[0].fig.texts
    )
    assert axis.get_xlabel() == "Mean on-target effect across relevant sensors (S_abs_AUC)"
    assert axis.get_ylabel() == "Mean burden penalty (-D_growth_AUC)"
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {"mono", "bi"}
    assert x_tick_sizes == {7.0}
    assert y_tick_sizes == {7.0}
    plt.close(figures[0].fig)


def test_retron_induced_effect_trace_derives_confidence_band_from_c_replicates() -> None:
    rows = []
    for time_value in (0.0, 1.0):
        for iptg, base in (("-IPTG", 0.2), ("+IPTG", -0.4)):
            for idx, delta in enumerate((0.00, 0.05, -0.04), start=1):
                rows.append(
                    {
                        "sensor": "spyP",
                        "sponge": "CpxR",
                        "stress_condition": "3% EtOH",
                        "time_from_stress": time_value,
                        "metric": "C",
                        "value": base + time_value + delta,
                        "IPTG": iptg,
                        "replicate_id": f"r{idx}",
                        "in_primary_post_stress": True,
                        "is_relevant_stress": True,
                        "relevant_sensor_pair": True,
                        "expected_decoy_sign": -1,
                    }
                )
        rows.append(
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "stress_condition": "3% EtOH",
                "time_from_stress": time_value,
                "metric": "D",
                "value": -0.6,
                "IPTG": pd.NA,
                "replicate_id": pd.NA,
                "in_primary_post_stress": True,
                "is_relevant_stress": True,
                "relevant_sensor_pair": True,
                "expected_decoy_sign": -1,
            }
        )
    trace = pd.DataFrame(rows)
    figures = plot_retron_sponge_trace(
        trace=trace,
        output_dir=None,
        metrics=["D"],
        title="IPTG-state effect kinetics",
        filename="induced_effect_kinetics",
        palette_book=None,
        relevant_only=True,
        panel_by="sponge",
    )

    figure = figures[0].fig
    axis = figure.axes[0]
    assert len(axis.collections) >= 1
    assert len(axis.patches) == 1
    assert axis.patches[0].get_alpha() == pytest.approx(0.14)
    assert any("New post-stress movement after preload is removed" in text.get_text() for text in figure.texts)
    assert not any("first 1.0 h after stress addition" in text.get_text() for text in figure.texts)
    assert not any("Dashed line = stress addition" in text.get_text() for text in figure.texts)
    assert axis.get_title() == "spyP · CpxR"
    assert not any(current.get_visible() for current in axis.child_axes)
    plt.close(figure)


def test_retron_absolute_effect_trace_derives_confidence_band_from_r_replicates() -> None:
    rows = []
    for time_value in (0.0, 1.0):
        for iptg, base in (("-IPTG", 0.9), ("+IPTG", 1.2)):
            for idx, delta in enumerate((0.00, 0.04, -0.03), start=1):
                rows.append(
                    {
                        "sensor": "sulAp",
                        "sponge": "LexA",
                        "stress_condition": "100 nM ciprofloxacin",
                        "time_from_stress": time_value,
                        "metric": "R",
                        "value": base + time_value + delta,
                        "IPTG": iptg,
                        "replicate_id": f"r{idx}",
                        "in_primary_post_stress": True,
                        "is_relevant_stress": True,
                        "relevant_sensor_pair": True,
                        "expected_decoy_sign": 1,
                        "configured_max_post_stress_hours": 4.0,
                    }
                )
        rows.append(
            {
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "time_from_stress": time_value,
                "metric": "D_abs",
                "value": 0.3,
                "IPTG": pd.NA,
                "replicate_id": pd.NA,
                "in_primary_post_stress": True,
                "is_relevant_stress": True,
                "relevant_sensor_pair": True,
                "expected_decoy_sign": 1,
                "configured_max_post_stress_hours": 4.0,
            }
        )
    trace = pd.DataFrame(rows)
    figures = plot_retron_sponge_trace(
        trace=trace,
        output_dir=None,
        metrics=["D_abs"],
        title="Absolute matched-control effect kinetics",
        filename="absolute_effect_kinetics",
        palette_book=None,
        relevant_only=True,
        panel_by="sponge",
    )

    figure = figures[0].fig
    axis = figure.axes[0]
    assert len(axis.collections) >= 1
    assert len(axis.patches) == 1
    assert axis.patches[0].get_alpha() == pytest.approx(0.14)
    assert any("Total +IPTG effect beyond matched tetO over time" in text.get_text() for text in figure.texts)
    assert not any("first 4.0 h after stress addition" in text.get_text() for text in figure.texts)
    assert not any("Dashed line = stress addition" in text.get_text() for text in figure.texts)
    assert axis.get_title() == "sulAp · LexA"
    assert not any(current.get_visible() for current in axis.child_axes)
    plt.close(figure)


def test_retron_matched_control_trace_can_facet_by_sponge() -> None:
    rows = []
    for sponge, offset in (("SoxR", 0.15), ("SoxS", -0.05)):
        for stress, stress_offset in (("H2O", 0.0), ("15 µM PMS", 0.25)):
            for iptg, iptg_offset in (("-IPTG", -0.08), ("+IPTG", 0.12)):
                for time_value in (-0.5, 0.0, 1.0):
                    rows.append(
                        {
                            "sensor": "soxSp",
                            "sponge": sponge,
                            "stress_condition": stress,
                            "time_from_stress": time_value,
                            "metric": "C",
                            "value": offset + stress_offset + iptg_offset + time_value * 0.05,
                            "IPTG": iptg,
                            "replicate_id": f"{sponge}-{stress}-{iptg}",
                            "in_primary_post_stress": time_value >= 0.0,
                            "relevant_sensor_pair": True,
                            "configured_max_post_stress_hours": 4.0,
                        }
                    )
    trace = pd.DataFrame(rows)
    figures = plot_retron_sponge_trace(
        trace=trace,
        output_dir=None,
        metrics=["C"],
        title="Matched-control-normalized kinetics",
        filename="matched_control_kinetics",
        palette_book=None,
        relevant_only=True,
        panel_by="sponge",
    )

    figure = figures[0].fig
    axes = [axis for axis in figure.axes if axis.get_visible()]
    assert [axis.get_title() for axis in axes] == ["SoxR", "SoxS"]
    assert all(axis.get_box_aspect() == pytest.approx(1.0) for axis in axes)
    assert len(figure.legends) == 1
    legend_labels = [text.get_text() for text in figure.legends[0].texts]
    assert legend_labels == ["H2O, -IPTG", "H2O, +IPTG", "15 µM PMS, -IPTG", "15 µM PMS, +IPTG"]
    assert any("Deviation from the matched tetO control over time" in text.get_text() for text in figure.texts)
    assert axes[0].get_xlim()[1] == pytest.approx(4.0)
    plt.close(figure)


def test_retron_trace_rejects_unknown_panel_by_mode() -> None:
    trace = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "stress_condition": "3% EtOH",
                "time_from_stress": 0.0,
                "metric": "D",
                "value": -0.3,
                "IPTG": pd.NA,
                "relevant_sensor_pair": True,
            }
        ]
    )

    with pytest.raises(ValueError, match="panel_by supports only 'stress' or 'sponge'"):
        plot_retron_sponge_trace(
            trace=trace,
            output_dir=None,
            metrics=["D"],
            title="IPTG-state effect kinetics",
            filename="induced_effect_kinetics",
            palette_book=None,
            relevant_only=True,
            panel_by="bad-mode",
        )


def test_retron_control_anchored_decomposition_renders_full_time_matched_traces() -> None:
    rows = []
    for stress_condition, is_relevant_stress, offset in (
        ("H2O", False, 0.00),
        ("100 nM ciprofloxacin", True, 0.12),
    ):
        for sponge, control_flag, minus_values, plus_values in (
            ("LexA", False, (0.85, 0.88, 0.83), (1.28, 1.31, 1.26)),
            ("tetO", True, (0.72, 0.75, 0.70), (0.90, 0.92, 0.89)),
        ):
            for iptg, values in (("-IPTG", minus_values), ("+IPTG", plus_values)):
                for idx, value in enumerate(values, start=1):
                    rows.append(
                        {
                            "plate_id": "plate-1",
                            "sensor": "sulAp",
                            "sponge": sponge,
                            "stress_condition": stress_condition,
                            "time_from_stress": 0.0,
                            "metric": "R",
                            "value": value + offset,
                            "IPTG": iptg,
                            "replicate_id": f"r{idx}",
                            "in_pre_window": True,
                            "in_primary_post_stress": True,
                            "is_relevant_stress": is_relevant_stress,
                            "relevant_sensor_pair": not control_flag,
                            "configured_max_post_stress_hours": 4.0,
                            "matched_control_key": f"plate-1::sulAp::{stress_condition}",
                            "summary_window_start_h": 0.0,
                            "summary_window_end_h": 4.0,
                            "summary_window_duration_h": 4.0,
                            "pre_stress_read_count": 1,
                            "post_stress_read_count": 2,
                            "matched_group_sample_count": 3,
                            "stress_addition_gap_h": 0.5,
                        }
                    )
                    rows.append(
                        {
                            "plate_id": "plate-1",
                            "sensor": "sulAp",
                            "sponge": sponge,
                            "stress_condition": stress_condition,
                            "time_from_stress": 4.0,
                            "metric": "R",
                            "value": value + offset + 0.10,
                            "IPTG": iptg,
                            "replicate_id": f"r{idx}",
                            "in_pre_window": False,
                            "in_primary_post_stress": True,
                            "is_relevant_stress": is_relevant_stress,
                            "relevant_sensor_pair": not control_flag,
                            "configured_max_post_stress_hours": 4.0,
                            "matched_control_key": f"plate-1::sulAp::{stress_condition}",
                            "summary_window_start_h": 0.0,
                            "summary_window_end_h": 4.0,
                            "summary_window_duration_h": 4.0,
                            "pre_stress_read_count": 1,
                            "post_stress_read_count": 2,
                            "matched_group_sample_count": 3,
                            "stress_addition_gap_h": 0.5,
                        }
                    )
    trace = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "P_pre",
                "value": 0.21,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "D_abs_AUC",
                "value": 0.42,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "D_AUC",
                "value": 0.18,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "D_growth_AUC",
                "value": -0.05,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "tetO",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "G_sensor",
                "value": 0.60,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "control",
            },
        ]
    )

    figures = plot_retron_sponge_summary(
        summary=summary,
        trace=trace,
        output_dir=None,
        view="decomposition",
        title="Sponge vs matched tetO",
        filename="control_anchored_decomposition",
        palette_book=None,
        control_name="tetO",
        fig_kwargs={},
    )

    figure = figures[0].fig
    h2o_axis, relevant_axis = figure.axes[:2]
    summary_axes = figure.axes[2:]
    assert len(figure.axes) == 5
    assert relevant_axis.get_xlabel() == "Time from stress addition (h)"
    assert h2o_axis.get_xlabel() == "Time from stress addition (h)"
    assert relevant_axis.get_ylabel() == "Reporter ratio R(t)"
    assert h2o_axis.get_ylabel() == "Reporter ratio R(t)"
    assert any("Stress addition" in text.get_text() for text in relevant_axis.texts)
    assert len(relevant_axis.patches) == 1
    assert relevant_axis.patches[0].get_alpha() == pytest.approx(0.14)
    assert len(h2o_axis.patches) == 1
    assert h2o_axis.get_title() == "sulAp · LexA · H2O"
    assert relevant_axis.get_title() == "sulAp · LexA · 100 nM ciprofloxacin"
    assert "\n" not in relevant_axis.get_title()
    assert "\n" not in h2o_axis.get_title()
    legend = relevant_axis.get_legend()
    assert legend is not None
    assert h2o_axis.get_legend() is None
    legend_labels = [text.get_text() for text in legend.texts]
    assert legend_labels == ["Sample -IPTG", "Sample +IPTG", "tetO -IPTG", "tetO +IPTG"]
    assert not any("R(t)=log2(YFP/CFP)" in text.get_text() for text in figure.texts)
    assert not any("D_abs_AUC=AUC_window[D_abs(t)]" in text.get_text() for text in figure.texts)
    assert len(summary_axes) == 3
    assert [axis.get_title(loc="left") for axis in summary_axes] == [
        "Preload shift",
        "Total effect beyond matched tetO",
        "Post-stress increment",
    ]
    assert [axis.get_xlabel() for axis in summary_axes] == [
        "log2 ratio",
        "log2 ratio x h",
        "log2 ratio x h",
    ]
    assert [axis.get_ylabel() for axis in summary_axes] == ["State", "State", "State"]
    assert [tick.get_text() for tick in summary_axes[0].get_yticklabels()] == ["-IPTG", "+IPTG", "ΔIPTG"]
    assert all(len(axis.collections) >= 2 for axis in summary_axes)
    assert any(
        "Matched-tetO reporter-ratio traces with preload and post-stress summaries" in text.get_text()
        for text in figure.texts
    )
    plt.close(figure)


def test_retron_decomposition_requires_trace_input() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "sulAp",
                "sponge": "LexA",
                "metric": "D_abs_AUC",
                "value": 0.42,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            }
        ]
    )

    with pytest.raises(ValueError, match="trace input is required"):
        plot_retron_sponge_summary(
            summary=summary,
            trace=None,
            output_dir=None,
            view="decomposition",
            title="Sponge vs matched tetO",
            filename="control_anchored_decomposition",
            palette_book=None,
            control_name="tetO",
            fig_kwargs={},
        )


def test_retron_decomposition_derives_missing_window_metadata_when_trace_shape_is_still_recoverable() -> None:
    trace = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "time_from_stress": 0.0,
                "metric": "R",
                "value": 1.0,
                "IPTG": "-IPTG",
                "replicate_id": "r1",
                "in_pre_window": True,
                "in_primary_post_stress": True,
                "is_relevant_stress": True,
                "relevant_sensor_pair": True,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "P_pre",
                "value": 0.1,
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "D_abs_AUC",
                "value": 0.2,
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "D_AUC",
                "value": 0.1,
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "D_growth_AUC",
                "value": -0.1,
            },
        ]
    )

    figures = plot_retron_sponge_summary(
        summary=summary,
        trace=trace,
        output_dir=None,
        view="decomposition",
        title="Sponge vs matched tetO",
        filename="control_anchored_decomposition",
        palette_book=None,
        control_name="tetO",
        fig_kwargs={},
    )

    figure = figures[0].fig
    assert len(figure.axes) == 5
    assert figure.axes[0].get_xlabel() == "Time from stress addition (h)"
    plt.close(figure)


def test_retron_decomposition_rejects_missing_required_decision_metrics() -> None:
    trace = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": sponge,
                "stress_condition": "100 nM ciprofloxacin",
                "time_from_stress": time_value,
                "metric": "R",
                "value": value,
                "IPTG": iptg,
                "replicate_id": "r1",
                "in_pre_window": time_value == 0.0,
                "in_primary_post_stress": True,
                "is_relevant_stress": sponge != "tetO",
                "relevant_sensor_pair": sponge != "tetO",
                "matched_control_key": "plate-1::sulAp::100 nM ciprofloxacin",
                "summary_window_start_h": 0.0,
                "summary_window_end_h": 4.0,
                "summary_window_duration_h": 4.0,
                "pre_stress_read_count": 1,
                "post_stress_read_count": 2,
                "matched_group_sample_count": 1,
                "stress_addition_gap_h": 0.5,
            }
            for sponge, iptg, value in (
                ("LexA", "-IPTG", 1.0),
                ("LexA", "+IPTG", 1.2),
                ("tetO", "-IPTG", 0.8),
                ("tetO", "+IPTG", 0.9),
            )
            for time_value in (0.0, 4.0)
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "P_pre",
                "value": 0.1,
            },
            {
                "plate_id": "plate-1",
                "sensor": "sulAp",
                "sponge": "LexA",
                "stress_condition": "100 nM ciprofloxacin",
                "metric": "D_abs_AUC",
                "value": 0.2,
            },
        ]
    )

    with pytest.raises(ValueError, match="missing required matched-tetO summary metrics"):
        plot_retron_sponge_summary(
            summary=summary,
            trace=trace,
            output_dir=None,
            view="decomposition",
            title="Sponge vs matched tetO",
            filename="control_anchored_decomposition",
            palette_book=None,
            control_name="tetO",
            fig_kwargs={},
        )


def test_retron_interaction_summary_uses_trace_replicates_and_renders_subplots() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "C_AUC",
                "value": 1.0,
                "relevant_sensor_pair": True,
                "stress_condition": "H2O",
                "IPTG": "-IPTG",
                "sponge_family_size": "mono",
            }
        ]
    )
    rows = []
    for stress in ("H2O", "3% EtOH"):
        for iptg, baseline in (("-IPTG", 0.2), ("+IPTG", -0.3)):
            for replicate_id, delta in (("r1", 0.00), ("r2", 0.05), ("r3", -0.04)):
                for time_value, value in ((0.0, baseline + delta), (1.0, baseline + delta + 0.2)):
                    rows.append(
                        {
                            "plate_id": "plate-1",
                            "sensor": "spyP",
                            "sponge": "CpxR",
                            "genotype_id": "spyP/CpxR",
                            "replicate_id": replicate_id,
                            "stress_condition": stress,
                            "IPTG": iptg,
                            "time": time_value,
                            "time_from_stress": time_value,
                            "metric": "C",
                            "value": value,
                            "in_primary_post_stress": True,
                            "in_endpoint_window": True,
                            "relevant_sensor_pair": True,
                            "is_relevant_stress": stress == "3% EtOH",
                            "sponge_family_size": "mono",
                        }
                    )
    trace = pd.DataFrame(rows)

    figures = plot_retron_sponge_summary(
        summary=summary,
        trace=trace,
        output_dir=None,
        view="interaction",
        title="IPTG and stress state summary",
        filename="interaction_summary",
        palette_book=None,
        control_name="tetO",
        metric="C_AUC",
        fig_kwargs={},
    )

    figure = figures[0].fig
    axes = [axis for axis in figure.axes if axis.get_visible()]
    assert len(axes) == 1
    assert axes[0].get_title() == "CpxR"
    assert len(axes[0].patches) == 4
    assert [label.get_text() for label in axes[0].get_xticklabels()] == [
        "H2O\n-IPTG",
        "H2O\n+IPTG",
        "3% EtOH\n-IPTG",
        "3% EtOH\n+IPTG",
    ]
    assert any("AUC of the matched tetO deviation trace" in text.get_text() for text in figure.texts)
    assert any(
        "Post-stress summary covers the first 1.0 h after stress addition" in text.get_text() for text in figure.texts
    )
    plt.close(figure)


def test_retron_interaction_summary_rejects_unsupported_metric() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "C_AUC",
                "value": 1.0,
                "relevant_sensor_pair": True,
            }
        ]
    )
    trace = pd.DataFrame(
        [
            {
                "plate_id": "plate-1",
                "sensor": "spyP",
                "sponge": "CpxR",
                "genotype_id": "spyP/CpxR",
                "replicate_id": "r1",
                "stress_condition": "H2O",
                "IPTG": "-IPTG",
                "time": 0.0,
                "metric": "C",
                "value": 0.2,
                "in_primary_post_stress": True,
                "in_endpoint_window": True,
                "relevant_sensor_pair": True,
            }
        ]
    )

    with pytest.raises(ValueError, match="unsupported metric"):
        plot_retron_sponge_summary(
            summary=summary,
            trace=trace,
            output_dir=None,
            view="interaction",
            title="IPTG and stress state summary",
            filename="interaction_summary",
            palette_book=None,
            control_name="tetO",
            metric="bad_metric",
            fig_kwargs={},
        )


def test_retron_pareto_rejects_missing_burden_metric() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "S_abs_AUC",
                "value": 0.40,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "L_pre",
                "value": 0.02,
                "relevant_sensor_pair": True,
                "is_relevant_stress": True,
                "sponge_family_size": "mono",
            },
        ]
    )

    with pytest.raises(ValueError, match="burden metric 'D_growth_AUC' is missing"):
        plot_retron_sponge_summary(
            summary=summary,
            trace=None,
            output_dir=None,
            view="pareto",
            title="Pareto ranking",
            filename="pareto_ranking",
            palette_book=None,
            control_name="tetO",
            fig_kwargs={},
        )


def test_retron_summary_rejects_unknown_view() -> None:
    summary = pd.DataFrame(
        [
            {
                "sensor": "spyP",
                "sponge": "CpxR",
                "metric": "S_abs_AUC",
                "value": 0.40,
            }
        ]
    )

    with pytest.raises(ValueError, match="unsupported view 'unknown'"):
        plot_retron_sponge_summary(
            summary=summary,
            trace=None,
            output_dir=None,
            view="unknown",
            title="Unknown view",
            filename=None,
            palette_book=None,
            fig_kwargs={},
        )
