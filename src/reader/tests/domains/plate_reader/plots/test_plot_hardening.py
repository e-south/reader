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

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from rich.console import Console

from reader.domains.plate_reader.plots.snapshot_barplot import plot_snapshot_barplot
from reader.domains.plate_reader.plots.snapshot_heatmap import plot_snapshot_heatmap, prepare_snapshot_heatmap_inputs
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
        "schema": "reader/v6",
        "experiment": {"id": "exp_semantics"},
        "protocol": {
            "id": "plate_reader/dual_reporter_screen",
            "analysis": {"include_fold_change": False},
            "deliverables": {
                "plots": {
                    "profile": "none",
                    "include": ["snapshot_heatmap_yfp_cfp"],
                    "settings": {
                        "snapshot_heatmap_yfp_cfp": {
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
