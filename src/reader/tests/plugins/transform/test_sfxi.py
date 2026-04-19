from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pandas.testing as pdt

from reader.domains.logic.sfxi.run import build_vec8_from_tidy
from reader.plugins.transform.sfxi import SFXICfg, SFXITransform
from reader.protocols import ProtocolBinding, ProtocolSemanticProgram
from reader.workbench.experiment import (
    AnnotationSemantics,
    ExperimentSemantics,
    LogicMaps,
    LogicMapSpec,
    OutputLayout,
    ResourceCatalog,
)


def _ctx():
    return SimpleNamespace(
        logger=logging.getLogger("reader.tests.sfxi"),
        experiment=ExperimentSemantics(
            protocol=ProtocolBinding(id="logic/sfxi_screen"),
            protocol_program=ProtocolSemanticProgram(protocol="logic/sfxi_screen"),
            annotations=AnnotationSemantics(
                logic_maps=LogicMaps(
                    by_id={
                        "screen": LogicMapSpec(
                            column="treatment",
                            corners={"00": "A", "10": "B", "01": "C", "11": "D"},
                            case_sensitive=True,
                        )
                    }
                )
            ),
            resources=ResourceCatalog(),
            layout=OutputLayout(
                outputs_dir=Path("."), plots_subdir="plots", exports_subdir="exports", notebooks_subdir="notebooks"
            ),
        ),
    )


def _input_df() -> pd.DataFrame:
    rows = []
    corners = {"00": "A", "10": "B", "01": "C", "11": "D"}
    logic_ref = {"00": 1.0, "10": 1.0, "01": 1.0, "11": 1.0}
    logic_g1 = {"00": 1.0, "10": 2.0, "01": 4.0, "11": 8.0}
    intensity_ref = {"00": 10.0, "10": 10.0, "01": 10.0, "11": 10.0}
    intensity_g1 = {"00": 20.0, "10": 20.0, "01": 20.0, "11": 20.0}
    for design_id, logic_values, intensity_values in (
        ("REF", logic_ref, intensity_ref),
        ("G1", logic_g1, intensity_g1),
    ):
        for corner, treatment in corners.items():
            for channel, value in (("YFP/CFP", logic_values[corner]), ("YFP/OD600", intensity_values[corner])):
                rows.append(
                    {
                        "position": f"{design_id}_{corner}",
                        "time": 12.0,
                        "channel": channel,
                        "value": value,
                        "treatment": treatment,
                        "design_id": design_id,
                    }
                )
    return pd.DataFrame(rows)


def test_sfxi_plugin_matches_build_vec8_from_tidy():
    ctx = _ctx()
    cfg = SFXICfg(
        response={"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        logic_map_ref="screen",
        reference={"design_id": "REF", "stat": "mean"},
        target_time_h=12.0,
    )
    df = _input_df()

    logic_map = ctx.experiment.annotations.resolve_logic_map(ref=cfg.logic_map_ref)
    run_cfg = cfg.model_dump()
    run_cfg["treatment_map"] = dict(logic_map.corners)
    run_cfg["treatment_case_sensitive"] = logic_map.case_sensitive
    expected = build_vec8_from_tidy(df.copy(), run_cfg).vec8.reset_index(drop=True)
    actual = SFXITransform().run(ctx, {"df": df}, cfg)["vec8"].reset_index(drop=True)

    pdt.assert_frame_equal(actual, expected)
