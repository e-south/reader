from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pandas.testing as pdt
import pytest

from reader.domains.logic.sfxi.run import build_vec8_from_tidy
from reader.plugins.transform.sfxi import SFXICfg, SFXITransform
from reader.protocols import ProtocolBinding, ProtocolSemanticProgram
from reader.workbench.experiment import (
    AnnotationSemantics,
    ExperimentSemantics,
    OrderedStateSpaces,
    OrderedStateSpaceSpec,
    OutputLayout,
    ResourceCatalog,
)


def _ctx(*, state_order: tuple[str, ...] = ("00", "10", "01", "11")):
    source_values = {state_id: state_id for state_id in state_order}
    source_values.update({"00": "A", "10": "B", "01": "C", "11": "D"})
    source_values = {state_id: source_values[state_id] for state_id in state_order}
    return SimpleNamespace(
        logger=logging.getLogger("reader.tests.sfxi"),
        experiment=ExperimentSemantics(
            protocol=ProtocolBinding(id="logic/sfxi_screen"),
            protocol_program=ProtocolSemanticProgram(protocol="logic/sfxi_screen"),
            annotations=AnnotationSemantics(
                ordered_state_spaces=OrderedStateSpaces(
                    by_id={
                        "screen": OrderedStateSpaceSpec(
                            column="treatment",
                            state_order=state_order,
                            source_values=source_values,
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
        state_map_ref="screen",
        reference={"design_id": "REF", "stat": "mean"},
        target_time_h=12.0,
    )
    df = _input_df()

    state_space = ctx.experiment.annotations.resolve_ordered_state_space(ref=cfg.state_map_ref)
    run_cfg = cfg.model_dump()
    run_cfg["treatment_map"] = dict(state_space.source_values)
    run_cfg["treatment_case_sensitive"] = state_space.case_sensitive
    expected = build_vec8_from_tidy(df.copy(), run_cfg).vec8.reset_index(drop=True)
    actual = SFXITransform().run(ctx, {"df": df}, cfg)["vec8"].reset_index(drop=True)

    pdt.assert_frame_equal(actual, expected)


def test_sfxi_transform_rejects_noncanonical_ordered_state_space() -> None:
    cfg = SFXICfg(
        response={"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        state_map_ref="screen",
        reference={"design_id": "REF", "stat": "mean"},
        target_time_h=12.0,
    )

    with pytest.raises(ValueError, match="SFXI state space must declare exactly 00, 10, 01, 11 in that order"):
        SFXITransform().run(_ctx(state_order=("00", "01", "10", "11")), {"df": _input_df()}, cfg)
