from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from reader_workbench.domains.logic.four_state_vector.builder import build_four_state_vector_from_tidy
from reader_workbench.domains.logic.four_state_vector.config import load_four_state_vector_config
from reader_workbench.plugins.transform.four_state_vector import FourStateVectorCfg, FourStateVectorTransform
from reader_workbench.protocols import ProtocolBinding, ProtocolSemanticProgram
from reader_workbench.workbench.experiment import (
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
        logger=logging.getLogger("reader_workbench.tests.four_state_vector"),
        experiment=ExperimentSemantics(
            protocol=ProtocolBinding(id="logic/four_state_vector_screen"),
            protocol_program=ProtocolSemanticProgram(protocol="logic/four_state_vector_screen"),
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
    intensity_g1 = {"00": 20.0, "10": 40.0, "01": 80.0, "11": 160.0}
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


def test_four_state_vector_plugin_matches_build_four_state_vector_from_tidy():
    ctx = _ctx()
    cfg = FourStateVectorCfg(
        response={"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        state_map_ref="screen",
        reference={"design_id": "REF", "observation_stat": "mean"},
        target_time_h=12.0,
    )
    df = _input_df()

    state_space = ctx.experiment.annotations.resolve_ordered_state_space(ref=cfg.state_map_ref)
    run_cfg = cfg.model_dump(exclude={"state_map_ref"})
    run_cfg["treatment_map"] = dict(state_space.source_values)
    run_cfg["treatment_case_sensitive"] = state_space.case_sensitive
    expected = build_four_state_vector_from_tidy(df.copy(), run_cfg).vector.reset_index(drop=True)
    actual = FourStateVectorTransform().run(ctx, {"df": df}, cfg)["vector"].reset_index(drop=True)

    pdt.assert_frame_equal(actual, expected)


def test_four_state_vector_preserves_frozen_pre_objective_coordinates() -> None:
    """The breaking terminology migration must not change the measurement math."""
    cfg = FourStateVectorCfg(
        response={"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        state_map_ref="screen",
        reference={"design_id": "REF", "observation_stat": "mean"},
        target_time_h=12.0,
    )

    vector = FourStateVectorTransform().run(_ctx(), {"df": _input_df()}, cfg)["vector"]

    assert vector[["design_id", "reference_design_id", "time_selected_h", "flat_logic"]].to_dict(orient="records") == [
        {
            "design_id": "G1",
            "reference_design_id": "REF",
            "time_selected_h": 12.0,
            "flat_logic": False,
        }
    ]
    np.testing.assert_allclose(
        vector.loc[
            0,
            [
                "v00",
                "v10",
                "v01",
                "v11",
                "y00_star",
                "y10_star",
                "y01_star",
                "y11_star",
            ],
        ].to_numpy(dtype=float),
        [
            0.0,
            0.333333333333222,
            0.666666666666444,
            0.999999999999667,
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        rtol=0.0,
        atol=1e-15,
    )


def test_four_state_vector_transform_rejects_noncanonical_ordered_state_space() -> None:
    cfg = FourStateVectorCfg(
        response={"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        state_map_ref="screen",
        reference={"design_id": "REF", "observation_stat": "mean"},
        target_time_h=12.0,
    )

    with pytest.raises(
        ValueError, match="four-state vector state space must declare exactly 00, 10, 01, 11 in that order"
    ):
        FourStateVectorTransform().run(_ctx(state_order=("00", "01", "10", "11")), {"df": _input_df()}, cfg)


def test_four_state_vector_plugin_logs_flat_logic_warning_on_canonical_path(caplog: pytest.LogCaptureFixture) -> None:
    cfg = FourStateVectorCfg(
        response={"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        state_map_ref="screen",
        reference={"design_id": "REF", "observation_stat": "mean"},
        target_time_h=12.0,
    )
    frame = _input_df()
    flat_candidate = (frame["design_id"] == "G1") & (frame["channel"] == "YFP/CFP")
    frame.loc[flat_candidate, "value"] = 5.0

    with caplog.at_level(logging.WARNING, logger="reader_workbench.tests.four_state_vector"):
        result = FourStateVectorTransform().run(_ctx(), {"df": frame}, cfg)

    assert result["vector"]["flat_logic"].tolist() == [True]
    assert "flat logic detected for 1/1 designs" in caplog.text


def test_four_state_vector_domain_config_uses_protocol_time_tolerance_default() -> None:
    cfg = load_four_state_vector_config(
        {
            "response": {"logic_channel": "logic", "intensity_channel": "intensity"},
            "treatment_map": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
            "reference": {"design_id": "reference"},
        }
    )

    assert cfg.time_tolerance_h == 0.5


def test_four_state_vector_domain_config_rejects_unknown_top_level_setting() -> None:
    with pytest.raises(ValueError, match="Unsupported four-state vector settings: unexpected"):
        load_four_state_vector_config(
            {
                "response": {"logic_channel": "logic", "intensity_channel": "intensity"},
                "treatment_map": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
                "reference": {"design_id": "reference"},
                "unexpected": "value",
            }
        )


@pytest.mark.parametrize(
    ("section", "unknown_key"),
    [
        ("response", "unexpected"),
        ("reference", "unexpected"),
    ],
)
def test_four_state_vector_domain_config_rejects_unknown_nested_settings(section: str, unknown_key: str) -> None:
    config = {
        "response": {"logic_channel": "logic", "intensity_channel": "intensity"},
        "treatment_map": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
        "reference": {"design_id": "reference"},
    }
    config[section][unknown_key] = "value"

    with pytest.raises(ValueError, match=rf"four_state_vector\.{section} has unsupported setting"):
        load_four_state_vector_config(config)


def test_four_state_vector_reference_rejects_legacy_stat_key() -> None:
    with pytest.raises(ValueError, match=r"four_state_vector\.reference has unsupported setting: stat"):
        load_four_state_vector_config(
            {
                "response": {"logic_channel": "logic", "intensity_channel": "intensity"},
                "treatment_map": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
                "reference": {"design_id": "reference", "stat": "mean"},
            }
        )


def test_four_state_vector_logs_undeclared_rows_as_observations(caplog: pytest.LogCaptureFixture) -> None:
    cfg = FourStateVectorCfg(
        response={"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        state_map_ref="screen",
        reference={"design_id": "REF", "observation_stat": "mean"},
        target_time_h=12.0,
    )

    with caplog.at_level(logging.INFO, logger="reader_workbench.tests.four_state_vector"):
        FourStateVectorTransform().run(_ctx(), {"df": _input_df()}, cfg)

    assert "observations (logic)=[1, 1, 1, 1]" in caplog.text
    assert "replicate" not in caplog.text.lower()
