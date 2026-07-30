from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pandas.testing as pdt
import pytest

from reader_workbench.domains.logic.sfxi.builder import build_vec8_from_tidy
from reader_workbench.domains.logic.sfxi.config import load_sfxi_config
from reader_workbench.plugins.transform.sfxi import SFXICfg, SFXITransform
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
        logger=logging.getLogger("reader_workbench.tests.sfxi"),
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
        reference={"design_id": "REF", "observation_stat": "mean"},
        target_time_h=12.0,
    )
    df = _input_df()

    state_space = ctx.experiment.annotations.resolve_ordered_state_space(ref=cfg.state_map_ref)
    run_cfg = cfg.model_dump(exclude={"state_map_ref"})
    run_cfg["treatment_map"] = dict(state_space.source_values)
    run_cfg["treatment_case_sensitive"] = state_space.case_sensitive
    expected = build_vec8_from_tidy(df.copy(), run_cfg).vec8.reset_index(drop=True)
    actual = SFXITransform().run(ctx, {"df": df}, cfg)["vec8"].reset_index(drop=True)

    pdt.assert_frame_equal(actual, expected)


def test_sfxi_transform_rejects_noncanonical_ordered_state_space() -> None:
    cfg = SFXICfg(
        response={"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        state_map_ref="screen",
        reference={"design_id": "REF", "observation_stat": "mean"},
        target_time_h=12.0,
    )

    with pytest.raises(ValueError, match="SFXI state space must declare exactly 00, 10, 01, 11 in that order"):
        SFXITransform().run(_ctx(state_order=("00", "01", "10", "11")), {"df": _input_df()}, cfg)


def test_sfxi_plugin_logs_flat_logic_warning_on_canonical_path(caplog: pytest.LogCaptureFixture) -> None:
    cfg = SFXICfg(
        response={"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        state_map_ref="screen",
        reference={"design_id": "REF", "observation_stat": "mean"},
        target_time_h=12.0,
    )
    frame = _input_df()
    flat_candidate = (frame["design_id"] == "G1") & (frame["channel"] == "YFP/CFP")
    frame.loc[flat_candidate, "value"] = 5.0

    with caplog.at_level(logging.WARNING, logger="reader_workbench.tests.sfxi"):
        result = SFXITransform().run(_ctx(), {"df": frame}, cfg)

    assert result["vec8"]["flat_logic"].tolist() == [True]
    assert "flat logic detected for 1/1 designs" in caplog.text


def test_sfxi_domain_config_uses_protocol_time_tolerance_default() -> None:
    cfg = load_sfxi_config(
        {
            "response": {"logic_channel": "logic", "intensity_channel": "intensity"},
            "treatment_map": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
            "reference": {"design_id": "reference"},
        }
    )

    assert cfg.time_tolerance_h == 0.5


def test_sfxi_domain_config_rejects_unknown_top_level_setting() -> None:
    with pytest.raises(ValueError, match="Unsupported SFXI settings: unexpected"):
        load_sfxi_config(
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
def test_sfxi_domain_config_rejects_unknown_nested_settings(section: str, unknown_key: str) -> None:
    config = {
        "response": {"logic_channel": "logic", "intensity_channel": "intensity"},
        "treatment_map": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
        "reference": {"design_id": "reference"},
    }
    config[section][unknown_key] = "value"

    with pytest.raises(ValueError, match=rf"sfxi\.{section} has unsupported setting"):
        load_sfxi_config(config)


def test_sfxi_reference_rejects_legacy_stat_key() -> None:
    with pytest.raises(ValueError, match=r"sfxi\.reference has unsupported setting: stat"):
        load_sfxi_config(
            {
                "response": {"logic_channel": "logic", "intensity_channel": "intensity"},
                "treatment_map": {"00": "none", "10": "a", "01": "b", "11": "a+b"},
                "reference": {"design_id": "reference", "stat": "mean"},
            }
        )


def test_sfxi_logs_undeclared_rows_as_observations(caplog: pytest.LogCaptureFixture) -> None:
    cfg = SFXICfg(
        response={"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"},
        state_map_ref="screen",
        reference={"design_id": "REF", "observation_stat": "mean"},
        target_time_h=12.0,
    )

    with caplog.at_level(logging.INFO, logger="reader_workbench.tests.sfxi"):
        SFXITransform().run(_ctx(), {"df": _input_df()}, cfg)

    assert "observations (logic)=[1, 1, 1, 1]" in caplog.text
    assert "replicate" not in caplog.text.lower()
