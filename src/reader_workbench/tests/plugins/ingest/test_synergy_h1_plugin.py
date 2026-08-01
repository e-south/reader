from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from reader_workbench.plugins.ingest.synergy_h1 import SynergyH1, SynergyH1UnifiedCfg
from reader_workbench.protocols import ProtocolBinding, builtin_protocol_catalog
from reader_workbench.workbench.graph import FileRef


def test_synergy_config_has_one_mixed_mode_and_single_workbook_default() -> None:
    cfg = SynergyH1UnifiedCfg(channel_map={"OD600": "OD600"})

    assert cfg.mode == "mixed"
    assert cfg.auto_pick == "single"
    assert cfg.auto_include == ["*.xlsx", "*.XLSX"]


def test_synergy_mixed_config_requires_explicit_channel_map() -> None:
    with pytest.raises(ValidationError, match="requires an explicit channel_map"):
        SynergyH1UnifiedCfg(channels=["OD600"])


@pytest.mark.parametrize(
    ("field", "value"),
    [("mode", "auto"), ("auto_pick", "merge")],
)
def test_synergy_config_rejects_removed_modes(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        SynergyH1UnifiedCfg.model_validate({field: value})


@pytest.mark.parametrize("field", ["add_source_column", "source_col", "add_sheet"])
def test_synergy_config_rejects_removed_fields(field: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SynergyH1UnifiedCfg.model_validate({field: True})


def test_synergy_preflight_opens_explicit_workbook_without_parsing(tmp_path: Path) -> None:
    workbook = tmp_path / "inputs" / "raw.xlsx"
    workbook.parent.mkdir()
    pd.DataFrame({"not_a_synergy_sheet": [1]}).to_excel(workbook, index=False)

    issues = SynergyH1.preflight_readiness(
        exp_dir=tmp_path,
        cfg=SynergyH1UnifiedCfg(channel_map={"OD600": "OD600"}),
        reads={"raw": FileRef(path=workbook)},
    )

    assert issues == ()


def test_synergy_preflight_reports_unreadable_selected_workbook(tmp_path: Path) -> None:
    workbook = tmp_path / "inputs" / "broken.xlsx"
    workbook.parent.mkdir()
    workbook.write_text("not an Excel workbook", encoding="utf-8")

    issues = SynergyH1.preflight_readiness(
        exp_dir=tmp_path,
        cfg=SynergyH1UnifiedCfg(channel_map={"OD600": "OD600"}),
        reads={"raw": FileRef(path=workbook)},
    )

    assert len(issues) == 1
    assert issues[0].kind == "file"
    assert "broken.xlsx" in issues[0].message
    assert "could not be opened" in issues[0].message


def test_synergy_resolves_one_discovered_workbook_as_runtime_input(tmp_path: Path) -> None:
    workbook = tmp_path / "inputs" / "raw.xlsx"
    workbook.parent.mkdir()
    workbook.touch()
    cfg = SynergyH1UnifiedCfg(channel_map={"OD600": "OD600"})

    resolved = SynergyH1.resolve_missing_file_inputs(exp_dir=tmp_path, cfg=cfg, inputs={})

    assert resolved == {"raw": workbook}
    assert SynergyH1.resolve_missing_file_inputs(exp_dir=tmp_path, cfg=cfg, inputs={"raw": workbook}) == {}


@pytest.mark.parametrize("protocol_id", ["plate_reader/dual_reporter_screen", "logic/four_state_vector_screen"])
def test_synergy_builtin_protocol_defaults_parse_declared_biotek_channels(
    tmp_path: Path,
    protocol_id: str,
) -> None:
    workbook = tmp_path / "common-synergy-export.xlsx"
    rows = [
        ["Date", "2026-07-07"],
        ["Time", "12:00:00"],
        ["Results"],
        [],
        [None, None, "1"],
        [None, "A", "1.0", "OD600:600"],
        [None, None, "2.0", "CFP:433,475"],
        [None, None, "3.0", "YFP:500,530"],
    ]
    for label, first_value, second_value in (
        ("OD600 B:600", "1.0", "1.1"),
        ("CFP B:433,475", "2.0", "2.1"),
        ("YFP B:500,530", "3.0", "3.1"),
    ):
        rows.extend(
            [
                [label],
                [],
                [None, "Time", "A1"],
                [None, "00:00:00", first_value],
                [None, "00:10:00", second_value],
            ]
        )
    pd.DataFrame(rows).to_excel(workbook, sheet_name="Assay", header=False, index=False)
    protocol = builtin_protocol_catalog().bind(ProtocolBinding(id=protocol_id))
    cfg = SynergyH1UnifiedCfg.model_validate(protocol.effective_plugin_config(plugin_id="ingest/synergy_h1"))

    result = SynergyH1().run(
        SimpleNamespace(logger=logging.getLogger("reader_workbench.tests.synergy")),
        {"raw": workbook},
        cfg,
    )["df"]

    assert set(result["channel"]) == {"OD600", "CFP", "YFP"}
    assert set(zip(result["source"], result["channel"], strict=True)) == {
        (source, channel) for source in ("snapshot", "kinetic") for channel in ("OD600", "CFP", "YFP")
    }
