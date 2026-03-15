"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_engine_contracts.py

Tests for engine contract enforcement and writes aliasing.
--------------------------------------------------------------------------------
"""

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from reader.core.engine import (
    _assert_input_contracts,
    _assert_output_contracts,
    _resolve_inputs,
    _resolve_output_labels,
    _resolve_runtime_output_contracts,
)
from reader.core.errors import ExecutionError
from reader.core.records import RecordStore
from reader.core.registry import Plugin
from reader.core.workbench import PluginSemantics


class DummyPlugin:
    @staticmethod
    def input_contracts():
        return {"df": "tidy.v1", "raw?": "none"}


class DummyOutputPlugin:
    @staticmethod
    def output_contracts():
        return {"df": "tidy.v1"}


class DummyPromotingPlugin(Plugin):
    key = "dummy_promoting"
    category = "transform"
    semantics = PluginSemantics(
        category="transform",
        domain="plate_reader",
        family="test_transform",
        summary="Test pass-through promotion plugin.",
    )

    @classmethod
    def input_contracts(cls):
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls):
        return {"df": "tidy.v1"}

    def resolve_output_contracts(self, *, inputs, outputs, cfg, where):
        del cfg
        return self.inherit_dataframe_output_contracts(
            inputs=inputs,
            outputs=outputs,
            passthrough={"df": "df"},
            where=where,
        )

    def run(self, ctx, inputs, cfg):
        del ctx, cfg
        return {"df": inputs["df"]}


def test_assert_input_contracts_rejects_extra_inputs():
    plugin = DummyPlugin()
    inputs = {
        "df": SimpleNamespace(contract_id="tidy.v1"),
        "extra": SimpleNamespace(contract_id="tidy.v1"),
    }
    with pytest.raises(ExecutionError):
        _assert_input_contracts(plugin, inputs, where="test")


def test_assert_input_contracts_allows_optional_missing():
    plugin = DummyPlugin()
    inputs = {"df": SimpleNamespace(contract_id="tidy.v1")}
    _assert_input_contracts(plugin, inputs, where="test")


def test_assert_input_contracts_accepts_stricter_contract():
    plugin = DummyPlugin()
    inputs = {"df": SimpleNamespace(contract_id="plate_reader.annotated.v1")}
    _assert_input_contracts(plugin, inputs, where="test")


def test_assert_input_contracts_allows_mismatch_when_non_strict(caplog):
    plugin = DummyPlugin()
    inputs = {"df": SimpleNamespace(contract_id="other.v1")}
    logger = logging.getLogger("reader.tests")
    with caplog.at_level(logging.WARNING, logger="reader.tests"):
        _assert_input_contracts(plugin, inputs, where="test", strict=False, logger=logger)
    assert any("contract relaxed" in rec.message for rec in caplog.records)


def test_assert_output_contracts_allows_mismatch_when_non_strict(caplog):
    bad_df = pd.DataFrame({"position": ["A1"]})
    logger = logging.getLogger("reader.tests")
    with caplog.at_level(logging.WARNING, logger="reader.tests"):
        _assert_output_contracts({"df": "tidy.v1"}, {"df": bad_df}, where="test", strict=False, logger=logger)
    assert any("contract relaxed" in rec.message for rec in caplog.records)


def test_resolve_runtime_output_contracts_accepts_promotion():
    plugin = DummyPromotingPlugin()
    inputs = {"df": SimpleNamespace(contract_id="plate_reader.annotated.v1")}
    outputs = {
        "df": pd.DataFrame(
            {
                "position": ["A1"],
                "time": [0.0],
                "channel": ["OD600"],
                "value": [1.0],
                "treatment": ["negative"],
                "design_id": ["ctrl"],
                "batch": [0.0],
            }
        )
    }
    resolved = _resolve_runtime_output_contracts(plugin, inputs=inputs, outputs=outputs, cfg=None, where="test")
    assert resolved == {"df": "plate_reader.annotated.v1"}


def test_resolve_output_labels_default_and_override():
    labels = _resolve_output_labels(
        step_id="ingest",
        output_contracts={"df": "tidy.v1"},
        writes={},
    )
    assert labels["df"] == "ingest/df"

    labels = _resolve_output_labels(
        step_id="ingest",
        output_contracts={"df": "tidy.v1"},
        writes={"df": "raw/df"},
    )
    assert labels["df"] == "raw/df"


def test_resolve_output_labels_rejects_unknown_and_duplicates():
    with pytest.raises(ExecutionError):
        _resolve_output_labels(
            step_id="ingest",
            output_contracts={"df": "tidy.v1"},
            writes={"unknown": "raw/df"},
        )

    with pytest.raises(ExecutionError):
        _resolve_output_labels(
            step_id="ingest",
            output_contracts={"a": "tidy.v1", "b": "tidy.v1"},
            writes={"a": "shared/df", "b": "shared/df"},
        )


def test_resolve_output_labels_rejects_none_contract_writes():
    with pytest.raises(ExecutionError):
        _resolve_output_labels(
            step_id="plot",
            output_contracts={"files": "none"},
            writes={"files": "plots"},
        )


def test_resolve_inputs_rejects_directory_file_input(tmp_path):
    store = RecordStore(tmp_path / "outputs")
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    with pytest.raises(ExecutionError):
        _resolve_inputs(store, {"raw": f"file:{data_dir}"})
