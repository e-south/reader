"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_engine_contracts.py

Tests for engine contract enforcement and writes aliasing.
--------------------------------------------------------------------------------
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from reader.contracts import builtin_contract_catalog
from reader.errors import ExecutionError
from reader.workbench.engine import (
    _assert_input_ports,
    _assert_output_ports,
    _resolve_inputs,
    _resolve_output_labels,
    _resolve_runtime_output_ports,
)
from reader.workbench.graph import FileRef, OutputRef
from reader.workbench.ports import dataframe_input, dataframe_output, file_bundle_output, file_path_input
from reader.workbench.records import RecordStore
from reader.workbench.registry import Plugin


class DummyPlugin:
    @staticmethod
    def input_ports():
        return {
            "df": dataframe_input("df", "tidy.v1"),
            "raw": file_path_input("raw", optional=True),
        }


class DummyOutputPlugin:
    @staticmethod
    def output_ports():
        return {"df": dataframe_output("df", "tidy.v1")}


class DummyPromotingPlugin(Plugin):
    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return cls.passthrough_output_ports(
            outputs={"df": dataframe_output("df", "tidy.v1")},
            passthrough={"df": "df"},
        )

    def resolve_output_ports(self, *, inputs, outputs, cfg, where):
        del cfg
        return self.inherit_dataframe_output_ports(
            inputs=inputs,
            outputs=outputs,
            passthrough={"df": "df"},
            where=where,
        )

    def run(self, ctx, inputs, cfg):
        del ctx, cfg
        return {"df": inputs["df"]}


def test_assert_input_ports_rejects_extra_inputs():
    plugin = DummyPlugin()
    contracts = builtin_contract_catalog()
    inputs = {
        "df": SimpleNamespace(contract_id="tidy.v1"),
        "extra": SimpleNamespace(contract_id="tidy.v1"),
    }
    with pytest.raises(ExecutionError):
        _assert_input_ports(plugin, inputs, contracts=contracts, where="test")


def test_assert_input_ports_allows_optional_missing():
    plugin = DummyPlugin()
    contracts = builtin_contract_catalog()
    inputs = {"df": SimpleNamespace(contract_id="tidy.v1")}
    _assert_input_ports(plugin, inputs, contracts=contracts, where="test")


def test_assert_input_ports_accepts_stricter_contract():
    plugin = DummyPlugin()
    contracts = builtin_contract_catalog()
    inputs = {"df": SimpleNamespace(contract_id="plate_reader.annotated.v1")}
    _assert_input_ports(plugin, inputs, contracts=contracts, where="test")


def test_assert_input_ports_allows_mismatch_when_non_strict(caplog):
    plugin = DummyPlugin()
    contracts = builtin_contract_catalog()
    inputs = {"df": SimpleNamespace(contract_id="other.v1")}
    logger = logging.getLogger("reader.tests")
    with caplog.at_level(logging.WARNING, logger="reader.tests"):
        _assert_input_ports(plugin, inputs, contracts=contracts, where="test", strict=False, logger=logger)
    assert any("contract relaxed" in rec.message for rec in caplog.records)


def test_assert_input_ports_rejects_file_for_dataframe_port():
    plugin = DummyPlugin()
    contracts = builtin_contract_catalog()
    with pytest.raises(ExecutionError):
        _assert_input_ports(plugin, {"df": Path("raw.csv")}, contracts=contracts, where="test")


def test_assert_output_ports_allows_mismatch_when_non_strict(caplog):
    bad_df = pd.DataFrame({"position": ["A1"]})
    contracts = builtin_contract_catalog()
    logger = logging.getLogger("reader.tests")
    with caplog.at_level(logging.WARNING, logger="reader.tests"):
        _assert_output_ports(
            {"df": dataframe_output("df", "tidy.v1")},
            {"df": bad_df},
            contracts=contracts,
            where="test",
            strict=False,
            logger=logger,
        )
    assert any("contract relaxed" in rec.message for rec in caplog.records)


def test_resolve_runtime_output_ports_accepts_promotion():
    plugin = DummyPromotingPlugin()
    plugin.bind_runtime(
        descriptor=type(
            "D",
            (),
            {"kind": "plugin", "cls": DummyPromotingPlugin, "plugin_id": "transform/dummy", "name": "transform/dummy"},
        )(),
        contracts=builtin_contract_catalog(),
    )
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
    resolved = _resolve_runtime_output_ports(
        plugin,
        inputs=inputs,
        outputs=outputs,
        cfg=None,
        contracts=builtin_contract_catalog(),
        where="test",
    )
    assert resolved["df"].contract == "plate_reader.annotated.v1"


def test_resolve_output_labels_default_and_override():
    labels = _resolve_output_labels(
        step_id="ingest",
        output_ports={"df": dataframe_output("df", "tidy.v1")},
        writes={},
    )
    assert labels["df"] == OutputRef(record_id="ingest/df")

    labels = _resolve_output_labels(
        step_id="ingest",
        output_ports={"df": dataframe_output("df", "tidy.v1")},
        writes={"df": OutputRef(record_id="raw/df")},
    )
    assert labels["df"] == OutputRef(record_id="raw/df")


def test_resolve_output_labels_rejects_unknown_and_duplicates():
    with pytest.raises(ExecutionError):
        _resolve_output_labels(
            step_id="ingest",
            output_ports={"df": dataframe_output("df", "tidy.v1")},
            writes={"unknown": OutputRef(record_id="raw/df")},
        )

    with pytest.raises(ExecutionError):
        _resolve_output_labels(
            step_id="ingest",
            output_ports={"a": dataframe_output("a", "tidy.v1"), "b": dataframe_output("b", "tidy.v1")},
            writes={"a": OutputRef(record_id="shared/df"), "b": OutputRef(record_id="shared/df")},
        )


def test_resolve_output_labels_rejects_file_output_writes():
    with pytest.raises(ExecutionError):
        _resolve_output_labels(
            step_id="plot",
            output_ports={"artifacts": file_bundle_output("artifacts")},
            writes={"artifacts": OutputRef(record_id="plots")},
        )


def test_resolve_inputs_rejects_directory_file_input(tmp_path):
    store = RecordStore(tmp_path / "outputs", contracts=builtin_contract_catalog())
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    with pytest.raises(ExecutionError):
        _resolve_inputs(store, {"raw": FileRef(path=data_dir)}, input_ports={"raw": file_path_input("raw")})
