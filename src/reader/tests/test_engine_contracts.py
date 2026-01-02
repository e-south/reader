"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_engine_contracts.py

Tests for engine contract enforcement and writes aliasing.
--------------------------------------------------------------------------------
"""

from types import SimpleNamespace

import pytest

from reader.core.engine import _assert_input_contracts, _resolve_output_labels
from reader.core.errors import ExecutionError


class DummyPlugin:
    @staticmethod
    def input_contracts():
        return {"df": "tidy.v1", "raw?": "none"}


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
