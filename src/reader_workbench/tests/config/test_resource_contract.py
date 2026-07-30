from __future__ import annotations

from pathlib import Path

import pytest

from reader_workbench.errors import ConfigError
from reader_workbench.tests.support.configs import base_reader_config, write_config
from reader_workbench.workbench.config import ReaderSpec


def test_resources_reject_directory_entries(tmp_path: Path) -> None:
    payload = base_reader_config(
        protocol_inputs={"discovery_root": "inputs/fcs"},
        resources={"raw_root": {"kind": "directory", "path": "inputs/fcs"}},
    )

    with pytest.raises(ConfigError, match=r"resources\.raw_root\.kind must be 'file'"):
        ReaderSpec.load(write_config(tmp_path, payload))


def test_protocol_inputs_may_still_express_discovery_roots(tmp_path: Path) -> None:
    payload = base_reader_config(protocol_inputs={"discovery_root": "inputs/fcs"})

    spec = ReaderSpec.load(write_config(tmp_path, payload))

    assert spec.protocol.inputs["discovery_root"] == "inputs/fcs"
