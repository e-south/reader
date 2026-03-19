from __future__ import annotations

from pathlib import Path

import pytest

from reader.errors import RecordError
from reader.runtime import builtin_runtime
from reader.workbench.cli.helpers import dataframe_record_contracts


def test_dataframe_record_contracts_raises_on_corrupt_catalog(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    manifests = outputs / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "records.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(RecordError, match="not valid JSON"):
        dataframe_record_contracts(outputs, runtime=builtin_runtime())
