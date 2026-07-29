"""Coverage for sample-metadata transform behavior."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from reader.errors import MergeError
from reader.plugins.transform.sample_metadata import SampleMetadataCfg, SampleMetadataMerge


def _ctx(exp_dir: Path):
    return SimpleNamespace(exp_dir=exp_dir, logger=logging.getLogger("reader.tests"))


def test_sample_metadata_output_stays_domain_neutral() -> None:
    port = SampleMetadataMerge.output_ports()["df"]

    assert port.contract == "tidy.v1"
    assert port.surface is None


def test_sample_metadata_merge_basic(tmp_path: Path):
    plugin = SampleMetadataMerge()
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "position": ["s1", "s2"],
            "time": [0.0, 0.0],
            "channel": ["A", "A"],
            "value": [1.0, 2.0],
        }
    )
    meta = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "design_id": ["d1", "d2"],
            "treatment": ["t1", "t2"],
        }
    )
    meta_path = tmp_path / "metadata.csv"
    meta.to_csv(meta_path, index=False)
    outputs = plugin.run(
        _ctx(tmp_path),
        {"df": df, "metadata": meta_path},
        SampleMetadataCfg(require_columns=["design_id", "treatment"], require_non_null=True),
    )
    merged = outputs["df"]
    assert {"design_id", "treatment"} <= set(merged.columns)


@pytest.mark.parametrize("name", ["metadata.xls", "metadata.tsv", "metadata"])
def test_sample_metadata_rejects_unsupported_formats(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("sample_id,design_id\ns1,d1\n", encoding="utf-8")

    with pytest.raises(MergeError, match="expected .csv or .xlsx"):
        SampleMetadataMerge().run(
            _ctx(tmp_path),
            {"df": pd.DataFrame({"sample_id": ["s1"]}), "metadata": path},
            SampleMetadataCfg(),
        )


def test_sample_metadata_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    pd.DataFrame({"sample_id": ["s1", "s1"], "design_id": ["d1", "d2"]}).to_csv(path, index=False)

    with pytest.raises(MergeError, match="must be unique"):
        SampleMetadataMerge().run(
            _ctx(tmp_path),
            {"df": pd.DataFrame({"sample_id": ["s1"]}), "metadata": path},
            SampleMetadataCfg(),
        )
