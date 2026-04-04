"""
--------------------------------------------------------------------------------
<reader project>
src/reader/tests/test_export_xlsx.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from reader.plugins.export.xlsx import ExportXlsx, ExportXlsxCfg
from reader.workbench.context import RunContext


def _ctx(tmp_path: Path) -> RunContext:
    outputs = tmp_path / "outputs"
    return RunContext(
        exp_dir=tmp_path,
        outputs_dir=outputs,
        artifacts_dir=outputs / "artifacts",
        plots_dir=outputs / "plots",
        exports_dir=outputs / "exports",
        records_path=outputs / "manifests" / "records.json",
        logger=logging.getLogger("test"),
        palette_book=None,
        strict=True,
    )


def test_export_xlsx_writes_readable_file(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3.0, 4.0]})
    cfg = ExportXlsxCfg(path="vec8.xlsx", sheet_name="vec8", index=False)

    out = ExportXlsx().run(ctx, {"df": df}, cfg)
    out_path = Path(out["artifact"])
    assert out_path.exists()

    back = pd.read_excel(out_path, sheet_name="vec8")
    assert list(back.columns) == ["col_a", "col_b"]
    assert back.shape == (2, 2)
