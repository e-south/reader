"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/export/excel.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig


class ExportExcelCfg(PluginConfig):
    path: str = Field(..., description="Output .xlsx path (relative to outputs/ if not absolute).")
    sheet: str = "Sheet1"
    index: bool = False


class ExportExcel(Plugin):
    key = "excel"
    category = "export"
    ConfigModel = ExportExcelCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "any"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def run(self, ctx, inputs, cfg: ExportExcelCfg):
        df: pd.DataFrame = inputs["df"]
        out = Path(cfg.path)
        if not out.is_absolute():
            out = Path(ctx.outputs_dir) / out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out, index=cfg.index, sheet_name=cfg.sheet)
        ctx.logger.info("export/excel → %s", out)
        return {"files": [out]}
