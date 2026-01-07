"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/export/xlsx.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from reader.core.errors import ExecutionError
from reader.core.registry import Plugin, PluginConfig


class ExportXlsxCfg(PluginConfig):
    path: str = Field(..., description="Output XLSX path (relative to outputs/ if not absolute).")
    sheet_name: str = "Sheet1"
    index: bool = False


class ExportXlsx(Plugin):
    key = "xlsx"
    category = "export"
    ConfigModel = ExportXlsxCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        # Accept any DataFrame artifact; validate type at runtime.
        return {"df": "none"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def run(self, ctx, inputs: dict[str, Any], cfg: ExportXlsxCfg) -> dict[str, Any]:
        df = inputs["df"]
        if not isinstance(df, pd.DataFrame):
            raise ExecutionError(f"export/xlsx expects a DataFrame input, got {type(df).__name__}")
        out_path = Path(cfg.path)
        if not out_path.is_absolute():
            out_path = ctx.exports_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=cfg.index, sheet_name=cfg.sheet_name)
        except ImportError as e:
            raise ExecutionError("export/xlsx requires the 'openpyxl' dependency to write .xlsx files.") from e
        ctx.logger.info("export • xlsx → %s", out_path)
        return {"files": [out_path]}
