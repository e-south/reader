"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/export/csv.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from reader.core.errors import ExecutionError
from reader.workbench.ports import dataframe_input, file_path_output
from reader.workbench.registry import Plugin, PluginConfig


class ExportCsvCfg(PluginConfig):
    path: str = Field(..., description="Output CSV path (relative to outputs/ if not absolute).")
    index: bool = False
    sep: str = ","
    na_rep: str | None = None


class ExportCsv(Plugin):
    ConfigModel = ExportCsvCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df")}

    @classmethod
    def output_ports(cls):
        return {"artifact": file_path_output("artifact")}

    def run(self, ctx, inputs: dict[str, Any], cfg: ExportCsvCfg) -> dict[str, Any]:
        df = inputs["df"]
        if not isinstance(df, pd.DataFrame):
            raise ExecutionError(f"export/csv expects a DataFrame input, got {type(df).__name__}")
        out_path = Path(cfg.path)
        if not out_path.is_absolute():
            out_path = ctx.exports_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=cfg.index, sep=cfg.sep, na_rep=cfg.na_rep)
        ctx.logger.info("export • csv → %s", out_path)
        return {"artifact": out_path}
