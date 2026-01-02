"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/export/csv.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig


class ExportCsvCfg(PluginConfig):
    path: str = Field(..., description="Output CSV path (relative to outputs/ if not absolute).")
    index: bool = False
    sep: str = ","
    na_rep: str | None = None


class ExportCsv(Plugin):
    key = "csv"
    category = "export"
    ConfigModel = ExportCsvCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def run(self, ctx, inputs: dict[str, Any], cfg: ExportCsvCfg) -> dict[str, Any]:
        df: pd.DataFrame = inputs["df"]
        out_path = Path(cfg.path)
        if not out_path.is_absolute():
            out_path = ctx.outputs_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=cfg.index, sep=cfg.sep, na_rep=cfg.na_rep)
        ctx.logger.info("export • csv → %s", out_path)
        return {"files": [out_path]}
