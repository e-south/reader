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

import pandas as pd
from pydantic import Field

from reader.core.registry import Plugin, PluginConfig


class ExportCsvCfg(PluginConfig):
    path: str = Field(..., description="Output CSV path (relative to outputs/ if not absolute).")
    index: bool = False


class ExportCsv(Plugin):
    key = "csv"
    category = "export"
    ConfigModel = ExportCsvCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "any"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"files": "none"}

    def run(self, ctx, inputs, cfg: ExportCsvCfg):
        df: pd.DataFrame = inputs["df"]
        out = Path(cfg.path)
        if not out.is_absolute():
            out = Path(ctx.outputs_dir) / out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=cfg.index)
        ctx.logger.info("export/csv → %s", out)
        return {"files": [out]}
