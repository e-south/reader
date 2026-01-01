"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/with_columns.py

Add constant columns to a tidy table (explicit, config-driven).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from reader.core.errors import TransformError
from reader.core.registry import Plugin, PluginConfig


class WithColumnsCfg(PluginConfig):
    """
    columns: mapping of column_name -> constant value to assign.
    mode: "create" (error if column exists) or "overwrite" (replace).
    """

    columns: dict[str, Any] = Field(default_factory=dict)
    mode: Literal["create", "overwrite"] = "create"


class WithColumns(Plugin):
    key = "with_columns"
    category = "transform"
    ConfigModel = WithColumnsCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    def run(self, ctx, inputs, cfg: WithColumnsCfg):
        df: pd.DataFrame = inputs["df"].copy()

        if not cfg.columns:
            raise TransformError("with_columns: columns must be non-empty")

        for col, value in cfg.columns.items():
            if col in df.columns and cfg.mode == "create":
                raise TransformError(f"with_columns: column already exists: {col!r}")
            df[col] = value

        try:
            ctx.logger.info(
                "with_columns • added=%d • mode=%s • columns=[%s]",
                len(cfg.columns),
                cfg.mode,
                ", ".join(sorted(cfg.columns.keys())),
            )
        except Exception:
            pass

        return {"df": df}
