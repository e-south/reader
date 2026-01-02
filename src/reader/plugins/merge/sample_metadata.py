"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/merge/sample_metadata.py

Merge tidy measurements with a sample metadata table (keyed by sample_id by default).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from pydantic import Field

from reader.core.errors import MergeError
from reader.core.registry import Plugin, PluginConfig


class SampleMetadataCfg(PluginConfig):
    key: str = "sample_id"
    require_columns: list[str] = Field(default_factory=list)
    require_non_null: bool = False


class SampleMetadataMerge(Plugin):
    key = "sample_metadata"
    category = "merge"
    ConfigModel = SampleMetadataCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1", "metadata": "none"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    def _load_metadata(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path)
        return pd.read_csv(path)

    def run(self, ctx, inputs, cfg: SampleMetadataCfg):
        df: pd.DataFrame = inputs["df"]
        meta_path: Path = inputs["metadata"]
        key = str(cfg.key)

        try:
            meta = self._load_metadata(meta_path)
        except Exception as e:
            raise MergeError(f"Failed to read metadata file {meta_path}: {e}") from e

        if key not in df.columns:
            raise MergeError(f"Metadata merge key '{key}' missing from input dataframe")
        if key not in meta.columns:
            raise MergeError(f"Metadata merge key '{key}' missing from metadata file")

        merged = df.merge(meta, on=key, how="left", validate="m:1")

        missing_cols = [c for c in cfg.require_columns if c not in merged.columns]
        if missing_cols:
            raise MergeError(f"Required metadata column(s) missing after merge: {missing_cols}")
        if cfg.require_non_null and cfg.require_columns:
            nulls = {c: int(merged[c].isna().sum()) for c in cfg.require_columns}
            bad = {c: n for c, n in nulls.items() if n > 0}
            if bad:
                raise MergeError(f"Required metadata column(s) contain NaN: {bad}")

        try:
            added_cols = [c for c in merged.columns if c not in df.columns]
            ctx.logger.info(
                "sample_metadata • rows=%d • added_cols=%d [%s]",
                len(merged),
                len(added_cols),
                ", ".join(added_cols[:6]) + (" …" if len(added_cols) > 6 else ""),
            )
        except Exception:
            pass

        return {"df": merged}
