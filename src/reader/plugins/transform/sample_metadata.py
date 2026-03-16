"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/sample_metadata.py

Merge tidy measurements with a sample metadata table (keyed by sample_id by default).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import Field

from reader.core.errors import MergeError
from reader.workbench.ports import dataframe_input, dataframe_output, file_path_input
from reader.workbench.registry import Plugin, PluginConfig


class SampleMetadataCfg(PluginConfig):
    key: str = "sample_id"
    require_columns: list[str] = Field(default_factory=list)
    require_non_null: bool = False


class SampleMetadataMerge(Plugin):
    ConfigModel = SampleMetadataCfg

    @classmethod
    def input_ports(cls):
        return {
            "df": dataframe_input("df", "tidy.v1"),
            "metadata": file_path_input("metadata"),
        }

    @classmethod
    def output_ports(cls):
        return cls.promoted_output_ports(
            outputs={"df": dataframe_output("df", "tidy.v1")},
            promotions={"df": ("plate_reader.annotated.v1",)},
            note="promotion requires merged metadata columns that satisfy the richer contract",
        )

    def resolve_output_ports(self, *, inputs, outputs, cfg, where):
        del inputs, cfg
        resolved = dict(self.output_ports())
        merged = outputs.get("df")
        if not isinstance(merged, pd.DataFrame):
            return resolved
        try:
            self.contracts.validate(merged, contract_id="plate_reader.annotated.v1", where=f"{where}:df")
        except Exception:
            return resolved
        resolved["df"] = dataframe_output("df", "plate_reader.annotated.v1", surface=resolved["df"].surface)
        return resolved

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
