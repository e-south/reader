"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/merge/sample_metadata.py

Merge event-level cytometry data with a sample metadata table keyed by sample_id.

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
from reader.parsers.sample_metadata import parse_sample_metadata


class SampleMetadataCfg(PluginConfig):
    """
    Merge metadata on sample_id.
    - require_columns: metadata columns that MUST exist after merge.
    - require_non_null: if true, also assert these columns are non-null.
    """

    require_columns: list[str] = Field(default_factory=list)
    require_non_null: bool = False


class SampleMetadataMerge(Plugin):
    key = "sample_metadata"
    category = "merge"
    ConfigModel = SampleMetadataCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "cyto.events.v1", "metadata": "none"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        # Extra metadata columns are allowed; keep base contract.
        return {"df": "cyto.events.v1"}

    def run(self, ctx, inputs, cfg: SampleMetadataCfg):
        df: pd.DataFrame = inputs["df"]
        meta_path: Path = inputs["metadata"]

        try:
            meta = parse_sample_metadata(str(meta_path))

            if meta["sample_id"].duplicated().any():
                dups = meta[meta["sample_id"].duplicated()]["sample_id"].astype(str).unique().tolist()
                raise MergeError(f"Metadata has duplicate sample_id values: {dups[:20]}{'…' if len(dups) > 20 else ''}")

            raw_ids = set(df["sample_id"].astype(str).unique())
            meta_ids = set(meta["sample_id"].astype(str).unique())
            missing = sorted(raw_ids - meta_ids)
            if missing:
                raise MergeError(
                    f"Metadata missing entries for sample_id: {missing[:40]}{'…' if len(missing) > 40 else ''}"
                )

            merged = df.merge(meta, on="sample_id", how="left", validate="m:1")

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
                    "sample_metadata • samples: raw=%d • meta=%d • added_cols=%d [%s]",
                    len(raw_ids),
                    len(meta_ids),
                    len(added_cols),
                    ", ".join(added_cols[:6]) + (" …" if len(added_cols) > 6 else ""),
                )
            except Exception:
                pass

        except MergeError:
            raise
        except Exception as e:
            raise MergeError(str(e)) from e

        return {"df": merged}
