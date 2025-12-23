"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/merge/sample_map.py

Merge tidy measurement table with a sample metadata map. Cleans the map by:
  1) dropping all-empty columns
  2) dropping positions that carry no metadata beyond 'position'
  3) asserting remaining raw positions exist in the map
Then merges many:1 on 'position'.

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
from reader.io.sample_map import parse_sample_map


class SampleMapCfg(PluginConfig):
    """
    Flexible plate-map merge.
    - require_columns: metadata columns that MUST exist after merge (presence-only by default).
    - require_non_null: if true, also assert these columns are non-null for all merged rows.
    """

    require_columns: list[str] = Field(default_factory=list)
    require_non_null: bool = False


class SampleMapMerge(Plugin):
    key = "sample_map"
    category = "merge"
    ConfigModel = SampleMapCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        # df: tidy measurements; sample_map: file path
        return {"df": "tidy.v1", "sample_map": "none"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        # Emit the essential tidy contract; extra metadata columns are allowed.
        # This keeps downstream transforms decoupled from experiment-specific metadata.
        return {"df": "tidy.v1"}

    def _clean_plate_map(self, plate_map: pd.DataFrame) -> pd.DataFrame:
        if "position" not in plate_map.columns:
            raise MergeError("Plate map must contain a 'position' column")

        # 1) drop all-empty columns (keeps 'position' even if empty by policy)
        pm = plate_map.copy()
        non_all_empty = [c for c in pm.columns if c == "position" or not pm[c].isna().all()]
        pm = pm[non_all_empty]

        # 2) drop rows with no metadata beyond 'position'
        meta_cols = [c for c in pm.columns if c != "position"]
        if not meta_cols:
            # map has only 'position' with no metadata → nothing to merge
            return pm.iloc[0:0].copy()

        no_meta = pm[meta_cols].isna().all(axis=1)
        pm = pm.loc[~no_meta].copy()

        return pm

    def run(self, ctx, inputs, cfg: SampleMapCfg):
        df: pd.DataFrame = inputs["df"]
        sm_path: Path = inputs["sample_map"]

        try:
            sm_raw = parse_sample_map(str(sm_path))
            sm = self._clean_plate_map(sm_raw)
            if sm.empty:
                raise MergeError("Plate map has no usable metadata rows after cleaning")

            # (legacy parity) Drop raw rows for positions that carried no metadata.
            removed_positions = sorted(set(sm_raw["position"].astype(str)) - set(sm["position"].astype(str)))
            if removed_positions:
                before = len(df)
                df = df[~df["position"].astype(str).isin(removed_positions)].copy()
                after = len(df)
                head = ", ".join(removed_positions[:20])
                tail = " …" if len(removed_positions) > 20 else ""
                try:
                    ctx.logger.info(
                        f"[muted]sample_map: dropped {before - after} raw rows for positions with no metadata: "
                        f"{head}{tail}[/muted]"
                    )
                    # Optional arithmetic trace (best-effort; relies on tidy schema)
                    try:
                        chans = df["channel"].astype(str).nunique()
                        # approximate number of time slices = (rows_per_pos / channels)
                        # snapshot rows are indistinguishable here; this is a hint, not a guarantee
                        avg_rows_per_pos = (before - after) / max(len(removed_positions), 1)
                        approx_time_slices = round(avg_rows_per_pos / max(chans, 1))
                        ctx.logger.info(
                            "sample_map: consistency hint • removed_positions=%d • channels=%d • ~time_slices_per_channel=%d",
                            len(removed_positions),
                            chans,
                            approx_time_slices,
                        )
                    except Exception:
                        pass
                except Exception:
                    pass

            # 3) ensure all remaining raw positions exist in the (cleaned) map
            raw_positions = set(df["position"].astype(str).unique())
            map_positions = set(sm["position"].astype(str).unique())
            missing = sorted(raw_positions - map_positions)
            if missing:
                raise MergeError(
                    f"Plate map missing entries for positions: {missing[:40]}{'…' if len(missing) > 40 else ''}"
                )

            merged = df.merge(sm, on="position", how="left", validate="m:1")

            # Optional dtype normalization for 'batch' when present
            if "batch" in merged.columns:
                try:
                    merged["batch"] = pd.to_numeric(merged["batch"], errors="raise").astype("Int64")
                except Exception as e:
                    raise MergeError(f"'batch' must be integer-typed: {e}") from e

            # Assert required metadata columns per-experiment (config-driven)
            missing_cols = [c for c in cfg.require_columns if c not in merged.columns]
            if missing_cols:
                raise MergeError(f"Required metadata column(s) missing after merge: {missing_cols}")
            if cfg.require_non_null and cfg.require_columns:
                nulls = {c: int(merged[c].isna().sum()) for c in cfg.require_columns}
                bad = {c: n for c, n in nulls.items() if n > 0}
                if bad:
                    raise MergeError(f"Required metadata column(s) contain NaN: {bad}")

            # ---- NEW: concise merge summary ----
            try:
                added_cols = [c for c in merged.columns if c not in df.columns]
                ctx.logger.info(
                    "sample_map • positions: raw=%d • map=%d • intersect=%d • added_cols=%d [%s]",
                    len(raw_positions),
                    len(map_positions),
                    len(raw_positions & map_positions),
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
