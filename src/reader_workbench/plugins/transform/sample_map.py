"""Merge tidy measurement table with a sample metadata map. Cleans the map by:
  1) dropping all-empty columns
  2) dropping positions that carry no metadata beyond 'position'
  3) asserting remaining raw positions exist in the map
Then merges many:1 on 'position'."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import Field

from reader_workbench.domains.plate_reader.io.sample_map import parse_sample_map
from reader_workbench.errors import MergeError
from reader_workbench.workbench.ports import dataframe_input, dataframe_output, file_path_input
from reader_workbench.workbench.registry import Plugin, PluginConfig


class SampleMapCfg(PluginConfig):
    """
    Flexible plate-map merge.
    - require_columns: metadata columns that MUST exist after merge (presence-only by default).
    - require_non_null: if true, also assert these columns are non-null for all merged rows.
    """

    require_columns: list[str] = Field(default_factory=list)
    require_non_null: bool = False


class SampleMapMerge(Plugin):
    ConfigModel = SampleMapCfg

    @classmethod
    def input_ports(cls):
        return {
            "df": dataframe_input("df", "tidy.v1"),
            "sample_map": file_path_input("sample_map"),
        }

    @classmethod
    def output_ports(cls):
        return cls.promoted_output_ports(
            outputs={"df": dataframe_output("df", "tidy.v1")},
            promotions={"df": ("plate_reader.annotated.v1",)},
            note="promotion requires mapped metadata columns that satisfy the richer contract",
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
            sm_raw = parse_sample_map(sm_path)
            sm = self._clean_plate_map(sm_raw)
            if sm.empty:
                raise MergeError("Plate map has no usable metadata rows after cleaning")

            # A position row with no metadata explicitly excludes that well from annotated output.
            removed_positions = sorted(set(sm_raw["position"].astype(str)) - set(sm["position"].astype(str)))
            if removed_positions:
                before = len(df)
                df = df[~df["position"].astype(str).isin(removed_positions)].copy()
                after = len(df)
                try:
                    ctx.logger.info(
                        "[muted]sample_map: dropped %d raw rows (%d positions without metadata)[/muted]",
                        before - after,
                        len(removed_positions),
                    )
                    head = ", ".join(removed_positions[:20])
                    tail = " …" if len(removed_positions) > 20 else ""
                    ctx.logger.debug("sample_map: removed positions: %s%s", head, tail)
                    # Optional arithmetic trace (best-effort; relies on tidy schema)
                    try:
                        chans = df["channel"].astype(str).nunique()
                        avg_rows_per_pos = (before - after) / max(len(removed_positions), 1)
                        approx_time_slices = round(avg_rows_per_pos / max(chans, 1))
                        ctx.logger.debug(
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

            # Concise merge summary.
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
