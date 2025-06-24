"""
--------------------------------------------------------------------------------
<reader project>
reader/parsers/synergy_h1_snapshot_and_time_series.py

Synergy H1 Snapshot + Time Series Parser

This module defines a parser for BioTek Synergy H1 plate reader exports,
handling both the initial snapshot (time-zero) and the subsequent kinetic reads.

Key Features:
- Dynamic detection of snapshot vs. kinetic blocks based on header keywords.
- Extraction of per-sheet timestamps from metadata rows (Date, Time) in column A.
- Snapshot detection by finding the first empty row after 'Results' in column A and ending at the next empty row.
- Kinetic detection by locating 'Time' in column B after the snapshot, then parsing bounded channel blocks.
- Channel labels cleaned via channel_map (e.g. 'YFP B' → 'YFP').
- Conversion to tidy long-form DataFrame with columns: position, time, value, channel.

Author: Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .raw import BaseRawParser, register_raw_parser

logger = logging.getLogger(__name__)

# ───────────────────────────── helpers ─────────────────────────────────────

def _dump_if_requested(df: pd.DataFrame, tag: str) -> None:
    if os.getenv("DEBUG_SNAPSHOT") == "1":
        fn = f"debug_{tag}.csv"
        df.to_csv(fn, index=False)
        logger.debug("   → dumped %s rows to %s", tag, fn)


def _row_has(df: pd.DataFrame, idx: int, keyword: str) -> bool:
    return df.iloc[idx].astype(str).str.contains(keyword, case=False, na=False).any()


def _normalise_channel(label: Any) -> str:
    if pd.isna(label):
        raise ValueError("Channel label is missing (NaN)")
    return str(label).split(":", 1)[0].strip()


def _extract_sheet_datetime(xl: pd.ExcelFile, sheet: str) -> datetime:
    meta = xl.parse(sheet_name=sheet, header=None, nrows=20, dtype=str)
    date_row = next((i for i in meta.index if meta.iloc[i].astype(str).str.fullmatch("Date", case=False).any()), None)
    time_row = next((i for i in meta.index if meta.iloc[i].astype(str).str.fullmatch("Time", case=False).any()), None)
    if date_row is None or time_row is None:
        raise ValueError(f"Cannot find 'Date'/'Time' in sheet {sheet!r}")
    d = pd.to_datetime(meta.iloc[date_row, 1]).date()
    t = pd.to_datetime(meta.iloc[time_row, 1]).time()
    return datetime.combine(d, t)

# ───────────────────────────── parser class ───────────────────────────────

@register_raw_parser("synergy_h1")
@register_raw_parser("synergy_h1_snapshot_and_timeseries")
class SynergyH1Parser(BaseRawParser):
    """Parser for BioTek **Synergy H1** snapshot + kinetic Excel exports."""

    def __init__(
        self,
        path: Path,
        *,
        channels: Optional[List[str]] = None,
        channel_map: Mapping[str, str] | None = None,
        sheet_names: Sequence[str] | None = None,
        add_sheet: bool = False,
    ) -> None:
        super().__init__(path, channel_map or {}, channels)
        self.sheet_names = sheet_names
        self.add_sheet = add_sheet
        # compile lowercase keys for case‑insensitive search
        self._channel_map_ci: Dict[str, str] = {k.lower(): v for k, v in self.channel_map.items()}

    # ------------------------------------------------------------------
    # Channel resolver (map → explicit list)
    # ------------------------------------------------------------------
    def _maybe_resolve_channel(self, raw: str) -> Optional[str]:
        raw_lc = raw.lower()
        # 1) explicit channel_map
        for needle, target in self._channel_map_ci.items():
            if needle in raw_lc:
                logger.debug("Mapped channel '%s' → '%s' via channel_map", raw, target)
                return target
        # 2) fall back to substring match against self.channels
        if not self.channels:
            return raw  # no filtering at all
        matches = [c for c in self.channels if c.lower() in raw_lc]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            logger.info("Skipping unmatched channel '%s'", raw)
            return None
        raise ValueError(f"Channel '{raw}' matched multiple configured channels {matches}")

    _resolve_channel = _maybe_resolve_channel  # legacy alias

    # ------------------------------------------------------------------
    # Main parse loop
    # ------------------------------------------------------------------
    def parse(self) -> pd.DataFrame:  # noqa: C901
        xl = pd.ExcelFile(self.path)
        sheets = self.sheet_names or xl.sheet_names
        frames: List[pd.DataFrame] = []
        start_dt: Optional[datetime] = None

        for sidx, sheet in enumerate(sheets):
            dt = _extract_sheet_datetime(xl, sheet)
            start_dt = start_dt or dt
            elapsed = (dt - start_dt).total_seconds() / 3600.0

            raw = xl.parse(sheet_name=sheet, header=None, dtype=str)
            snap, kin = self._split_snapshot_vs_kinetic(raw)

            if snap is not None and not snap.empty:
                frames.append(self._tidy_snapshot(snap, elapsed, sidx, sheet))
            if kin is not None and not kin.empty:
                frames.append(self._tidy_kinetics(kin, elapsed, sidx, sheet))

        if not frames:
            raise ValueError(f"No data parsed from {self.path}")

        out = pd.concat(frames, ignore_index=True)

        # remove per‑sheet t=0 duplicates
        if "sheet_index" in out.columns:
            out = out.loc[out.groupby("sheet_index")["time"].transform("min") < out["time"]]

        if self.channels:
            out = out[out["channel"].isin(self.channels)]
        return out

    # ------------------------------------------------------------------
    # Snapshot / kinetic split
    # ------------------------------------------------------------------
    @staticmethod
    def _split_snapshot_vs_kinetic(df: pd.DataFrame) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
        res = next((i for i in df.index if _row_has(df, i, "Results")), None)
        if res is None:
            return None, None
        start = res + 1
        for i in df.index[res + 1 :]:
            if df.iloc[i].dropna(how="all").empty:
                start = i + 1
                break
        t_idx = next((i for i in df.index[start:] if _row_has(df, i, "Time")), None)
        s_idx = next((i for i in df.index[start:] if df.iloc[i].dropna().size == 1), None)
        cut = min(x for x in (t_idx, s_idx) if x is not None) if any([t_idx, s_idx]) else None
        snap = df.iloc[start:cut].reset_index(drop=True) if cut else df.iloc[start:].reset_index(drop=True)
        kin = df.iloc[cut:].reset_index(drop=True) if cut else None
        return snap, kin

    # ------------------------------------------------------------------
    # Snapshot tidy
    # ------------------------------------------------------------------
    def _tidy_snapshot(self, snap: pd.DataFrame, elapsed: float, sidx: int, sname: str) -> pd.DataFrame:
        hdr = snap.iloc[0]
        well_cols = [i for i, v in enumerate(hdr) if str(v).strip().isdigit()]
        well_nums = [str(int(hdr[i])) for i in well_cols]
        row_col = next(i for i, v in enumerate(snap.iloc[1]) if isinstance(v, str) and not str(v).strip().isdigit())

        if not self.channels:
            raise ValueError("Snapshot parsing requires 'channels' list when channel_map is used")
        gs = len(self.channels)
        recs: List[Dict[str, Any]] = []
        for grp in range((snap.shape[0] - 1) // gs):
            block = snap.iloc[1 + grp * gs : 1 + (grp + 1) * gs]
            row_letter = block.iat[0, row_col]
            for ci, chan in enumerate(self.channels):
                vals = block.iloc[ci, well_cols].tolist()
                for w, v in zip(well_nums, vals):
                    recs.append({
                        "position": f"{row_letter}{w}",
                        "time": elapsed,
                        "value": v,
                        "channel": chan,
                        "sheet_index": sidx,
                        "sheet_name": sname,
                        **({"sheet": sname} if self.add_sheet else {}),
                    })
        return pd.DataFrame(recs)

    # ------------------------------------------------------------------
    # Kinetic tidy
    # ------------------------------------------------------------------
    def _tidy_kinetics(self, kin: pd.DataFrame, elapsed: float, sidx: int, sname: str) -> pd.DataFrame:
        label_rows = [i for i, r in kin.iterrows() if isinstance(r.iat[0], str) and ":" in r.iat[0]]
        out_blocks: List[pd.DataFrame] = []
        for idx, start in enumerate(label_rows):
            end = label_rows[idx + 1] if idx + 1 < len(label_rows) else len(kin)
            blk = kin.iloc[start:end].reset_index(drop=True)

            chan_raw = _normalise_channel(blk.iat[0, 0])
            chan = self._maybe_resolve_channel(chan_raw)
            if chan is None:
                continue

            hdr_idx = next(i for i in range(1, len(blk)) if any(str(v).strip().lower().startswith("time") for v in blk.iloc[i]))
            header = blk.iloc[hdr_idx].astype(str).tolist()
            time_col = next(h for h in header if h.strip().lower().startswith("time"))
            data = blk.iloc[hdr_idx + 1 :].reset_index(drop=True)
            data.columns = header

            wells = [h for h in header if re.fullmatch(r"[A-H][0-9]{1,2}", h)]
            melted = (
                data.melt(id_vars=[time_col], value_vars=wells, var_name="position", value_name="value")
                .dropna(subset=["value"])
            )

            ts_hr = pd.to_timedelta(melted[time_col], errors="coerce").dt.total_seconds() / 3600.0
            melted["time"] = elapsed + (ts_hr - ts_hr.iloc[0])
            melted["channel"] = chan
            melted["sheet_index"] = sidx
            melted["sheet_name"] = sname
            if self.add_sheet:
                melted["sheet"] = sname
            out_blocks.append(melted.drop(columns=[time_col]))

        return pd.concat(out_blocks, ignore_index=True) if out_blocks else pd.DataFrame()
