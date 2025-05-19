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
import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, List, Optional

import pandas as pd

from .raw import BaseRawParser, register_raw_parser

logger = logging.getLogger(__name__)

def _dump_if_requested(df: pd.DataFrame, name: str) -> None:
    if os.getenv("DEBUG_SNAPSHOT") == "1":
        fn = f"debug_{name}.csv"
        df.to_csv(fn, index=False)
        logger.debug("   → dumped %s rows to %s", name, fn)


def _debug_df(df: pd.DataFrame | None, name: str) -> None:
    if df is None:
        logger.debug("=== [%s] is None ===", name)
        return
    logger.debug("=== [%s] shape=%s cols=%s ===", name, df.shape, list(df.columns))
    if not df.empty:
        logger.debug(df.head(5).to_string(index=False))
    _dump_if_requested(df, name)


def _normalise_channel(label: Any) -> str:
    if pd.isna(label):
        raise ValueError("Channel label is missing (NaN)")
    s = str(label).strip()
    base = s.split(":", 1)[0].strip()
    if not base:
        raise ValueError(f"Empty channel after normalization of {label!r}")
    return base


def _row_has(df: pd.DataFrame, idx: int, keyword: str) -> bool:
    row = df.iloc[idx].astype(str)
    return row.str.contains(keyword, case=False, na=False).any()


def _extract_sheet_datetime(xl: pd.ExcelFile, sheet: str) -> datetime:
    meta = xl.parse(sheet_name=sheet, header=None, nrows=20, dtype=str)
    _debug_df(meta, f"meta[{sheet}]")

    date_row = next(
        (i for i in meta.index if meta.iloc[i].astype(str).str.fullmatch("Date", case=False).any()),
        None
    )
    time_row = next(
        (i for i in meta.index if meta.iloc[i].astype(str).str.fullmatch("Time", case=False).any()),
        None
    )
    if date_row is None or time_row is None:
        raise ValueError(f"Cannot find 'Date' or 'Time' in sheet {sheet!r}")

    d = meta.iloc[date_row, 1]
    t = meta.iloc[time_row, 1]
    if not isinstance(d, (pd.Timestamp, datetime)):
        d = pd.to_datetime(d).date()
    if not isinstance(t, (pd.Timestamp, datetime)):
        t = pd.to_datetime(t).time()

    dt = datetime.combine(d, t)
    logger.debug("Sheet %s → %s", sheet, dt)
    return dt


@register_raw_parser("synergy_h1")
@register_raw_parser("synergy_h1_snapshot_and_timeseries")
class SynergyH1Parser(BaseRawParser):
    """
    Parser for BioTek Synergy H1 snapshot + kinetic exports,
    with an optional 'sheet' column. Ensures channels match user parameters.
    """

    def __init__(
        self,
        path: Path,
        *,
        channels: Optional[List[str]] = None,
        channel_map: Mapping[str, str] | None = None,
        sheet_names: Sequence[str] | None = None,
        add_sheet: bool = False,
    ):
        super().__init__(path, channel_map or {}, channels)
        self.sheet_names = sheet_names
        self.add_sheet = add_sheet

    def _resolve_channel(self, raw: str) -> str:
        """
        Match a raw channel string to one of the user-defined channels via substring.
        """
        if not self.channels:
            return raw
        matches = [p for p in self.channels if p.lower() in raw.lower()]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Channel '{raw}' did not match uniquely any of {self.channels}")

    def parse(self) -> pd.DataFrame:
        xl = pd.ExcelFile(self.path)
        sheets = self.sheet_names or xl.sheet_names
        frames: List[pd.DataFrame] = []
        start_dt: Optional[datetime] = None

        for sidx, sheet in enumerate(sheets):
            sheet_dt = _extract_sheet_datetime(xl, sheet)
            if sidx == 0:
                start_dt = sheet_dt
            # convert elapsed time to hours
            elapsed = (sheet_dt - start_dt).total_seconds() / 3600.0 if start_dt else 0.0

            raw = xl.parse(sheet_name=sheet, header=None, dtype=str)
            _debug_df(raw, f"raw[{sheet}]")

            snap, kin = self._split_snapshot_vs_kinetic(raw)
            _debug_df(snap, f"snap[{sheet}]")
            _debug_df(kin, f"kin[{sheet}]")

            if snap is not None and not snap.empty:
                frames.append(self._tidy_snapshot(snap, elapsed, sidx, sheet))
            if kin is not None and not kin.empty:
                frames.append(self._tidy_kinetics(kin, elapsed, sidx, sheet))

        if not frames:
            raise ValueError(f"No data found in {self.path}")

        out = pd.concat(frames, ignore_index=True)
        # filter to only configured channels
        if self.channels:
            out = out[out['channel'].isin(self.channels)]
        return out

    @staticmethod
    def _split_snapshot_vs_kinetic(
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
        res = next((i for i in df.index if _row_has(df, i, "Results")), None)
        if res is None:
            return None, None
        start = res + 1
        for i in df.index[res+1:]:
            if df.iloc[i].dropna(how="all").empty:
                start = i + 1
                break
        t_idx = next((i for i in df.index[start:] if _row_has(df, i, "Time")), None)
        s_idx = next((i for i in df.index[start:] if df.iloc[i].dropna().size == 1), None)
        cut = min(x for x in (t_idx, s_idx) if x is not None) if any([t_idx, s_idx]) else None
        snap = df.iloc[start:cut].reset_index(drop=True) if cut else df.iloc[start:].reset_index(drop=True)
        kin = df.iloc[cut:].reset_index(drop=True) if cut else None
        return snap, kin

    def _tidy_snapshot(
        self,
        snap: pd.DataFrame,
        elapsed: float,
        sidx: int,
        sname: str,
    ) -> pd.DataFrame:
        # infer wells from header row
        hdr = snap.iloc[0]
        well_cols = [i for i,v in enumerate(hdr) if str(v).strip().isdigit()]
        well_nums = [str(int(hdr[i])) for i in well_cols]
        # find row-letter column
        second = snap.iloc[1]
        row_col = next(i for i,v in enumerate(second) if isinstance(v,str) and not str(v).strip().isdigit())
        # group rows by channel count
        channels = list(self.channels or [])
        gs = len(channels)
        recs: List[dict] = []
        for grp in range((snap.shape[0]-1)//gs):
            block = snap.iloc[1+grp*gs:1+(grp+1)*gs]
            row_letter = block.iat[0, row_col]
            for ci, chan in enumerate(channels):
                vals = block.iloc[ci, well_cols].tolist()
                for w,v in zip(well_nums, vals):
                    recs.append({
                        'position': f"{row_letter}{w}",
                        'time': elapsed,
                        'value': v,
                        'channel': chan,
                        'sheet_index': sidx,
                        'sheet_name': sname,
                        **({'sheet': sname} if self.add_sheet else {})
                    })
        df = pd.DataFrame(recs)
        _dump_if_requested(df, f"tidy_snapshot_{sname}")
        return df

    def _tidy_kinetics(
        self,
        kin: pd.DataFrame,
        elapsed: float,
        sidx: int,
        sname: str,
    ) -> pd.DataFrame:
        label_rows = [i for i,row in kin.iterrows() if isinstance(row.iat[0],str) and ':' in row.iat[0]]
        blocks: List[pd.DataFrame] = []
        for idx in range(len(label_rows)):
            start = label_rows[idx]
            end = label_rows[idx+1] if idx+1<len(label_rows) else len(kin)
            blk = kin.iloc[start:end].reset_index(drop=True)
            raw_lbl = blk.iat[0,0]
            chan_raw = _normalise_channel(raw_lbl)
            chan = self._resolve_channel(chan_raw)
            # header detection
            hdr_idx = next(i for i in range(1, len(blk))
                           if any(str(v).strip().lower().startswith('time') for v in blk.iloc[i]))
            header = blk.iloc[hdr_idx].astype(str).tolist()
            time_col = next(h for h in header if h.strip().lower().startswith('time'))
            data = blk.iloc[hdr_idx+1:].reset_index(drop=True)
            data.columns = header
            _debug_df(data, f"kinetics data[{chan}]")
            wells = [h for h in header if re.fullmatch(r"[A-H][0-9]{1,2}",h)]
            melted = data.melt(id_vars=[time_col], value_vars=wells,
                                var_name='position', value_name='value')
            # drop blanks
            melted = melted.dropna(subset=['value'])
            # convert times: seconds→hours
            ts_sec = pd.to_timedelta(melted[time_col].astype(str), errors='coerce').dt.total_seconds()
            ts_hr = ts_sec / 3600.0
            melted['time'] = elapsed + (ts_hr - ts_hr.iloc[0])
            melted['channel'] = chan
            melted['sheet_index'] = sidx
            melted['sheet_name'] = sname
            if self.add_sheet:
                melted['sheet'] = sname
            blocks.append(melted.drop(columns=[time_col]))
        result = pd.concat(blocks, ignore_index=True)
        _dump_if_requested(result, f"tidy_kinetics_{sname}")
        return result
