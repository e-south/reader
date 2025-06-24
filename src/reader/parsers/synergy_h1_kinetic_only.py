"""
--------------------------------------------------------------------------------
<reader project>
reader/parsers/synergy_h1_kinetic_only.py

Synergy H1 Time Series Parser

This module defines a parser for BioTek Synergy H1 plate reader exports,
handling both the initial snapshot (time-zero) and the subsequent kinetic reads.

- Ignores snapshot tables completely.
- Uses `channel_map` or the configured `channels` list to detect the start
  of the kinetic section on each sheet.
- Fails early if any expected channel is missing.

Author: Eric J. South
--------------------------------------------------------------------------------
"""


from __future__ import annotations
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
from .raw import BaseRawParser, register_raw_parser

LOG = logging.getLogger(__name__)


# ───────────────────────────── helpers ──────────────────────────────
def _row_has(df: pd.DataFrame, idx: int, pat: str) -> bool:
    return df.iloc[idx].astype(str).str.contains(pat, case=False, na=False).any()


def _norm_chan(lbl: Any) -> str:
    if pd.isna(lbl):
        raise ValueError("Channel label missing (NaN)")
    return str(lbl).split(":", 1)[0].strip()


def _sheet_datetime(xl: pd.ExcelFile, sheet: str) -> datetime:
    meta = xl.parse(sheet, header=None, nrows=20, dtype=str)
    d_row = next((i for i in meta.index if _row_has(meta, i, r"^Date$")), None)
    t_row = next((i for i in meta.index if _row_has(meta, i, r"^Time$")), None)
    if d_row is None or t_row is None:
        raise ValueError(f"'Date' / 'Time' rows not found in {sheet!r}")
    d = pd.to_datetime(meta.iloc[d_row, 1]).date()
    t = pd.to_datetime(meta.iloc[t_row, 1]).time()
    return datetime.combine(d, t)


# ───────────────────────────── parser ───────────────────────────────
@register_raw_parser("synergy_h1_kinetic")
@register_raw_parser("synergy_h1_kinetic_only")
class SynergyH1KineticParser(BaseRawParser):
    """
    BioTek **Synergy H1** – *kinetic reads only*

    *   Ignores snapshots completely  
    *   `channel_map` may be either **mapping** ({needle: canonical}) **or list**
        ([canonical, …]).  Matching is case-insensitive substring search.
    """

    # -----------------------------------------------------------------
    def __init__(
        self,
        path: Path,
        *,
        channels: Optional[List[str]] = None,
        channel_map: Mapping[str, str] | Sequence[str] | None = None,
        sheet_names: Sequence[str] | None = None,
        add_sheet: bool = False,
    ) -> None:
        # normalise channel_map → Dict[str, str]
        if channel_map is None:
            channel_map = {}
        if isinstance(channel_map, Sequence) and not isinstance(channel_map, Mapping):
            # user passed a list of “canonical” names – use each lowercase name as its own needle
            channel_map = {c.lower(): c for c in channel_map}
        super().__init__(path, channel_map, channels)

        self.sheet_names = sheet_names
        self.add_sheet   = add_sheet
        self._map_ci: Dict[str, str] = {k.lower(): v for k, v in self.channel_map.items()}

    # -----------------------------------------------------------------
    # channel resolver
    def _resolve(self, raw: str) -> Optional[str]:
        rl = raw.lower()
        for needle, target in self._map_ci.items():
            if needle in rl:
                return target
        if not self.channels:
            return raw
        hits = [c for c in self.channels if c.lower() in rl]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            return None
        raise ValueError(f"Channel '{raw}' ambiguous → {hits}")

    # -----------------------------------------------------------------
    # kinetic locator
    def _kinetic_section(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        for i in df.index:
            cell = str(df.iat[i, 0]).strip()
            if cell and self._resolve(cell):
                return df.iloc[i:].reset_index(drop=True)
        return None

    # -----------------------------------------------------------------
    def parse(self) -> pd.DataFrame:  # noqa: C901
        xl = pd.ExcelFile(self.path)
        sheets = self.sheet_names or xl.sheet_names
        blocks: List[pd.DataFrame] = []
        t0: Optional[datetime] = None

        for sidx, sheet in enumerate(sheets):
            dt = _sheet_datetime(xl, sheet)
            t0  = t0 or dt
            elapsed = (dt - t0).total_seconds() / 3600.0

            raw = xl.parse(sheet, header=None, dtype=str)
            kin = self._kinetic_section(raw)
            if kin is None:
                LOG.warning("No kinetic data in sheet %s – skipped", sheet)
                continue
            tidy = self._tidy(kin, elapsed, sidx, sheet)
            if not tidy.empty:
                blocks.append(tidy)

        if not blocks:
            raise ValueError("No kinetic readings found – check channel_map / channels")

        out = pd.concat(blocks, ignore_index=True)

        # final sanity-check – ignore derived / ratio names
        expected = {c for c in (self.channels or self._map_ci.values()) if "/" not in c}
        missing  = expected - set(out["channel"].unique())
        if missing:
            raise ValueError(f"Kinetic data missing for channels: {sorted(missing)}")

        return out

    # -----------------------------------------------------------------
    def _tidy(self, kin: pd.DataFrame, elapsed: float, sidx: int, sname: str) -> pd.DataFrame:
        label_rows = [i for i, r in kin.iterrows() if isinstance(r.iat[0], str) and self._resolve(r.iat[0])]
        parts: List[pd.DataFrame] = []

        for idx, start in enumerate(label_rows):
            end = label_rows[idx + 1] if idx + 1 < len(label_rows) else len(kin)
            blk = kin.iloc[start:end].reset_index(drop=True)

            chan_raw = _norm_chan(blk.iat[0, 0])
            chan     = self._resolve(chan_raw)
            if chan is None:
                continue

            hdr_idx  = next(i for i in range(1, len(blk)) if _row_has(blk, i, r"^Time"))
            header   = blk.iloc[hdr_idx].astype(str).tolist()
            time_col = next(c for c in header if c.strip().lower().startswith("time"))

            data = blk.iloc[hdr_idx + 1:].reset_index(drop=True)
            data.columns = header
            wells  = [h for h in header if re.fullmatch(r"[A-H][0-9]{1,2}", h)]

            molten = (data.melt(id_vars=[time_col], value_vars=wells,
                                var_name="position", value_name="value")
                      .dropna(subset=["value"]))

            ts_hr  = pd.to_timedelta(molten[time_col], errors="coerce").dt.total_seconds() / 3600.0
            molten["time"]        = elapsed + (ts_hr - ts_hr.iloc[0])
            molten["channel"]     = chan
            molten["sheet_index"] = sidx
            molten["sheet_name"]  = sname
            if self.add_sheet:
                molten["sheet"] = sname

            parts.append(molten.drop(columns=[time_col]))

        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
