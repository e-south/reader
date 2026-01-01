"""
--------------------------------------------------------------------------------
<reader project>
src/reader/parsers/synergy_h1.py

This module defines a parser for BioTek Synergy H1 plate reader exports.

Author: Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------- helpers -------------------------------


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _drop_all_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize whitespace-only strings to NA without triggering deprecation warnings.
    # We opt into the future behavior for this call (no silent downcast), then do an
    # explicit infer_objects to retain legacy semantics deterministically.
    with pd.option_context("future.no_silent_downcasting", True):
        tmp = df.replace(r"^\s*$", pd.NA, regex=True)
    tmp = tmp.infer_objects(copy=False)
    return tmp.dropna(how="all").reset_index(drop=True)


def _ensure_excel_path(path: Path) -> None:
    _require(path.exists(), f"Input file not found: {path}")
    _require(path.suffix.lower() in {".xlsx", ".xls"}, f"Unsupported file type for Synergy H1: {path.suffix}")


def _extract_sheet_datetime(xl: pd.ExcelFile, sheet: str) -> datetime:
    meta = xl.parse(sheet_name=sheet, header=None, nrows=20, dtype=str)

    def _row_has(idx: int, pat: str) -> bool:
        return meta.iloc[idx].astype(str).str.fullmatch(pat, case=False).any()

    d_row = next((i for i in meta.index if _row_has(i, r"Date")), None)
    t_row = next((i for i in meta.index if _row_has(i, r"Time")), None)
    _require(d_row is not None and t_row is not None, f"Missing 'Date'/'Time' rows in sheet {sheet!r}")
    d = pd.to_datetime(meta.iloc[d_row, 1]).date()
    t = pd.to_datetime(meta.iloc[t_row, 1]).time()
    return datetime.combine(d, t)


# ---- channel canonicalization tolerant to "OD600 B", trailing letters, spaces, colons


def _canon(s: str) -> str:
    s0 = str(s or "").strip()
    s0 = s0.split(":", 1)[0]  # drop trailing colon sections
    s0 = re.sub(r"(?<=\d)\s+[A-Za-z]$", "", s0)  # remove final single letter after digits (e.g., "OD600 B")
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0


# --- overflow token detection ---
_OVERFLOW_PATTERNS = ("overflow", "ovrflw", "ovr", "over", "inf", "infinity", "∞")


def _is_overflow_token(x: object) -> bool:
    s = str(x or "").strip()
    if not s:
        return False
    if s.startswith(">"):  # e.g., "> 65000"
        return True
    t = s.lower()
    return any(k in t for k in _OVERFLOW_PATTERNS)


def _normalize_channel_map(channel_map: Mapping[str, str] | None) -> Mapping[str, str]:
    if not channel_map:
        return {}
    # match on canonicalized, case-insensitive keys; keep user-provided canonical names as values
    return {_canon(k).lower(): str(v) for k, v in channel_map.items()}


def _resolve_channel(
    raw_label: str, *, channels: Sequence[str] | None, channel_map_ci: Mapping[str, str]
) -> str | None:
    rl = _canon(raw_label).lower()

    # 1) mapping takes precedence (substring on canonicalized lower)
    for needle, canonical in channel_map_ci.items():
        if needle and needle in rl:
            return canonical

    # 2) otherwise match against provided channels (canonicalized, tolerant substring)
    if channels:
        canon_to_orig = {_canon(c).lower(): c for c in channels}
        # exact canonical match first
        if rl in canon_to_orig:
            return canon_to_orig[rl]
        # tolerant substring either way
        hits = [orig for ccanon, orig in canon_to_orig.items() if ccanon in rl or rl in ccanon]
        if len(hits) == 1:
            return hits[0]
        if len(hits) == 0:
            return None
        raise ValueError(f"Ambiguous channel label {raw_label!r} → matches {hits}")

    # 3) require at least one of (channels, channel_map)
    raise ValueError("Provide at least one of: channels or channel_map")


def _time_col(header: Iterable[str]) -> str:
    for h in header:
        if str(h).strip().lower().startswith("time"):
            return str(h)
    raise ValueError("Time column not found in kinetic block header")


def _well_headers(header: Iterable[str]) -> list[str]:
    return [str(h) for h in header if re.fullmatch(r"[A-H][0-9]{1,2}", str(h))]


# ---------------------------- snapshot tidy ----------------------------


def _tidy_snapshot_block(
    snap: pd.DataFrame,
    *,
    elapsed_h: float,
    sheet_idx: int,
    sheet_name: str,
    channels: Sequence[str],
    add_sheet: bool,
) -> pd.DataFrame:
    _require(len(channels) > 0, "Snapshot parsing requires a non-empty 'channels' list")
    _require(not snap.empty, "Empty snapshot block")

    hdr = snap.iloc[0]
    well_cols = [i for i, v in enumerate(hdr) if str(v).strip().isdigit()]
    _require(well_cols, "Snapshot header did not contain numbered well columns")
    well_nums = [str(int(hdr[i])) for i in well_cols]

    data_rows = _drop_all_empty_rows(snap.iloc[1:].copy())
    _require(not data_rows.empty, "Snapshot block contains a header but no data rows")

    row_col = next(
        (i for i, v in enumerate(data_rows.iloc[0]) if isinstance(v, str) and not str(v).strip().isdigit()),
        None,
    )
    _require(row_col is not None, "Could not infer row-letter column in snapshot block")

    rows = []
    group_height = len(channels)
    n_rows = data_rows.shape[0]
    _require(
        n_rows % group_height == 0,
        f"Snapshot rows do not align with channel count (rows={n_rows}, channels/group={group_height}).",
    )

    for grp_idx in range(n_rows // group_height):
        block = data_rows.iloc[grp_idx * group_height : (grp_idx + 1) * group_height]
        row_letter = str(block.iat[0, row_col]).strip()
        _require(row_letter and re.fullmatch(r"[A-H]", row_letter), f"Invalid row letter in snapshot: {row_letter!r}")

        for ci, chan in enumerate(channels):
            vals = block.iloc[ci, well_cols].tolist()
            for w, v in zip(well_nums, vals, strict=False):
                rows.append(
                    {
                        "position": f"{row_letter}{w}",
                        "time": float(elapsed_h),
                        "channel": str(chan),
                        "value": v,
                        "sheet_index": sheet_idx,
                        "sheet_name": sheet_name,
                        "source": "snapshot",
                        **({"sheet": sheet_name} if add_sheet else {}),
                    }
                )

    out = pd.DataFrame(rows)
    # Preserve overflow tokens as +inf and mark them.
    rawv = out["value"].astype(str)
    over_mask = rawv.map(_is_overflow_token)
    out["overflow"] = over_mask
    out["value"] = pd.to_numeric(rawv, errors="coerce")
    out.loc[over_mask, "value"] = np.inf
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    return out


# ----------------------------- kinetic tidy -----------------------------


def _tidy_kinetic_blocks(
    kin: pd.DataFrame,
    *,
    elapsed_h: float,
    sheet_idx: int,
    sheet_name: str,
    channels: Sequence[str] | None,
    channel_map_ci: Mapping[str, str],
    add_sheet: bool,
) -> pd.DataFrame:
    log = logging.getLogger("reader")
    raw_to_resolved: dict[str, str] = {}
    raw_to_canon: dict[str, str] = {}

    def _row_has(df: pd.DataFrame, idx: int, pat: str) -> bool:
        return df.iloc[idx].astype(str).str.contains(pat, case=False, na=False).any()

    # --- tolerant label row detection (keeps legacy ':' behavior) ---
    def _looks_like_label(cell0: object) -> bool:
        s = str(cell0 or "").strip()
        if not s or s.lower() == "nan":
            return False
        if ":" in s:
            return True  # legacy exports: "OD600: …"
        # Newer exports: plain channel name in col A (e.g., "OD600", "RFP")
        try:
            _ = _resolve_channel(s, channels=channels, channel_map_ci=channel_map_ci)
            return True
        except Exception:
            return False

    label_rows = [i for i, r in kin.iterrows() if _looks_like_label(r.iat[0])]
    _require(
        label_rows,
        "No kinetic blocks found: expected a channel label in column A "
        "(e.g., 'OD600' or 'OD600: …') preceding a 'Time' header row.",
    )

    parts: list[pd.DataFrame] = []
    for idx, start in enumerate(label_rows):
        end = label_rows[idx + 1] if idx + 1 < len(label_rows) else len(kin)
        blk = kin.iloc[start:end].reset_index(drop=True)

        chan_raw = str(blk.iat[0, 0]).split(":", 1)[0].strip()
        canon = _canon(chan_raw)
        chan = _resolve_channel(chan_raw, channels=channels, channel_map_ci=channel_map_ci)
        if chan is None:
            continue

        # Track mapping
        raw_to_canon.setdefault(chan_raw, canon)
        prev = raw_to_resolved.get(chan_raw)
        if prev is not None and prev != chan:
            log.warning(
                "[warn]inconsistent channel resolution[/warn] • raw=%r canon=%r previously→%r now→%r",
                chan_raw,
                canon,
                prev,
                chan,
            )
        raw_to_resolved[chan_raw] = chan

        hdr_idx = next((i for i in range(1, len(blk)) if _row_has(blk, i, r"^Time")), None)
        _require(hdr_idx is not None, f"Time header not found in kinetic block for channel {chan!r}")
        header = blk.iloc[hdr_idx].astype(str).tolist()
        time_col = _time_col(header)
        wells = _well_headers(header)
        _require(wells, f"No well columns (A1..H12) in kinetic header for {chan!r}")

        data = blk.iloc[hdr_idx + 1 :].reset_index(drop=True)
        data.columns = header
        ts = pd.to_timedelta(data[time_col], errors="coerce").dt.total_seconds() / 3600.0
        data = data.assign(__time_hr=ts).loc[lambda d: d["__time_hr"].notna()].reset_index(drop=True)
        _require(not data.empty, f"Non-parsable time values in kinetic block for {chan!r}")

        t0_rel = float(data["__time_hr"].iloc[0])
        data["__time_hr"] = float(elapsed_h) + (data["__time_hr"] - t0_rel)

        melted = (
            data.melt(id_vars=["__time_hr"], value_vars=wells, var_name="position", value_name="value")
            .dropna(subset=["value"])
            .reset_index(drop=True)
            .rename(columns={"__time_hr": "time"})
        )
        melted["channel"] = chan
        melted["sheet_index"] = sheet_idx
        melted["sheet_name"] = sheet_name
        if add_sheet:
            melted["sheet"] = sheet_name
        melted["source"] = "kinetic"
        parts.append(melted)

    out = pd.concat(parts, ignore_index=True)
    rawv = out["value"].astype(str)
    over_mask = rawv.map(_is_overflow_token)
    out["overflow"] = over_mask
    out["value"] = pd.to_numeric(rawv, errors="coerce")
    out.loc[over_mask, "value"] = np.inf
    out = out.dropna(subset=["value"]).reset_index(drop=True)

    # Emit the mapping once per call (sheet‑scoped)
    if raw_to_resolved:
        pairs = [
            f"{raw!r} → {raw_to_canon.get(raw, '?')!r} → {raw_to_resolved[raw]!r}" for raw in sorted(raw_to_resolved)
        ]
        log.info("channel normalization (kinetic, sheet %s): %s", sheet_name, "; ".join(pairs))

    return out


# ----------------------------- public API ------------------------------


def parse_snapshot_and_timeseries(
    path: str | Path,
    *,
    channels: Sequence[str] | None = None,
    channel_map: Mapping[str, str] | None = None,
    sheet_names: Sequence[str] | None = None,
    add_sheet: bool = False,
    include_snapshot: bool = True,
    include_kinetic: bool = True,
) -> pd.DataFrame:
    """
    Parse Synergy H1 export containing snapshot and/or kinetic blocks into tidy long form.

    Output columns: position:str, time:float(h), channel:str, value:float

    Requirements:
      - Provide at least one of (channels, channel_map). No hidden fallbacks.
      - Input must be .xlsx/.xls.
    """
    p = Path(path)
    _ensure_excel_path(p)
    ch_map_ci = _normalize_channel_map(channel_map)
    _require(channels or ch_map_ci, "Provide either 'channels' or 'channel_map'")

    xl = pd.ExcelFile(p)
    sheets = list(sheet_names or xl.sheet_names)
    _require(sheets, "Workbook has no sheets")
    for s in sheets:
        _require(s in xl.sheet_names, f"Sheet {s!r} not found in workbook")

    frames: list[pd.DataFrame] = []
    t0: datetime | None = None

    def _split_snapshot_vs_kinetic(df: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        def _row_has(idx: int, keyword: str) -> bool:
            return df.iloc[idx].astype(str).str.contains(keyword, case=False, na=False).any()

        res = next((i for i in df.index if _row_has(i, "Results")), None)
        if res is None:
            return None, None
        start = res + 1
        for i in df.index[res + 1 :]:
            if df.iloc[i].dropna(how="all").empty:
                start = i + 1
                break
        t_idx = next((i for i in df.index[start:] if _row_has(i, "Time")), None)
        s_idx = next((i for i in df.index[start:] if df.iloc[i].dropna().size == 1), None)
        cut = min(x for x in (t_idx, s_idx) if x is not None) if (t_idx is not None or s_idx is not None) else None
        snap = df.iloc[start:cut].reset_index(drop=True) if cut else df.iloc[start:].reset_index(drop=True)
        kin = df.iloc[cut:].reset_index(drop=True) if cut else None
        return snap, kin

    for sidx, sheet in enumerate(sheets):
        dt = _extract_sheet_datetime(xl, sheet)
        t0 = t0 or dt
        elapsed = (dt - t0).total_seconds() / 3600.0

        raw = xl.parse(sheet_name=sheet, header=None, dtype=str)
        snap, kin = _split_snapshot_vs_kinetic(raw)

        if include_snapshot and snap is not None and not snap.empty:
            frames.append(
                _tidy_snapshot_block(
                    snap,
                    elapsed_h=elapsed,
                    sheet_idx=sidx,
                    sheet_name=sheet,
                    channels=list(channels or []),
                    add_sheet=add_sheet,
                )
            )
            # Snapshot uses configured channels; log them for transparency
            logging.getLogger("reader").info(
                "channel normalization (snapshot, sheet %s): configured=%s", sheet, list(channels or [])
            )
        if include_kinetic and kin is not None and not kin.empty:
            frames.append(
                _tidy_kinetic_blocks(
                    kin,
                    elapsed_h=elapsed,
                    sheet_idx=sidx,
                    sheet_name=sheet,
                    channels=channels,
                    channel_map_ci=ch_map_ci,
                    add_sheet=add_sheet,
                )
            )

    _require(frames, f"No parsable data found in {p.name}")
    out = pd.concat(frames, ignore_index=True)

    # If a sheet contains a snapshot, drop overlapping t=0 kinetic points for that sheet only.
    if {"sheet_index", "source"}.issubset(out.columns):
        sheet_has_snapshot = out.groupby("sheet_index")["source"].transform(lambda s: (s == "snapshot").any())
        sheet_min_time = out.groupby("sheet_index")["time"].transform("min")
        drop_mask = (out["source"] == "kinetic") & sheet_has_snapshot & (out["time"] == sheet_min_time)
        out = out.loc[~drop_mask]

    if channels:
        out = out[out["channel"].isin(channels)].reset_index(drop=True)

    out["time"] = pd.to_numeric(out["time"], errors="raise")
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    _require(out["time"].ge(0).all(), "Internal error: negative time encountered after alignment")
    _require(out["time"].notna().all(), "Internal error: time contains NaN after alignment")

    expected = set(channels or _normalize_channel_map(channel_map).values())
    if expected:
        missing = expected - set(out["channel"].unique())
        _require(not missing, f"Missing data for channels: {sorted(missing)}")

    out["position"] = out["position"].astype(str)
    out["channel"] = out["channel"].astype(str)
    return out.reset_index(drop=True)


def parse_kinetic_only(
    path: str | Path,
    *,
    channels: Sequence[str] | None = None,
    channel_map: Mapping[str, str] | None = None,
    sheet_names: Sequence[str] | None = None,
    add_sheet: bool = False,
) -> pd.DataFrame:
    p = Path(path)
    _ensure_excel_path(p)
    ch_map_ci = _normalize_channel_map(channel_map)
    _require(channels or ch_map_ci, "Provide either 'channels' or 'channel_map'")

    xl = pd.ExcelFile(p)
    sheets = list(sheet_names or xl.sheet_names)
    _require(sheets, "Workbook has no sheets")
    for s in sheets:
        _require(s in xl.sheet_names, f"Sheet {s!r} not found in workbook")

    frames: list[pd.DataFrame] = []
    t0: datetime | None = None

    def _find_kinetic_section(df: pd.DataFrame) -> pd.DataFrame | None:
        for i in df.index:
            cell = str(df.iat[i, 0]).strip()
            if not cell:
                continue
            if ":" in cell:
                return df.iloc[i:].reset_index(drop=True)
            try:
                _ = _resolve_channel(cell, channels=channels, channel_map_ci=ch_map_ci)
                return df.iloc[i:].reset_index(drop=True)
            except Exception:
                continue
        return None

    for sidx, sheet in enumerate(sheets):
        dt = _extract_sheet_datetime(xl, sheet)
        t0 = t0 or dt
        elapsed = (dt - t0).total_seconds() / 3600.0

        raw = xl.parse(sheet_name=sheet, header=None, dtype=str)
        kin = _find_kinetic_section(raw)
        _require(kin is not None, f"No kinetic data found in sheet {sheet!r}")
        frames.append(
            _tidy_kinetic_blocks(
                kin,
                elapsed_h=elapsed,
                sheet_idx=sidx,
                sheet_name=sheet,
                channels=channels,
                channel_map_ci=ch_map_ci,
                add_sheet=add_sheet,
            )
        )

    _require(frames, f"No kinetic readings found in {p.name}")
    out = pd.concat(frames, ignore_index=True)

    expected_vals = set(channels or ch_map_ci.values())
    expected = {str(c) for c in expected_vals if "/" not in str(c)}
    if expected:
        missing = expected - set(out["channel"].astype(str).unique())
        _require(not missing, f"Kinetic data missing for channels: {sorted(missing)}")

    out["position"] = out["position"].astype(str)
    out["channel"] = out["channel"].astype(str)
    out["time"] = pd.to_numeric(out["time"], errors="raise")
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    _require(out["time"].ge(0).all(), "Internal error: negative time encountered after alignment")
    _require(out["time"].notna().all(), "Internal error: time contains NaN after alignment")
    return out.reset_index(drop=True)
