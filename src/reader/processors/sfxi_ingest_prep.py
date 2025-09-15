"""
--------------------------------------------------------------------------------
<reader project>
src/reader/processors/sfxi_ingest_prep.py

sfxi_ingest_prep: snapshot workbook builder

• Minimal YAML:
    - workbook (Path or str; written under output.dir if relative)
    - target_time_h (float)
    - channels: ["YFP","CFP","OD600"]  (include-list; YFP must be present)
    - logic_state_map: {"00": "<exact label>", "10": ..., "01": ..., "11": ...}

• Sheets written:
  1) Replicates  - one row per well with YFP/CFP/OD600 at a single anchor time,
                   plus ratio_yfp_cfp and yfp_od600
  2) Samples     - replicate-aggregated by idxbatchxtreatment (means, SDs, N)
  3) States (wide) - one row per idxbatch; columns for each corner 00/10/01/11
                     for yfp_raw, yfp_od600, ratio_yfp_cfp; also log2_yfp_od600_*
  4) Metadata    - provenance (requested vs. selected time stats, channels, map)

Assumptions / hardcoded columns in input tidy frame:
    required: position, time, channel, value, treatment
    metadata: batch (numeric), id or genotype (at least one), is_reference (optional)
The module will prefer 'id' and fall back to 'genotype' for grouping.


Author(s): Eric J. South (new)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# ---- helpers -----------------------------------------------------------------

CORE_COLS = {"position", "time", "channel", "value", "treatment"}

def _coerce_numeric(series: pd.Series, name: str) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="raise")
    except Exception as e:
        raise ValueError(f"Column '{name}' must be numeric.") from e

def _require(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"sfxi_ingest_prep: missing required columns: {missing}")

def _pick_id_col(df: pd.DataFrame) -> str:
    if "id" in df.columns:
        return "id"
    if "genotype" in df.columns:
        return "genotype"
    raise ValueError("sfxi_ingest_prep: need an 'id' or 'genotype' column.")

def _reverse_logic_map(logic_state_map: Mapping[str, str], case_sensitive: bool = True) -> Dict[str, str]:
    # validate keys
    keys = set(logic_state_map.keys())
    req = {"00","10","01","11"}
    if keys != req:
        raise ValueError(f"logic_state_map must have exactly {sorted(req)}; got {sorted(keys)}")
    # build value->corner mapping
    rev: Dict[str, str] = {}
    for k, v in logic_state_map.items():
        key = v if case_sensitive else str(v).strip().casefold()
        if key in rev:
            raise ValueError(f"Duplicate treatment label in logic_state_map: {v!r}")
        rev[key] = k
    return rev

def _slugify_value(x: object) -> str:
    s = str(x if x is not None else "").strip()
    # normalize spaces/unsafe chars
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def _build_exp_details(row: pd.Series, id_col: str) -> str:
    """
    Build a stable slug from non-id plate-map-ish columns present on the row.
    Hardcode: exclude core numeric signal columns and a few known fields.
    """
    exclude = CORE_COLS | {id_col, "is_reference", "ratio_yfp_cfp", "yfp_od600",
                           "YFP", "CFP", "OD600", "well", "logic_state", "snapshot_time_h"}
    # deterministic order by column name
    parts: List[str] = []
    for col in sorted(c for c in row.index if c not in exclude):
        val = row[col]
        # skip NaN / empty
        if pd.isna(val) or (isinstance(val, str) and not val.strip()):
            continue
        parts.append(_slugify_value(val))
    return "__".join(parts) if parts else "na"

def _nearest(arr: np.ndarray, target: float) -> float:
    idx = int(np.argmin(np.abs(arr - target)))
    return float(arr[idx])

def _select_anchor_time(df: pd.DataFrame, target: float) -> float:
    """Choose anchor time from the available 'time' values (nearest to target)."""
    times = df["time"].to_numpy(dtype=float)
    return _nearest(times, float(target))

def _maybe_time_for_channel(df: pd.DataFrame, ch: str, anchor: float, fallback: float) -> Tuple[Optional[float], Optional[float]]:
    """Return (time_used, value) for channel at time closest to anchor (falls back to 'fallback' if needed)."""
    sub = df[df["channel"] == ch]
    if sub.empty:
        return None, None
    # prefer anchor; if exact not present, pick nearest overall
    t_avail = sub["time"].to_numpy(dtype=float)
    t_sel = anchor if anchor in t_avail else _nearest(t_avail, anchor)
    val = float(pd.to_numeric(sub.loc[sub["time"] == t_sel, "value"], errors="coerce").dropna().iloc[0])
    return t_sel, val

# ---- core --------------------------------------------------------------------

def run_sfxi_ingest_prep(
    tidy_df: pd.DataFrame,
    xform_cfg,                           # raw XForm (pydantic BaseModel with .dict())
    out_dir: Path | str,
    *,
    case_sensitive_treatments: bool = True
) -> Path:
    """
    Build the Excel workbook for SFXI ingestion and return the written path.
    """
    # ----- validate inputs
    df = tidy_df.copy()
    _require(df, CORE_COLS | {"batch", "treatment"})
    id_col = _pick_id_col(df)

    # channels config
    cfg = getattr(xform_cfg, "model_dump", None)
    cfg = cfg(exclude_none=True) if callable(cfg) else dict(xform_cfg.__dict__)  # pydantic or SimpleNamespace
    workbook = cfg.get("workbook")
    target_time = float(cfg.get("target_time_h"))
    include_channels: List[str] = list(cfg.get("channels") or [])
    if not include_channels:
        raise ValueError("sfxi_ingest_prep: 'channels' include-list is required (e.g., ['YFP','CFP','OD600']).")
    logic_state_map: Mapping[str, str] = cfg.get("logic_state_map") or {}
    if not workbook or target_time is None or not logic_state_map:
        raise ValueError("sfxi_ingest_prep requires: workbook, target_time_h, channels, logic_state_map.")

    yfp = next((c for c in include_channels if c.upper() == "YFP"), None)
    if yfp is None:
        raise ValueError("sfxi_ingest_prep: channels must include 'YFP' (case-insensitive).")

    include_channels = [str(c) for c in include_channels]
    # filter to channels of interest only
    df = df[df["channel"].isin(include_channels)].copy()
    if df.empty:
        raise ValueError("sfxi_ingest_prep: after filtering to requested channels, no rows remain.")

    # enforce batch numeric & time numeric
    df["batch"] = _coerce_numeric(df["batch"], "batch")
    df["time"]  = _coerce_numeric(df["time"], "time")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # fill optional
    if "is_reference" not in df.columns:
        df["is_reference"] = False

    # logic-state resolver
    rev_map = _reverse_logic_map(logic_state_map, case_sensitive=case_sensitive_treatments)
    def _to_corner(lbl: str) -> Optional[str]:
        key = lbl if case_sensitive_treatments else str(lbl).strip().casefold()
        return rev_map.get(key)

    # ---- Replicates sheet (one row per well at single anchor time) -----------
    keep_cols = [id_col, "is_reference", "batch", "treatment", "position"]
    extra_cols = [c for c in df.columns if c not in set(keep_cols) | CORE_COLS | {"is_reference"}]
    # Build unique replicate rows using YFP presence to anchor times reliably
    ydf = df[df["channel"] == yfp].copy()
    if ydf.empty:
        raise ValueError(f"sfxi_ingest_prep: no rows for YFP channel {yfp!r}.")
    # group by replicate identity (position×id×batch×treatment)
    gkeys = [id_col, "is_reference", "batch", "treatment", "position"]
    base = (ydf[gkeys + extra_cols]
            .drop_duplicates(subset=gkeys)
            .reset_index(drop=True))

    rows: List[Dict[str, object]] = []
    sel_times: List[float] = []

    for _, r in base.iterrows():
        mask = (
            (df[id_col] == r[id_col]) &
            (df["batch"] == r["batch"]) &
            (df["treatment"] == r["treatment"]) &
            (df["position"] == r["position"])
        )
        sub = df.loc[mask].copy()
        if sub.empty:
            # defensive; should not happen
            continue

        # anchor from YFP times
        sub_y = sub[sub["channel"] == yfp]
        t_anchor = _select_anchor_time(sub_y, target_time)
        # values for each requested channel (nearest to anchor)
        vals: Dict[str, Optional[float]] = {}
        t_used: Dict[str, Optional[float]] = {}
        for ch in include_channels:
            t_u, v = _maybe_time_for_channel(sub, ch, anchor=t_anchor, fallback=target_time)
            t_used[ch] = t_u
            vals[ch]   = v

        # pick the consensus snapshot_time_h as the YFP anchor (what users care about)
        snapshot_time_h = float(t_anchor)
        sel_times.append(snapshot_time_h)

        row: Dict[str, object] = {
            "id": r[id_col],
            "is_reference": bool(r.get("is_reference", False)),
            "batch": int(r["batch"]),
            "treatment": r["treatment"],
            "logic_state": _to_corner(r["treatment"]),
            "exp_details": None,  # fill below
            "well": r["position"],
            "snapshot_time_h": snapshot_time_h,
        }
        # add channels as columns
        for ch in include_channels:
            row[ch] = vals.get(ch, np.nan)

        # convenience metrics (only if both present)
        y = row.get("YFP")
        c = row.get("CFP")
        o = row.get("OD600")
        row["ratio_yfp_cfp"] = (float(y) / float(c)) if (y not in (None, np.nan) and c not in (None, np.nan) and float(c) != 0.0) else np.nan
        row["yfp_od600"]     = (float(y) / float(o)) if (y not in (None, np.nan) and o not in (None, np.nan) and float(o) != 0.0) else np.nan

        # build exp_details from remaining metadata on r
        row["exp_details"] = _build_exp_details(r, id_col=id_col)
        rows.append(row)

    replicates = pd.DataFrame.from_records(rows)
    # assert logic_state all resolved
    if replicates["logic_state"].isna().any():
        unresolved = replicates.loc[replicates["logic_state"].isna(), ["treatment"]]["treatment"].unique().tolist()
        raise ValueError(f"sfxi_ingest_prep: some treatments did not resolve to 00/10/01/11 via logic_state_map. Unresolved examples: {unresolved[:6]}")

    # ---- Samples sheet (aggregate by id×batch×treatment) ---------------------
    agg_keys = ["id", "is_reference", "batch", "treatment", "logic_state", "exp_details"]
    def _agg_cols(prefix: str) -> List[Tuple[str, str]]:
        return [(prefix, "mean"), (prefix, "std")]
    agg_spec = {
        "snapshot_time_h": ["mean"],
        "YFP":  ["mean", "std"],
        "CFP":  ["mean", "std"] if "CFP" in include_channels else [],
        "OD600":["mean", "std"] if "OD600" in include_channels else [],
        "ratio_yfp_cfp": ["mean", "std"] if "CFP" in include_channels else [],
        "yfp_od600":     ["mean", "std"] if "OD600" in include_channels else [],
    }
    # flatten agg spec removing empties
    clean_spec = {k:v for k,v in agg_spec.items() if v}
    samples = (replicates
               .groupby(agg_keys, dropna=False)
               .agg(clean_spec))
    # flatten multiindex columns
    samples.columns = ["_".join([c for c in col if c]).replace("__","_") for col in samples.columns.to_flat_index()]
    samples = samples.reset_index()
    samples["n"] = (replicates
                    .groupby(agg_keys, dropna=False)
                    .size()
                    .reindex(samples.set_index(agg_keys).index, fill_value=0)
                    .to_numpy())

    # ---- States (wide) -------------------------------------------------------
    # pivot sample means to per-corner columns
    def _pivot(metric: str, src: pd.DataFrame) -> pd.DataFrame:
        if metric not in src.columns:
            return pd.DataFrame()
        tbl = (src.pivot_table(index=["id","is_reference","batch","exp_details"],
                               columns="logic_state", values=metric, aggfunc="first")
               .rename(columns={"00": f"{metric}_00","10": f"{metric}_10","01": f"{metric}_01","11": f"{metric}_11"}))
        return tbl

    base_idx = samples[["id","is_reference","batch","exp_details"]].drop_duplicates().set_index(["id","is_reference","batch","exp_details"])
    pieces = [base_idx]

    for metric in ["ratio_yfp_cfp_mean", "YFP_mean", "yfp_od600_mean"]:
        p = _pivot(metric, samples)
        if not p.empty:
            pieces.append(p)

    states = pieces[0]
    for p in pieces[1:]:
        states = states.join(p, how="left")

    states = states.reset_index()

    # convenience: log2 of yfp_od600_mean_* if present, else log2 of YFP_mean_*
    def _add_log2(dfw: pd.DataFrame, src_prefix: str, dst_prefix: str) -> None:
        for corner in ["00","10","01","11"]:
            col = f"{src_prefix}_{corner}"
            if col in dfw.columns:
                dfw[f"{dst_prefix}_{corner}"] = np.log2(dfw[col]).replace([np.inf, -np.inf], np.nan)

    if any(c.startswith("yfp_od600_mean_") for c in states.columns):
        _add_log2(states, "yfp_od600_mean", "log2_yfp_od600")
    elif any(c.startswith("YFP_mean_") for c in states.columns):
        _add_log2(states, "YFP_mean", "log2_yfp_raw")

    # ---- Metadata sheet ------------------------------------------------------
    t_arr = np.asarray(sel_times, dtype=float) if sel_times else np.array([np.nan])
    meta_rows = [
        {"key": "requested_time_h", "value": target_time},
        {"key": "selected_time_min_h", "value": float(np.nanmin(t_arr))},
        {"key": "selected_time_median_h", "value": float(np.nanmedian(t_arr))},
        {"key": "selected_time_max_h", "value": float(np.nanmax(t_arr))},
        {"key": "channels", "value": ", ".join(include_channels)},
        {"key": "n_replicate_rows", "value": int(len(replicates))},
        {"key": "n_sample_groups", "value": int(len(samples))},
    ]
    meta_map = pd.DataFrame([
        {"key": f"logic_state_map[{k}]", "value": v}
        for k, v in logic_state_map.items()
    ])
    metadata = pd.concat([pd.DataFrame(meta_rows), meta_map], ignore_index=True)

    # ---- write workbook ------------------------------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(workbook)
    if not out_path.is_absolute():
        out_path = out_dir / out_path
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        replicates.to_excel(xw, sheet_name="Replicates", index=False)
        samples.to_excel(xw,    sheet_name="Samples",    index=False)
        states.to_excel(xw,     sheet_name="States (wide)", index=False)
        metadata.to_excel(xw,   sheet_name="Metadata",   index=False)

    # human-friendly printout
    LOG.info("sfxi_ingest_prep: requested t=%.3g h → selected [%s, %s] h (median=%s) over %d wells",
             target_time,
             f"{np.nanmin(t_arr):.3g}", f"{np.nanmax(t_arr):.3g}", f"{np.nanmedian(t_arr):.3g}",
             len(replicates))
    print(f"✓ sfxi_ingest_prep → {out_path.name}  (requested {target_time:g} h; "
          f"selected median {np.nanmedian(t_arr):g} h)")

    return out_path
