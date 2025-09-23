"""
--------------------------------------------------------------------------------
<reader project>
reader/processors/sfxi/api.py

Canonical config loader for the SFXI pipeline.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional


def _get(obj: Any, key: str, default=None):
    """Robust getter for dict-like OR object-with-attributes."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if hasattr(obj, key):
        val = getattr(obj, key)
        return default if val is None else val
    try:
        return obj[key]  # type: ignore[index]
    except Exception:
        return default


def _sub(obj: Any, key: str) -> dict:
    v = _get(obj, key, None)
    if v is None:
        return {}
    if isinstance(v, Mapping):
        return dict(v)
    if hasattr(v, "model_dump"):
        return dict(v.model_dump())
    # last resort: collect public attrs
    return {
        k: getattr(v, k)
        for k in dir(v)
        if not k.startswith("_") and not callable(getattr(v, k))
    }


@dataclass(frozen=True)
class SFXIResponseCfg:
    # What to use for the 4-state logic shape
    logic_channel: str        # e.g. "YFP/CFP"
    # What to use for absolute intensity & reference anchor
    intensity_channel: str    # e.g. "YFP/OD600"


@dataclass(frozen=True)
class SFXIReferenceCfg:
    genotype: Optional[str] = None        # name of reference genotype in tidy data
    scope: Literal["batch", "global"] = "batch"
    stat: Literal["mean", "median"] = "mean"
    on_missing: Literal["error", "skip"] = "error"


@dataclass(frozen=True)
class SFXIConfig:
    # experiment structure
    design_by: List[str]
    batch_col: str
    time_column: str

    # channels
    response: SFXIResponseCfg

    # mapping
    treatment_map: Dict[str, str]              # keys = {"00","10","01","11"}
    treatment_case_sensitive: bool = True

    # snapshot/time picking
    target_time_h: Optional[float] = None
    time_mode: Literal["nearest", "last_before", "first_after", "exact"] = "nearest"
    time_tolerance_h: Optional[float] = 0.51
    time_per_batch: bool = True
    on_missing_time: Literal["error", "skip-batch", "drop-all"] = "error"

    # misc rules
    require_all_corners_per_design: bool = True

    # reference normalization
    reference: SFXIReferenceCfg = SFXIReferenceCfg()

    # numerical guards
    eps_ratio: float = 1e-9   # ratio/log guard for logic & intensity
    eps_range: float = 1e-12  # min range for min-max (logic)
    eps_ref: float = 1e-9     # reference denom guard
    eps_abs: float = 0.0      # tiny add in numerator for intensity

    # output
    output_subdir: str = "sfxi"
    vec8_filename: str = "vec8.csv"
    log_filename: str = "sfxi_log.json"

    # optional cosmetics
    name: str = "sfxi"
    filename_prefix: Optional[str] = None


def load_sfxi_config(xform_cfg: Any) -> SFXIConfig:
    """
    Accept the YAML transform stanza or a pydantic XForm and build an SFXIConfig.
    Requires explicit 'response.logic_channel' and 'response.intensity_channel'.
    """
    # top-level
    design_by = list(_get(xform_cfg, "design_by", ["genotype"]))
    batch_col = str(_get(xform_cfg, "batch_col", "batch"))
    time_column = str(_get(xform_cfg, "time_column", "time"))

    # response (explicit only; no back-compat keys)
    resp_in = _sub(xform_cfg, "response")
    logic_channel = resp_in.get("logic_channel", None)
    intensity_channel = resp_in.get("intensity_channel", None)
    if not isinstance(logic_channel, str) or not logic_channel:
        raise ValueError("sfxi.response.logic_channel must be provided (e.g. 'YFP/CFP').")
    if not isinstance(intensity_channel, str) or not intensity_channel:
        raise ValueError("sfxi.response.intensity_channel must be provided (e.g. 'YFP/OD600').")
    response = SFXIResponseCfg(
        logic_channel=str(logic_channel),
        intensity_channel=str(intensity_channel),
    )

    # mapping
    tmap = dict(_get(xform_cfg, "treatment_map", {}))
    if set(tmap.keys()) != {"00", "10", "01", "11"}:
        raise ValueError("sfxi.treatment_map must have exactly the keys {'00','10','01','11'}.")
    tcase = bool(_get(xform_cfg, "treatment_case_sensitive", True))

    # time picking
    target_time_h = _get(xform_cfg, "target_time_h", None)
    time_mode = str(_get(xform_cfg, "time_mode", "nearest")).lower()
    tol = _get(xform_cfg, "time_tolerance_h", 0.51)
    time_per_batch = bool(_get(xform_cfg, "time_per_batch", True))
    on_missing_time = str(_get(xform_cfg, "on_missing_time", "error")).lower()
    if time_mode not in {"nearest", "last_before", "first_after", "exact"}:
        raise ValueError(f"Invalid sfxi.time_mode='{time_mode}'")

    # misc
    require_all = bool(_get(xform_cfg, "require_all_corners_per_design", True))

    # reference
    ref = _sub(xform_cfg, "reference")
    reference = SFXIReferenceCfg(
        genotype=ref.get("genotype"),
        scope=str(ref.get("scope", "batch")).lower() if ref.get("scope") else "batch",
        stat=str(ref.get("stat", "mean")).lower() if ref.get("stat") else "mean",
        on_missing=str(ref.get("on_missing", "error")).lower() if ref.get("on_missing") else "error",
    )

    # eps / output
    eps_ratio = float(_get(xform_cfg, "eps_ratio", 1e-9))
    eps_range = float(_get(xform_cfg, "eps_range", 1e-12))
    eps_ref   = float(_get(xform_cfg, "eps_ref", 1e-9))
    eps_abs   = float(_get(xform_cfg, "eps_abs", 0.0))

    output_subdir   = str(_get(xform_cfg, "output_subdir", "sfxi"))
    vec8_filename   = str(_get(xform_cfg, "vec8_filename", "vec8.csv"))
    log_filename    = str(_get(xform_cfg, "log_filename", "sfxi_log.json"))
    name            = str(_get(xform_cfg, "name", "sfxi"))
    filename_prefix = _get(xform_cfg, "filename_prefix", None)
    if filename_prefix is not None:
        filename_prefix = str(filename_prefix)

    cfg = SFXIConfig(
        design_by=design_by,
        batch_col=batch_col,
        time_column=time_column,
        response=response,
        treatment_map=tmap,
        treatment_case_sensitive=tcase,
        target_time_h=(float(target_time_h) if target_time_h is not None else None),
        time_mode=time_mode, time_tolerance_h=(float(tol) if tol is not None else None),
        time_per_batch=time_per_batch, on_missing_time=on_missing_time,
        require_all_corners_per_design=require_all,
        reference=reference,
        eps_ratio=eps_ratio, eps_range=eps_range, eps_ref=eps_ref, eps_abs=eps_abs,
        output_subdir=output_subdir, vec8_filename=vec8_filename, log_filename=log_filename,
        name=name, filename_prefix=filename_prefix,
    )
    return cfg


__all__ = ["SFXIConfig", "SFXIResponseCfg", "SFXIReferenceCfg", "load_sfxi_config"]
