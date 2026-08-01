"""Strict configuration model for the four-state vector transform."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

_CONFIG_KEYS = frozenset(
    {
        "carry_metadata",
        "design_by",
        "eps_abs",
        "eps_range",
        "eps_ratio",
        "eps_ref",
        "exclude_reference_from_output",
        "log2_offset_delta",
        "ref_add_alpha",
        "reference",
        "require_all_corners_per_design",
        "response",
        "target_time_h",
        "time_column",
        "time_mode",
        "time_tolerance_h",
        "treatment_case_sensitive",
        "treatment_column",
        "treatment_map",
    }
)


def _sub(obj: Mapping[str, Any], key: str, *, allowed: frozenset[str]) -> dict[str, Any]:
    value = obj.get(key)
    if value is None:
        return {}
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise ValueError(f"four_state_vector.{key} must be a mapping.")
    unsupported = sorted(str(item) for item in payload if item not in allowed)
    if unsupported:
        noun = "setting" if len(unsupported) == 1 else "settings"
        raise ValueError(f"four_state_vector.{key} has unsupported {noun}: {', '.join(unsupported)}.")
    return payload


def _nonempty_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"four_state_vector.{field_name} must be a non-empty string.")
    return value.strip()


def _string_list(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"four_state_vector.{field_name} must be a list of non-empty strings.")
    return [item.strip() for item in value]


@dataclass(frozen=True)
class FourStateVectorResponseCfg:
    """Channels used for logic shape and reference-anchored intensity."""

    logic_channel: str
    intensity_channel: str


@dataclass(frozen=True)
class FourStateVectorReferenceCfg:
    design_id: str
    observation_stat: Literal["mean", "median"] = "mean"


@dataclass(frozen=True)
class FourStateVectorConfig:
    # experiment structure
    design_by: list[str]
    time_column: str

    # channels
    response: FourStateVectorResponseCfg

    # mapping
    treatment_map: dict[str, str]  # keys = {"00","10","01","11"}
    reference: FourStateVectorReferenceCfg
    treatment_case_sensitive: bool = True
    treatment_column: str | None = None

    # snapshot/time picking
    target_time_h: float | None = None
    time_mode: Literal["nearest", "last_before", "first_after", "exact"] = "nearest"
    time_tolerance_h: float | None = 0.5
    # misc rules
    require_all_corners_per_design: bool = True

    # intensity math knobs (spec §1.1b)
    ref_add_alpha: float = 0.0  # α: additive to A_i in denominator
    log2_offset_delta: float = 0.0  # δ: additive inside log2(y_linear + δ)

    # numerical guards
    eps_ratio: float = 1e-9  # ratio/log guard for logic & intensity
    eps_range: float = 1e-12  # min range for min-max (logic)
    eps_ref: float = 1e-9  # reference denom guard
    eps_abs: float = 0.0  # tiny add in numerator for intensity

    # output table policy
    exclude_reference_from_output: bool = True
    carry_metadata: list[str] = field(default_factory=lambda: ["sequence", "id"])


def load_four_state_vector_config(xform_cfg: Mapping[str, Any]) -> FourStateVectorConfig:
    """
    Validate an explicit transform mapping and build an FourStateVectorConfig.
    Requires explicit 'response.logic_channel' and 'response.intensity_channel'.
    """
    if not isinstance(xform_cfg, Mapping):
        raise ValueError("four-state vector settings must be a mapping.")
    unsupported = sorted(str(key) for key in xform_cfg if key not in _CONFIG_KEYS)
    if unsupported:
        raise ValueError(f"Unsupported four-state vector settings: {', '.join(unsupported)}.")

    design_by = _string_list(xform_cfg.get("design_by", ["design_id"]), field_name="design_by")
    if not design_by or design_by[0] != "design_id":
        raise ValueError(
            "four_state_vector.design_by must start with 'design_id' to align with the four-state vector spec."
        )
    time_column = _nonempty_text(xform_cfg.get("time_column", "time"), field_name="time_column")

    resp_in = _sub(xform_cfg, "response", allowed=frozenset({"logic_channel", "intensity_channel"}))
    logic_channel = resp_in.get("logic_channel", None)
    intensity_channel = resp_in.get("intensity_channel", None)
    response = FourStateVectorResponseCfg(
        logic_channel=_nonempty_text(logic_channel, field_name="response.logic_channel"),
        intensity_channel=_nonempty_text(intensity_channel, field_name="response.intensity_channel"),
    )

    # mapping
    raw_treatment_map = xform_cfg.get("treatment_map", {})
    if not isinstance(raw_treatment_map, Mapping):
        raise ValueError("four_state_vector.treatment_map must be a mapping.")
    tmap = dict(raw_treatment_map)
    if set(tmap.keys()) != {"00", "10", "01", "11"}:
        raise ValueError("four_state_vector.treatment_map must have exactly the keys {'00','10','01','11'}.")
    if any(not isinstance(value, str) or not value.strip() for value in tmap.values()):
        raise ValueError("four_state_vector.treatment_map values must be non-empty strings.")
    tmap = {key: value.strip() for key, value in tmap.items()}
    tcase = bool(xform_cfg.get("treatment_case_sensitive", True))
    treatment_column_raw = xform_cfg.get("treatment_column")
    if treatment_column_raw is not None and (
        not isinstance(treatment_column_raw, str) or not treatment_column_raw.strip()
    ):
        raise ValueError("four_state_vector.treatment_column must be a non-empty string when provided.")
    treatment_column = str(treatment_column_raw).strip() if isinstance(treatment_column_raw, str) else None

    # time picking
    target_time_h = xform_cfg.get("target_time_h")
    time_mode = str(xform_cfg.get("time_mode", "nearest")).lower()
    tol = xform_cfg.get("time_tolerance_h", 0.5)
    if time_mode not in {"nearest", "last_before", "first_after", "exact"}:
        raise ValueError(f"Invalid four_state_vector.time_mode='{time_mode}'")
    require_all = bool(xform_cfg.get("require_all_corners_per_design", True))

    ref = _sub(xform_cfg, "reference", allowed=frozenset({"design_id", "observation_stat"}))
    ref_label = _nonempty_text(ref.get("design_id"), field_name="reference.design_id")
    reference = FourStateVectorReferenceCfg(
        design_id=ref_label,
        observation_stat=(str(ref.get("observation_stat", "mean")).lower() if ref.get("observation_stat") else "mean"),
    )
    if reference.observation_stat not in {"mean", "median"}:
        raise ValueError(f"Invalid four_state_vector.reference.observation_stat='{reference.observation_stat}'")
    # eps / table policy
    eps_ratio = float(xform_cfg.get("eps_ratio", 1e-9))
    eps_range = float(xform_cfg.get("eps_range", 1e-12))
    eps_ref = float(xform_cfg.get("eps_ref", 1e-9))
    eps_abs = float(xform_cfg.get("eps_abs", 0.0))

    ref_add_alpha = float(xform_cfg.get("ref_add_alpha", 0.0))
    log2_offset_delta = float(xform_cfg.get("log2_offset_delta", 0.0))
    exclude_ref = bool(xform_cfg.get("exclude_reference_from_output", True))
    carry_metadata = _string_list(
        xform_cfg.get("carry_metadata", ["sequence", "id"]),
        field_name="carry_metadata",
    )
    cfg = FourStateVectorConfig(
        design_by=design_by,
        time_column=time_column,
        response=response,
        treatment_map=tmap,
        treatment_case_sensitive=tcase,
        treatment_column=treatment_column,
        target_time_h=(float(target_time_h) if target_time_h is not None else None),
        time_mode=time_mode,
        time_tolerance_h=(float(tol) if tol is not None else None),
        require_all_corners_per_design=require_all,
        reference=reference,
        eps_ratio=eps_ratio,
        eps_range=eps_range,
        eps_ref=eps_ref,
        eps_abs=eps_abs,
        ref_add_alpha=ref_add_alpha,
        log2_offset_delta=log2_offset_delta,
        exclude_reference_from_output=exclude_ref,
        carry_metadata=carry_metadata,
    )
    return cfg


__all__ = [
    "FourStateVectorConfig",
    "FourStateVectorResponseCfg",
    "FourStateVectorReferenceCfg",
    "load_four_state_vector_config",
]
