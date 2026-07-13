"""
SFXI setpoint-scatter scoring and plotting helpers.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from reader.errors import SFXIError
from reader.plotting.sinks import PlotFigure

from .validation import require_intensity_delta_column

VEC8_COLUMNS = (
    "v00",
    "v10",
    "v01",
    "v11",
    "y00_star",
    "y10_star",
    "y01_star",
    "y11_star",
)
READER_SUPPORTED_SFXI_API_VERSION = "1"


def _load_dnadesign_sfxi_api():
    try:
        module = importlib.import_module("dnadesign.opal.api.sfxi")
    except ImportError as exc:
        _raise_dnadesign_sfxi_import_error(exc)
    api_version = getattr(module, "SFXI_API_VERSION", None)
    if str(api_version) != READER_SUPPORTED_SFXI_API_VERSION:
        raise SFXIError(
            "Unsupported dnadesign SFXI API version "
            f"{api_version!r}; reader expects {READER_SUPPORTED_SFXI_API_VERSION!r}. "
            "Update the reader lockfile or install a compatible dnadesign build."
        )
    for attr in ("SFXIScoringConfig", "score_vec8"):
        _require_public_attr(module, attr)
    return module


def _require_public_attr(module, attr: str) -> None:
    try:
        getattr(module, attr)
    except AttributeError as exc:
        raise SFXIError(f"dnadesign.opal.api.sfxi is missing required public API: {attr}.") from exc
    except ImportError as exc:
        _raise_dnadesign_sfxi_import_error(exc)


def _raise_dnadesign_sfxi_import_error(exc: ImportError) -> None:
    raise SFXIError(
        "dnadesign SFXI API unavailable. Reader checkout: `uv sync --locked --group dnadesign`. "
        "Packaged installs require a compatible dnadesign build that exposes dnadesign.opal.api.sfxi."
    ) from exc


def require_dnadesign_sfxi_api():
    return _load_dnadesign_sfxi_api()


def _require_vec8_columns(df: pd.DataFrame) -> None:
    missing = [col for col in VEC8_COLUMNS if col not in df.columns]
    if missing:
        raise SFXIError(f"SFXI setpoint scatter requires vec8 columns: {', '.join(missing)}.")


def _require_intensity_delta_matches(vec8: pd.DataFrame, *, expected: float) -> None:
    expected_value = float(expected)
    if not math.isfinite(expected_value) or expected_value < 0.0:
        raise SFXIError("SFXI setpoint scatter intensity_log2_offset_delta must be finite and nonnegative.")
    values = pd.to_numeric(vec8["intensity_log2_offset_delta"], errors="coerce")
    invalid = values.isna() | ~values.map(lambda value: math.isfinite(float(value)))
    if invalid.any():
        raise SFXIError("SFXI setpoint scatter vec8 intensity_log2_offset_delta values must be finite.")
    mismatched = ~values.map(lambda value: math.isclose(float(value), expected_value, rel_tol=0.0, abs_tol=1e-12))
    if mismatched.any():
        observed = ", ".join(f"{float(value):g}" for value in sorted(values.drop_duplicates().tolist()))
        raise SFXIError(
            "SFXI setpoint scatter intensity_log2_offset_delta mismatch: "
            f"vec8 has [{observed}], scorer configured {expected_value:g}."
        )


def _coerce_setpoints(setpoints: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    if not isinstance(setpoints, Mapping) or not setpoints:
        raise SFXIError("SFXI setpoint scatter requires at least one named setpoint.")
    out: dict[str, list[float]] = {}
    for raw_name, raw_vector in setpoints.items():
        name = str(raw_name).strip()
        if not name:
            raise SFXIError("SFXI setpoint names must be non-empty strings.")
        vector = [float(value) for value in raw_vector]
        if len(vector) != 4 or not all(math.isfinite(value) for value in vector):
            raise SFXIError(f"SFXI setpoint {name!r} must be a finite length-4 vector.")
        out[name] = vector
    return out


def _metadata_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "source_id",
        "source_path",
        "table_path",
        "source_kind",
        "source_row_index",
        "row_label",
        "design_id",
        "sequence",
        "id",
        "sequence_source_id",
        "experiment_id",
        "experiment_date",
        "time_selected_h",
        "reference_design_id",
        "intensity_log2_offset_delta",
        "r_logic",
        "flat_logic",
    ]
    return [col for col in preferred if col in df.columns]


def score_sfxi_setpoints(
    vec8: pd.DataFrame,
    *,
    setpoints: Mapping[str, Sequence[float]],
    scaling_percentile: int = 95,
    scaling_min_n: int = 5,
    scaling_eps: float = 1.0e-8,
    logic_exponent_beta: float = 1.0,
    intensity_exponent_gamma: float = 1.0,
    intensity_log2_offset_delta: float = 0.0,
) -> pd.DataFrame:
    _require_vec8_columns(vec8)
    require_intensity_delta_column(vec8)
    _require_intensity_delta_matches(vec8, expected=intensity_log2_offset_delta)
    setpoint_map = _coerce_setpoints(setpoints)
    api = require_dnadesign_sfxi_api()

    vec8_values = vec8.loc[:, VEC8_COLUMNS].astype(float).to_numpy()
    metadata_cols = _metadata_columns(vec8)
    metadata = vec8.loc[:, metadata_cols].reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for setpoint_name, setpoint_vector in setpoint_map.items():
        cfg = api.SFXIScoringConfig(
            setpoint_vector=tuple(setpoint_vector),
            scaling_percentile=int(scaling_percentile),
            scaling_min_n=int(scaling_min_n),
            scaling_eps=float(scaling_eps),
            logic_exponent_beta=float(logic_exponent_beta),
            intensity_exponent_gamma=float(intensity_exponent_gamma),
            intensity_log2_offset_delta=float(intensity_log2_offset_delta),
        )
        result = api.score_vec8(vec8_values, cfg, scaling_vec8=vec8_values)
        scored = pd.DataFrame(result.to_records())
        scored.insert(0, "setpoint_name", setpoint_name)
        frames.append(pd.concat([metadata.copy(), scored], axis=1))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def _layout_for_panels(n_panels: int) -> tuple[int, int]:
    if n_panels <= 1:
        return 1, 1
    if n_panels <= 4:
        return 2, 2
    return math.ceil(n_panels / 3), 3


def _setpoint_label(row: pd.Series) -> str:
    vector = row.get("setpoint_vector")
    if isinstance(vector, list):
        return "[" + ",".join(f"{float(value):g}" for value in vector) + "]"
    return str(vector)


def render_sfxi_setpoint_scatter(
    *,
    vec8: pd.DataFrame,
    setpoints: Mapping[str, Sequence[float]],
    scaling_percentile: int = 95,
    scaling_min_n: int = 5,
    scaling_eps: float = 1.0e-8,
    logic_exponent_beta: float = 1.0,
    intensity_exponent_gamma: float = 1.0,
    intensity_log2_offset_delta: float = 0.0,
    fig_kwargs: Mapping[str, Any] | None = None,
    filename: str | None = None,
    formats: Sequence[str] = ("pdf",),
    dpi: int = 300,
    label_points: bool = False,
) -> list[PlotFigure]:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - dependency guard
        raise SFXIError("SFXI setpoint scatter requires matplotlib.") from exc

    scored = score_sfxi_setpoints(
        vec8,
        setpoints=setpoints,
        scaling_percentile=scaling_percentile,
        scaling_min_n=scaling_min_n,
        scaling_eps=scaling_eps,
        logic_exponent_beta=logic_exponent_beta,
        intensity_exponent_gamma=intensity_exponent_gamma,
        intensity_log2_offset_delta=intensity_log2_offset_delta,
    )
    if scored.empty:
        raise SFXIError("SFXI setpoint scatter has no rows to plot.")

    setpoint_names = list(dict.fromkeys(scored["setpoint_name"].astype(str).tolist()))
    n_rows, n_cols = _layout_for_panels(len(setpoint_names))
    fig_opts = dict(fig_kwargs or {})
    figsize = fig_opts.pop("figsize", (4.4 * n_cols, 3.9 * n_rows))
    fig_opts.setdefault("constrained_layout", True)
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, **fig_opts)
    axes = [ax for row in axes_grid for ax in row]
    mappable = None
    for ax, setpoint_name in zip(axes, setpoint_names, strict=False):
        subset = scored[scored["setpoint_name"].astype(str) == setpoint_name].copy()
        sort_cols = [col for col in ("design_id", "time_selected_h") if col in subset.columns]
        if sort_cols:
            subset = subset.sort_values(sort_cols)
        if "design_id" in subset.columns and "time_selected_h" in subset.columns:
            for _, group in subset.groupby("design_id", dropna=False):
                if len(group) > 1:
                    ax.plot(
                        group["logic_fidelity"],
                        group["effect_scaled"],
                        color="0.72",
                        linewidth=0.9,
                        zorder=1,
                    )
        mappable = ax.scatter(
            subset["logic_fidelity"],
            subset["effect_scaled"],
            c=subset["sfxi"],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=42,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )
        if label_points and "design_id" in subset.columns:
            for _, row in subset.iterrows():
                ax.annotate(
                    str(row["design_id"]),
                    (float(row["logic_fidelity"]), float(row["effect_scaled"])),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
        label = _setpoint_label(subset.iloc[0])
        ax.set_title(f"{setpoint_name} {label}")
        ax.set_xlabel("logic_fidelity")
        ax.set_ylabel("effect_scaled")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, color="0.9", linewidth=0.8)

    for ax in axes[len(setpoint_names) :]:
        ax.axis("off")
    if mappable is not None:
        fig.colorbar(mappable, ax=axes[: len(setpoint_names)], label="sfxi", shrink=0.88)

    base = filename or "sfxi_setpoint_scatter"
    return [PlotFigure(fig=fig, filename=base, ext=str(ext).lstrip(".").lower(), dpi=dpi) for ext in formats]


__all__ = [
    "READER_SUPPORTED_SFXI_API_VERSION",
    "VEC8_COLUMNS",
    "render_sfxi_setpoint_scatter",
    "require_dnadesign_sfxi_api",
    "score_sfxi_setpoints",
]
