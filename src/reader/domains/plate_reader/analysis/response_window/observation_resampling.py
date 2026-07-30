"""Descriptive resampling primitives for within-experiment observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ObservationStat, ResponseWindowAnalysisSpec
from .seeds import stable_seed
from .sources import STATE_ORDER


def descriptive_resampling_records(wells: pd.DataFrame, *, request: ResponseWindowAnalysisSpec) -> pd.DataFrame:
    """Resample observed wells jointly without implying population inference."""

    anchor_id = request.source.reference_design_id
    anchor = wells.loc[wells["design_id"].astype(str).eq(anchor_id)]
    records: list[pd.DataFrame] = []
    for design_id, design_wells in wells.groupby("design_id", sort=True):
        state_draws: dict[str, np.ndarray] = {}
        for state in STATE_ORDER:
            state_wells = design_wells.loc[design_wells["state"].astype(str).eq(state)]
            anchor_wells = anchor.loc[anchor["state"].astype(str).eq(state)]
            response_values = state_wells["response_well"].to_numpy(dtype=float)
            magnitude_values = state_wells["magnitude_well"].to_numpy(dtype=float)
            anchor_values = anchor_wells["magnitude_well"].to_numpy(dtype=float)
            minimum = request.quality.min_observations_per_state
            if min(len(response_values), len(magnitude_values), len(anchor_values)) < minimum:
                raise ValueError(f"{design_id}:{state} lacks support for within-experiment descriptive resampling.")
            reduction_id = str(state_wells["reduction_id"].iloc[0])
            experiment_id = str(state_wells["experiment_id"].iloc[0])
            rng = np.random.default_rng(
                stable_seed(request.aggregation.random_seed, experiment_id, str(design_id), state, reduction_id)
            )
            response_draws, anchored_magnitude_draws = joint_state_descriptive_resampling_draws(
                response_values,
                magnitude_values,
                anchor_values,
                samples=request.aggregation.descriptive_resampling_draws,
                stat=request.aggregation.observation_stat,
                rng=rng,
                paired_anchor=str(design_id) == anchor_id,
            )
            state_draws[f"r{state}"] = response_draws
            state_draws[f"b{state}"] = anchored_magnitude_draws
        frame = pd.DataFrame(state_draws)
        frame.insert(0, "draw_index", np.arange(len(frame), dtype=int))
        frame.insert(0, "reduction_id", str(design_wells["reduction_id"].iloc[0]))
        frame.insert(0, "design_id", str(design_id))
        frame.insert(0, "experiment_id", str(design_wells["experiment_id"].iloc[0]))
        frame["is_reference"] = str(design_id) == anchor_id
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def joint_state_descriptive_resampling_draws(
    response_values: np.ndarray,
    magnitude_values: np.ndarray,
    anchor_values: np.ndarray,
    *,
    samples: int,
    stat: ObservationStat,
    rng: np.random.Generator,
    paired_anchor: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample paired observed wells and either paired or independent reference wells."""

    response = np.asarray(response_values, dtype=float)
    magnitude = np.asarray(magnitude_values, dtype=float)
    anchor = np.asarray(anchor_values, dtype=float)
    if stat not in {"mean", "median"}:
        raise ValueError("joint descriptive resampling requires stat to be 'mean' or 'median'.")
    if response.ndim != 1 or magnitude.ndim != 1 or anchor.ndim != 1:
        raise ValueError("joint descriptive-resampling inputs must be one-dimensional arrays.")
    if len(response) != len(magnitude) or len(response) == 0 or len(anchor) == 0:
        raise ValueError(
            "joint descriptive resampling requires aligned design observations and non-empty reference observations."
        )
    if samples < 2:
        raise ValueError("joint descriptive resampling requires at least two draws.")
    if paired_anchor and not np.array_equal(magnitude, anchor):
        raise ValueError(
            "paired reference descriptive resampling requires identical ordered magnitude and anchor observations."
        )
    design_indexes = rng.integers(0, len(response), size=(samples, len(response)))
    anchor_indexes = design_indexes if paired_anchor else rng.integers(0, len(anchor), size=(samples, len(anchor)))
    aggregate = np.median if stat == "median" else np.mean
    response_draws = aggregate(response[design_indexes], axis=1)
    magnitude_draws = aggregate(magnitude[design_indexes], axis=1)
    anchor_draws = aggregate(anchor[anchor_indexes], axis=1)
    return response_draws, magnitude_draws - anchor_draws


__all__ = [
    "descriptive_resampling_records",
    "joint_state_descriptive_resampling_draws",
]
