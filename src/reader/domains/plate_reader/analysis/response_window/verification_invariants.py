"""Cross-record invariants for response-window bundles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sources import STATE_ORDER

_PLOT_MANIFEST_FIELDS = {
    "plot_id",
    "tier",
    "title",
    "premise",
    "decision_value",
    "rationale",
    "alt_text",
    "non_claim_boundary",
    "data_table",
    "path",
}


def verify_record_invariants(
    *,
    root: Path,
    manifest: dict[str, object],
    artifacts: dict[str, dict[str, object]],
    counts: dict[str, int],
    display: dict[str, object],
    frames: dict[str, pd.DataFrame],
) -> None:
    wells = frames["wells"]
    designs = frames["designs"]
    draws = frames["bootstrap_draws"]
    traces = frames["traces"]
    events = frames["events"]
    expected_experiments = set(events["experiment_id"].astype(str))
    for record_id, frame in frames.items():
        observed = set(frame["experiment_id"].astype(str))
        if observed != expected_experiments:
            raise ValueError(f"response-window {record_id} experiment universe disagrees with events.")

    key_columns = ["experiment_id", "design_id", "reduction_id"]
    design_keys = _keys(designs, key_columns)
    if _keys(wells, key_columns) != design_keys or _keys(draws, key_columns) != design_keys:
        raise ValueError("response-window design, well, and bootstrap-draw universes disagree.")
    _verify_state_coverage(wells)
    _verify_reduction_contract(designs, manifest)
    _verify_reference_contract(designs, display)
    _verify_bootstrap_contract(designs, draws)
    _verify_well_counts(designs, wells)
    _verify_event_contract(events)
    _verify_display_examples(designs, display)
    _verify_counts(designs, events, counts)
    _verify_plot_manifest(root, artifacts, counts)
    if set(traces["signal_kind"].astype(str)) != {"response", "magnitude", "growth"}:
        raise ValueError("response-window traces must preserve response, magnitude, and growth signals.")


def _verify_state_coverage(wells: pd.DataFrame) -> None:
    states = wells.groupby(["experiment_id", "design_id", "reduction_id"], sort=False)["state"].agg(
        lambda values: frozenset(map(str, values))
    )
    expected = frozenset(STATE_ORDER)
    if any(value != expected for value in states):
        raise ValueError("every response-window design and reduction must contain all four conditions.")


def _verify_reduction_contract(designs: pd.DataFrame, manifest: dict[str, object]) -> None:
    fields = [
        "reduction_method",
        "response_basis",
        "reduction_role",
        "window_start_event_h",
        "window_end_event_h",
    ]
    if (designs["window_end_event_h"] <= designs["window_start_event_h"]).any():
        raise ValueError("response-window reductions require window end after window start.")
    if (designs.groupby("reduction_id")[fields].nunique(dropna=False) != 1).any().any():
        raise ValueError("response-window reduction semantics drift across design rows.")
    primary_ids = designs.loc[designs["reduction_role"].eq("primary"), "reduction_id"].unique().tolist()
    if primary_ids != [manifest.get("primary_reduction_id")]:
        raise ValueError("response-window primary reduction disagrees with the bundle manifest.")


def _verify_reference_contract(designs: pd.DataFrame, display: dict[str, object]) -> None:
    channels = display.get("channels")
    if not isinstance(channels, dict):
        raise ValueError("response-window display channels are malformed.")
    reference_id = channels["reference_design_id"]
    if set(designs["reference_design_id"].astype(str)) != {reference_id}:
        raise ValueError("response-window design rows disagree with the display reference anchor.")
    reference_rows = designs.loc[designs["is_reference"].astype(bool)]
    if set(reference_rows["design_id"].astype(str)) != {reference_id}:
        raise ValueError("response-window reference rows disagree with reference_design_id.")
    counts = reference_rows.groupby(["experiment_id", "reduction_id"]).size()
    if counts.empty or not counts.eq(1).all():
        raise ValueError("response-window requires one reference row per experiment and reduction.")


def _verify_bootstrap_contract(designs: pd.DataFrame, draws: pd.DataFrame) -> None:
    observed = draws.groupby(["experiment_id", "design_id", "reduction_id"]).size().sort_index()
    expected = designs.set_index(["experiment_id", "design_id", "reduction_id"])["bootstrap_samples"].sort_index()
    if not observed.index.equals(expected.index) or not np.array_equal(observed.to_numpy(), expected.to_numpy()):
        raise ValueError("response-window bootstrap draw counts disagree with design metadata.")


def _verify_well_counts(designs: pd.DataFrame, wells: pd.DataFrame) -> None:
    observed = wells.groupby(["experiment_id", "design_id", "reduction_id", "state"])["position"].nunique()
    for state in STATE_ORDER:
        expected = designs.set_index(["experiment_id", "design_id", "reduction_id"])[f"n{state}"].sort_index()
        state_observed = observed.xs(state, level="state").sort_index()
        if not state_observed.index.equals(expected.index) or not np.array_equal(
            state_observed.to_numpy(), expected.to_numpy()
        ):
            raise ValueError(f"response-window well counts disagree with design metadata for state {state}.")


def _verify_event_contract(events: pd.DataFrame) -> None:
    midpoint = (events["event_interval_start_assay_h"] + events["event_interval_end_assay_h"]) / 2.0
    half_range = (events["event_interval_end_assay_h"] - events["event_interval_start_assay_h"]) / 2.0
    if not np.allclose(events["event_time_estimate_assay_h"], midpoint, rtol=0.0, atol=1.0e-12):
        raise ValueError("response-window event estimates must equal interval midpoints.")
    if not np.allclose(events["event_time_uncertainty_h"], half_range, rtol=0.0, atol=1.0e-12):
        raise ValueError("response-window event uncertainty must equal the interval half-range.")


def _verify_display_examples(designs: pd.DataFrame, display: dict[str, object]) -> None:
    raw_examples = display.get("examples")
    if not isinstance(raw_examples, list):
        raise ValueError("response-window display examples are malformed.")
    examples = {str(item["design_id"]) for item in raw_examples if isinstance(item, dict)}
    missing = sorted(examples - set(designs["design_id"].astype(str)))
    if missing:
        raise ValueError(f"response-window bundle lacks configured display examples: {missing}.")


def _verify_counts(designs: pd.DataFrame, events: pd.DataFrame, counts: dict[str, int]) -> None:
    primary = designs.loc[designs["reduction_role"].eq("primary") & ~designs["is_reference"].astype(bool)]
    expected = {
        "experiments": int(events["experiment_id"].nunique()),
        "unique_design_ids": int(designs["design_id"].nunique()),
        "repeated_design_ids": int((primary.groupby("design_id")["experiment_id"].nunique() > 1).sum()),
        "reductions": int(designs["reduction_id"].nunique()),
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise ValueError(f"response-window count {key!r} disagrees with persisted records.")


def _verify_plot_manifest(
    root: Path,
    artifacts: dict[str, dict[str, object]],
    counts: dict[str, int],
) -> None:
    plots = pd.read_csv(root / "tables" / "plot_manifest.csv")
    if set(plots.columns) != _PLOT_MANIFEST_FIELDS or len(plots) != counts["plots"]:
        raise ValueError("response-window plot manifest disagrees with its public field or count contract.")
    if plots["plot_id"].duplicated().any() or plots.isna().any().any():
        raise ValueError("response-window plot manifest contains duplicate IDs or missing metadata.")
    if any(not str(title).strip() or str(title).endswith(".") for title in plots["title"]):
        raise ValueError("response-window plot titles must be non-empty sentences without terminal periods.")
    for row in plots.itertuples(index=False):
        if str(row.path) not in artifacts or str(row.data_table) not in artifacts:
            raise ValueError(f"response-window plot {row.plot_id!r} references an untracked artifact.")


def _keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple[object, ...]]:
    return set(frame.loc[:, columns].itertuples(index=False, name=None))


__all__ = ["verify_record_invariants"]
