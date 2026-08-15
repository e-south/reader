from __future__ import annotations

import pandas as pd

from reader_workbench.domains.plate_reader.analysis.four_state_event_window.contracts import (
    QualitySpec,
    ReductionSpec,
)
from reader_workbench.domains.plate_reader.analysis.four_state_event_window.well_exclusion_validation import (
    require_well_exclusion_support_failure,
)
from reader_workbench.domains.plate_reader.analysis.four_state_event_window.well_exclusions import WellExclusion


def test_declared_exclusion_uses_the_reducers_interior_gap_contract() -> None:
    frame = pd.DataFrame(
        {
            "position": ["D7"] * 4,
            "time": [0.0, 1.0, 2.0, 3.0],
            "value": [2.0, 2.0, 2.0, 2.0],
            "value_policy_clipped": [False] * 4,
            "value_instrument_overflow": [False] * 4,
            "value_bound_kind": ["exact"] * 4,
        }
    )
    reduction = ReductionSpec.from_mapping(
        {
            "id": "primary",
            "window_start_event_h": 1.0,
            "window_end_event_h": 2.0,
            "method": "geometric_time_mean",
            "response_basis": "post_window",
            "role": "primary",
        },
        index=0,
    )
    quality = QualitySpec.from_mapping(
        {
            "positive_floor": 1.0e-12,
            "max_interior_gap_h": 0.75,
            "min_observations_per_state": 2,
        }
    )

    require_well_exclusion_support_failure(
        (
            WellExclusion(
                experiment_id="experiment-a",
                design_id="candidate",
                state="01",
                position="D7",
                reduction_id="primary",
                reason="insufficient_event_window_coverage",
            ),
        ),
        reductions={"primary": reduction},
        quality=quality,
        response=frame,
        magnitude=frame,
        event_estimates_h=(0.0,),
        pre_window_end_h=0.0,
        experiment_id="experiment-a",
    )


def test_declared_exclusion_accepts_missing_pre_event_support() -> None:
    frame = pd.DataFrame(
        {
            "position": ["D7"] * 3,
            "time": [2.0, 3.0, 4.0],
            "value": [2.0, 2.0, 2.0],
            "value_policy_clipped": [False] * 3,
            "value_instrument_overflow": [False] * 3,
            "value_bound_kind": ["exact"] * 3,
        }
    )
    reduction = ReductionSpec.from_mapping(
        {
            "id": "primary",
            "window_start_event_h": 1.0,
            "window_end_event_h": 2.0,
            "method": "geometric_time_mean",
            "response_basis": "post_minus_pre",
            "pre_window_duration_h": 1.0,
            "role": "primary",
        },
        index=0,
    )
    quality = QualitySpec.from_mapping(
        {
            "positive_floor": 1.0e-12,
            "max_interior_gap_h": 1.0,
            "min_observations_per_state": 2,
        }
    )

    require_well_exclusion_support_failure(
        (
            WellExclusion(
                experiment_id="experiment-a",
                design_id="candidate",
                state="01",
                position="D7",
                reduction_id="primary",
                reason="insufficient_event_window_coverage",
            ),
        ),
        reductions={"primary": reduction},
        quality=quality,
        response=frame,
        magnitude=frame,
        event_estimates_h=(2.0,),
        pre_window_end_h=2.0,
        experiment_id="experiment-a",
    )
