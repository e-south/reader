from __future__ import annotations

import pandas as pd

from reader.domains.plate_reader.analysis.timepoints import infer_acquisition_transition_time_h


def test_acquisition_transition_uses_first_time_in_later_sheet() -> None:
    frame = pd.DataFrame(
        {
            "sheet_index": [0, 0, 1, 1],
            "time": [0.0, 2.0, 4.5, 6.5],
        }
    )

    assert infer_acquisition_transition_time_h(frame, time_col="time") == 4.5


def test_acquisition_transition_does_not_interpret_biological_event_columns() -> None:
    frame = pd.DataFrame({"induction_time_h": [3.0], "time": [4.0]})

    assert infer_acquisition_transition_time_h(frame, time_col="time") is None
