"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/analysis/__init__.py

Plate-reader derived-table and timepoint-selection helpers.
--------------------------------------------------------------------------------
"""

from .fold_change import compute_fold_change_table
from .retron_sponge import compute_retron_sponge_metrics
from .spop import (
    SPOP_ACRONYM,
    SPOP_DEFAULT_LAMBDA,
    SPOP_METRIC_ID,
    SPOP_NORMALIZATION_BASIS,
    SPOP_NUMERIC_SCOPE,
    SPOP_REPORTER_READOUT,
    SPOP_VIABILITY_READOUT,
    SpopDoseValue,
    SpopEndpointScore,
    SpopScoringError,
    score_spop_endpoint,
)
from .timepoints import choose_nearest_time, nearest_time_per_key

__all__ = [
    "SPOP_ACRONYM",
    "SPOP_DEFAULT_LAMBDA",
    "SPOP_METRIC_ID",
    "SPOP_NORMALIZATION_BASIS",
    "SPOP_NUMERIC_SCOPE",
    "SPOP_REPORTER_READOUT",
    "SPOP_VIABILITY_READOUT",
    "SpopDoseValue",
    "SpopEndpointScore",
    "SpopScoringError",
    "choose_nearest_time",
    "compute_fold_change_table",
    "compute_retron_sponge_metrics",
    "nearest_time_per_key",
    "score_spop_endpoint",
]
