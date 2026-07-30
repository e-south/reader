"""Public event-table preparation and analysis for cytometry workflows."""

from reader.domains.cytometry.analysis.events import (
    CytometryAnalysis,
    CytometryAnalysisError,
    GateSpec,
    ThresholdSpec,
    prepare_event_table,
)
from reader.domains.cytometry.analysis.gating import analyze_events
from reader.domains.cytometry.analysis.workflow import (
    CytometryGatingRequest,
    CytometryGatingResult,
    CytometryQCSpec,
    run_cytometry_gating,
)

__all__ = [
    "CytometryAnalysis",
    "CytometryAnalysisError",
    "CytometryGatingRequest",
    "CytometryGatingResult",
    "CytometryQCSpec",
    "GateSpec",
    "ThresholdSpec",
    "analyze_events",
    "prepare_event_table",
    "run_cytometry_gating",
]
