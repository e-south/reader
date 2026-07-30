"""Public event-table preparation and analysis for cytometry workflows."""

from reader_workbench.domains.cytometry.analysis.events import (
    CytometryAnalysis,
    CytometryAnalysisError,
    GateSpec,
    ThresholdSpec,
    prepare_event_table,
)
from reader_workbench.domains.cytometry.analysis.gating import analyze_events
from reader_workbench.domains.cytometry.analysis.workflow import (
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
