"""Source-level readiness checks for response-window requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import load_response_window_request
from .provenance import sha256_file
from .sources import ExperimentSource, ExperimentSourceLoader


@dataclass(frozen=True)
class ExperimentPreflight:
    experiment_id: str
    response_designs: int
    magnitude_designs: int
    trajectory_designs: int
    response_rows: int
    magnitude_rows: int
    trajectory_rows: int
    event_interval_start_assay_h: float
    event_interval_end_assay_h: float
    event_time_estimate_assay_h: float
    event_time_uncertainty_h: float
    post_event_coverage_h: float
    record_digests: dict[str, str]


@dataclass(frozen=True)
class ResponseWindowPreflight:
    ready: bool
    request_id: str
    request_path: Path
    request_sha256: str
    schema_version: str
    state_order: tuple[str, str, str, str]
    primary_reduction_id: str
    reduction_ids: tuple[str, ...]
    experiments: tuple[ExperimentPreflight, ...]
    observed_design_ids: tuple[str, ...]
    missing_display_examples: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["request_path"] = str(self.request_path)
        return payload


def preflight_response_window_request(
    *,
    request_path: Path,
    source_loader: ExperimentSourceLoader,
) -> ResponseWindowPreflight:
    """Verify source records and report whether one request can be built."""

    request_file = Path(request_path).expanduser().resolve()
    request = load_response_window_request(request_file)
    sources = tuple(
        source_loader(experiment_id, request.source, request.event) for experiment_id in request.experiment_ids
    )
    observed_design_ids = tuple(
        sorted({str(value) for source in sources for value in source.response["design_id"].unique()})
    )
    required_examples = {example.design_id for example in request.display.examples}
    missing_examples = tuple(sorted(required_examples - set(observed_design_ids)))
    return ResponseWindowPreflight(
        ready=not missing_examples,
        request_id=request.request_id,
        request_path=request_file,
        request_sha256=sha256_file(request_file),
        schema_version=request.schema_version,
        state_order=request.state_order,
        primary_reduction_id=request.primary_reduction.id,
        reduction_ids=tuple(spec.id for spec in request.reductions),
        experiments=tuple(_experiment_preflight(source) for source in sources),
        observed_design_ids=observed_design_ids,
        missing_display_examples=missing_examples,
    )


def _experiment_preflight(source: ExperimentSource) -> ExperimentPreflight:
    event = source.event
    return ExperimentPreflight(
        experiment_id=source.experiment_id,
        response_designs=int(source.response["design_id"].nunique()),
        magnitude_designs=int(source.magnitude["design_id"].nunique()),
        trajectory_designs=int(source.trajectory["design_id"].nunique()),
        response_rows=len(source.response),
        magnitude_rows=len(source.magnitude),
        trajectory_rows=len(source.trajectory),
        event_interval_start_assay_h=event.interval_start_assay_h,
        event_interval_end_assay_h=event.interval_end_assay_h,
        event_time_estimate_assay_h=event.estimate_assay_h,
        event_time_uncertainty_h=event.uncertainty_h,
        post_event_coverage_h=event.post_event_coverage_h,
        record_digests=dict(sorted(source.record_digests.items())),
    )


__all__ = [
    "ExperimentPreflight",
    "ResponseWindowPreflight",
    "preflight_response_window_request",
]
