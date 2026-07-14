"""Request-to-bundle parity checks for response-window evidence."""

from __future__ import annotations

import pandas as pd

from .contracts import ResponseWindowRequest
from .verification_request_payload import verify_request_payload_parity


def verify_request_parity(
    manifest: dict[str, object],
    request: ResponseWindowRequest,
    *,
    frames: dict[str, pd.DataFrame],
    source_records: tuple[dict[str, object], ...],
) -> None:
    if manifest.get("study_id") != request.study_id:
        raise ValueError("response-window bundle study identity disagrees with bundled request.yaml.")
    if manifest.get("request_id") != request.request_id:
        raise ValueError("response-window bundle request identity disagrees with bundled request.yaml.")
    if manifest.get("primary_reduction_id") != request.primary_reduction.id:
        raise ValueError("response-window bundle primary reduction disagrees with bundled request.yaml.")
    if manifest.get("state_order") != list(request.state_order):
        raise ValueError("response-window bundle state order disagrees with bundled request.yaml.")
    expected_display = request.display.to_manifest(
        response_ratio=request.source.response_channel,
        magnitude_ratio=request.source.magnitude_channel,
        growth=request.source.growth_channel,
        reference_design_id=request.source.reference_design_id,
    )
    if manifest.get("display") != expected_display:
        raise ValueError("response-window bundle display disagrees with bundled request.yaml.")
    verify_request_payload_parity(request, frames=frames, source_records=source_records)


__all__ = ["verify_request_parity"]
