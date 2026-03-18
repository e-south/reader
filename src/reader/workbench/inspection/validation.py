from __future__ import annotations

from copy import deepcopy
from typing import Any


def validation_surface_payload(
    *,
    experiment: dict[str, object],
    check_files: bool,
    summary: dict[str, Any],
) -> dict[str, object]:
    payload = deepcopy(summary)
    return {
        "experiment": deepcopy(experiment),
        "selection": {
            "check_files": check_files,
        },
        "summary": {
            "status": payload.pop("status"),
            "checks": payload.pop("checks"),
            "counts": payload.pop("counts"),
        },
        "validation": {
            "files": payload.pop("files"),
            "tip": payload.pop("tip"),
        },
    }
