from __future__ import annotations

import pytest

from reader.core.engine.setup import slice_pipeline_steps
from reader.core.errors import ConfigError
from reader.core.workbench import WorkbenchSpec


def _steps(*ids: str) -> list[WorkbenchSpec]:
    return [WorkbenchSpec(kind="pipeline", id=step_id, uses="transform/ratio") for step_id in ids]


def test_slice_pipeline_steps_rejects_reversed_range() -> None:
    with pytest.raises(ConfigError, match="comes after"):
        slice_pipeline_steps(_steps("a", "b", "c"), resume_from="c", until="a")
