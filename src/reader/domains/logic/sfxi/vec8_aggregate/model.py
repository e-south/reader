from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import VEC8_CHANNELS


@dataclass(frozen=True)
class SFXIVec8Source:
    source_id: str
    source_path: Path
    table_path: Path
    source_kind: str
    row_count: int
    record_id: str | None = None
    record_metadata: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_id": self.source_id,
            "source_path": str(self.source_path),
            "table_path": str(self.table_path),
            "source_kind": self.source_kind,
            "row_count": self.row_count,
        }
        if self.record_id is not None:
            payload["record_id"] = self.record_id
        if self.record_metadata is not None:
            payload["record"] = dict(self.record_metadata)
        return payload


@dataclass(frozen=True)
class SFXIVec8Aggregate:
    frame: pd.DataFrame
    sources: tuple[SFXIVec8Source, ...]

    @property
    def summary(self) -> dict[str, int]:
        return {"sources": len(self.sources), "rows": len(self.frame), "channels": len(VEC8_CHANNELS)}

    @property
    def intensity_log2_offset_deltas(self) -> tuple[float, ...]:
        values = self.frame["intensity_log2_offset_delta"].drop_duplicates().astype(float).sort_values()
        return tuple(float(value) for value in values.tolist())


@dataclass(frozen=True)
class LoadedSFXIVec8Source:
    source_id: str
    source_path: Path
    table_path: Path
    source_kind: str
    frame: pd.DataFrame
    record_id: str | None = None
    record_metadata: dict[str, Any] | None = None
