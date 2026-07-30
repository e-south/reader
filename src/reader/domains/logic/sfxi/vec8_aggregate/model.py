from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SFXIVec8Source:
    resource_id: str
    experiment_id: str
    record_id: str
    revision_digest: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class SFXIVec8Aggregate:
    frame: pd.DataFrame
