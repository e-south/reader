"""
--------------------------------------------------------------------------------
<reader project>
src/reader/config.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, model_validator

class CustomParameter(BaseModel):
    name: str
    type: str
    parameters: List[str]

class PlotSpec(BaseModel):
    name: str
    module: str
    filename: Optional[str] = None
    subplots: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    groups: Optional[List[Dict[str, List[str]]]] = None
    iterate_genotypes: bool = False
    fig: Optional[Dict[str, Any]] = None

    class Config:
        extra = "forbid"

class ReaderConfig(BaseModel):
    raw_data: Path
    plate_map: Path
    output_dir: Path

    data_parser: str
    parameters: Optional[List[str]] = None

    sheet_names: Optional[List[str]] = Field(
        default=None,
        description="If provided, only parse these sheets (in this order)."
    )

    add_sheet_column: bool = Field(
        default=False,
        description="If true, the parser will add a 'sheet' column (0-based index)."
    )

    time_column: str = "time"
    blank_correction: str = "avg_blank"
    overflow_action: str = "max"

    custom_parameters: List[CustomParameter] = Field(default_factory=list)

    use_short_genotype_names: bool = False
    short_genotype_names: List[Dict[str, str]] = Field(default_factory=list)

    plots: List[PlotSpec] = Field(default_factory=list)

    class Config:
        extra = "forbid"

    @model_validator(mode="before")
    def _coerce_channel_list(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        # support legacy 'channels' → 'parameters'
        if "channels" in data and not data.get("parameters"):
            data["parameters"] = data.pop("channels")
        return data

    @model_validator(mode="after")
    def _expand_raw_data_dir(self):
        rd = self.raw_data
        if rd.is_dir():
            candidates = [f for f in rd.glob("*.csv")] + [f for f in rd.glob("*.xlsx")]
            candidates = [f for f in candidates if not f.name.startswith("~$")]
            if len(candidates) == 1:
                object.__setattr__(self, "raw_data", candidates[0])
            elif not candidates:
                raise ValueError(f"No CSV/XLSX in raw_data folder: {rd}")
            else:
                raise ValueError(f"Multiple CSV/XLSX in {rd}: {candidates}")
        return self

    @model_validator(mode="after")
    def _normalize_plate_map(self):
        pm = self.plate_map
        if not pm.is_file():
            alt = pm.with_suffix(".xlsx")
            if alt.is_file():
                object.__setattr__(self, "plate_map", alt)
            else:
                stem = pm.stem
                cand = [p for p in pm.parent.glob(f"{stem}.*") if p.suffix in {".csv", ".xlsx"}]
                if len(cand) == 1:
                    object.__setattr__(self, "plate_map", cand[0])
                elif not cand:
                    raise ValueError(f"plate_map not found: {pm}")
                else:
                    raise ValueError(f"Multiple plate_map: {cand}")
        return self

    @model_validator(mode="after")
    def _prepare_output_dir(self):
        od = self.output_dir
        od.mkdir(parents=True, exist_ok=True)
        return self