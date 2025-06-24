"""
--------------------------------------------------------------------------------
<reader project>
reader/config.py

A Pydantic schema that mirrors one idea:

    RAW DATA  →  PARSER  →  TRANSFORMS  →  PLOTS

Blocks in YAML should include:

data:              ↔ DataCfg      (raw files + optional sample_map)
output:            ↔ OutputCfg    (output folder)
parsing:           ↔ ParsingCfg   (which parser + free-form knobs)
transformations:   ↔ List[XForm]  (ordered, plug-in defined)
naming:            ↔ NamingCfg
plotting:          ↔ PlottingCfg  (defaults + per-plot specs)

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# helpers
def _expand_path(value: str | Path) -> Path:
    """Expand ~/ and $ENV and return a resolved Path."""
    return Path(value).expanduser().resolve()


def _single_raw(dir_: Path) -> Path:
    """Return the single CSV/XLSX/FCS file in *dir_* or raise."""
    files = [
        p for p in dir_.iterdir()
        if p.suffix.lower() in {".csv", ".xlsx", ".fcs"} and not p.name.startswith("~$")
    ]
    if len(files) != 1:
        raise ValueError(
            f"'{dir_}' must contain exactly one raw file (found {files})"
        )
    return files[0]


# data / I-O
class SheetCfg(BaseModel):
    add_column: bool = False
    names: Optional[List[str]] = None


def _alias_plate_map(value, info):  # noqa: D401
    """Allow legacy ``plate_map:`` key to act as ``sample_map:``."""
    return info.data.get("plate_map", value)


class DataCfg(BaseModel):
    raw: Union[Path, List[Path]]
    sample_map: Optional[Path] = None

    _alias_plate = field_validator("sample_map", mode="before")(_alias_plate_map)
    _expand      = field_validator("raw", "sample_map", mode="before")(_expand_path)

    @model_validator(mode="after")
    def _resolve(self) -> "DataCfg":  # noqa: D401
        raw_list = self.raw if isinstance(self.raw, list) else [self.raw]
        resolved = [_single_raw(p) if p.is_dir() else p for p in raw_list]
        object.__setattr__(self, "raw", resolved)

        if self.sample_map and not self.sample_map.is_file():
            stem = self.sample_map.stem
            hits = [
                p for p in self.sample_map.parent.glob(f"{stem}.*")
                if p.suffix in {".csv", ".xlsx"}
            ]
            if len(hits) != 1:
                raise FileNotFoundError(f"sample_map ambiguous: {hits}")
            object.__setattr__(self, "sample_map", hits[0])
        return self


class OutputCfg(BaseModel):
    dir: Path

    _expand = field_validator("dir", mode="before")(_expand_path)

    @model_validator(mode="after")
    def _mkdir(self) -> "OutputCfg":  # noqa: D401
        self.dir.mkdir(parents=True, exist_ok=True)
        return self


# parsing block
class ParsingCfg(BaseModel):
    parser: str
    time_column: str = "time"
    sheet: SheetCfg = Field(default_factory=SheetCfg)
    extra: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


# misc blocks
class XForm(BaseModel):
    type: str

    class Config:
        extra = "allow"


class CustomParameter(BaseModel):
    name: str
    type: str          # e.g. "ratio"
    parameters: List[str]


class PlotSpec(BaseModel):
    name: str
    module: str
    params: Dict[str, Any] = Field(default_factory=dict)
    subplots: Optional[str] = None
    filename: Optional[str] = None
    groups: Optional[List[Dict[str, List[str]]]] = None
    fig: Optional[Dict[str, Any]] = None
    iterate_genotypes: bool = False

    class Config:
        extra = "forbid"


class PlotDefaults(BaseModel):
    channels: List[str] = Field(default_factory=list)
    groups: Optional[List[Dict[str, List[str]]]] = None
    fig: Dict[str, Any] = Field(default_factory=dict)


class PlottingCfg(BaseModel):
    defaults: PlotDefaults
    plots: List[PlotSpec] = Field(default_factory=list)


class NamingCfg(BaseModel):
    use_short: bool = False
    map: Dict[str, str] = Field(default_factory=dict)


# root
class ReaderCfg(BaseModel):
    data: DataCfg
    output: OutputCfg
    parsing: ParsingCfg
    transformations: List[XForm] = Field(default_factory=list)
    naming: NamingCfg = Field(default_factory=NamingCfg)
    plotting: PlottingCfg

    @property
    def raw_files(self) -> List[Path]:
        return self.data.raw

    @property
    def channels(self) -> List[str]:
        return self.plotting.defaults.channels

    def xform(self, x_type: str) -> Optional[XForm]:
        """Return the first transformation matching *x_type*."""
        return next((x for x in self.transformations if x.type == x_type), None)