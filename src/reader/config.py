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
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _single_raw(dir_: Path) -> Path:
    files = [
        p for p in dir_.iterdir()
        if p.suffix.lower() in {".csv", ".xlsx", ".fcs"} and not p.name.startswith("~$")
    ]
    if len(files) != 1:
        raise ValueError(f"'{dir_}' must contain exactly one raw file (found {files})")
    return files[0]


class SheetCfg(BaseModel):
    add_column: bool = False
    names: Optional[List[str]] = None


def _alias_plate_map(value, info):
    return info.data.get("plate_map", value)


class DataCfg(BaseModel):
    raw: Union[Path, List[Path]]
    sample_map: Optional[Path] = None

    _alias_plate = field_validator("sample_map", mode="before")(_alias_plate_map)
    _expand      = field_validator("raw", "sample_map", mode="before")(_expand_path)

    @model_validator(mode="after")
    def _resolve(self) -> "DataCfg":
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
    def _mkdir(self) -> "OutputCfg":
        self.dir.mkdir(parents=True, exist_ok=True)
        return self


class ParsingCfg(BaseModel):
    parser: str
    time_column: str = "time"
    sheet: SheetCfg = Field(default_factory=SheetCfg)
    extra: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class XForm(BaseModel):
    type: str
    class Config:
        extra = "allow"


class CustomParameter(BaseModel):
    name: str
    type: str
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


# ── Typed schema for logic_symmetry ───────────────────────────────────────────

class LogicSymmetryPrepCfg(BaseModel):
    enable: bool = False
    align_corners: bool = False
    mode: Literal["nearest", "first", "last", "median", "exact"] = "last"
    target_time: Optional[float] = None
    tolerance: float = 0.51
    case_sensitive_treatments: bool = True
    time_column: str = "time"


class LogicSymmetryAggregationCfg(BaseModel):
    replicate_stat: Literal["mean", "median"] = "mean"
    uncertainty:   Literal["none", "errorbars", "halo"] = "halo"


class LogicSymmetryEncodingsCfg(BaseModel):
    size_by: Literal["log_r", "cv", "fixed"] = "log_r"
    size_fixed: float = 80.0
    hue: Optional[str] = None
    alpha_by: Optional[str] = "batch"
    alpha_min: float = 0.35
    alpha_max: float = 1.0
    shape_by: Optional[str] = None
    shape_cycle: List[str] = Field(default_factory=lambda: ["o","s","^","D","P","X","v","*"])
    shape_max_categories: Optional[int] = None


class LogicSymmetryOverlayStyleCfg(BaseModel):
    # shared (both dot & tiles)
    alpha: float = 0.25
    size:  float = 40.0
    face_color: Optional[str] = "#FFFFFF"
    edge_color: Optional[str] = "#888888"
    show_labels: bool = True
    label_offset: float = 0.02
    label_line_height: float = 0.018

    # tiles extras
    mode: Optional[Literal["dot", "tiles"]] = None
    tile_cell_w: Optional[float] = None
    tile_cell_h: Optional[float] = None
    tile_gap: Optional[float] = None
    tile_edge_width: Optional[float] = None
    tiles_stack_multiple: Optional[bool] = None

    class Config:
        extra = "allow"


class LogicSymmetryVisualsCfg(BaseModel):
    color: str = "#6e6e6e"
    xlim: List[float] = Field(default_factory=lambda: [-1.02, 1.02])
    ylim: List[float] = Field(default_factory=lambda: [-1.02, 1.02])
    grid: bool = True
    annotate_designs: bool = False
    design_label_col: Optional[str] = None
    label_fontsize: int = 9
    label_offset: float = 0.02

    class Config:
        extra = "allow"


class LogicSymmetryOverlayCfg(BaseModel):
    enable: bool = False
    gate_set: Literal["core", "logic_family", "full16", "tiles_dual"] = "logic_family"
    style: LogicSymmetryOverlayStyleCfg = Field(default_factory=LogicSymmetryOverlayStyleCfg)


class LogicSymmetryOutputCfg(BaseModel):
    format: List[Literal["pdf", "png"]] = Field(default_factory=lambda: ["pdf"])
    dpi: int = 300
    figsize: List[float] = Field(default_factory=lambda: [7.0, 6.0])


class LogicSymmetryParams(BaseModel):
    response_channel: str
    design_by: List[str] = Field(default_factory=lambda: ["genotype"])
    batch_col: str = "batch"
    treatment_map: Dict[str, str]
    treatment_case_sensitive: bool = True

    aggregation: LogicSymmetryAggregationCfg = Field(default_factory=LogicSymmetryAggregationCfg)
    encodings:   LogicSymmetryEncodingsCfg   = Field(default_factory=LogicSymmetryEncodingsCfg)
    ideals_overlay: LogicSymmetryOverlayCfg  = Field(default_factory=LogicSymmetryOverlayCfg)
    visuals: LogicSymmetryVisualsCfg         = Field(default_factory=LogicSymmetryVisualsCfg)
    output:  LogicSymmetryOutputCfg          = Field(default_factory=LogicSymmetryOutputCfg)

    prep: Optional[LogicSymmetryPrepCfg] = None

    @model_validator(mode="after")
    def _check_tmap_keys(self) -> "LogicSymmetryParams":
        keys = set(self.treatment_map.keys())
        required = {"00", "10", "01", "11"}
        if keys != required:
            raise ValueError(f"treatment_map must have exactly the keys {sorted(required)}; got {sorted(keys)}")
        return self


def normalize_plot_params(spec: PlotSpec) -> Dict[str, Any]:
    if spec.module == "logic_symmetry":
        params = LogicSymmetryParams.model_validate(spec.params)
        return params.model_dump(exclude_none=True)
    return dict(spec.params)


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
        return next((x for x in self.transformations if x.type == x_type), None)
