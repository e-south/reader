## reader

A workbench for turning instrument exports into tidy tables and plots.

- **Plugin engine**: `ingest → merge → transform → plot`
- **CLI**: `reader ls | explain | validate | run | artifacts | steps | plugins`
- **Auditable**: artifacts live next to the config; a manifest tracks revisions.

**Example usage**

You have an instrument that produces raw files. You add a small parser whose only job is to read that raw format and return a tidy table. From there, you declare a sequence of **transforms** that operate on columns to produce derived values (e.g., ratios) and to apply cleanups (e.g., blank subtraction or overflow handling). Finally, **plots** consume the cleaned/transformed tables and write figures.

This flow is described once in a YAML spec and executed by the CLI.

---

### Install with `uv`

```bash
# install uv (once)
# macOS/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:      iwr https://astral.sh/uv/install.ps1 | iex

uv venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
uv sync
uv pip install -e .
```

---

### Architecture

```bash
src/reader/
core/     # engine, registry, contracts, artifacts, cli, errors
plugins/  # built-ins: ingest/, merge/, transform/, plot/
io/       # instrument/file helpers (pure funcs: raw → tidy)
lib/      # domain algorithms & plotting utilities
```

- **Core** runs steps, enforces **contracts**, persists artifacts, and logs.
- **Plugins** are thin orchestration layers with explicit inputs/outputs.
- **io/** holds pure parsing helpers; **lib/** holds reusable logic.

Quick mental model:
```bash
raw/*.xlsx
└─▶ [ingest]        → tidy.v1
└─▶ [merge]         → tidy+map.v1
└─▶ [transforms...] → tidy.v1 / ...
└─▶ [plots]         → .pdf/.png (flat files)
````

Artifacts are written under `outputs/artifacts/<stepId.plugin>/__rN/…`.
`outputs/manifest.json` records the latest `<step_id>/<output_name>` and history.

---

### Usage

```bash
# scaffold an experiment
cp -r experiments/template experiments/20250512_my_assay

# add raw files under experiments/.../(raw|raw_data)/
# update sample_map.(csv|xlsx) and config.yaml

# inspect & run
reader ls --root ./experiments
reader explain   experiments/.../config.yaml
reader validate  experiments/.../config.yaml
reader run       experiments/.../config.yaml

# inspect results
reader artifacts experiments/.../config.yaml
ls experiments/.../outputs/plots
```

### Commands

* `reader ls --root DIR` — find experiments (`**/config.yaml`) under `DIR`.
* `reader explain CONFIG` — show the plan with plugin contracts.
* `reader validate CONFIG` — validate schema and per‑step plugin configs.
* `reader run CONFIG [--resume-from STEP_ID] [--until STEP_ID] [--dry-run] [--log-level LEVEL]` — execute the plan.
* `reader artifacts CONFIG [--all]` — list latest artifacts (or revision counts).
* `reader steps CONFIG` — list step IDs in order.
* `reader plugins [--category ingest|merge|transform|plot]` — show discovered plugins.
* `reader run-step` — Execute exactly one step by ID or 1-based index, using existing artifacts for inputs.

For example:
```bash
# From anywhere inside the repo:
reader ls

# Run the 7th experiment (20250620_sensor_panel_crosstalk) by index:
reader explain 7
reader validate 7
reader run 7

# Run a single step of that experiment by index:
reader run-step 1 --config 7     # runs the 'ingest' step only

# Show artifacts for the 7th experiment:
reader artifacts 7
```

---

### Experiment configuration

A compact ReaderSpec showing Synergy H1 ingest, plate‑map merge, a small transform chain, and two plots.

```yaml
experiment:
  id: "20250512_panel_M9_glu_araBAD_pspA_marRAB_umuDC_alaS_phoA"
  name: "Retrons panel — M9 + glucose"
  outputs: "./outputs"
  palette: "colorblind"

runtime:
  strict: true
  log_level: "INFO"

steps:
  # 1) Ingest (auto-discovery if 'raw' is absent)
  - id: "ingest"
    uses: "ingest/synergy_h1_snapshot_and_timeseries"
    with:
      channels: ["OD600", "CFP", "YFP"]
      sheet_names: ["Plate 1 - Sheet1"]
      add_sheet: true
      auto_roots: ["./raw_data", "./raw"]
      auto_include: ["*.xlsx", "*.xls"]
      auto_exclude: ["~$*", "._*", "#*#", "*.tmp"]
      auto_pick: "single"      # single | latest | merge

  # 2) Enrich with sample metadata
  - id: "merge_map"
    uses: "merge/sample_map"
    reads:
      df: "ingest/df"
      sample_map: "file:./sample_map.xlsx"

  # 3) Transforms — flexible sequence based on the experiment
  - id: "blank"
    uses: "transform/blank_correction"
    reads: { df: "merge_map/df" }
    with:  { method: "disregard", capture_blanks: true }

  - id: "overflow"
    uses: "transform/overflow_handling"
    reads: { df: "blank/df" }
    with:  { action: "max", clip_quantile: 0.999 }

  - id: "ratio_yfp_cfp"
    uses: "transform/ratio"
    reads: { df: "overflow/df" }
    with:  { name: "YFP/CFP", numerator: "YFP", denominator: "CFP" }

  - id: "ratio_yfp_od600"
    uses: "transform/ratio"
    reads: { df: "ratio_yfp_cfp/df" }
    with:  { name: "YFP/OD600", numerator: "YFP", denominator: "OD600" }

  # 4) Plots — consume cleaned/transformed tables
  - id: "plot_time_series"
    uses: "plot/time_series"
    reads: { df: "ratio_yfp_od600/df" }
    with:
      x: "time"
      y: ["OD600", "YFP", "YFP/CFP", "YFP/OD600"]
      hue: "treatment"
      subplots: "group"
      groups:
        - { "AraC-targeting retron": ["araBADp", "araBADp - msd[AraC]"] }
        - { "PspF-targeting retron": ["pspAp", "pspAp - msd[PspF]"] }
      add_sheet_line: true
      fig: { dpi: 300 }

  - id: "plot_snapshot_multi_genotype"
    uses: "plot/snapshot_multi_genotype"
    reads: { df: "ratio_yfp_od600/df" }
    with:
      x: "genotype"
      y: ["OD600", "YFP/OD600"]
      hue: "treatment"
      time: 14
      groups:
        - { "AraC-targeting retron": ["araBADp", "araBADp - msd[AraC]"] }
        - { "PspF-targeting retron": ["pspAp", "pspAp - msd[PspF]"] }
      fig: { figsize: [10, 6], dpi: 300 }
````

**How to read this**

* Steps run in order. `uses:` selects a plugin.
* `reads:` binds a plugin input to either a prior output (`<step>/<output>`, e.g., `ingest/df`) or a file via `file:<path>`.
* `with:` holds plugin configuration.
* Transforms operate on the tidy table to create derived **channels** (e.g., `YFP/CFP`, `YFP/OD600`) or to clean values (e.g., blank handling, overflow clipping).
* Plot steps take a tidy table and write figures into `outputs/plots/`.

---

### Extending

#### New ingest (new instrument/file format)

Keep parsing logic in **io/**; the plugin just wires it up.

```python
# src/reader/io/my_format.py
import pandas as pd
from pathlib import Path

def parse_my_format(path: str | Path) -> pd.DataFrame:
    # return tidy: columns = position, time, channel, value
    ...
    return df
```

```python
# src/reader/plugins/ingest/my_format.py
from typing import Mapping, Dict, Any
from reader.core.registry import Plugin, PluginConfig
from reader.io.my_format import parse_my_format

class MyCfg(PluginConfig): pass

class MyIngest(Plugin):
    key = "my_format"
    category = "ingest"
    ConfigModel = MyCfg
    @classmethod
    def input_contracts(cls) -> Mapping[str,str]:  return {"raw": "none"}
    @classmethod
    def output_contracts(cls) -> Mapping[str,str]: return {"df": "tidy.v1"}
    def run(self, ctx, inputs: Dict[str, Any], cfg: MyCfg):
        return {"df": parse_my_format(inputs["raw"])}
```

Use it:

```yaml
- id: "ingest_custom"
  uses: "ingest/my_format"
  reads: { raw: "file:./raw_data/run001.ext" }
```

#### New transform (operate on columns to create derived columns)

```python
# my_pkg/my_transform.py
from typing import Mapping, Dict, Any
import pandas as pd
from reader.core.registry import Plugin, PluginConfig

class Cfg(PluginConfig):
    factor: float = 2.0

class ScaleValues(Plugin):
    key = "scale"
    category = "transform"
    ConfigModel = Cfg
    @classmethod
    def input_contracts(cls) -> Mapping[str,str]:  return {"df": "tidy.v1"}
    @classmethod
    def output_contracts(cls) -> Mapping[str,str]: return {"df": "tidy.v1"}
    def run(self, ctx, inputs: Dict[str, Any], cfg: Cfg):
        df = inputs["df"].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce") * cfg.factor
        return {"df": df}
```

Register via entry point:

```toml
[project.entry-points."reader.transform"]
scale = "my_pkg.my_transform:ScaleValues"
```

---

**Sequence design panel (baserender) — setup & usage**

This plot depends on [`dnadesign.baserender`](./dnadesign/baserender), installed in **editable mode** into the same env as `reader`.

### 1) Install

```bash
git clone https://github.com/e-south/dnadesign.git
cd dnadesign
uv pip install -e .[dev]
```
---


@e-south
