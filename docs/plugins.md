## Extending reader with plugins

Plugins exist so that repeated parsing/transforms/plots can be reused across experiments.
They should be **thin orchestration**: keep real parsing logic in `io/` and reusable computation in `lib/`.

Built-in plugins live under:

```bash
src/reader/plugins/<category>/
```

See also:
- [Pipeline config and deliverables](./pipeline.md)
- [CLI reference](./cli.md)
- [Spec / architecture](./spec.md)
- [Marimo notebook reference](./marimo_reference.md)

### New ingest (new instrument/file format)

Keep parsing logic in `io/`:

```python
# src/reader/io/my_format.py
import pandas as pd
from pathlib import Path

def parse_my_format(path: str | Path) -> pd.DataFrame:
    # return tidy long table
    # required columns depend on your chosen contract(s)
    ...
    return df
```

Wire it up as a plugin:

```python
# src/reader/plugins/ingest/my_format.py
from typing import Mapping, Dict, Any
from reader.core.registry import Plugin, PluginConfig
from reader.io.my_format import parse_my_format

class MyCfg(PluginConfig):
    pass

class MyIngest(Plugin):
    key = "my_format"
    category = "ingest"
    ConfigModel = MyCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"raw": "none"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    def run(self, ctx, inputs: Dict[str, Any], cfg: MyCfg):
        return {"df": parse_my_format(inputs["raw"])}
```

Use it in an experiment:

```yaml
- id: "ingest_custom"
  uses: "ingest/my_format"
  reads: { raw: "file:./inputs/run001.ext" }
```

### Built-in cytometry ingest (FCS)

For flow cytometry `.fcs` files, use `ingest/flow_cytometer`. It emits a tidy table with:
- `sample_id` (from filename) and `position` = `sample_id`
- `time` set to a constant (default `0.0`, since cytometry is snapshot data)
- long-form `channel` / `value` pairs per event

Example:

```yaml
- id: ingest_cytometer
  uses: ingest/flow_cytometer
  with:
    auto_roots: ["./inputs"]
    channel_name_field: "pns"
    auto_pick: "merge"
```

To attach metadata keyed by `sample_id`:

```yaml
- id: merge_metadata
  uses: merge/sample_metadata
  reads:
    df: "ingest_cytometer/df"
    metadata: "file:./metadata.csv"
  with:
    require_columns: ["design_id", "treatment"]
```

Note: install cytometry extras with `uv sync --locked --group cytometry`.

### New transform (operate on a tidy table)

```python
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
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    def run(self, ctx, inputs: Dict[str, Any], cfg: Cfg):
        df = inputs["df"].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce") * cfg.factor
        return {"df": df}
```

### New deliverable (plot/export)

Deliverable steps live under `deliverables:` in config (optionally bundled via
`deliverable_presets` + `deliverable_overrides`) and are run by `reader deliverables`
(or automatically after `reader run` unless `--no-deliverables` is set).

Guidelines:
- Deliverable plugins should be deterministic and pure: read declared inputs, write files.
- Avoid experiment‑specific logic inside plot plugins; keep bespoke logic in `lib/`.
- Declare input/output contracts; write under `outputs/plots` or `outputs/exports`.
- Plot steps are assertive: missing required columns raise an error; empty selections emit a warning and skip.
- Deliverable outputs are tracked in `outputs/deliverables_manifest.json`.

Common plot config knobs (shared across most plot plugins):
- `filename`: override the output filename stub.
- `fig.ext`: file extension (default `pdf`).
- `fig.dpi`: raster resolution for PNGs (ignored for vector PDFs).

Inspect plugins:

```bash
uv run reader plugins
uv run reader plugins --category plot
uv run reader plugins --category export
```

Example export step:

```yaml
- id: export_vec8
  uses: "export/csv"
  reads: { df: "sfxi_vec8/df" }
  with: { path: "exports/sfxi_vec8.csv" }
```

---

@e-south
