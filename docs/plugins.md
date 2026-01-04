
# Extending reader with plugins

Plugins exist so repeated parsing/transforms/plots can be reused across experiments.

### Contents

1. [Plugin categories](#plugin-categories)
2. [Example of adding new plugins](#example-of-adding-new-plugins)
3. [Flow cytometry ingest plugin](#flow-cytometry-ingest-plugin)
4. [Adding a transform plugin](#adding-a-transform-plugin)
5. [Adding a plot/export plugin](#adding-a-plotexport-plugin)

---

### Plugin categories

A good plugin is thin orchestration:

- keep instrument/file parsing in `io/` (raw → tidy tables)
- keep reusable computation in `lib/` (domain logic)
- keep plugins focused on wiring inputs → computation → declared outputs

Built-in plugins live under:

```bash
src/reader/plugins/<category>/
```

You’ll typically see plugins grouped as:

* `ingest/*` — read raw instrument/files into a tidy table
* `merge/*` — attach metadata or mapping tables
* `transform/*` — operate on tidy tables (derive new channels, filter, normalize, etc.)
* `validator/*` — enforce or upgrade schema/shape
* `plot/*` — render plots (plot specs)
* `export/*` — write exports (export specs)

---

### Example of adding new plugins

**Generic ingestion**

1. Keep parsing logic in `io/`:

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

2. Wire it up as a plugin:

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

3. Use it in an experiment:

  ```yaml
  - id: "ingest_custom"
    uses: "ingest/my_format"
    reads: { raw: "file:./inputs/run001.ext" }
  ```

### Flow cytometry ingest plugin

For flow cytometry `.fcs` files, use `ingest/flow_cytometer`. It emits a tidy table with:

* `sample_id` (from filename) and `position = sample_id`
* `time` set to a constant (default `0.0`, since cytometry is snapshot data)
* long-form `channel` / `value` pairs per event

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

**Note:** install cytometry extras with `uv sync --locked --group cytometry`.

---

### Adding a transform plugin

Transforms typically accept a `tidy.v1` table and emit a `tidy.v1` table.

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

---

### Adding a plot/export plugin

Plot specs live under `plots:` and export specs under `exports:` in config (optionally bundled via
`plots.presets` / `plots.overrides` and `exports.presets` / `exports.overrides`).

They are run by:

* `reader plot --mode save` (plots only)
* `reader export` (exports only)
* `reader plot --mode notebook` (scaffolds a plot-focused notebook)

Guidelines:

* Plot/export plugins should be deterministic and pure: read declared inputs, write files.
* Avoid experiment-specific logic inside plot plugins; keep bespoke logic in `lib/`.
* Declare input/output contracts; write under `outputs/plots` or `outputs/exports`.
* Plot specs are assertive: missing required columns raise an error.
* If a selection is empty, emit a warning and skip (don’t silently write an empty plot).
* Plot/export outputs are tracked in `outputs/plots_manifest.json` and `outputs/exports_manifest.json`.

Common plot config knobs (shared across most plot plugins):

* `filename`: override the output filename stub
* `fig.ext`: file extension (default `pdf`)
* `fig.dpi`: raster resolution for PNGs (ignored for vector PDFs)

Inspect plugins:

```bash
uv run reader plugins
uv run reader plugins --category plot
uv run reader plugins --category export
```

Example export spec:

```yaml
exports:
  specs:
    - id: export_vec8
      uses: "export/csv"
      reads: { df: "sfxi_vec8/df" }
      with: { path: "sfxi_vec8.csv" }
```

---

@e-south
