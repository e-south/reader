## Extending reader with plugins

Plugins exist so that repeated parsing/transforms/plots can be reused across experiments.
They should be **thin orchestration**: keep real parsing logic in `parsers/`, reusable computation in `domain/`, and plot helpers in `plotting/`.

Built-in plugins live under:

```bash
src/reader/plugins/<category>/
```

Plugins are discovered in two ways:

- built-ins under `reader.plugins.*`
- optional **entry points**: `reader.ingest`, `reader.merge`, `reader.transform`, `reader.plot`, `reader.validator`

Entry points let external packages add plugins without modifying this repo.

### Flow cytometry ingest (built-in)

```yaml
steps:
  - id: "ingest_cytometer"
    uses: "ingest/flow_cytometer"
    with:
      auto_roots: ["./inputs"]
      channel_name_field: "pns"

  - id: "merge_metadata"
    uses: "merge/sample_metadata"
    reads:
      df: "ingest_cytometer/df"
      metadata: "file:./metadata.csv"
    with:
      require_columns: ["design_id", "treatment"]
      require_non_null: true
```

### New ingest (new instrument/file format)

Keep parsing logic in `parsers/`:

```python
# src/reader/parsers/my_format.py
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
from reader.parsers.my_format import parse_my_format

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

### External plugins (entry points)

If you want a plugin in another package, expose it via entry points:

```toml
[project.entry-points."reader.ingest"]
my_ingest = "my_pkg.reader_plugins:MyIngest"

[project.entry-points."reader.validator"]
my_validator = "my_pkg.reader_plugins:MyValidator"

[project.entry-points."reader.export"]
my_export = "my_pkg.reader_plugins:MyExport"
```

Then `reader plugins` will discover it alongside built-ins.

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

### Inspect what’s available

```bash
uv run reader plugins
uv run reader plugins --category transform
```

### Exports (reports)

Exports are report‑only steps that write deliverables (CSV/Excel/TSV) from artifacts:

```yaml
reports:
  - id: sfxi_vec8_csv
    uses: export/csv
    reads: { df: "sfxi_vec8/df" }
    with: { path: "exports/sfxi_vec8.csv" }
```

---

@e-south

---

See also:

- `docs/pipeline.md`
- `README.md`
