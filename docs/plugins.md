## Extending reader with plugins

Plugins exist so that repeated parsing/transforms/plots can be reused across experiments.
They should be **thin orchestration**: keep real parsing logic in `io/` and reusable computation in `lib/`.

Built-in plugins live under:

```bash
src/reader/plugins/<category>/
```

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
  reads: { raw: "file:./raw_data/run001.ext" }
```

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

---

@e-south
