
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
- if multiple plugins in one category share orchestration-only behavior, keep
  that in `plugins/<category>/_*.py`; do not duplicate file discovery,
  partition resolution, or figure-save plumbing across plugin modules
- for plotting, keep figure selection/layout in `lib/microplates/support/` or a
  figure-specific package, keep axes rendering in `lib/microplates/panels/`,
  and keep plugin modules as config adapters only
- if a plot needs semantic input preparation, keep that in the plotting library
  next to the figure package rather than in a plugin-private helper

Current examples of this convention:

- `plugins/ingest/_discovery.py` owns shared ingest auto-discovery and pick
  logic
- `plugins/plot/_shared.py` owns shared figure-plot adapter behavior
- `plugins/transform/_*.py` owns transform-local support logic when extraction
  into `lib/` would be premature

Each plugin now also declares a small workbench ontology entry:

- `category` = execution stage (`ingest`, `merge`, `transform`, `validator`, `plot`, `export`)
- `domain` = problem domain (`plate_reader`, `cytometry`, `logic`, `generic`, ...)
- `family` = semantic plugin type within that domain (`time_series`, `metadata_merge`, `derived_channel`, ...)

That ontology is first-class in the registry and CLI. `reader plugins` is no longer
just a flat key dump; it is the package’s semantic catalog.

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

External plugins use matching entry-point groups for every advertised category:
`reader.ingest`, `reader.merge`, `reader.transform`, `reader.validator`,
`reader.plot`, and `reader.export`.

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
  from reader.core.workbench import PluginSemantics
  from reader.io.my_format import parse_my_format

  class MyCfg(PluginConfig):
      pass

  class MyIngest(Plugin):
      key = "my_format"
      category = "ingest"
      semantics = PluginSemantics(
          category="ingest",
          domain="plate_reader",
          family="workbook_ingest",
          summary="Parse my custom workbook format into tidy traces.",
      )
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

The raw FCS parsing lives in `reader.io.flow_cytometer`; the plugin is just the workbench adapter
for config, auto-discovery, output contracts, and logging.

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

If the merged table satisfies the annotated plate-reader contract, reader
stores it as `plate_reader.annotated.v1` instead of plain `tidy.v1`.

`reader explain` shows this as a minimum contract with a possible runtime
promotion; execution decides the actual stored contract from the emitted data.

**Note:** install cytometry extras with `uv sync --locked --group cytometry`.

---

### Adding a transform plugin

Transforms typically declare a minimal table contract such as `tidy.v1`.
When a transform preserves richer metadata semantics, it can resolve a stricter
runtime output contract instead of collapsing back to the minimum.
If that promotion matters to users, expose it in `output_contract_surfaces()`
so `reader explain` reports the planned semantic range instead of only the floor.

```python
from typing import Mapping, Dict, Any
import pandas as pd
from reader.core.registry import Plugin, PluginConfig
from reader.core.workbench import PluginSemantics

class Cfg(PluginConfig):
    factor: float = 2.0

class ScaleValues(Plugin):
    key = "scale"
    category = "transform"
    semantics = PluginSemantics(
        category="transform",
        domain="generic",
        family="value_transform",
        summary="Scale tidy values by a constant factor.",
    )
    ConfigModel = Cfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contract_surfaces(cls):
        return cls.passthrough_output_contract_surfaces(
            passthrough={"df": "df"},
            promoted_examples={"df": ("plate_reader.annotated.v1",)},
        )

    def resolve_output_contracts(self, *, inputs, outputs, cfg, where):
        del cfg
        return self.inherit_dataframe_output_contracts(
            inputs=inputs,
            outputs=outputs,
            passthrough={"df": "df"},
            where=where,
        )

    def run(self, ctx, inputs: Dict[str, Any], cfg: Cfg):
        df = inputs["df"].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce") * cfg.factor
        return {"df": df}
```

---

### Crosstalk pairing (transform/crosstalk_pairs)

Compute pairwise crosstalk-safe design pairings using a `fold_change.v1` table. This transform
summarizes per-design selectivity and evaluates pairs where each design responds strongly to its
own treatment while responding weakly to others (including non-self treatments).
If your design-to-treatment mapping lives in metadata, include that column in the fold-change
step via `attach_metadata` so it is available to this transform.

Time selection is explicit and assertive:
- `time_mode: single` requires exactly one time in the fold-change table.
- `time_mode: exact|nearest` requires `time` or `times` to be provided (tolerance applies only to `nearest`).
- `time_mode: latest` uses the latest time present in the fold-change table.
- `time_mode: all` evaluates every time present in the fold-change table.
- `time_policy: all` (optional) keeps only pairs that pass at *every* evaluated time.

Mapping strategies are explicit and documented in config:
- `mapping_mode: explicit` uses `design_treatment_map` (stable, recommended for ground-truth mapping).
- `mapping_mode: column` uses a metadata column (keeps mapping in data; good for reuse).
- `mapping_mode: top1` uses the top response in the data (data-driven, but can change across runs/time).
  Use `top1_tie_policy` and `top1_tie_tolerance` to control how ties are handled.

For library-level API details and column semantics, see `docs/lib/crosstalk_pairs.md`.

```yaml
pipeline:
  steps:
    - id: fold_change__yfp_over_cfp
      uses: transform/fold_change
      reads: { df: ratio_yfp_cfp/df }
      with:
        target: YFP/CFP
        report_times: [12.0]
        treatment_column: treatment
        group_by: [design_id]
        use_global_baseline: true
        global_baseline_value: negative

    - id: crosstalk_pairs
      uses: transform/crosstalk_pairs
      reads: { table: fold_change__yfp_over_cfp/table }
      with:
        value_column: log2FC
        value_scale: log2
        target: YFP/CFP
        time_mode: all
        time_policy: per_time
        mapping_mode: column
        design_treatment_column: cognate_treatment   # mapping_mode: explicit -> use design_treatment_map
        min_self: 1.0
        max_cross: 0.5
        max_other: 0.5             # max response to any non-self treatment
        min_self_minus_best_other: 1.0
        min_selectivity_delta: 1.0
        require_self_is_top1: true
```

To export pairings:

```yaml
exports:
  specs:
    - id: export_crosstalk_pairs
      uses: export/csv
      reads: { df: crosstalk_pairs/table }
      with: { path: "crosstalk_pairs.csv" }
```

---

### Adding a plot/export plugin

Plot specs live under `plots:` and export specs under `exports:` in config (optionally bundled via
`plots.presets` / `plots.overrides` and `exports.presets` / `exports.overrides`).

They are run by:

* `reader plot` (save plot files only)
* `reader export` (exports only)

Guidelines:

* Plot/export plugins should be deterministic and pure: read declared inputs, produce deterministic outputs.
* Avoid experiment-specific logic inside plot plugins; keep bespoke logic in `lib/`.
* Declare input/output contracts; write under `outputs/plots` or `outputs/exports`.
* Plot specs are assertive: missing required columns raise an error.
* If a selection is empty, emit a warning and skip (don’t silently write an empty plot).
* Plot/export outputs are tracked as `file_bundle` records in `outputs/manifests/records.json`.

Plot plugins implement a **single render path** that powers file output:

* `render(ctx, inputs, cfg) -> PlotFigure | list[PlotFigure]`
* `run(...)` should call `render(...)` and then save via `save_plot_figures(...)`.

Minimal plot plugin pattern:

```python
from reader.core.plot_sinks import PlotFigure, normalize_plot_figures, save_plot_figures

class MyPlot(Plugin):
    key = "my_plot"
    category = "plot"
    ConfigModel = MyCfg

    def render(self, ctx, inputs, cfg: MyCfg) -> list[PlotFigure]:
        fig = build_plot(inputs["df"])
        return [PlotFigure(fig=fig, filename=cfg.filename or "my_plot")]

    def run(self, ctx, inputs, cfg: MyCfg):
        figures = normalize_plot_figures(self.render(ctx, inputs, cfg), where=f"plot/{self.key}")
        saved = save_plot_figures(figures, ctx.plots_dir)
        return {"files": [str(p) for p in saved] if saved else None}
```

Common plot config knobs (shared across most plot plugins):

* `filename`: override the output filename stub
* `fig.ext`: file extension (default `pdf`)
* `fig.dpi`: raster resolution for PNGs (ignored for vector PDFs)

Inspect plugins:

```bash
uv run reader plugins
uv run reader plugins --category plot
uv run reader plugins --domain plate_reader
uv run reader plugins --family time_series
```

Export plugins are intentionally permissive about input contracts; the built‑in
`export/csv` and `export/xlsx` accept any dataframe record and write it to disk.

Example export spec:

```yaml
exports:
  specs:
    - id: export_vec8
      uses: "export/csv"
      reads: { df: "sfxi_vec8/df" }
      with: { path: "sfxi_vec8.csv" }
    - id: export_vec8_xlsx
      uses: "export/xlsx"
      reads: { df: "sfxi_vec8/df" }
      with: { path: "sfxi_vec8.xlsx", sheet_name: "vec8" }
```

---

@e-south
