
# Extending reader with plugins

Plugins exist so repeated parsing/transforms/plots can be reused across experiments.
This is a maintainer-facing surface. Ordinary experiment authors should stay in
`config.yaml`, `reader init`, `reader protocols`, `reader inspect`,
`reader plot --list`, and `reader export --list`; they should not need plugin
ids for normal workbench use. When you do need registry-level inspection, start
with `reader plugins --protocol <protocol-id>` so you see the plugin kernel in
assay context instead of as a flat global dump.

### Contents

1. [Plugin categories](#plugin-categories)
2. [Example of adding new plugins](#example-of-adding-new-plugins)
3. [Flow cytometry ingest plugin](#flow-cytometry-ingest-plugin)
4. [Adding a transform plugin](#adding-a-transform-plugin)
5. [Adding a plot/export plugin](#adding-a-plotexport-plugin)

---

### Plugin categories

A good plugin is thin orchestration:

- keep instrument/file parsing in `domains/<domain>/io/` when it is domain-owned,
  or in `plugins/ingest/discovery_policy.py` when it is genuinely shared raw-file
  autodiscovery policy for ingest adapters
- keep reusable computation in `domains/<domain>/...` instead of `plugins/`
- keep derived domain tables with the producing domain, not in shared contract buckets
- keep plugins focused on wiring inputs → computation → declared outputs
- if multiple plugins in one category share orchestration-only behavior, keep
  that in `plugins/<category>/_*.py`; do not duplicate file discovery,
  partition resolution, or figure-save plumbing across plugin modules
- for plotting, keep figure selection/layout in the domain package
  (for example `domains/plate_reader/analysis/`, `domains/plate_reader/ordering.py`,
  and `domains/plate_reader/plots/`), keep axes rendering in
  `domains/<domain>/plots/panels/`, and keep plugin modules as config adapters only
- if a plot needs semantic input preparation, keep that in the plotting library
  next to the figure package rather than in a plugin-private helper

Current examples of this convention:

- `plugins/ingest/_discovery.py` owns shared ingest auto-discovery and pick
  logic
- `plugins/ingest/discovery_policy.py` owns raw-file discovery defaults and search helpers
- `plugins/plot/_shared.py` owns shared figure-plot adapter behavior
- `plotting/style.py` owns shared palette/style helpers used by workbench and domain plot code
- `plugins/transform/_labeling.py` owns reusable dataframe label-application mechanics for generic labeling transforms
- `domains/semantics.py` owns the shared domain-semantic access surface used by workbench and plugins
- `plugins/transform/_*.py` owns transform-local adapter support only when the code is not shared outside one plugin

Each plugin now has a small workbench ontology entry declared in the explicit
built-in plugin manifest:

- `category` = execution stage (`ingest`, `transform`, `validator`, `plot`, `export`)
- `domain` = canonical problem domain (`plate_reader`, `cytometry`, `logic`, `generic`)
- `family` = semantic plugin type within that domain (`time_series`, `metadata_merge`, `derived_channel`, ...)

That ontology is first-class in the registry and CLI. `reader plugins` is no longer
just a flat key dump; it is the package’s semantic catalog.

Built-in plugins live under:

```bash
src/reader/plugins/<category>/
```

You’ll typically see plugins grouped as:

* `ingest/*` — read raw instrument/files into a tidy table
* `transform/*` — operate on tidy tables (derive new channels, attach metadata, filter, normalize, etc.)
* `validator/*` — enforce or upgrade schema/shape
* `plot/*` — render plots (plot specs)
* `export/*` — write exports (export specs)

Built-in plugin registration is explicit in:

```bash
src/reader/workbench/assets/plugin_manifest.py
```

`src/reader/plugins/` now contains implementations only; the runtime does not
discover built-ins by scanning that package tree.

External plugins use the `reader.plugins` entry-point group and must expose an
explicit plugin descriptor, not just a `Plugin` subclass.

Plugin I/O is now declared through the explicit port kernel in
`reader.workbench.ports`, not through string conventions. That means:

- input optionality is `optional=True`, not a `?` suffix
- dataframe ports declare `contract="tidy.v1"` or `contract=None`
- file inputs use `file_path` ports
- plot/export outputs use explicit `file_path` or `file_bundle` ports
- the removed legacy conventions `"none"` and `"files"` are not valid plugin API
  surface anymore

---

### Example of adding new plugins

**Generic ingestion**

1. Keep parsing logic in a domain package:

  ```python
  # src/reader/domains/plate_reader/io/my_format.py
  import pandas as pd
  from pathlib import Path

  def parse_my_format(path: str | Path) -> pd.DataFrame:
      # return tidy long table
      # required columns depend on your chosen contract(s)
      ...
      return df
  ```

2. Wire it up as a plugin implementation:

  ```python
  # src/reader/plugins/ingest/my_format.py
  from typing import Any
  from reader.workbench.ports import dataframe_output, file_path_input
  from reader.workbench.registry import Plugin, PluginConfig
  from reader.domains.plate_reader.io.my_format import parse_my_format

  class MyCfg(PluginConfig):
      pass

  class MyIngest(Plugin):
      ConfigModel = MyCfg

      @classmethod
      def input_ports(cls):
          return {"raw": file_path_input("raw")}

      @classmethod
      def output_ports(cls):
          return {"df": dataframe_output("df", "tidy.v1")}

      def run(self, ctx, inputs: dict[str, Any], cfg: MyCfg):
          return {"df": parse_my_format(inputs["raw"])}
  ```

3. Register it explicitly in the built-in manifest:

  ```python
  # src/reader/workbench/assets/plugin_manifest.py
  from reader.plugins.ingest.my_format import MyIngest
  from reader.workbench import PluginSemantics
  from reader.workbench.assets import build_plugin_asset

  build_plugin_asset(
      plugin_id="ingest/my_format",
      semantics=PluginSemantics(
          domain="plate_reader",
          family="workbook_ingest",
          summary="Parse my custom workbook format into tidy traces.",
      ),
      plugin_cls=MyIngest,
  )
  ```

4. Use it in an experiment:

  ```yaml
  - id: "ingest_custom"
    plugin: "ingest/my_format"
    reads:
      raw:
        file: "./inputs/run001.ext"
  ```

### Flow cytometry ingest plugin

For flow cytometry `.fcs` files, use `ingest/flow_cytometer`. It emits a tidy table with:

* `sample_id` (from filename) and `position = sample_id`
* `time` set to a constant (default `0.0`, since cytometry is snapshot data)
* long-form `channel` / `value` pairs per event

The raw FCS parsing currently lives in `reader.domains.cytometry.io.fcs`; the plugin is just the workbench adapter
for config, auto-discovery, output contracts, and logging.

Example:

```yaml
- id: ingest_cytometer
  plugin: ingest/flow_cytometer
  with:
    auto_roots: ["./inputs"]
    channel_name_field: "pns"
    auto_pick: "merge"
```

To attach metadata keyed by `sample_id`:

```yaml
- id: attach_metadata
  plugin: transform/sample_metadata
  reads:
    df:
      record: "ingest_cytometer/df"
    metadata:
      file: "./metadata.csv"
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
If that promotion matters to users, expose it through the dataframe output
port surface
so `reader explain` reports the planned semantic range instead of only the floor.

```python
import pandas as pd
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig

class Cfg(PluginConfig):
    factor: float = 2.0

class ScaleValues(Plugin):
    ConfigModel = Cfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return cls.passthrough_output_ports(
            outputs={"df": dataframe_output("df", "tidy.v1")},
            passthrough={"df": "df"},
            promoted_examples={"df": ("plate_reader.annotated.v1",)},
        )

    def resolve_output_ports(self, *, inputs, outputs, cfg, where):
        del cfg
        return self.inherit_dataframe_output_ports(
            inputs=inputs,
            outputs=outputs,
            passthrough={"df": "df"},
            where=where,
        )

    def run(self, ctx, inputs: dict[str, Any], cfg: Cfg):
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
protocol:
  id: plate_reader/dual_reporter_screen
  with:
    workflow:
      include_crosstalk_pairs: true
      include_crosstalk_export: true
      plot_set: yfp_time_series
    fold_change:
      report_times: [12.0]
      use_global_baseline: true
      global_baseline_value: negative
    plugins:
      transform/crosstalk_pairs:
        value_column: log2FC
        value_scale: log2
        target: YFP/CFP
        time_mode: all
        time_policy: per_time
        mapping_mode: column
        design_treatment_column: cognate_treatment
        min_self: 1.0
        max_cross: 0.5
        max_other: 0.5
        min_self_minus_best_other: 1.0
        min_selectivity_delta: 1.0
        require_self_is_top1: true
```

To export pairings:

```yaml
protocol:
  id: plate_reader/dual_reporter_screen
  analysis:
    crosstalk_pairs:
      enabled: true
      export: true
  outputs:
    exports:
      include: [crosstalk_pairs_table]
      artifacts:
        crosstalk_pairs_table:
          path: crosstalk_pairs.csv
```

---

### Adding a plot/export plugin

Plot and export plugins now enter configs through protocol compilation plus
`protocol.outputs` selection/settings.

They are run by:

* `reader plot` (save plot files only)
* `reader export` (exports only)

Guidelines:

* Plot/export plugins should be deterministic and pure: read declared inputs, produce deterministic outputs.
* Avoid experiment-specific logic inside plot plugins; keep bespoke logic in `domains/<domain>/`.
* Declare typed input/output ports; write under `outputs/plots` or `outputs/exports`.
* Plot specs are assertive: missing required columns raise an error.
* If a selection is empty, emit a warning and skip (don’t silently write an empty plot).
* Plot/export outputs are tracked as `file_bundle` records in `outputs/manifests/records.json`.

Plot plugins implement a **single render path** that powers file output:

* `render(ctx, inputs, cfg) -> PlotFigure | list[PlotFigure]`
* `run(...)` should call `render(...)` and then save via `save_plot_figures(...)`.

Minimal plot plugin pattern:

```python
from reader.plotting.sinks import PlotFigure, normalize_plot_figures, save_plot_figures
from reader.workbench.ports import dataframe_input, file_bundle_output

class MyPlot(Plugin):
    ConfigModel = MyCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return {"artifacts": file_bundle_output("artifacts")}

    def render(self, ctx, inputs, cfg: MyCfg) -> list[PlotFigure]:
        fig = build_plot(inputs["df"])
        return [PlotFigure(fig=fig, filename=cfg.filename or "my_plot")]

    def run(self, ctx, inputs, cfg: MyCfg):
        figures = normalize_plot_figures(self.render(ctx, inputs, cfg), where=f"plot/{self.plugin_key}")
        saved = save_plot_figures(figures, ctx.plots_dir)
        return {"artifacts": [str(p) for p in saved]}
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
protocol:
  id: logic/sfxi_screen
  outputs:
    exports:
      include: [logic_summary_workbook]
      artifacts:
        logic_summary_workbook:
          path: sfxi_vec8.xlsx
          sheet_name: vec8
```

---

@e-south
