## reader pipelines (config.yaml)

A `config.yaml` is the “repeatable” part of the workbench: it encodes pipeline steps you want to run the same way each time (parsing, metadata merges, common transforms) and optional report steps (plots/exports). Notebooks can then do added exploratory work alongside this.

Config validation is strict: unknown keys are errors.

### The basics

An experiment is a directory. Paths in `reads: file:...` are resolved relative to the config file.

```bash
experiments/<year>/<yyyymmdd_shortslug>/
  config.yaml
  inputs/
  metadata.csv
  notebooks/
  outputs/
````

If a single experiment uses multiple instruments, you may subdivide:

```
inputs/
  plate_reader/
  cytometer/
```

Run the pipeline:

```bash
uv run reader init  experiments/<exp> --preset plate_reader/basic
uv run reader explain  experiments/<exp>/config.yaml
uv run reader validate experiments/<exp>/config.yaml
uv run reader config   experiments/<exp>/config.yaml
uv run reader run      experiments/<exp>/config.yaml
```

Run reports (plots/exports) without re‑running the pipeline:

```bash
uv run reader report experiments/<exp>/config.yaml
```

Inspect the config schema:

```bash
uv run reader config --schema
```

Generate a minimal config from a preset:

```bash
uv run reader presets plate_reader/basic --write > config.yaml
```

You can also include report presets at init/template time:

```bash
uv run reader init experiments/<exp> --preset plate_reader/basic --report-preset plots/plate_reader_yfp_full
```

Top‑level keys are intentionally small: `experiment`, `collections` (optional), `steps`, and optional `reports`.
`presets`/`overrides` (and `report_presets`/`report_overrides`) are optional sugar.

### Presets (optional)

Presets let you reuse common step bundles without bloating each config.
List them:

```bash
uv run reader presets
uv run reader presets plate_reader/synergy_h1
```

Inline a preset in `steps` and override only what differs:

```yaml
overrides:
  ingest:
    with:
      mode: auto
      channels: ["OD600", "CFP", "YFP"]
      sheet_names: ["Plate 1 - Sheet1"]

steps:
  - preset: plate_reader/synergy_h1
  - preset: plate_reader/sample_map
  - id: aliases
    uses: transform/alias
    reads: { df: merge_map/df }
    with:
      aliases: { design_id: {}, treatment: {} }
      in_place: false
      case_insensitive: true
  - preset: plate_reader/blank_overflow
  - preset: plate_reader/ratios_yfp_cfp_od600
  - preset: plots/plate_reader_yfp_full
```

### Reports (optional)

Reports are **deliverables** built from pipeline artifacts. They run after `steps` and are
restricted to `plot/*` and `export/*` plugins. Reports **must** read artifacts
(no `file:` reads).

You can also use `report_presets:` and `report_overrides:` the same way you use
`presets:` and `overrides:` for pipeline steps.

```yaml
reports:
  - preset: plots/plate_reader_yfp_full
  - id: sfxi_vec8_csv
    uses: export/csv
    reads: { df: "sfxi_vec8/df" }
    with: { path: "exports/sfxi_vec8.csv" }
```

### Outputs: artifacts + revisions

**reader** writes into `experiment.outputs`.

```bash
outputs/
  manifest.json
  report_manifest.json
  reader.log
  artifacts/
    <step_id>.<plugin_key>/        # first revision
      <output>.parquet
      meta.json
    <step_id>.<plugin_key>__r2/    # later revision if config changed
      <output>.parquet
      meta.json
  plots/                           # optional; controlled by experiment.plots_dir
```

Use:

```bash
uv run reader artifacts experiments/<exp>/config.yaml
```

`reader` also appends a short command log to `JOURNAL.md` in the experiment directory
when you run `explain`, `validate`, `run`, or `run-step`.

`report_manifest.json` tracks files emitted by report steps (plots/exports).

### Example config

Below is an example configuration showing a Synergy H1 ingest, sample-map merge, a small transform chain, and two plots.

How to intepret the configuration:

* Steps run in order.
* `uses:` chooses a plugin (`<category>/<key>`).
* `reads:` binds plugin inputs to either:

  * a prior output (`<step_id>/<output>`, e.g. `ingest/df`)
  * a file path: `file:./something.xlsx` (only when the plugin input contract is `none`)
* `with:` is plugin-specific configuration.

```yaml
experiment:
  id: "20250512_panel_M9_glu_araBAD_pspA_marRAB_umuDC_alaS_phoA"
  name: "Retrons panel — M9 + glucose"
  outputs: "./outputs"
  palette: "colorblind"   # optional: plot palette name
  plots_dir: "plots"      # optional: set to null to write plots into outputs/

steps:
  - id: "ingest"
    uses: "ingest/synergy_h1"
    with:
      mode: "mixed"
      channels: ["OD600", "CFP", "YFP"]
      sheet_names: ["Plate 1 - Sheet1"]
      add_sheet: true
      auto_roots: ["./inputs"]
      auto_include: ["*.xlsx", "*.xls"]
      auto_exclude: ["~$*", "._*", "#*#", "*.tmp"]
      auto_pick: "single"      # single | latest | merge

  - id: "merge_map"
    uses: "merge/sample_map"
    reads:
      df: "ingest/df"
      sample_map: "file:./metadata.csv"

  - id: "blank"
    uses: "transform/blank_correction"
    reads: { df: "merge_map/df" }
    with:  { method: "disregard" }

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

reports:
  - id: "plot_time_series"
    uses: "plot/time_series"
    reads: { df: "ratio_yfp_od600/df" }
    with:
      x: "time"
      y: ["OD600", "YFP", "YFP/CFP", "YFP/OD600"]
      hue: "treatment"
      subplots: "group"
      add_sheet_line: true
      fig: { dpi: 300 }

  - id: "plot_snapshot_barplot"
    uses: "plot/snapshot_barplot"
    reads: { df: "ratio_yfp_od600/df" }
    with:
      x: "design_id"
      y: ["OD600", "YFP/OD600"]
      hue: "treatment"
      time: 14
      fig: { figsize: [10, 6], dpi: 300 }
```

**Note:** Steps like `transform/sfxi` and `plot/logic_symmetry` (pipeline or reports) require `design_id` and `treatment`.  
`batch` is optional; if it isn’t provided, a constant batch of `0` is assumed for grouping.
Use `validator/to_tidy_plus_map` after your metadata merge to emit `tidy+map.v2` before those steps.

### Collections + pool_sets (grouping helper)

Some plot plugins accept `pool_sets` to group categories. Define them once under `collections`
and reference them by name:

```yaml
collections:
  design_id:
    panel_a:
      - { "Group 1": ["D1", "D2"] }
      - { "Group 2": ["D3"] }

reports:
  - id: "plot_ts"
    uses: "plot/time_series"
    reads: { df: "ratio_yfp_od600/df" }
    with:
      group_on: "design_id"
      pool_sets: "design_id:panel_a"
```

### Running slices

Useful during iteration:

```bash
uv run reader run <CONFIG> --dry-run
uv run reader run <CONFIG> --step 1
uv run reader run <CONFIG> --resume-from ingest --until merge_map
uv run reader run-step merge_map --config <CONFIG>
uv run reader config <CONFIG>
```

### Exit codes

- `0` success
- `2` config invalid (schema, plugin config, or registry issues)
- `3` contract mismatch
- `4` input/merge/transform validation failures
- `5` runtime/internal errors

---

@e-south

---

See also:

- `docs/plugins.md`
- `docs/sfxi_vec8.md`
- `docs/logic_symmetry.md`
- `README.md`
