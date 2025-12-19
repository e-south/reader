## reader pipelines (config.yaml)

A `config.yaml` is the “repeatable” part of the workbench: it encodes steps you want to run the same way each time (parsing, metadata merges, common transforms, optional plots). Notebooks can then do added exploratory work alongside this.

### The basics

An experiment is a directory. Paths in `reads: file:...` are resolved relative to the config file.

```bash
experiments/<exp>/
  config.yaml
  raw_data/
  notebooks/
  outputs/
````

Run the pipeline:

```bash
uv run reader explain  experiments/<exp>/config.yaml
uv run reader validate experiments/<exp>/config.yaml
uv run reader run      experiments/<exp>/config.yaml
```

### Outputs: artifacts + revisions

**reader** writes into `experiment.outputs`.

```bash
outputs/
  manifest.json
  reader.log
  artifacts/
    <step_id>.<plugin_key>/        # first revision
      <output>.parquet
      meta.json
    <step_id>.<plugin_key>__r2/    # later revision if config changed
      <output>.parquet
      meta.json
  plots/                           # optional; only if plot steps write figures
```

Use:

```bash
uv run reader artifacts experiments/<exp>/config.yaml
```

### Example config

Below is an example configuration showing a Synergy H1 ingest, sample-map merge, a small transform chain, and two plots.

How to intepret the configuration:

* Steps run in order.
* `uses:` chooses a plugin (`<category>/<key>`).
* `reads:` binds plugin inputs to either:

  * a prior output (`<step_id>/<output>`, e.g. `ingest/df`)
  * a file path: `file:./something.xlsx`
* `with:` is plugin-specific configuration.

```yaml
experiment:
  id: "20250512_panel_M9_glu_araBAD_pspA_marRAB_umuDC_alaS_phoA"
  name: "Retrons panel — M9 + glucose"
  outputs: "./outputs"
  palette: "colorblind"

runtime:
  strict: true

steps:
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

  - id: "merge_map"
    uses: "merge/sample_map"
    reads:
      df: "ingest/df"
      sample_map: "file:./sample_map.xlsx"

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

  - id: "plot_snapshot_multi_genotype"
    uses: "plot/snapshot_multi_genotype"
    reads: { df: "ratio_yfp_od600/df" }
    with:
      x: "genotype"
      y: ["OD600", "YFP/OD600"]
      hue: "treatment"
      time: 14
      fig: { figsize: [10, 6], dpi: 300 }
```

### Running slices

Useful during iteration:

```bash
uv run reader run <CONFIG> --dry-run
uv run reader run <CONFIG> --step 1
uv run reader run <CONFIG> --resume-from ingest --until merge_map
uv run reader run-step merge_map --config <CONFIG>
```

---

@e-south
