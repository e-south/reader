## reader pipelines (config.yaml)

A `config.yaml` is the “repeatable” part of the workbench: it encodes steps you want to run the same way each time (parsing, metadata merges, common transforms), plus optional **reports** (plots/exports). Reports are deterministic deliverables; notebooks are for interactive exploration.

### The basics

An experiment is a directory. Paths in `reads: file:...` are resolved relative to the config file.

```bash
experiments/<exp>/
  config.yaml
  inputs/
  notebooks/
  outputs/
```

Run the pipeline:

```bash
uv run reader demo
uv run reader explain  experiments/<exp>/config.yaml
uv run reader validate experiments/<exp>/config.yaml
uv run reader check-inputs experiments/<exp>/config.yaml   # verify reads: file: inputs exist
uv run reader run      experiments/<exp>/config.yaml        # runs pipeline + reports
uv run reader run      experiments/<exp>/config.yaml --no-reports
uv run reader report   experiments/<exp>/config.yaml        # reports only (uses existing artifacts)
uv run reader explore  experiments/<exp>/config.yaml        # scaffold marimo notebook for EDA
```

Use `reader report` to render plots/exports from the **reports** section (add `--list` to show report steps),
and `reader explore` to scaffold a notebook (no execution) for interactive analysis. `reader validate` only checks
config + plugin params; it does not touch input files or data.

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
  plots/                           # optional; only if plot steps write figures
  exports/                         # optional; only if export steps write files
```

Use:

```bash
uv run reader artifacts experiments/<exp>/config.yaml
```

### Config keys (single-page glance)

| Key | Type | Required | Purpose |
| --- | --- | --- | --- |
| `experiment.id` | string | no | Stable identifier for the experiment. |
| `experiment.name` | string | no | Human-friendly name (defaults to directory name). |
| `experiment.outputs` | string | yes | Output directory (relative to config). |
| `experiment.palette` | string/null | no | Plot palette name (set null to disable). |
| `experiment.plots_dir` | string/null | no | Subdir for plots under outputs (null/"" = outputs root). |
| `runtime.strict` | bool | no | Fail on contract/validation errors (default true). |
| `steps` | list | yes | Pipeline steps (ingest/merge/transform/validate). |
| `reports` | list | no | Report steps (plots/exports). |
| `report_presets` | list | no | Preset bundles for reports. |
| `report_overrides` | map | no | Per-report step overrides by id. |
| `overrides` | map | no | Per-step overrides by id (pipeline steps). |
| `collections` | map | no | Named groupings (used by some plots). |

### Example config

Below is an example configuration showing a Synergy H1 ingest, sample-map merge, a small transform chain, and a report preset.

How to interpret the configuration:

* Steps run in order.
* `uses:` chooses a plugin (`<category>/<key>`).
* `reads:` binds plugin inputs to either:

  * a prior output (`<step_id>/<output>`, e.g. `ingest/df`)
  * a file path: `file:./something.xlsx`
* `with:` is plugin-specific configuration.
* `preset:` expands to one or more steps (use for shared bundles).
* `overrides:` / `report_overrides:` apply per-step config overrides by id.

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
    uses: "ingest/synergy_h1"
    with:
      channels: ["OD600", "CFP", "YFP"]
      sheet_names: ["Plate 1 - Sheet1"]
      add_sheet: true
      auto_roots: ["./inputs", "./raw_data", "./raw"]
      auto_include: ["*.xlsx", "*.xls"]
      auto_exclude: ["~$*", "._*", "#*#", "*.tmp"]
      auto_pick: "single"      # single | latest | merge

  - id: "merge_map"
    uses: "merge/sample_map"
    reads:
      df: "ingest/df"
      sample_map: "file:./metadata.xlsx"

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

report_presets:
  - plots/plate_reader_yfp_full

reports:
  - id: export_ratios
    uses: export/csv
    reads: { df: "ratio_yfp_od600/df" }
    with: { path: "exports/ratio_yfp_od600.csv" }

report_overrides:
  snapshot_bars_by_channel:
    with:
      time: 14.0

Notes:
- Snapshot plots require an explicit `time` (and `snap_time` for `plot/ts_and_snap`).
- Use `reader report --list <config>` to list report steps for a config.
- `report_presets` is a shortcut for common report bundles; add or override steps under `reports`/`report_overrides`.

You can list available presets with:

```bash
uv run reader presets
```
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
