## Pipelines and deliverables (config.yaml)

A `config.yaml` is the repeatable part of the workbench: it defines pipeline steps you want to run the same way each time (parsing, metadata merges, common transforms), plus optional **deliverables** (plots/exports). Pipeline steps produce artifacts; deliverables render outputs from those artifacts. Notebooks are for interactive exploration.

**Quick links**

- [README](../README.md)
- [CLI reference](./cli.md)
- [Notebooks](./notebooks.md)
- [Plugin development](./plugins.md)
- [Spec / architecture](./spec.md)
- [Marimo notebook reference](./marimo_reference.md)
- [End-to-end demo](./demo.md)
- [Outputs and revisions](#outputs-and-revisions)
- [Config keys](#config-keys-single-page-glance)
- [Example config](#example-config)

### Mental model

- An experiment is a directory (inputs + config + outputs).
- Steps run in order and write versioned artifacts into `outputs/artifacts/`.
- Deliverables read artifacts and write plots/exports (tracked in `deliverables_manifest.json`).
- Notebooks are optional and read from the same outputs.

Paths in `reads: file:...` are resolved relative to the config file.

### Directory layout

```bash
experiments/<exp>/
  config.yaml
  inputs/
  notebooks/
  outputs/
```

### Where commands live

Command usage is documented in [docs/cli.md](./cli.md), and an end‑to‑end example lives in
[docs/demo.md](./demo.md). A typical loop is:

1) `reader explain` → sanity check the plan  
2) `reader validate` → config + plugin params  
3) `reader run` → artifacts + deliverables  
4) `reader deliverables` → re-render plots/exports

### Outputs: artifacts + revisions

**reader** writes into `experiment.outputs`.

```bash
outputs/
  manifest.json
  deliverables_manifest.json
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

See [docs/cli.md](./cli.md) for the `reader artifacts` command.

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
| `deliverables` | list | no | Deliverable steps (plots/exports). |
| `deliverable_presets` | list | no | Preset bundles for deliverables. |
| `deliverable_overrides` | map | no | Per-deliverable step overrides by id. |
| `overrides` | map | no | Per-step overrides by id (pipeline steps). |
| `collections` | map | no | Named groupings (used by some plots). |

### Example config

Below is an example configuration showing a Synergy H1 ingest, sample-map merge, a small transform chain, and a deliverable preset.

How to interpret the configuration:

* Steps run in order.
* `uses:` chooses a plugin (`<category>/<key>`).
* `reads:` binds plugin inputs to either:

  * a prior output (`<step_id>/<output>`, e.g. `ingest/df`)
  * a file path: `file:./something.xlsx`
* `with:` is plugin-specific configuration.
* `writes:` optionally maps plugin outputs to stable artifact labels (decouples downstream steps from step ids). Downstream `reads:` must reference the mapped label.
* `preset:` expands to one or more steps (use for shared bundles).
* `overrides:` / `deliverable_overrides:` apply per-step config overrides by id.

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

deliverable_presets:
  - plots/plate_reader_yfp_full

deliverables:
  - id: export_ratios
    uses: export/csv
    reads: { df: "ratio_yfp_od600/df" }
    with: { path: "exports/ratio_yfp_od600.csv" }

deliverable_overrides:
  snapshot_bars_by_channel:
    with:
      time: 14.0
```

Notes:
- Snapshot plots require an explicit `time` (and `snap_time` for `plot/ts_and_snap`).
- `deliverable_presets` is a shortcut for common deliverable bundles; add or override steps under `deliverables`/`deliverable_overrides`.
- `reads` labels must refer to outputs from prior steps (or `file:`). If you use `writes`, make sure downstream steps read the new label.
- Output labels must be unique across steps; use `writes` to avoid accidental clobbering.

### Deliverables vs notebooks

Deliverables are deterministic outputs (plots/exports) that run from existing artifacts. Notebooks are interactive and can
mix ad-hoc analysis with saved outputs. Use `reader explore` to scaffold a notebook with the right paths and manifest
lookups for an experiment.

---

@e-south
