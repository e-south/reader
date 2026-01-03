
## Configuring pipelines

Pipelines are defined within `config.yaml` and detail steps you want to run the same way each time (ingest/merge/transform/validate). Outputs derived from pipelines can then feed into other deliverables (plots/exports).

### Contents

1. [Configuration keys](#configuration-keys)
2. [Example configuration](#example-configuration)

---

### Configuration keys

| Key                     | Type        | Required | Purpose                                                                                                                                                |
| ----------------------- | ----------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `experiment.id`         | string      | no       | Stable identifier for the experiment.                                                                                                                  |
| `experiment.name`       | string      | no       | Verbose name (defaults to directory name).                                                                                                      |
| `experiment.outputs`    | string      | yes      | Output directory (relative to config).                                                                                                                 |
| `experiment.palette`    | string/null | no       | Plot palette name (one of: `colorblind`, `muted`, `tableau`) or null to disable. Unknown names error.                                                  |
| `experiment.plots_dir`  | string/null | no       | Subdir for plots under outputs (null/"" = outputs root).                                                                                               |
| `runtime.strict`        | bool        | no       | Enforce contracts during execution (default true). If false, contract mismatches become warnings and execution continues (missing inputs still error). |
| `steps`                 | list        | yes      | Pipeline steps (ingest/merge/transform/validate).                                                                                                      |
| `deliverables`          | list        | no       | Deliverable steps (plots/exports).                                                                                                                     |
| `deliverable_presets`   | list        | no       | Preset bundles for deliverables.                                                                                                                       |
| `deliverable_overrides` | map         | no       | Per-deliverable step overrides by id.                                                                                                                  |
| `overrides`             | map         | no       | Per-step overrides by id (pipeline steps).                                                                                                             |
| `collections`           | map         | no       | Named groupings (used by some plots).                                                                                                                  |

How to interpret the configuration:

* Steps run top-to-bottom, in order.
* `uses:` selects a plugin (`<category>/<key>`).
* `reads:` binds plugin inputs to either:

  * a prior step output (`<step_id>/<output>`, e.g. `ingest/df`)
  * a file path (`file:./something.xlsx`)
* `with:` is plugin-specific configuration (unknown keys error).
* `writes:` optionally maps plugin outputs to stable artifact labels (decouples downstream steps from step ids).

  * downstream `reads:` must reference the mapped label
* `preset:` expands to one or more steps (use for shared bundles).
* `overrides:` / `deliverable_overrides:` apply per-step config overrides by id.

Practical guardrails:

* Step ids must be unique across **pipeline steps and deliverables**.
* `reads:` labels must refer to outputs from prior steps (or `file:`).
* Output labels must be unique across steps; use `writes:` to avoid accidental clobbering.

---

### Example configuration

Below is an example configuration showing a Synergy H1 ingest, sample-map merge, a small transform chain, and a deliverable preset.

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

---

@e-south
