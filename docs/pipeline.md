# Configuring pipelines

Pipelines are defined in `config.yaml` and detail steps you want to run the same way each time (ingest/merge/transform/validate). Outputs derived from pipelines can then feed into plots and exports.

### Contents

1. [Schema marker](#schema-marker)
2. [Top-level structure](#top-level-structure)
3. [Step shape](#step-shape)
4. [Example configuration](#example-configuration)

---

### Schema marker

Every config must declare the schema at the top:

```yaml
schema: "reader/v2"
```

If the schema is missing or not `reader/v2`, `reader` errors immediately.

---

### Top-level structure

```yaml
schema: "reader/v2"

experiment:
  id: <string>                 # required
  title: <string | null>       # optional

paths:
  outputs: "./outputs"        # default
  plots: "plots"              # default (relative to outputs; use "." to flatten)
  exports: "exports"          # default (relative to outputs; use "." to flatten)
  notebooks: "notebooks"      # default (relative to outputs)

plotting:
  palette: "colorblind"       # string or null

data:
  groupings: {}                # optional (used by some plots)
  aliases: {}                  # optional (alias maps for transform/alias)

pipeline:
  presets: []                  # optional
  runtime: {}                  # optional (e.g., strict: true)
  overrides: {}                # optional per-step overrides by id
  steps: []                    # required (use empty list if none)

plots:
  presets: []                  # optional
  defaults:                    # optional defaults applied to all plot specs
    reads: {}                  # e.g., { df: "ratios/yfp_od600" }
    with:  {}                  # shallow-merged into spec.with
  overrides: {}                # optional per-plot overrides by id
  specs: []                    # optional (unordered)

exports:
  presets: []                  # optional
  defaults:                    # optional defaults applied to all export specs
    reads: {}                  # e.g., { df: "ratios/yfp_od600" }
    with:  {}
  overrides: {}                # optional per-export overrides by id
  specs: []                    # optional (unordered)

notebook:
  preset: "notebook/basic"     # optional (default for `reader notebook`)
```

Notes:

- `paths.outputs` is resolved relative to the config file and stored as an absolute path.
- `paths.plots`, `paths.exports`, and `paths.notebooks` must be relative to `paths.outputs`.
- `pipeline.steps` is required (use `[]` if you have no pipeline steps yet).
- Step ids must be unique across pipeline, plots, and exports.
- Inline `preset:` entries inside `steps` are not supported. Use `pipeline.presets`, `plots.presets`, or `exports.presets` instead.
- Plot/export defaults apply after preset expansion and before per-id overrides.

---

### Outputs layout

By default, outputs are written under `outputs/`:

```
outputs/
  artifacts/
  plots/
  exports/
  notebooks/
  manifests/
    manifest.json
    plots_manifest.json
    exports_manifest.json
```
- `notebook.preset` controls the default preset used by `reader notebook` when `--preset` is omitted.

---

### Step shape

A step object (used in `pipeline.steps`, `plots.specs`, and `exports.specs`) looks like:

```yaml
- id: <string>
  uses: "<category>/<key>"     # ingest/merge/transform/validator/plot/export
  reads: {}                    # optional (input bindings)
  with:  {}                    # optional (plugin params)
  writes: {}                   # optional (stable output labels)
```

Rules:

- `reads` can bind inputs to a prior output (e.g., `merge/df`) or to a file path using `file:`.
- `writes` maps outputs to stable labels (so downstream steps can avoid tight coupling to step ids).
- `pipeline` steps may not use `plot/*` or `export/*` plugins.
- `plots` specs must use `plot/*` plugins and are unordered.
- `exports` specs must use `export/*` plugins and are unordered.

**Aliases in steps**

The `transform/alias` plugin can pull alias maps from `data.aliases` using `aliases_ref`:

```yaml
- id: alias_design_id
  uses: transform/alias
  reads: { df: "final/df" }
  with:
    aliases_ref: "design_id"     # pulls from data.aliases.design_id
```

For multiple columns, add multiple alias steps (one per column):

```yaml
- id: alias_design_id
  uses: transform/alias
  reads: { df: "final/df" }
  with:
    aliases_ref: "design_id"

- id: alias_treatment
  uses: transform/alias
  reads: { df: "alias_design_id/df" }
  with:
    aliases_ref: "treatment"
```

---

### Example configuration

```yaml
schema: "reader/v2"

experiment:
  id: "20250512_panel_M9_glu"
  title: "Cell line panel — M9 + glucose"

paths:
  outputs: "./outputs"
  plots: "plots"
  exports: "exports"
  notebooks: "notebooks"

plotting:
  palette: "colorblind"

data:
  groupings:
    genotype:
      group_ab:
        - { "Group A": ["g1", "g2"] }
        - { "Group B": ["g3"] }
  aliases:
    design_id:
      "ctrl": "control"

pipeline:
  runtime:
    strict: true
  steps:
    - id: ingest
      uses: ingest/synergy_h1
      with:
        channels: ["OD600", "CFP", "YFP"]
        auto_roots: ["./inputs"]
        auto_pick: "single"

    - id: merge_map
      uses: merge/sample_map
      reads:
        df: "ingest/df"
        sample_map: "file:./metadata.xlsx"

    - id: ratio_yfp_od600
      uses: transform/ratio
      reads: { df: "merge_map/df" }
      with:  { name: "YFP/OD600", numerator: "YFP", denominator: "OD600" }
      writes: { df: "ratios/yfp_od600" }

plots:
  presets:
    - plots/plate_reader_yfp_full
  defaults:
    reads:
      df: "ratios/yfp_od600"
  specs:
    - id: plot_ts
      uses: plot/time_series
      with:
        x: time
        y: ["OD600", "YFP"]
        hue: treatment

exports:
  defaults:
    reads:
      df: "ratios/yfp_od600"
  specs:
    - id: export_ratios
      uses: export/csv
      with: { path: "ratios.csv" }

notebook:
  preset: "notebook/eda"
```

---

@e-south
