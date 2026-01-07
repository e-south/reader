# Configuring pipelines

Pipelines are defined in `config.yaml` and detail steps you want to run the same way each time (ingest/merge/transform/validate). Outputs derived from pipelines can then feed into plots, notebooks, and exports.

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

---

### Inputs + metadata placement

By default, place **raw inputs and metadata under `inputs/`**. Auto-discovery for ingest plugins
(`ingest/synergy_h1`, `ingest/flow_cytometer`) scans `inputs/` by default and **excludes common
metadata filenames** to avoid accidental ingestion:

- `metadata.*`
- `metadata_filtered.*`
- `sample_map.*`
- `sample_metadata.*`
- `plate_map.*`

If your metadata uses different names, either pass an explicit `reads.raw` file path or add those
names to the ingest step’s `auto_exclude` list.

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
schema: "reader/v2"                 # required schema marker

experiment:
  id: "20250512_panel_M9_glu"       # short unique experiment id
  title: "Cell line panel — M9"     # optional display name

paths:
  outputs: "./outputs"              # base output directory (relative to config)
  plots: "plots"                    # subdir under outputs/
  exports: "exports"                # subdir under outputs/
  notebooks: "notebooks"            # subdir under outputs/

plotting:
  palette: "colorblind"             # palette name (or null)

data:
  groupings:
    genotype:                       # grouping name used by plots
      group_ab:
        - {"Group A": ["g1", "g2"]} # label -> members
        - {"Group B": ["g3"]}
  aliases:
    design_id:
      "ctrl": "control"             # rename raw labels

pipeline:
  runtime:
    strict: true                    # fail fast on missing inputs/columns
  steps:
    - id: ingest                    # unique step id
      uses: ingest/synergy_h1       # plugin to read plate reader files
      with:
        channels: ["OD600", "CFP"]  # measurements to ingest
        auto_roots: ["./inputs"]    # where to look for raw files
        auto_pick: "single"         # pick one file if multiple

    - id: merge_map
      uses: merge/sample_map        # attach metadata columns
      reads:
        df: "ingest/df"             # from prior step
        sample_map: "file:./inputs/metadata.xlsx"  # metadata file

    - id: ratio_yfp_od600
      uses: transform/ratio
      reads: { df: "merge_map/df" } # input dataframe
      with:  { name: "YFP/OD600", numerator: "YFP", denominator: "OD600" }  # new column
      writes: { df: "ratios/yfp_od600" }  # stable label for downstream

plots:
  presets:
    - plots/plate_reader_yfp_full   # bundle of plot specs
  defaults:
    reads:
      df: "ratios/yfp_od600"        # default plot input
  specs:
    - id: plot_ts
      uses: plot/time_series
      with:
        x: time                     # x-axis column
        y: ["OD600", "YFP"]         # y-series
        hue: treatment              # color by treatment

exports:
  defaults:
    reads:
      df: "ratios/yfp_od600"        # default export input
  specs:
    - id: export_ratios
      uses: export/csv
      with: { path: "ratios.csv" }  # file name under outputs/exports/

notebook:
  preset: "notebook/eda"            # default notebook scaffold
```

---

@e-south
