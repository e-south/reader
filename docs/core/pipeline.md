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
schema: "reader/v3"
```

---

### Top-level structure

```yaml
schema: "reader/v3"

experiment:                     # optional; omit entirely to derive id/title from the directory
  id: <string>                 # optional; defaults to the experiment directory name
  title: <string | null>       # optional; defaults to experiment.id

paths:
  outputs: "./outputs"        # optional default
  plots: "plots"              # optional default (relative to outputs; use "." to flatten)
  exports: "exports"          # optional default (relative to outputs; use "." to flatten)
  notebooks: "notebooks"      # optional default (relative to outputs)

plotting:
  palette: "colorblind"       # optional default; string or null

semantics:
  groups: {}                   # optional generic plot grouping sets

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

notebooks:
  defaults:
    with: {}
  overrides: {}
  specs:
    - id: "default"
      uses: "notebook/basic"   # optional default for `reader notebook`

resources:
  sample_map:
    kind: file
    path: "./inputs/metadata.xlsx"

assay:
  labels: {}                   # optional reusable label maps for transform/alias
  orders: {}                   # optional reusable categorical orders for plots
  logic_maps: {}               # optional reusable corner/state maps (for SFXI etc.)
```

Notes:

- `paths.outputs` is resolved relative to the config file and stored as an absolute path.
- `paths.plots`, `paths.exports`, and `paths.notebooks` must be relative to `paths.outputs`.
- Omit `paths` entirely when you want the defaults shown above.
- Omit `plotting.palette` when you want the default `colorblind` palette.
- Omit `experiment` entirely when the directory name is the id you want and you do not need a custom title.
- Omit `experiment.title` when it would just repeat `experiment.id`.
- Omit `experiment.id` only when the experiment directory name is the canonical id you want.
- `pipeline.steps` is required (use `[]` if you have no pipeline steps yet).
- Step/spec ids must be unique across pipeline, plots, exports, and notebooks.
- Inline `preset:` entries inside `steps` are not supported. Use `pipeline.presets`, `plots.presets`, or `exports.presets` instead.
- `notebooks.specs` is declarative today: use `id`, `uses`, and keep `with` empty until notebook config semantics are implemented.
- Plot/export defaults apply after preset expansion and before per-id overrides.
- `resources` is the canonical place for reusable external inputs such as `sample_map` or cytometer files; use `resource:<id>` in `reads` instead of repeating `file:./...` in multiple places.
- `assay` is the canonical place for reusable experiment semantics. Keep resource handles under `resources:` and reusable meaning under `assay:` rather than hiding both inside free-form step config.
- Internally, `reader` materializes all three sections into one shared
  `WorkbenchSpec` model before planning, validation, runtime execution, and CLI
  inspection. That keeps spec semantics consistent across the workbench instead
  of re-deriving `pipeline` vs `plot` vs `export` vs `notebook` shape in multiple places.

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
    records.json
```
- The first `notebooks.specs` entry controls the default notebook template used by `reader notebook` when `--preset` is omitted.

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

- `reads` can bind inputs to a prior output (e.g., `merge/df`), to an explicit file path using `file:`, or to a named resource using `resource:`.
- `writes` maps outputs to stable labels (so downstream steps can avoid tight coupling to step ids).
- `pipeline` steps may not use `plot/*` or `export/*` plugins.
- `plots` specs must use `plot/*` plugins and are unordered.
- `exports` specs must use `export/*` plugins and are unordered.
- `notebooks` specs must use `notebook/*` templates, are unordered, and currently do not support `reads` or `writes`.

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

**Resources**

Use `resources` for files that multiple steps or presets need:

```yaml
resources:
  sample_map:
    kind: file
    path: "./inputs/metadata.xlsx"

pipeline:
  steps:
    - id: merge_map
      uses: merge/sample_map
      reads:
        df: ingest/df
        sample_map: resource:sample_map
```

**Assay labels**

The `transform/alias` plugin now resolves reusable label maps from `assay.labels` via `refs`:

```yaml
assay:
  labels:
    design_id:
      source: design_id
      output: design_id_alias
      values:
        ctrl: control
    treatment:
      source: treatment
      output: treatment_alias
      values:
        IPTG_0: -IPTG
        IPTG_500: +IPTG

pipeline:
  steps:
    - id: labels
      uses: transform/alias
      reads: { df: "final/df" }
      with:
        refs: ["design_id", "treatment"]
```

One alias step can apply multiple reusable label refs. Keep one `labels` step unless there is a concrete need to split the transformations.

**Assay orders**

Plots can pull reusable category orders from `assay.orders`:

```yaml
assay:
  orders:
    induction_stress_2x2:
      column: treatment_alias
      values:
        - "-IPTG/-stress"
        - "+IPTG/-stress"
        - "-IPTG/+stress"
        - "+IPTG/+stress"

plots:
  specs:
    - id: heatmap
      uses: plot/snapshot_heatmap
      with:
        x: treatment_alias
        y: design_id_alias
        order_x_ref: "induction_stress_2x2"
```

`order_x_ref` and `order_y_ref` fail fast if the named order is missing or if the declared labels do not exist in the rendered surface.

**Assay logic maps**

SFXI and logic-symmetry use reusable corner/state maps from `assay.logic_maps`:

```yaml
assay:
  logic_maps:
    induction_logic:
      column: treatment_alias
      corners:
        "00": "-IPTG/-stress"
        "10": "+IPTG/-stress"
        "01": "-IPTG/+stress"
        "11": "+IPTG/+stress"
      case_sensitive: true

pipeline:
  steps:
    - id: sfxi_vec8
      uses: transform/sfxi
      reads: { df: promote_to_tidy_plus_map/df }
      with:
        response:
          logic_channel: YFP/CFP
          intensity_channel: YFP/OD600
        design_by: [design_id]
        reference: { design_id: REF, stat: mean }
        logic_map_ref: induction_logic
```

Use `semantics.groups` only for generic plotting groups that are not assay-specific, such as “Group A/Group B” slices reused across multiple plot types.

---

### Example configuration

```yaml
schema: "reader/v3"                 # required schema marker

experiment:
  id: "20250512_panel_M9_glu"       # optional if the directory name already matches
  title: "Cell line panel — M9"     # optional display name

semantics:
  groups:
    genotype:                       # grouping name used by plots
      group_ab:
        - {"Group A": ["g1", "g2"]} # label -> members
        - {"Group B": ["g3"]}

resources:
  sample_map:
    kind: file
    path: "./inputs/metadata.xlsx"

assay:
  labels:
    design_id:
      source: design_id
      output: design_id_alias
      values:
        ctrl: control               # rename raw labels once
  orders:
    induction_stress_2x2:
      column: treatment
      values:
        - "-IPTG/-stress"
        - "+IPTG/-stress"
        - "-IPTG/+stress"
        - "+IPTG/+stress"

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
        sample_map: "resource:sample_map"          # named external resource
      # If the merged table contains the required mapped metadata
      # (for example design_id + treatment), reader promotes the stored
      # dataframe-record contract to plate_reader.annotated.v1.
      # `reader explain` will show the minimum contract plus this
      # runtime promotion path when the plugin advertises it.

    - id: labels
      uses: transform/alias
      reads: { df: "merge_map/df" }
      with:
        refs: ["design_id"]

    - id: ratio_yfp_od600
      uses: transform/ratio
      reads: { df: "labels/df" }    # input dataframe
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

    - id: heatmap
      uses: plot/snapshot_heatmap
      with:
        channel: "YFP/OD600"
        time: 14.0
        x: treatment
        y: design_id_alias
        order_x_ref: "induction_stress_2x2"  # pulls from assay.orders.induction_stress_2x2

exports:
  defaults:
    reads:
      df: "ratios/yfp_od600"        # default export input
  specs:
    - id: export_ratios
      uses: export/csv
      with: { path: "ratios.csv" }  # file name under outputs/exports/

notebooks:
  specs:
    - id: "default"
      uses: "notebook/eda"          # default notebook scaffold
```

---

@e-south
