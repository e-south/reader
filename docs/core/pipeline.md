# Configuring pipelines

Pipelines are defined in `config.yaml` and detail steps you want to run the same way each time (ingest/transform/validate). Outputs derived from pipelines can then feed into plots, notebooks, and exports.

### Contents

1. [Schema marker](#schema-marker)
2. [Top-level structure](#top-level-structure)
3. [Step shape](#step-shape)
4. [Example configuration](#example-configuration)

---

### Schema marker

Every config must declare the schema at the top:

```yaml
schema: "reader/v4"
```

---

### Top-level structure

```yaml
schema: "reader/v4"

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

pipeline:
  recipes: []                  # optional
  runtime: {}                  # optional (e.g., strict: true)
  overrides: {}                # optional per-step overrides by id
  steps: []                    # required (use empty list if none)

plots:
  recipes: []                  # optional
  defaults:                    # optional defaults applied to all plot specs
    reads: {}                  # e.g., { df: { record: "ratios/yfp_od600" } }
    with:  {}                  # shallow-merged into spec.with
  overrides: {}                # optional per-plot overrides by id
  specs: []                    # optional (unordered)

exports:
  recipes: []                  # optional
  defaults:                    # optional defaults applied to all export specs
    reads: {}                  # e.g., { df: { record: "ratios/yfp_od600" } }
    with:  {}
  overrides: {}                # optional per-export overrides by id
  specs: []                    # optional (unordered)

notebooks:
  specs:
    - id: "default"
      template: "notebook/basic"   # optional default for `reader notebook`

resources:
  optional_extra_input:
    kind: file
    path: "./inputs/custom.xlsx"

assay:
  labels: {}                   # optional reusable label maps for transform/assay_labels
  collections: {}              # optional reusable plot partitions for conditions/designs/treatments
  orders: {}                   # optional reusable categorical orders for plots
  logic_maps: {}               # optional reusable corner/state maps (for SFXI etc.)
```

Notes:

- `paths.outputs` is resolved relative to the config file when `reader` builds the internal workbench declaration.
- `paths.plots`, `paths.exports`, and `paths.notebooks` must be relative to `paths.outputs`.
- `paths.plots`, `paths.exports`, and `paths.notebooks` may not contain `..` segments.
- Omit `paths` entirely when you want the defaults shown above.
- Omit `plotting.palette` when you want the default `colorblind` palette.
- Omit `experiment` entirely when the directory name is the id you want and you do not need a custom title.
- Omit `experiment.title` when it would just repeat `experiment.id`.
- Omit `experiment.id` only when the experiment directory name is the canonical id you want.
- `pipeline.steps` is required (use `[]` if you have no pipeline steps yet).
- Step/spec ids must be unique across pipeline, plots, exports, and notebooks.
- Inline `recipe:` entries inside `steps` are not supported. Use `pipeline.recipes`, `plots.recipes`, or `exports.recipes` instead.
- `notebooks.specs` is declarative-only in `reader/v4`: use `id` and `template`.
- Plot/export defaults apply after recipe expansion and before per-id overrides.
- `reads.*` bindings are explicit mappings: `{record: ...}`, `{file: ...}`, or `{resource: ...}`.
- Every `resource:` binding must resolve through an explicit `resources:` declaration. There are no hidden conventional resources.
- `assay` is the canonical place for reusable assay semantics; `resources` is the canonical place for named external inputs; `paths` owns only output layout.
- Plugin interfaces are typed internally through `workbench/ports/`; config
  wiring no longer depends on `?` suffixes, `"none"` sentinels, or the legacy
  `"files"` output convention.
- Internally, `reader` materializes plugin-backed steps and notebook templates
  through an explicit staged model:
  `config -> decl -> experiment -> graph -> engine -> records`.
  That keeps wire syntax, internal authored structure, experiment-local
  semantics, executable graph nodes, and persisted provenance semantically
  distinct.

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
- The first `notebooks.specs` entry controls the default notebook template used by `reader notebook` when `--template` is omitted. If no notebook spec is configured, `reader` selects a default template from template capabilities instead of hardcoded CLI branches.

---

### Step shape

A step object (used in `pipeline.steps`, `plots.specs`, and `exports.specs`) looks like:

```yaml
- id: <string>
  plugin: "<category>/<key>"     # ingest/transform/validator/plot/export
  reads: {}                      # optional (input bindings)
  with:  {}                    # optional (plugin params)
  writes: {}                     # optional (stable output labels)
```

Rules:

- `reads` binds each input name to exactly one ref shape:
  `{record: "step_id/df"}`, `{file: "./inputs/run001.ext"}`, or `{resource: "sample_map"}`.
- `writes` maps outputs to stable labels (so downstream steps can avoid tight coupling to step ids).
- `writes` binds each output name to `{record: "stable/id"}`.
- `pipeline` steps may not use `plot/*` or `export/*` plugins.
- `plots` specs must use `plot/*` plugins and are unordered.
- `exports` specs must use `export/*` plugins and are unordered.
- `notebooks` specs must use `template: notebook/*`, are unordered, and currently do not support `reads` or `writes`.

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

That autodiscovery policy now lives with ingest plugins under
`plugins/ingest/discovery_policy.py`, not under `workbench/`.

**Resources**

Declare every named external input under `resources`:

```yaml
resources:
  custom_map:
    kind: file
    path: "./inputs/custom_map.xlsx"

pipeline:
  steps:
    - id: merge_map
      plugin: transform/sample_map
      reads:
        df:
          record: ingest/df
        sample_map:
          resource: custom_map
```

Common cases still use explicit resources:

```yaml
resources:
  sample_map:
    kind: file
    path: "./inputs/metadata.xlsx"
  metadata:
    kind: file
    path: "./inputs/metadata.csv"
```

**Assay labels**

The `transform/assay_labels` plugin materializes reusable label maps from `assay.labels`. By default it applies every defined label:

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
      plugin: transform/assay_labels
      reads:
        df:
          record: "final/df"
```

If you only want a subset, pass `with.refs`. Keep one `labels` step unless there is a concrete need to split the transformations.

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
      plugin: plot/snapshot_heatmap
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
      plugin: transform/sfxi
      reads: { df: promote_to_tidy_plus_map/df }
      with:
        response:
          logic_channel: YFP/CFP
          intensity_channel: YFP/OD600
        design_by: [design_id]
        reference: { design_id: REF, stat: mean }
        logic_map_ref: induction_logic
```

**Assay collections**

Grouped plot plugins (`plot/time_series`, `plot/distributions`, `plot/ts_and_snap`, `plot/snapshot_barplot`) now use an explicit `partition` block plus reusable `assay.collections` when you want experiment-specific plot composition:

```yaml
assay:
  collections:
    group_ab:
      column: design_id
      items:
        Group A: ["g1", "g2"]
        Group B: ["g3"]

plots:
  specs:
    - id: plot_ts
      plugin: plot/time_series
      with:
        partition:
          collection_ref: group_ab
        hue: treatment
        y: ["OD600", "YFP"]
```

- `partition.by` groups one figure/panel per distinct value of a column.
- `partition.collection_ref` reuses a named assay collection with labeled subsets.
- `partition.match` controls membership matching (`exact`, `contains`, `startswith`, `endswith`, `regex`).
- `partition.collection_ref` fails fast if the named collection is missing or targets a different column than `partition.by`.

---

### Example configuration

```yaml
schema: "reader/v4"                 # required schema marker

experiment:
  id: "20250512_panel_M9_glu"       # optional if the directory name already matches
  title: "Cell line panel — M9"     # optional display name

assay:
  labels:
    design_id:
      source: design_id
      output: design_id_alias
      values:
        ctrl: control               # rename raw labels once
  collections:
    group_ab:
      column: design_id_alias
      items:
        Group A: ["control", "g1"]
        Group B: ["g2"]
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
      plugin: ingest/synergy_h1       # plugin to read plate reader files
      with:
        channels: ["OD600", "CFP"]  # measurements to ingest
        auto_roots: ["./inputs"]    # where to look for raw files
        auto_pick: "single"         # pick one file if multiple

    - id: merge_map
      plugin: transform/sample_map        # attach metadata columns
      reads:
        df:
          record: "ingest/df"       # from prior step
        sample_map:
          resource: "sample_map"    # named external resource
      # If the merged table contains the required mapped metadata
      # (for example design_id + treatment), reader promotes the stored
      # dataframe-record contract to plate_reader.annotated.v1.
      # `reader explain` will show the minimum contract plus this
      # runtime promotion path when the plugin advertises it.

    - id: labels
      plugin: transform/assay_labels
      reads:
        df:
          record: "merge_map/df"

    - id: ratio_yfp_od600
      plugin: transform/ratio
      reads:
        df:
          record: "labels/df"       # input dataframe
      with:  { name: "YFP/OD600", numerator: "YFP", denominator: "OD600" }  # new column
      writes:
        df:
          record: "ratios/yfp_od600"  # stable label for downstream

plots:
  recipes:
    - plots/plate_reader_yfp_full   # bundle of plot specs
  defaults:
    reads:
      df:
        record: "ratios/yfp_od600"  # default plot input
  specs:
    - id: plot_ts
      plugin: plot/time_series
      with:
        x: time                     # x-axis column
        y: ["OD600", "YFP"]         # y-series
        hue: treatment              # color by treatment
        partition:
          collection_ref: group_ab  # reusable labeled plot partition

    - id: heatmap
      plugin: plot/snapshot_heatmap
      with:
        channel: "YFP/OD600"
        time: 14.0
        x: treatment
        y: design_id_alias
        order_x_ref: "induction_stress_2x2"  # pulls from assay.orders.induction_stress_2x2

exports:
  defaults:
    reads:
      df:
        record: "ratios/yfp_od600"  # default export input
  specs:
    - id: export_ratios
      plugin: export/csv
      with: { path: "ratios.csv" }  # file name under outputs/exports/

notebooks:
  specs:
    - id: "default"
      template: "notebook/eda"        # default notebook scaffold
```

---

@e-south
