# Configuring Reader v6

`reader` now has one normal authoring surface:

- `protocols/` own assay semantics and default workflow assembly
- experiment configs bind semantic protocol inputs
- compiled workbench graphs are runtime IR, not user-authored config

There is no public `graph_patch`, no top-level `pipeline` / `plots` / `exports`,
and no `protocol.with`.

## Minimal shape

```yaml
schema: reader/v6

experiment:
  id: 20250614_sensor_panel_M9_glu

protocol:
  id: plate_reader/dual_reporter_screen
  parameters:
    ingest:
      mode: auto
      channels: [OD600, CFP, YFP]
    fold_change:
      report_times: [8.0, 14.0]

resources:
  sample_map:
    kind: file
    path: ./inputs/metadata.xlsx
```

## Top-Level Surface

- `schema`
  Must be `reader/v6`.

- `experiment`
  Explicit experiment identity. `experiment.id` is required. There is no
  directory-name fallback.

- `protocol`
  The assay/workflow binding.

- `resources`
  Explicit external inputs such as `sample_map` or `metadata`.

- `annotations`
  Experiment-local labels, orders, collections, and logic maps.

- `paths`
  Optional output-layout override. Omit when using the defaults.

- `plotting`
  Optional plotting palette override. Omit when using the default.

## Protocol Surface

The protocol block is split by semantic role:

- `protocol.parameters`
  Experiment parameters for the assay family.

- `protocol.analysis`
  Analysis-plan toggles such as preprocessing, strictness, measurement mode,
  and protocol-defined feature gates.

- `protocol.deliverables`
  User-facing plot/export/notebook selection.

### Plate-reader example

```yaml
protocol:
  id: plate_reader/dual_reporter_screen
  parameters:
    ingest:
      mode: auto
      channels: [OD600, CFP, YFP]
      sheet_names: ["Plate 1 - Sheet1", "Plate 2 - Sheet1"]
    fold_change:
      report_times: [8.0, 14.0]
      treatment_column: treatment
      group_by: [design_id]
  analysis:
    crosstalk_pairs:
      enabled: true
      export: true
  deliverables:
    plots:
      profile: yfp_time_series
      include: [snapshot_heatmap_yfp_cfp]
      settings:
        snapshot_heatmap_yfp_cfp:
          channel: YFP/CFP
          time: 12.0
    exports:
      include: [crosstalk_pairs_csv]
```

### Logic/SFXI example

```yaml
protocol:
  id: logic/sfxi_screen
  parameters:
    ingest:
      mode: mixed
      channels: [OD600, CFP, YFP]
    response:
      logic_channel: YFP/CFP
      intensity_channel: YFP/OD600
    reference:
      design_id: REF
      stat: mean
    design_by: [design_id]
    logic_map_ref: induction_logic
  analysis:
    include_export: true
  deliverables:
    notebook:
      template: notebook/sfxi_eda
    exports:
      include: [vec8_xlsx]
```

### Cytometry example

```yaml
protocol:
  id: cytometry/flow_panel
  parameters:
    ingest:
      auto_roots: ["./inputs"]
      channel_name_field: pns
    metadata:
      require_columns: [design_id, treatment]
      require_non_null: true
```

## Deliverable IDs

Plot and export selection use stable public ids, not internal compiled step ids.

Current built-in ids include:

- plate reader / logic plots:
  - `time_series`
  - `snapshot_by_channel`
  - `snapshot_by_design`
  - `snapshot_state`
  - `ts_and_snap_intensity`
  - `ts_and_snap_ratio`
  - `distributions`
  - `snapshot_heatmap_yfp_cfp`
  - `snapshot_heatmap_cfp_od600`
  - `logic_symmetry_yfp_cfp`
- exports:
  - `crosstalk_pairs_csv`
  - `vec8_xlsx`

Use `protocol.deliverables.<surface>.settings.<deliverable_id>` for
deliverable-specific config.

## Validation Rules

- No legacy top-level workflow sections.
- No implicit protocol inference.
- No implicit experiment id inference.
- No public raw graph mutation surface.
- Unknown resources, labels, logic maps, deliverable ids, or notebook templates
  fail fast.

## Mental Model

The authoritative flow is:

`config -> protocol compiler -> workbench decl -> graph -> engine -> records`

That keeps:

- assay semantics in `protocols/`
- experiment-local metadata in `annotations/` and `resources/`
- execution IR in `decl/` and `graph/`
- plugin mechanics in `plugins/`

separate and explicit.
