# Configuring Reader v7

`reader` now has three explicit layers:

- authoring: experiment `config.yaml`
- semantics: `reader.protocols`
- execution: compiled workbench IR

The config surface is for human authors. It should describe assay inputs,
analysis choices, and requested outputs in domain terms, not plugin or graph
terms.

## Minimal shape

```yaml
schema: reader/v7

experiment:
  id: 20250614_sensor_panel_M9_glu

protocol:
  id: plate_reader/dual_reporter_screen
  inputs:
    ingest:
      mode: auto
    fold_change:
      report_times: [8.0, 14.0]

resources:
  sample_map:
    kind: file
    path: ./inputs/metadata.xlsx
```

## Top-level surface

- `schema`
  Must be `reader/v7`.
- `experiment`
  Explicit experiment identity. `experiment.id` is required. Optional `experiment.lifecycle`
  may be `draft` or `template` for intentionally non-runnable configs; omit it for
  normal active experiments.
- `protocol`
  Assay binding plus human-facing semantic config.
- `resources`
  External files such as `sample_map` or `metadata`.
- `annotations`
  Labels, orders, collections, and logic maps.
- `paths`
  Optional output layout override.
- `plotting`
  Optional shared plotting palette override.

There is no public `graph_patch`, no top-level `pipeline` / `plots` /
`exports`, and no `protocol.with`.

## Protocol surface

The protocol block is split by role:

- `protocol.inputs`
  Assay-family input bindings and protocol-owned knobs.
- `protocol.analysis`
  Analysis toggles and semantic policy choices.
- `protocol.outputs`
  Notebook, plot, and export selection.

## Plot and artifact registries

Protocols expose two user-facing registries:

- plot outputs
  Named figures such as `raw_kinetics`, `endpoint_by_condition`, or
  `logic_symmetry`
- export artifacts
  Named files such as `crosstalk_pairs_table` or
  `logic_summary_workbook`

Plot outputs can also be grouped into named plot profiles. A profile is just a
semantic bundle of figure ids chosen by the protocol author.

Users do not select plugins directly. They choose:

- a plot profile
- optional `include` / `exclude` figure ids
- optional per-figure `views`
- optional export `include` / `exclude`
- optional per-artifact `artifacts` config

Unknown keys on the public authoring surface now fail fast. `reader/v7` no
longer silently drops misspelled `protocol` keys, unknown plot/export output
blocks, or malformed annotation collections.

## Plate-reader example

```yaml
protocol:
  id: plate_reader/dual_reporter_screen
  inputs:
    ingest:
      mode: auto
      sheet_names: ["Plate 1 - Sheet1", "Plate 2 - Sheet1"]
    fold_change:
      report_times: [8.0, 14.0]
      treatment_column: treatment
      group_by: [design_id]
  analysis:
    crosstalk_pairs:
      enabled: true
      export: true
  outputs:
    notebook:
      template: notebook/eda
    plots:
      profile: screen_overview
      include: [ratio_heatmap]
      views:
        ratio_heatmap:
          time: 12.0
    exports:
      include: [crosstalk_pairs_table]
      artifacts:
        crosstalk_pairs_table:
          path: crosstalk_pairs.csv
```

## Logic/SFXI example

```yaml
protocol:
  id: logic/sfxi_screen
  inputs:
    ingest:
      mode: mixed
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
  outputs:
    notebook:
      template: notebook/sfxi_eda
    plots:
      profile: logic_geometry
    exports:
      include: [logic_summary_workbook]
```

## Cytometry example

```yaml
protocol:
  id: cytometry/flow_panel
  inputs:
    ingest:
      auto_roots: ["./inputs"]
      channel_name_field: pns
    metadata:
      require_columns: [design_id, treatment]
      require_non_null: true
```

## Discoverability

Use the CLI to inspect the semantic surface:

```bash
uv run reader ls --details
uv run reader ls --details --format json
uv run reader init ./experiments/20260317_new_assay --protocol plate_reader/dual_reporter_screen
uv run reader init ./experiments/20260317_new_assay --protocol plate_reader/single_reporter_screen
uv run reader init ./experiments/20260317_new_assay --protocol plate_reader/retron_sponge_screen
uv run reader inspect experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
uv run reader inspect experiments/2025/20250614_sensor_panel_M9_glu/config.yaml --format json
uv run reader protocols plate_reader/dual_reporter_screen
uv run reader protocols plate_reader/single_reporter_screen
uv run reader protocols plate_reader/retron_sponge_screen
uv run reader protocols plate_reader/dual_reporter_screen --format json
uv run reader protocols plate_reader/single_reporter_screen --format json
uv run reader protocols plate_reader/retron_sponge_screen --format json
uv run reader protocols plate_reader/dual_reporter_screen --example-config
uv run reader protocols plate_reader/retron_sponge_screen --example-config
uv run reader plugins --protocol plate_reader/dual_reporter_screen --category transform
uv run reader plugins --protocol plate_reader/dual_reporter_screen --category transform --format json
uv run reader plugins --protocol plate_reader/single_reporter_screen --category plot --format json
uv run reader plugins --protocol plate_reader/retron_sponge_screen --category transform --format json
uv run reader explain experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
uv run reader plot experiments/2025/20250614_sensor_panel_M9_glu/config.yaml --list
uv run reader export experiments/2025/20250614_sensor_panel_M9_glu/config.yaml --list
```

For plate-reader assays, the protocol boundary matters:

- `plate_reader/dual_reporter_screen` owns CFP/YFP-style dual-reporter panels.
- `plate_reader/single_reporter_screen` owns single-reporter panels such as RFP/OD600 screens.
- `plate_reader/retron_sponge_screen` owns matched-control sponge assays with explicit control/window/comparison/ranking semantics.

These commands show:

- what experiments exist, which protocol each one binds, and how many outputs already exist
- the same experiment inventory as a machine-readable contract for agents and automation
- a starter experiment directory for a chosen protocol
- one bound experiment as explicit `authoring`, `semantics`, and `implementation` layers, including inputs/resources, generated output examples, and the compiled runtime chain
- the same three-layer JSON contract for `reader config`, `reader steps`, `reader inspect`, and `reader explain`, so automation does not need to reconcile multiple experiment surfaces
- the authoring inputs and analysis surface for the selected protocol, plus the default compiled pipeline and output implementations, in either table or JSON form
- the plugin registry filtered to the transform kernel a given protocol actually uses
- a starter YAML outline for that protocol
- the bound protocol
- the compiled pipeline chain and record/resource daisy chain
- the selected plot outputs
- the selected export artifacts
- the notebook template policy

## Mental model

The authoritative flow is:

`config -> protocol binding -> protocol compiler -> workbench decl -> graph -> engine -> records`

That keeps:

- human authoring in config
- assay semantics in `reader.protocols`
- execution IR in `decl/` and `graph/`
- plugin mechanics in `plugins/`

separate and explicit.
