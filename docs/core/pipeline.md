---
doc_id: reader-v8-config
surface: config-reference
owner: reader-maintainers
last_verified: 2026-07-28
summary: Public Reader v8 configuration reference for experiments, protocols, resources, annotations, paths, and outputs.
---

# Configuring Reader v8

Reader v8 has three explicit layers:

- config: experiment `config.yaml`
- protocol: `reader.protocols`
- execution: compiled workbench plan

The config should describe assay inputs, analysis choices, and requested outputs
in domain terms, not plugin or graph terms.

## Minimal shape

```yaml
schema: reader/v8

experiment:
  id: 20250614_sensor_panel_M9_glu

evidence:
  data_class: plate_reader_screen
  data_class_reason: Time-series plate-reader assay with an explicit sample map.
  replicate_kind: biological
  replicate_identity_field: position

protocol:
  id: plate_reader/dual_reporter_screen
  inputs:
    ingest:
      mode: mixed
      channel_map:
        "OD600:600": OD600
        "CFP:433,475": CFP
        "YFP:500,530": YFP
    fold_change:
      report_times: [8.0, 14.0]

resources:
  sample_map:
    kind: file
    path: ./inputs/metadata.xlsx
```

`channel_map` keys are exact canonical channel labels from the Synergy export.
Keep wavelength suffixes when the workbook supplies them; the values are the
stable Reader channel names used by downstream analyses.

## Top-level keys

- `schema`
  Must be `reader/v8`.
- `experiment`
  Explicit experiment identity. `experiment.id` is required. Optional `experiment.lifecycle`
  may be `draft` or `template` for intentionally non-runnable configs; omit it for
  normal active experiments.
- `evidence`
  Optional experiment evidence for a registered DOP `data_class`, a short
  selection reason, replicate kind, and a stable replicate identity column when
  one exists. Reader records this evidence; it does not infer it from notes or
  filenames. When present, the selected class must list the bound protocol under
  its registry-owned `protocol_candidates` (`uv run reader dop classes --format json`).
- `protocol`
  Assay binding plus semantic config.
- `resources`
  Declared files such as `sample_map` or `metadata`. Resources are file-only;
  directory discovery roots belong under protocol inputs such as
  `protocol.inputs.ingest.auto_roots`.
- `annotations`
  Labels, orders, collections, and metric-neutral ordered state spaces.
- `paths`
  Optional output layout override.
- `plotting`
  Optional shared plotting palette override.

There is no public `graph_patch`, no top-level `pipeline` / `plots` /
`exports`, and no `protocol.with`.

## Output ownership

Reader's configured output paths are relative to one experiment directory.
That experiment's `outputs/` contains generated records, plots, exports,
notebook scaffolds, and manifests. A repository-root `outputs/` directory is
invalid because it has no experiment owner.

Cross-experiment reviews and aggregates are experiments in their own right.
Their publishing commands therefore require an explicit output experiment and
resolve its configured `outputs/` directory. Domain-level Python writers remain
path-agnostic so downstream repositories can publish into their own owned
workspaces without inheriting Reader's repository layout.

## Ordered state-space annotation

`annotations.ordered_state_spaces` binds ordered state ids to exact values in
one metadata column. It contains no target mask or metric semantics. Analyses
resolve it and enforce their own state requirements.

```yaml
annotations:
  ordered_state_spaces:
    stress_states:
      column: treatment
      state_order: ["00", "10", "01", "11"]
      values:
        "00": no stress
        "10": ethanol
        "01": ciprofloxacin
        "11": ethanol plus ciprofloxacin
      case_sensitive: true
```

See [Ordered state spaces](./ordered_state_spaces.md) for validation and
ownership rules.

## Protocol block

The protocol block is split by role:

- `protocol.inputs`
  Assay-family input bindings and protocol-owned knobs.
- `protocol.analysis`
  Analysis toggles and protocol policy choices.
- `protocol.outputs`
  Notebook, plot, and export selection.

## Plot and export choices

Protocols expose two user-facing registries:

- plot outputs
  Named figures such as `raw_kinetics`, `endpoint_by_condition`, or
  `logic_symmetry`
- export artifacts
  Named files such as `crosstalk_pairs_table` or
  `logic_summary_workbook`

Plot outputs can also be grouped into named plot profiles. A profile is just a
named group of figure ids chosen by the protocol author.

Users do not select plugins directly. They choose:

- a plot profile
- optional `include` / `exclude` figure ids
- optional per-figure `views`
- optional export `include` / `exclude`
- optional per-artifact `artifacts` config

Unknown keys in the public config fail fast. `reader/v8` does not
longer silently drops misspelled `protocol` keys, unknown plot/export output
blocks, or malformed annotation collections.

## Plate-reader example

```yaml
protocol:
  id: plate_reader/dual_reporter_screen
  inputs:
    ingest:
      mode: mixed
      channel_map:
        "OD600:600": OD600
        "CFP:433,475": CFP
        "YFP:500,530": YFP
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
      profile: heatmap_review
      views:
        ratio_heatmap:
          time: 12.0
        support_heatmap:
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
    state_map_ref: induction_logic
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

## Inspect the config

Use the CLI to inspect one protocol or one experiment:

```bash
uv run reader protocols <protocol-id>
uv run reader protocols <protocol-id> --example-config
uv run reader inspect <config|dir|index>
uv run reader config <config|dir|index> --format json
uv run reader explain <config|dir|index>
uv run reader plot <config|dir|index> --list
uv run reader export <config|dir|index> --list
```

For setup and task-oriented command routes, use
[Getting started](../guides/getting_started.md),
[Common tasks](../guides/common_routes.md), and the
[CLI reference](./cli.md). For machine-readable inspection, use
[Automation and JSON](../guides/automation.md).

For plate-reader assays, the protocol boundary matters:

- `plate_reader/dual_reporter_screen` owns CFP/YFP-style dual-reporter panels.
- `plate_reader/single_reporter_screen` owns single-reporter panels such as RFP/OD600 screens.
- `plate_reader/retron_sponge_screen` owns matched-control sponge assays with explicit control/window/comparison/ranking semantics.

For the direct-ratio retron workflow, including the compiled `R -> B -> C -> D -> M -> O` program and semantic-table exports, see the [Retron sponge screen guide](../guides/retron_sponge_screen.md).

## Mental model

The flow is:

`config -> protocol binding -> protocol compiler -> workbench decl -> graph -> engine -> records`

That keeps:

- authored config in `config.yaml`
- assay semantics in `reader.protocols`
- execution IR in `decl/` and `graph/`
- plugin mechanics in `plugins/`

separate and explicit.
