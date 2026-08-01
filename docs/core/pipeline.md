---
doc_id: reader-v8-config
surface: config-reference
owner: reader-maintainers
last_verified: 2026-08-01
summary: Public Reader v8 configuration reference for experiments, protocols, resources, annotations, paths, and outputs.
---

# Configuring Reader v8

Reader v8 has three explicit layers:

- config: experiment `config.yaml`
- protocol: `reader_workbench.protocols`
- execution: compiled workbench plan

The config should describe assay inputs, analysis choices, and requested outputs
in domain terms, not plugin or graph terms.

## Minimal shape

```yaml
schema: reader/v8

experiment:
  id: example_plate_reader

evidence:
  data_class: plate_reader_screen
  data_class_reason: Time-series plate-reader assay with an explicit sample map.
  replicate_kind: biological

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
      observation_stat: median

resources:
  sample_map:
    kind: file
    path: ./inputs/metadata.xlsx
```

`replicate_kind` records what the source establishes: `biological`,
`technical`, `mixed`, `unknown`, or `not_applicable`. Declare
`replicate_identity_field` only when a persisted source field identifies the
replicate units inside one experiment record. Without that field, the
declaration applies to the experiment itself. For example, when one Reader
experiment represents one physical plate and plates are biological replicates,
declare `replicate_kind: biological` and omit `replicate_identity_field`. Plate
positions remain observations; they are not thereby technical replicates. Use
`unknown` when the replicate kind is not established, and omit the identity
field as well when no grouping relationship is established.

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
  Declared files and exact records owned by other Reader experiments. Directory
  discovery roots belong under protocol inputs such as
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
They declare source records, compile through a protocol, and publish only to
their configured `outputs/` directory through the normal engine.

## Resource kinds

A file resource is confined to its owning experiment:

```yaml
resources:
  sample_map:
    kind: file
    path: ./inputs/metadata.xlsx
```

A record resource names one current dataframe record in another Reader
experiment. The referenced experiment must live beneath the same canonical
`experiments/` owner. Directory names and year partitions are not part of the
contract.

```yaml
resources:
  source_a:
    kind: record
    experiment: source_experiment_a
    record: annotated/df
```

Protocols bind one or more record-resource ids to typed collection ports.
Before a real run creates output state, Reader verifies that every source
record exists, satisfies the port's dataframe contract, and still matches its
recorded content digest. The produced record then captures each exact source
revision. Reader does not copy source configs or record catalogs into the
aggregate experiment.

## Ordered state-space annotation

`annotations.ordered_state_spaces` binds ordered state ids to exact values in
one metadata column. It contains no target mask or metric semantics. Analyses
resolve it and enforce their own state requirements.

```yaml
annotations:
  ordered_state_spaces:
    four_state_conditions:
      column: treatment
      state_order: ["00", "10", "01", "11"]
      values:
        "00": baseline
        "10": condition A
        "01": condition B
        "11": conditions A and B
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
  Plot and export selection.

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

Unknown keys in the public config fail fast, including misspelled protocol
keys, unknown plot or export blocks, and malformed annotation collections.

Reader does not choose a scientific endpoint. The neutral plate-reader starter
selects kinetics and QC views only. Enabling fold-change requires explicit
`protocol.inputs.fold_change.report_times`; selecting an endpoint figure
requires its `time` or `snap_time` under `protocol.outputs.plots.views`.

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
      observation_stat: median
      treatment_column: treatment
      group_by: [design_id]
  analysis:
    include_fold_change: true
    crosstalk_pairs:
      enabled: true
      export: true
  outputs:
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

## Four-state logic-vector example

```yaml
protocol:
  id: logic/four_state_vector_screen
  inputs:
    ingest:
      mode: mixed
    reference:
      design_id: REF
      observation_stat: mean
    design_by: [design_id]
    state_map_ref: induction_logic
  analysis:
    include_export: true
  outputs:
    plots:
      profile: logic_geometry
    exports:
      include: [logic_summary_workbook]
```

## Cytometry example

The gating policy is required and has no scientific defaults. Replace the
example channel names and bounds with values declared for the experiment.

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
    gating:
      cells_enabled: true
      cells_x_channel: FSC-A
      cells_y_channel: SSC-A
      cells_x_range: [100.0, 100000.0]
      cells_y_range: [100.0, 100000.0]
      singlets_enabled: true
      singlet_x_channel: FSC-H
      singlet_y_channel: FSC-A
      singlet_ratio_range: [0.8, 1.2]
      fluorescence_channel: GFP-A
      threshold_mode: manual
      threshold_value: 1000.0
      threshold_group_column: null
      threshold_control_value: null
      threshold_quantile: null
      group_column: treatment
      minimum_final_events: 100
      minimum_final_percent: 1.0
      maximum_nonpositive_percent: 5.0
      nonpositive_scope: all_events
  outputs:
    plots:
      profile: gating_review
    exports:
      include: [gate_definition_table, sample_stats_table, group_stats_table, qc_table]
```

The transform persists the resolved gate, gated events, per-sample statistics,
optional group statistics, and QC. The diagnostic plot and CSV exports consume
those records through the same lifecycle; the notebook does not recompute or
publish cytometry results.

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
- `logic/four_state_vector_screen` owns the ordered four-state vector measurement contract.

## Mental model

The flow is:

`config -> protocol binding -> protocol compiler -> workbench decl -> graph -> engine -> records`

That keeps:

- authored config in `config.yaml`
- assay semantics in `reader_workbench.protocols`
- execution IR in `decl/` and `graph/`
- plugin mechanics in `plugins/`

separate and explicit.
