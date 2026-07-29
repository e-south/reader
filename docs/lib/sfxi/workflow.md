---
doc_id: reader-sfxi-workflow
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-07-28
summary: Configure, preflight, run, inspect, export, and open notebooks for a Reader SFXI experiment.
---

# SFXI experiment workflow

This guide covers the Reader-owned path from `reader/v8` configuration to the
typed vec8 record. For calculation details, use the
[vec8 contract](./vec8.md). For optional figures, use
[SFXI plot surfaces](./plots.md). For the shared plate-reader ingest boundary,
use [Plate-reader metric outputs](../plate_reader/metric_outputs.md).

## Configure the protocol

The following fragment matches the `logic/sfxi_screen` authoring
surface. The treatment labels and `J23105` reference are taken from
`experiments/2026/20260707_sfxi_sensor-panel-m9-glu-secg/config.yaml`; use the
values that are present in the target experiment's metadata.

```yaml
schema: reader/v8
protocol:
  id: logic/sfxi_screen
  inputs:
    response:
      logic_channel: YFP/CFP
      intensity_channel: YFP/OD600
    design_by: [design_id]
    state_map_ref: induction_logic
    reference:
      design_id: J23105
      stat: mean
    target_time_h: 12.0
    time_mode: nearest
    time_tolerance_h: 0.51
  analysis:
    include_vec8: true
    include_export: true
    sfxi_objective:
      intensity_log2_offset_delta: 0.0

annotations:
  ordered_state_spaces:
    induction_logic:
      column: treatment_alias
      state_order: ["00", "10", "01", "11"]
      values:
        "00": EtOH 0%, 0 nM cipro
        "10": EtOH 3%, 0 nM cipro
        "01": EtOH 0%, 100 nM cipro
        "11": EtOH 3%, 100 nM cipro
      case_sensitive: true
```

The experiment still needs its `experiment`, `resources`, ingest, and metadata
sections. Keep the ordered state space in `annotations`; do not add a hand-authored
`transform/sfxi` step or duplicate `treatment_map` in protocol inputs.

`analysis.sfxi_objective.intensity_log2_offset_delta` is compiled into vec8
generation and the setpoint scorer. Keeping it in one semantic field prevents
the two paths from using different intensity inverses.

## Preflight without writing

```bash
uv run reader validate <config-or-experiment>
uv run reader inspect <config-or-experiment> --format json
uv run reader explain <config-or-experiment> --format json
uv run reader run <config-or-experiment> --dry-run --format json
```

Use these surfaces to confirm the protocol, ordered state space, selected records, and
optional dependency checks before execution. `reader explain` should show a
transform named `sfxi_vec8` writing `sfxi_vec8/vec8`.

## Run and verify the record

```bash
uv run reader run <config-or-experiment>
uv run reader records <config-or-experiment> --format json
uv run reader verify <config-or-experiment> --format json
```

The run writes generated records under the experiment's `outputs/` directory.
The vec8 dataframe is a manifest-backed Parquet artifact with contract
`sfxi.vec8.v3`. Use `records` to discover it and `verify` to check recorded
digests; do not guess an artifact path.

## Export a workbook

When vec8 and export generation are enabled, the protocol compiles the semantic
export `logic_summary_workbook`. Inspect and preflight it before writing:

```bash
uv run reader export <config-or-experiment> --list --format json
uv run reader export <config-or-experiment> --dry-run
uv run reader export <config-or-experiment> --only logic_summary_workbook
```

The default workbook path is `outputs/exports/sfxi/vec8.xlsx`, with a `vec8`
worksheet. The typed record remains the source for Reader automation; the
workbook is an explicit presentation/export artifact.

## Open the SFXI notebook

```bash
uv run reader notebook <config-or-experiment> \
  --template notebook/sfxi_eda \
  --mode run
```

The template reads experiment records and supports interactive vec8 review. Its
acquisition-time selector lists only times observed across every required SFXI
state, so each selection is a valid snapshot. Time-series lines show means with
95% bootstrap confidence intervals. Snapshot panels show exact well values as
hollow points, a short mean line, and sample-standard-deviation whiskers. A
selected time produces a non-persistent review table; it does not replace
`sfxi_vec8/vec8` or write a second handoff. Use `reader export` when a durable
workbook is required.

Reader scaffolds the notebook under `outputs/notebooks/`, so change the template
or a hand-authored notebook when behavior must be durable. Do not hand-edit a
generated scaffold as the package implementation.

## Operating sequence

1. Validate metadata, state-space values, channels, and reference identity.
2. Inspect or explain the compiled record flow.
3. Dry-run the pipeline.
4. Run the experiment and inspect `sfxi_vec8/vec8`.
5. List and generate only the plots, exports, or notebook needed for the
   selected review.
6. Regenerate outputs after config or code changes; do not patch files under
   `outputs/`.

For the package-wide command contract, see
[Preflight, run, verify](../../guides/preflight_run_verify.md).
