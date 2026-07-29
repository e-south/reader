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

`logic/sfxi_screen` is the concrete dual-reporter plate-reader adapter. It
materializes `YFP/CFP` for logic shape and `YFP/OD600` for anchored intensity,
then passes those channels to the generic SFXI transform. State labels and the
reference identity remain experiment-authored.

```yaml
schema: reader/v8
protocol:
  id: logic/sfxi_screen
  inputs:
    design_by: [design_id]
    state_map_ref: logic_states
    reference:
      design_id: reference
      stat: mean
    target_time_h: 12.0
    time_mode: nearest
    time_tolerance_h: 0.5
  analysis:
    include_vec8: true
    include_export: true
    sfxi_vec8:
      intensity_log2_offset_delta: 0.0

annotations:
  ordered_state_spaces:
    logic_states:
      column: condition
      state_order: ["00", "10", "01", "11"]
      values:
        "00": baseline
        "10": input_a
        "01": input_b
        "11": input_a_and_b
      case_sensitive: true
```

The experiment still needs its `experiment`, `resources`, ingest, and metadata
sections. Its ingest channel list must include `OD600`, `CFP`, and `YFP`. Keep
the ordered state space in `annotations`; do not add a hand-authored
`transform/sfxi` step or duplicate `treatment_map` in protocol inputs.

`analysis.sfxi_vec8.intensity_log2_offset_delta` is compiled into vec8
generation and persisted in the typed record. Downstream consumers can verify
that field before applying any objective that inverts the intensity transform.

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
