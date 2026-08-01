---
doc_id: reader-four-state-vector-workflow
surface: operator-guide
owner: reader-maintainers
last_verified: 2026-08-01
summary: Configure, preflight, run, inspect, export, and open notebooks for a Reader four-state vector experiment.
---

# Four-state vector experiment workflow

This guide covers the Reader-owned path from `reader/v8` configuration to the
typed vector record. For calculation details, use the
[vector contract](./vector.md). For optional figures, use
[four-state vector plot surfaces](./plots.md). For the shared plate-reader
ingest boundary, use [Plate-reader metric outputs](../plate_reader/metric_outputs.md).

## Configure the protocol

`logic/four_state_vector_screen` is the concrete dual-reporter plate-reader
adapter. It materializes `YFP/CFP` for logic shape and `YFP/OD600` for anchored
intensity, then passes those channels to the four-state vector transform. State
labels and the reference identity remain experiment-authored.

```yaml
schema: reader/v8
protocol:
  id: logic/four_state_vector_screen
  inputs:
    design_by: [design_id]
    state_map_ref: logic_states
    reference:
      design_id: reference
      observation_stat: mean
    target_time_h: 12.0
    time_mode: nearest
    time_tolerance_h: 0.5
  analysis:
    include_four_state_vector: true
    include_export: true
    four_state_vector:
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
`transform/four_state_vector` step or duplicate `treatment_map` in protocol inputs.

`analysis.four_state_vector.intensity_log2_offset_delta` is compiled into vector
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
transform named `four_state_vector` writing `four_state_vector/vector`.

## Run and verify the record

```bash
uv run reader run <config-or-experiment>
uv run reader records <config-or-experiment> --format json
uv run reader verify <config-or-experiment> --format json
```

The run writes generated records under the experiment's `outputs/` directory.
The vector dataframe is a manifest-backed Parquet artifact with contract
`logic.four_state_vector.v1`. Use `records` to discover it and `verify` to check recorded
digests; do not guess an artifact path.

## Export a workbook

When vector and export generation are enabled, the protocol compiles the semantic
export `logic_summary_workbook`. Inspect and preflight it before writing:

```bash
uv run reader export <config-or-experiment> --list --format json
uv run reader export <config-or-experiment> --dry-run
uv run reader export <config-or-experiment> --only logic_summary_workbook
```

The default workbook path is `outputs/exports/four_state_vector/vector.xlsx`, with a `vector`
worksheet. The typed record remains the source for Reader automation; the
workbook is an explicit presentation/export artifact.

## Review persisted four-state vector diagnostics

```bash
uv run reader plot <config-or-experiment> --only four_state_vector_diagnostic
uv run reader verify <config-or-experiment> --format json
uv run reader notebook <config-or-experiment> --mode run
```

The diagnostic is a normal plot step. It reads the manifest-backed
`promote_to_tidy_plus_map/df` and `four_state_vector/vector` records, renders one artifact
per persisted design by default, and uses each vector row's `time_selected_h` as
the trajectory marker. The first two panels show growth and response
trajectories with descriptive resampling intervals. The remaining panels show
the persisted logic-shape and relative-intensity components on separate scales.
It does not choose a new time or recompute vector.

The canonical `notebook/eda` workbench discovers the resulting plot bundle and
the vector dataframe through the record catalog. Use `reader export` when a
durable workbook is required. Do not hand-edit generated plot or notebook
artifacts.

## Operating sequence

1. Validate metadata, state-space values, channels, and reference identity.
2. Inspect or explain the compiled record flow.
3. Dry-run the pipeline.
4. Run the experiment and inspect `four_state_vector/vector`.
5. List and generate only the plots, exports, or notebook needed for the
   selected review.
6. Regenerate outputs after config or code changes; do not patch files under
   `outputs/`.

For the package-wide command contract, see
[Preflight, run, verify](../../guides/preflight_run_verify.md).
