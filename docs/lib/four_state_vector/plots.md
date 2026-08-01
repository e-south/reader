---
doc_id: reader-four-state-vector-plots
surface: plot-reference
owner: reader-maintainers
last_verified: 2026-08-01
summary: Reader-owned four-state vector plot inputs, outputs, and failure boundaries.
---

# Four-state vector plot surfaces

Reader renders four-state vector figures only from typed Reader records. Plot
plugins own configuration and publication; the four-state vector domain owns
measurement calculations.

## Per-design diagnostic

`four_state_vector_diagnostic` reads both `promote_to_tidy_plus_map/df` and
`four_state_vector/vector`. It renders one artifact per persisted vector design by default.
Each figure shows growth and response trajectories, marks the exact
`time_selected_h` carried by that design's vector row, and displays the persisted
logic-shape and relative-intensity components on separate scales. The plot does
not select another time or recompute vector.

Generate the diagnostic and inspect it in the canonical record-driven
workbench:

```bash
uv run reader plot <config-or-experiment> --only four_state_vector_diagnostic
uv run reader verify <config-or-experiment> --format json
uv run reader notebook <config-or-experiment> --mode run
```

To render a focused subset, use the semantic figure view:

```yaml
protocol:
  id: logic/four_state_vector_screen
  outputs:
    plots:
      profile: none
      include: [four_state_vector_diagnostic]
      views:
        four_state_vector_diagnostic:
          design_ids: [design-a, design-b]
          trajectory_interval_mass: 0.95
          trajectory_resamples: 300
          format: [png]
```

These intervals describe observation dispersion; they do not establish a
replicate relationship among positions.

## Per-experiment heatmap

`four_state_vector_heatmap` reads `four_state_vector/vector` and writes
`outputs/plots/four_state_vector_heatmap.pdf` by default. Each row is one
`design_id`. The first four columns show `v00`, `v10`, `v01`, and
`v11` on a unit-interval scale. The remaining columns show
`y00_star`, `y10_star`, `y01_star`, and `y11_star` on a separate log2
scale centered at zero.

Select the plot explicitly:

```yaml
protocol:
  id: logic/four_state_vector_screen
  outputs:
    plots:
      profile: none
      include: [four_state_vector_heatmap]
```

Then preflight and run it:

```bash
uv run reader plot <config-or-experiment> --dry-run
uv run reader plot <config-or-experiment> --only four_state_vector_heatmap
```

## Cross-experiment heatmap

`logic/four_state_vector_collection` collects typed vector records. The
collection is itself an experiment-owned unit of work.

```bash
uv run reader init experiments/collection \
  --protocol logic/four_state_vector_collection \
  --title "Four-state vector collection"
# Declare record resources and protocol.inputs.record_resources in config.yaml.
uv run reader validate experiments/collection
uv run reader run experiments/collection
uv run reader verify experiments/collection
uv run reader plot experiments/collection
uv run reader export experiments/collection
uv run reader notebook experiments/collection --mode none
```

`reader run` writes the collection record under the
`logic.four_state_vector_collection.v1` contract and a digest-bearing manifest. The
explicit plot, export, and notebook commands materialize their own requested
surfaces below the same experiment's `outputs/` directory. Reader rejects
unknown experiments, missing or changed records, incompatible contracts,
duplicate upstream `(experiment, record)` identities, missing columns,
negative offsets, and incomplete vectors. The collection table keeps the
consumer-local alias (`source_resource_id`) separate from the upstream
`source_experiment_id`, `source_record_id`, and exact
`source_record_revision_digest`.

## Related references

- [Four-state vector contract](vector.md)
- [Four-state vector workflow](workflow.md)
- [Plate-reader metric outputs](../plate_reader/metric_outputs.md)
