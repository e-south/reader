---
doc_id: reader-sfxi-plots
surface: plot-reference
owner: reader-maintainers
last_verified: 2026-07-29
summary: Reader-owned SFXI vec8 plot inputs, outputs, and failure boundaries.
---

# SFXI plot surfaces

Reader renders SFXI figures only from typed Reader records. Plot plugins own
configuration and publication; the SFXI domain owns measurement calculations.

## Per-design diagnostic

`sfxi_diagnostic` reads both `promote_to_tidy_plus_map/df` and
`sfxi_vec8/vec8`. It renders one artifact per persisted vec8 design by default.
Each figure shows growth and response trajectories, marks the exact
`time_selected_h` carried by that design's vec8 row, and displays the persisted
logic-shape and relative-intensity components on separate scales. The plot does
not select another time or recompute vec8.

Generate the diagnostic and inspect it in the canonical record-driven
workbench:

```bash
uv run reader plot <config-or-experiment> --only sfxi_diagnostic
uv run reader verify <config-or-experiment> --format json
uv run reader notebook <config-or-experiment> --mode run
```

To render a focused subset, use the semantic figure view:

```yaml
protocol:
  id: logic/sfxi_screen
  outputs:
    plots:
      profile: none
      include: [sfxi_diagnostic]
      views:
        sfxi_diagnostic:
          design_ids: [design-a, design-b]
          trajectory_interval_mass: 0.95
          trajectory_resamples: 300
          format: [png]
```

These intervals describe observation dispersion; they do not establish a
replicate relationship among positions.

## Per-experiment heatmap

`sfxi_vec8_heatmap` reads `sfxi_vec8/vec8` and writes
`outputs/plots/sfxi_vec8_heatmap.pdf` by default. Each row is one
`design_id`. The first four columns show `v00`, `v10`, `v01`, and
`v11` on a unit-interval scale. The remaining columns show
`y00_star`, `y10_star`, `y01_star`, and `y11_star` on a separate log2
scale centered at zero.

Select the plot explicitly:

```yaml
protocol:
  id: logic/sfxi_screen
  outputs:
    plots:
      profile: none
      include: [sfxi_vec8_heatmap]
```

Then preflight and run it:

```bash
uv run reader plot <config-or-experiment> --dry-run
uv run reader plot <config-or-experiment> --only sfxi_vec8_heatmap
```

## Cross-experiment heatmap

`logic/sfxi_vec8_collection` combines typed vec8 records. The aggregate is
itself an experiment-owned unit of work.

```bash
uv run reader init experiments/vec8_aggregate \
  --protocol logic/sfxi_vec8_collection \
  --title "Vec8 aggregate"
# Declare record resources and protocol.inputs.record_resources in config.yaml.
uv run reader validate experiments/vec8_aggregate
uv run reader run experiments/vec8_aggregate
uv run reader verify experiments/vec8_aggregate
uv run reader plot experiments/vec8_aggregate
uv run reader export experiments/vec8_aggregate
uv run reader notebook experiments/vec8_aggregate --mode none
```

`reader run` writes the collection record under the
`sfxi.vec8_collection.v2` contract and a digest-bearing manifest. The
explicit plot, export, and notebook commands materialize their own requested
surfaces below the same experiment's `outputs/` directory. Reader rejects
unknown experiments, missing or changed records, incompatible contracts,
duplicate upstream `(experiment, record)` identities, missing columns,
negative offsets, and incomplete vectors. The collection table keeps the
consumer-local alias (`source_resource_id`) separate from the upstream
`source_experiment_id`, `source_record_id`, and exact
`source_record_revision_digest`.

## Related references

- [SFXI vec8 contract](vec8.md)
- [SFXI workflow](workflow.md)
- [Plate-reader metric outputs](../plate_reader/metric_outputs.md)
