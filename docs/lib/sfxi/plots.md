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

`reader aggregate-sfxi-vec8` combines typed vec8 records or explicitly named
vec8 tables. The aggregate is itself an experiment-owned unit of work.

```bash
uv run reader init experiments/2026/20260708_vec8_aggregate \
  --protocol workbench/generic \
  --title "Vec8 aggregate"

uv run reader aggregate-sfxi-vec8 \
  EXPERIMENT_OR_VEC8_TABLE [EXPERIMENT_OR_VEC8_TABLE ...] \
  --output-experiment experiments/2026/20260708_vec8_aggregate
```

Reader writes the heatmap, tidy CSV, and digest-bearing manifest below that
experiment's `outputs/` directory. It rejects ambiguous sources, incompatible
contracts, missing columns, negative offsets, incomplete vectors, and
unconfined destinations.

## Related references

- [SFXI vec8 contract](vec8.md)
- [SFXI workflow](workflow.md)
- [Plate-reader metric outputs](../plate_reader/metric_outputs.md)
