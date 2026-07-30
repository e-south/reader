---
doc_id: reader-sfxi-vec8
surface: library-router
owner: reader-maintainers
last_verified: 2026-07-29
summary: Canonical route to Reader SFXI vec8 semantics, operation, and plots.
---

# SFXI in Reader

Reader converts annotated four-state measurements into one measured SFXI vec8
row per `design_id`. It persists the result as `sfxi_vec8/vec8` under the
`sfxi.vec8.v3` contract.

## Choose a route

- [Vec8 contract](./sfxi/vec8.md): input requirements, ordered-state
  selection, reference normalization, equations, output columns, and failures.
- [Experiment workflow](./sfxi/workflow.md): configuration, preflight,
  execution, records, workbook export, and canonical EDA review.
- [Plot surfaces](./sfxi/plots.md): per-design diagnostics plus per-experiment
  and aggregate heatmaps.
- [Plate-reader metric outputs](./plate_reader/metric_outputs.md): the shared
  measurement boundary and independent analysis lanes.

## Stable terms

| Term | Meaning | Owner |
| --- | --- | --- |
| SFXI vec8 | Measured `[v00, v10, v01, v11, y00_star, y10_star, y01_star, y11_star]` row | Reader |
| `design_id` | Experiment-scoped grouping identity | Experiment config |
| ordered state space | Exact mapping from observed labels to `00, 10, 01, 11` | Experiment config |
| `sequence` | Optional carried metadata; never identity authority | Source experiment |

Reader stops at measured records and visual review. Any external
classification, optimization, or campaign interpretation consumes the public
record and remains outside this package.

## Shortest path

```bash
uv run reader validate <config-or-experiment>
uv run reader run <config-or-experiment> --dry-run --format json
uv run reader run <config-or-experiment>
uv run reader records <config-or-experiment> --format json
```

Generated artifacts stay below the owning experiment's `outputs/` directory
and are regenerated through Reader rather than edited by hand.
