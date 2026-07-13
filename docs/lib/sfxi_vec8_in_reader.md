---
doc_id: reader-sfxi-vec8
surface: library-router
owner: reader-maintainers
last_verified: 2026-07-11
summary: Canonical route to Reader SFXI vec8 semantics, operation, plots, and the OPAL handoff boundary.
---

# SFXI in Reader

Reader converts annotated plate-reader measurements into one measured
Setpoint Fidelity x Intensity (SFXI) 8-vector per `design_id`. It persists that table as the typed record
`sfxi_vec8/vec8` with contract `sfxi.vec8.v3`.

This page is the stable entry point. Detailed reference material is split by
task so calculation rules, operating steps, plot behavior, and downstream
ownership can change independently.

## Choose a route

- [Vec8 contract](./sfxi/vec8.md): input requirements, time and corner
  selection, reference normalization, equations, output columns, and failure
  rules.
- [Plate-reader response-window analysis](./plate_reader/response_window.md):
  event-relative response and reference-relative fluorescence records.
- [Experiment workflow](./sfxi/workflow.md): `reader/v7` configuration,
  preflight, execution, records, workbook export, and the SFXI marimo notebook.
- [Plot surfaces](./sfxi/plots.md): the per-experiment heatmap, setpoint
  scatter, triptych sequence bundle, and cross-experiment aggregate heatmap.
- [Reader-to-OPAL handoff](./plate_reader/opal_handoff.md): ownership of
  `design_id`, sequence/X readiness, batch0 staging, OPAL validation, and
  mutation boundaries.

## Aggregate vec8 heatmap

For multi-experiment measured vec8 review, use the
[cross-experiment aggregate heatmap](./sfxi/plots.md#cross-experiment-aggregate-heatmap).
The aggregate command consumes typed Reader records or explicitly named table
snapshots; it does not run OPAL scoring.

## Stable terms

| Term | Meaning in this workflow | Owner |
| --- | --- | --- |
| SFXI vec8 | Measured `[v00, v10, v01, v11, y00_star, y10_star, y01_star, y11_star]` row | Reader |
| `design_id` | Reader experiment identity used to group measurements | Reader |
| `sequence` on vec8 | Optional experiment metadata carried with a measured row | Reader |
| sequence/X readiness | Exact study join to sequence, candidate identity, and OPAL X | Owning study and OPAL candidate table |
| batch0 | Study-owned round-0 label input and evidence manifest | Owning study |
| scalar SFXI objective | OPAL scoring over a measured vec8 | OPAL |
| response window | Event-relative well reductions and reference-relative fluorescence summaries | Reader |

A Reader `design_id` is not automatically an OPAL candidate ID. A Reader
alias or `sequence` value also does not replace the study-owned sequence/X
readiness check.

## Shortest Reader path

```bash
uv run reader validate <config-or-experiment>
uv run reader explain <config-or-experiment> --format json
uv run reader run <config-or-experiment> --dry-run --format json
uv run reader run <config-or-experiment>
uv run reader records <config-or-experiment> --format json
```

Use `reader plot --list`, `reader export --list`, and `reader notebook
--list-templates` to discover optional outputs without writing them. Generated
files under `experiments/**/outputs/` are regenerated through Reader and are
not hand-edited.

For the general execution contract, see
[Preflight, run, verify](../guides/preflight_run_verify.md).
