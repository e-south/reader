---
doc_id: reader-crosstalk-pairs
surface: library-reference
owner: reader-maintainers
last_verified: 2026-07-10
summary: Contract, computation, and output reference for Reader crosstalk-pair ranking.
---

# Crosstalk pairs

## Contents

- [Inputs](#inputs)
- [Time selection](#time-selection)
- [Mapping modes](#mapping-modes)
- [Filters and scoring](#filters-and-scoring)
- [Outputs](#outputs)
- [Example config](#example-config)
- [Pipeline run example (pairwise)](#pipeline-run-example-pairwise)
- [Groups of 3 (triads) from passing pairs](#groups-of-3-triads-from-passing-pairs)
- [Export to the experiment exports directory](#export-to-the-experiment-exports-directory)
- [See also](#see-also)

This document describes the crosstalk-pair transform compiled by
`plate_reader/dual_reporter_screen`. Reader owns the generic pairwise math;
integrations configure it through `reader/v8` and consume its typed record
through `reader.api` rather than importing implementation modules.

The library computes:
- Per-design selectivity summary (top-1 vs top-2 treatment response).
- Pairwise crosstalk compatibility between designs.

---

## Inputs

You supply a fold-change table (typically `fold_change.v1`) with at least:
- `design_col` (default `design_id`)
- `treatment_col` (default `treatment`)
- `value_col` (e.g., `log2FC`)
- `time_column` (default `time`)
- Optional `target` column (for multi-target experiments)

Rows are aggregated per (design, treatment) before analysis using `agg`.
If you pass `target`, the table must include a `target` column.

---

## Time selection

Use `time_mode` to control how times are selected:
- `single`: require exactly one time in the table.
- `exact`: use exact values provided via `time` or `times`.
- `nearest`: snap to nearest available time (within `time_tolerance`).
- `latest`: use the latest time in the table.
- `all`: evaluate every time in the table.

If multiple times are evaluated, `time_policy: all` keeps only pairs that pass at every time.
Time selection is strict to avoid mismatches across pipeline steps.

---

## Mapping modes

Choose how each design maps to its "self" treatment:
- `explicit`: pass `design_treatment_map` (recommended when you have ground truth).
- `column`: pass `design_treatment_column` in the data (one value per design).
- `top1`: derive mapping from the highest response in the data.

Notes:
- `explicit` forbids `design_treatment_column` and `design_treatment_overrides`.
- `top1` forbids all explicit mapping inputs; ties are handled by
  `top1_tie_policy` + `top1_tie_tolerance`.
- Mapping values must exist in the `treatment_col` values.

---

## Filters and scoring

Pairs are evaluated using these criteria:
- `min_self`: minimum self response for each design.
- `max_cross`: maximum allowed cross response between treatments.
- `max_other`: maximum allowed response to any non-self treatment.
- `min_self_minus_best_other`: minimum (self - best_other) per design.
- `min_self_ratio_best_other`: minimum self/best_other ratio per design.
- `min_selectivity_delta`: minimum top1 - top2 delta for each design.
- `min_selectivity_ratio`: minimum top1/top2 ratio for each design.
- `require_self_treatment`: require both self and cross values to exist.
- `require_self_is_top1`: require each design's mapped treatment to be top-1.

Scores:
- `selectivity_delta` and `selectivity_ratio` are computed per design.
- `pair_score` is always **min(self) - max(cross)** (in the units of `value_scale`).
- `pair_ratio` expresses separation as a ratio.

`value_scale` controls ratio calculations:
- `log2`: ratios are computed as powers of 2 (e.g., `2 ** delta`).
- `linear`: ratios are computed directly (e.g., `top1 / top2`).

---

## Outputs

The transform writes `crosstalk_pairs/table`, containing:
- `pairs`: pairwise table with `design_a`, `design_b`, self/cross values,
  self-vs-other metrics, and pass/fail flags.
- `designs`: per-design summary table with top1/top2 and selectivity info.
- the evaluated time, target, value column, and scale on each relevant row.

---

## Example config

```yaml
protocol:
  id: plate_reader/dual_reporter_screen
  analysis:
    include_fold_change: true
    crosstalk_pairs:
      enabled: true
      export: true
      time_mode: exact
      time: 12.0
      mapping_mode: explicit
      design_treatment_map:
        design_a: treatment_1
        design_b: treatment_2
      min_self: 1.0
      max_cross: 0.5
      min_selectivity_delta: 1.0
      require_self_is_top1: true
```

---

## Pipeline run example (pairwise)

Run the step directly (starting from fold-change):

```bash
uv run reader run experiments/my_experiment --until crosstalk_pairs
uv run reader verify experiments/my_experiment
```

---

## Groups of 3 (triads) from passing pairs

There is no built-in `crosstalk_groups` step yet, but you can lift triads from the pair table.
This example finds groups of 3 where **all three pairwise edges pass**:

```bash
uv run python - <<'PY'
from itertools import combinations
from reader.api import open_experiment, read_dataframe, verify

experiment = open_experiment("experiments/my_experiment")
verification = verify(experiment)
if verification.status != "ok":
    raise RuntimeError(f"Reader verification failed: {verification.issues}")
df = read_dataframe(experiment, "crosstalk_pairs/table").dataframe

if "passes_filters" in df.columns:
    df = df[df["passes_filters"]]

edges = {tuple(sorted(pair)) for pair in zip(df["design_a"], df["design_b"])}
designs = sorted(set(df["design_a"]).union(df["design_b"]))

triads = []
for a, b, c in combinations(designs, 3):
    if (a, b) in edges and (a, c) in edges and (b, c) in edges:
        triads.append((a, b, c))

print("triads:", triads)
PY
```

---

## Export to the experiment exports directory

Bind the export through protocol outputs (relative to `outputs/exports/`):

```yaml
protocol:
  id: plate_reader/dual_reporter_screen
  analysis:
    crosstalk_pairs:
      enabled: true
      export: true
  outputs:
    exports:
      include: [crosstalk_pairs_table]
      artifacts:
        crosstalk_pairs_table:
          path: crosstalk_pairs.csv
```

Then run:

```bash
uv run reader export experiments/my_experiment
```

## See also

- [Pipeline configuration](../core/pipeline.md)
- [Python API](../core/python_api.md)
