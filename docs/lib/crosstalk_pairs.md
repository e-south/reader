## Crosstalk pairs pipeline step

### Contents

- [Inputs](#inputs)
- [Time selection](#time-selection)
- [Mapping modes](#mapping-modes)
- [Filters and scoring](#filters-and-scoring)
- [Outputs](#outputs)
- [Example](#example)
- [Pipeline run example (pairwise)](#pipeline-run-example-pairwise)
- [Groups of 3 (triads) from passing pairs](#groups-of-3-triads-from-passing-pairs)
- [Export to the experiment exports directory](#export-to-the-experiment-exports-directory)
- [See also](#see-also)

This document describes the low-level crosstalk pairing utilities in `reader.lib.crosstalk.pairs`. Use this when you want bespoke control or to embed the logic outside the pipeline plugin.

The library computes:
- Per-design selectivity summary (top-1 vs top-2 treatment response).
- Pairwise crosstalk compatibility between designs.

---

### Inputs

You supply a fold-change table (typically `fold_change.v1`) with at least:
- `design_col` (default `design_id`)
- `treatment_col` (default `treatment`)
- `value_col` (e.g., `log2FC`)
- `time_column` (default `time`)
- Optional `target` column (for multi-target experiments)

Rows are aggregated per (design, treatment) before analysis using `agg`.
If you pass `target`, the table must include a `target` column.

---

### Time selection

Use `time_mode` to control how times are selected:
- `single`: require exactly one time in the table.
- `exact`: use exact values provided via `time` or `times`.
- `nearest`: snap to nearest available time (within `time_tolerance`).
- `latest`: use the latest time in the table.
- `all`: evaluate every time in the table.

If multiple times are evaluated, `time_policy: all` keeps only pairs that pass at every time.
Time selection is strict to avoid mismatches across pipeline steps.

---

### Mapping modes

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

### Filters and scoring

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

### Outputs

`compute_crosstalk_pairs` returns a `CrosstalkResult`:
- `pairs`: pairwise table with `design_a`, `design_b`, self/cross values,
  self-vs-other metrics, and pass/fail flags.
- `designs`: per-design summary table with top1/top2 and selectivity info.
- `times_used`: list of times actually evaluated.
- `target_used`, `value_column`, `value_scale`.

---

### Example config

```python
from reader.lib.crosstalk.pairs import compute_crosstalk_pairs

result = compute_crosstalk_pairs(
    df,
    design_col="design_id",
    treatment_col="treatment",
    value_col="log2FC",
    value_scale="log2",
    time_mode="exact",
    time=12.0,
    mapping_mode="explicit",
    design_treatment_map={
        "design_a": "treatment_1",
        "design_b": "treatment_2",
    },
    min_self=1.0,
    max_cross=0.5,
    min_selectivity_delta=1.0,
    require_self_is_top1=True,
)

pairs = result.pairs
per_design = result.designs
```

---

### Pipeline run example (pairwise)

Run the step directly (starting from fold-change):

```bash
uv run reader run experiments/2025/20250620_sensor_panel_crosstalk/config.yaml --from fold_change__yfp_over_cfp --until crosstalk_pairs
```

Artifact location:

```
experiments/2025/20250620_sensor_panel_crosstalk/outputs/artifacts/crosstalk_pairs.crosstalk_pairs/table.parquet
```

Quick readout from that run:
- Evaluated time: 12.0 (only time in table; time_mode=all)
- Designs: 6 -> candidate pairs: 15 -> passing pairs: 2 (only_passing: true)
- Table shape: 2 rows x 37 columns

Example row (interpreted):

1) pDual-10-soxSp <-> pDual-10-spyp
   - **Self** (log2FC): 6.11 / 4.46 (~69x / ~22x)
   - **Cross:** 0.392 / -0.052 (~1.31x / ~0.96x)
   - **pair_score** = 4.07 -> pair_ratio ~= 16.8x separation

Interpretation: both designs show strong response to their own treatments and minimal response to each other's treatments.

---

### Groups of 3 (triads) from passing pairs

There is no built-in `crosstalk_groups` step yet, but you can lift triads from the pair table.
This example finds groups of 3 where **all three pairwise edges pass**:

```bash
uv run python - <<'PY'
from itertools import combinations
import pandas as pd

path = "experiments/2025/20250620_sensor_panel_crosstalk/outputs/artifacts/crosstalk_pairs.crosstalk_pairs/table.parquet"
df = pd.read_parquet(path)

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

### Export to the experiment exports directory

Add an export spec (relative to `outputs/exports/`):

```yaml
exports:
  specs:
    - id: export_crosstalk_pairs
      uses: export/csv
      reads: { df: crosstalk_pairs/table }
      with: { path: "crosstalk_pairs.csv" }
```

Then run:

```bash
uv run reader export experiments/2025/20250620_sensor_panel_crosstalk/config.yaml
```

### See also

- Pipeline usage: `docs/core/plugins.md` (transform/crosstalk_pairs)
