---
doc_id: reader-sfxi-vec8
surface: library-reference
owner: reader-maintainers
last_verified: 2026-07-08
summary: Reader-owned SFXI vec8 generation, records, plots, exports, and aggregate heatmap review.
---

## Generating SFXI 8-vectors in `reader`

This document describes how **reader** processes **Setpoint Fidelity x Intensity** (SFXI) 8-vectors from experimental measurements. The objective/scalar spec is outside of reader and is owned by **dnadesign** (see `dnadesign/src/dnadesign/opal/docs/plugins/objectives/sfxi.md`). The process here involves collecting microplate reader data, selecting a timepoint, and then deriving an 8‑vector per *design_id* in the fixed state order **00, 10, 01, 11**.

8-vector definition:

```
[v00, v10, v01, v11, y00_star, y10_star, y01_star, y11_star]
```

* `v..` encodes the **logic shape** in `[0,1]` (derived from a **logic channel**, e.g. `YFP/CFP`).
* `y*_..` encodes **absolute intensity**, normalized to a reference design_id and stored in **log2** (derived from an **intensity channel**, e.g. `YFP/OD600`).

### Contents

1. [Reader hand-off to OPAL](#reader-hand-off-to-opal)
2. [Scope (vec8 vs objective)](#scope-vec8-vs-objective)
3. [Relevant modules](#relevant-modules)
4. [Input contract](#input-contract)
5. [Time selection](#time-selection)
6. [Corner mapping](#corner-mapping)
7. [Logic channel](#logic-channel)
8. [Intensity channel](#intensity-channel)
9. [Output](#output)
10. [Setpoint scatter plot](#setpoint-scatter-plot)
11. [Aggregate vec8 heatmap](#aggregate-vec8-heatmap)
12. [Configuration entry point](#configuration-entry-point)
13. [Usage demo](#usage-demo)

---

### reader hand-off to OPAL

- **reader** is the source of truth for vec8 math. It computes:
  * `u = log2(ratio)` for logic, then per-design min–max into `v ∈ [0,1]`.
  * `y* = log2(y_linear + delta)` for intensity, where `y_linear` is reference-normalized.

- **OPAL is the source of truth for the scalar objective.** It expects `y*_..` columns already in log2 and converts back to linear using its `intensity_log2_offset_delta` parameter.
  * **Important:** The `delta` used in **reader** (`log2_offset_delta`) must match OPAL’s `intensity_log2_offset_delta`.
  * **Reader makes this explicit** by writing an `intensity_log2_offset_delta` column into every vec8 row. When `log2_offset_delta` is left at its default (`0.0`), this column will be all zeros.
  * **If OPAL uses a different delta, recovered linear intensities and downstream scores will be inconsistent.** Keep the values in sync (preferably by validating against the vec8 column at ingest time).
  * Reader plot code imports only the public scoring boundary `dnadesign.opal.api.sfxi`, never OPAL internals.

- The **reader** transform plugin (`src/reader/plugins/transform/sfxi.py`) delegates to `reader.domains.logic.sfxi.*` and adds pipeline plumbing and logging.

---

### Scope (vec8 vs objective)

Reader owns measured vec8 generation from plate-reader data, the typed
`sfxi_vec8/vec8` record, reader plots over that record, workbook export, and
the aggregate heatmap described below.

Reader does not own OPAL campaign scoring, active-learning round scaling,
selection policy, or OPAL ledger plots. Those semantics stay in dnadesign. When
reader needs scalar SFXI scores for a plot, it calls the public
`dnadesign.opal.api.sfxi` boundary and keeps OPAL internals out of reader.

---

### Relevant modules

Key modules in `src/reader/domains/logic/sfxi/`:

* **Selection + cornerization + aggregation:** `selection.py`

  * `cornerize_and_aggregate(...)`
  * `select_times(...)`
* **Vec8 math:** `math.py`

  * `compute_vec8(...)`
* **Reference label resolution:** `reference.py`

  * `resolve_reference_design_id(...)`
* **Config loader (canonical):** `api.py`

  * `load_sfxi_config(...)`
* **Orchestration + output writing:** `run.py` and `writer.py`

  * `build_vec8_from_tidy(...)`, `run_sfxi(...)`
  * `write_outputs(...)`
* **Setpoint scoring plot prep/rendering:** `setpoint_scatter.py`

  * `score_sfxi_setpoints(...)`
  * `render_sfxi_setpoint_scatter(...)`
* **Single-experiment vec8 heatmap adapter:** `vec8_heatmap.py`

  * `normalize_experiment_vec8_heatmap_frame(...)`
  * `render_experiment_sfxi_vec8_heatmap(...)`
* **Cross-experiment aggregate heatmap:** `reader.domains.logic.sfxi.vec8_aggregate`

  * `load_sfxi_vec8_sources(...)`
  * `write_sfxi_vec8_aggregate(...)`

---

### Input contract

SFXI consumes an **annotated plate-reader** DataFrame (the typical source is the `validator/to_tidy_plus_map` dataframe record).

#### Required columns

The selector enforces the following base columns (see `selection.REQUIRED_COLS`
and `_enforce_columns`):

* `position` — well/position identifier (not used in vec8 math, but required for tidy contract consistency)
* `time` — measurement time (numeric; cast to float during selection)
* `channel` — channel name (matched by exact string equality)
* `value` — numeric measurement value

Treatment labels are resolved from `treatment` or `treatment_alias`. At least
one of those columns must contain labels that match the configured logic map.

#### Design identity

* `design_by` — one or more columns that define a design grouping (default: `["design_id"]`)

Notes:

* The **first** `design_by` column (i.e. `design_by[0]`) is treated as the primary design label for:

  * reference design_id lookup
  * sequence attachment
* Reader enforces `design_by[0] == "design_id"` for SFXI to keep outputs and logs consistent.

#### Optional columns

* `sequence` — if present, it is attached into the vec8 output (per `design_by[0]`)

#### Sequence metadata

Reader treats `sequence` as sample metadata, not as a derived SFXI value. When a
study has an authoritative sequence table, fill `sequence` in
`inputs/metadata.xlsx` by exact `design_id` match before regenerating outputs.
Do not infer sequences from partial aliases. If a design is intentionally
outside the downstream candidate/X universe, leave that decision documented in
the study-owned handoff rather than encoding it in reader math.

#### Numerical guards (recorded in logs)

SFXI uses a small set of numerical stabilizers; these values are echoed in `sfxi_log.json`:

* `eps_ratio` (log/ratio floor)
* `eps_range` (flat-logic threshold)
* `eps_ref` (denominator floor for anchors)
* `eps_abs` (numerator additive)
* `ref_add_alpha` (α, additive to anchors)
* `log2_offset_delta` (δ, additive inside the log argument). If this is `0.0` (default), the exported `intensity_log2_offset_delta` column will be all zeros.

---

### Time selection

Time selection happens inside `selection.cornerize_and_aggregate(...)`, which calls `select_times(...)`. It is driven by config:

* `target_time_h`: target snapshot time (float hours).

  * `None` means “use the latest available time”.
* `time_mode`: one of `nearest | last_before | first_after | exact`
* `time_tolerance_h`: soft warning threshold (does **not** change the chosen time)

#### What rows are considered when picking times?

Time selection is performed **after filtering** to:

1. the requested `channel`, and
2. rows whose treatment label matches the configured `treatment_map` values (using either `treatment` or `treatment_alias`, as described in [Corner mapping](#corner-mapping)).

This keeps the “chosen time” decision tied to the same subset of rows that will become the corner aggregates.

#### Selection behavior

* If `target_time_h is None`: choose the **maximum** available time.
* If `time_mode` is:

  * `exact`: require an exact match (using `np.isclose(..., atol=1e-12)` for the target comparison)
  * `nearest`: choose the closest time to target
  * `last_before`: choose the closest time **≤ target**
  * `first_after`: choose the closest time **≥ target**
#### Missing-time policy

If no time can be chosen under the configured mode, SFXI fails immediately with:

`SFXI: could not choose a global time.`

#### Tolerance warnings (soft)

If both `target_time_h` and `time_tolerance_h` are set, the selector records a warning when:

```
abs(time_selected - target_time_h) > time_tolerance_h
```

These warnings are:

* stored on the selection result as `CornerizeResult.time_warning`
* included in the run log payload under `time.out_of_tolerance`
* emitted as runtime warnings by `run.run_sfxi(...)`

#### Same-time requirement across channels

Time selection is run independently for the **logic channel** and the **intensity channel**. After both selections, `run._assert_same_times(...)` enforces that the chosen times match exactly (within `atol=1e-9`).

If your tidy data contains different time grids by channel, you will see an explicit error such as:

> `SFXI: logic and intensity channels selected different times: ...`

---

### Corner mapping

Corner mapping is handled in `selection.cornerize_and_aggregate(...)`.

#### Treatment map

At the library layer, SFXI still consumes a plain `treatment_map` that assigns experimental treatment labels to the four logic corners:

```yaml
with:
  treatment_map:
    "00": <label for 00>
    "10": <label for 10>
    "01": <label for 01>
    "11": <label for 11>
```

* Keys **must be exactly**: `{"00","10","01","11"}` (`api.load_sfxi_config` enforces this).
* Values are the treatment labels expected to appear in the tidy data.

In `reader/v7` experiment configs, you do **not** write `treatment_map`
directly on a hand-authored `transform/sfxi` step. Instead, define the mapping
once under `annotations.logic_maps.<name>` and reference it with
`protocol.inputs.logic_map_ref`. The protocol materializes the lower-level
mapping before running the SFXI transform.

Duplicate values are rejected (after optional normalization) to avoid ambiguous mapping:

* If `treatment_case_sensitive: true`, duplicates are checked on raw strings.
* If `false`, duplicates are checked after `strip()` + `casefold()`.

#### Which column is used: `treatment` vs `treatment_alias`?

If both columns exist, SFXI chooses the one that matches **more** configured `treatment_map` values within the selected channel. This is implemented in `_choose_treatment_column(...)`.

Tie-break rule:

* If both score equally, SFXI prefers the raw `treatment` column.

#### Case sensitivity and normalization

`treatment_case_sensitive` controls whether matching uses:

* exact string equality (`true`), or
* `strip()` + `casefold()` normalization (`false`)

#### After mapping: aggregation to per-corner means

Once a single snapshot time is selected and treatments are mapped to corners, replicate rows are aggregated per:

```
(design_by..., corner)
```

The aggregated per-corner table contains:

* `time` — first time value in the group (after selection, the code asserts there is only one)
* `y_mean` — mean of numeric `value`
* `y_sd` — sample standard deviation (`ddof=1`), with:

  * `0.0` when there is only one replicate
* `y_n` — count of numeric (non-NaN) values

This table is returned as `CornerizeResult.per_corner`.

#### Wide “points” table

SFXI also produces a wide table with one row per *design*:

* corner means:

  * `b00, b10, b01, b11`
* corner standard deviations:

  * `sd00, sd10, sd01, sd11`
* corner counts:

  * `n00, n10, n01, n11`

This table is returned as `CornerizeResult.points` and is the direct input to vec8 computation.

#### Completeness rule: all corners present

If `require_all_corners_per_design: true`, SFXI requires that every *design* has **all four** corners. Otherwise it raises a detailed error listing missing corners.

---

### Logic channel

The **logic channel** is typically a ratio such as `YFP/CFP` (often computed upstream). SFXI uses the per-corner means (`b00..b11`) from `CornerizeResult.points` for the configured `response.logic_channel`.

Let the corner means be:

* `L00, L10, L01, L11` (in linear space)

#### Dynamic range diagnostic: `r_logic`

SFXI reports the dynamic range of the four logic corner means in linear space after an ε guard:

* Guard: `L_i_guard = max(L_i, eps_ratio)`
* Then:

```
r_logic = max(L_i_guard) / min(L_i_guard)
```

This is computed in `math._logic_minmax_from_four(...)` and written to output as:

* `r_logic`
* plus supporting diagnostics:

  * `r_logic_min`, `r_logic_max`
  * `r_logic_corner_min`, `r_logic_corner_max`
  * `logic_span_log2` (defined below)

#### Shape mapping to `[0,1]`: `v00..v11`

To make logic shapes comparable across designs, SFXI performs:

1. **Log2 transform** (with ε guard):

```
u_i = log2(max(L_i, eps_ratio))
```

2. **Flat-logic check**:

```
span = max(u) - min(u)
if span <= eps_range:
    v_i = 0.25   (for all i)
    flat_logic = True
else:
    v_i = (u_i - u_min) / (span + eps_range)
    flat_logic = False
```

Notes:

* The flat-logic value of `0.25` is a neutral, symmetry-preserving choice: it does not imply a preferred corner when there is no measurable separation.
* `logic_span_log2` is the `span` value above (the log2-space separation across corners).
* Reader uses `eps_range` as the denominator guard in the non-flat branch.

All of this logic is implemented in `math._logic_minmax_from_four(...)` and applied per *design* in `math.compute_vec8(...)`.

---

### Intensity channel

The **intensity channel** is typically `YFP/OD600`. It is used to compute the four `y*_..` values after normalizing by a **reference design_id** (“anchor strain”).

#### Reference design_id requirement and label resolution

Reader’s vec8 generation requires a reference design_id label in config:

* `reference.design_id` (required; `reference.genotype` is not supported)

Internally, the configured reference label is resolved to a *raw* design label using `reference.resolve_reference_design_id(...)`:

Policy:

1. If the reference label matches `design_by[0]` values exactly, use it.
2. Otherwise, if `<design_by[0]>_alias` exists and maps uniquely to a raw label, use that raw label.
3. Otherwise, raise a clear error (no silent fallback).

This ensures the anchor is tied to the correct design label even when aliases are used for display.

For the current 2026 SFXI pDual-10 panels, configs use
`reference.design_id: J23105`. The metadata maps the raw `pDual-10` vector row
to alias `J23105`, so reader resolves the configured J23105 reference to the raw
`pDual-10` row before computing anchors. The emitted `reference_design_id`
therefore records the raw design row used for math, while the config preserves
the biological reference label.

#### Anchor computation (per corner)

Anchors are computed from the **intensity** per-corner table (`CornerizeResult.per_corner`) for the reference design_id:

* `reference.stat: "mean" | "median"`

  * applied to the reference’s per-corner `y_mean` values

Missing anchors are treated as **errors**. There is no silent fallback: if an anchor is missing for any corner needed by a sample, vec8 generation fails.

#### Intensity normalization and log2 storage

For each corner `i ∈ {00,10,01,11}`:

* `I_i` = intensity-channel corner mean for the sample (from `points_intensity`)
* `A_i` = reference anchor for that corner (from the reference design_id)
* Config knobs / numerical guards (from `SFXIConfig`):

  * `eps_abs` (added to numerator)
  * `ref_add_alpha` (α, added to anchor in denominator)
  * `eps_ref` (lower bound for denominator)
  * `log2_offset_delta` (δ, added inside the log argument)
  * `eps_ratio` (lower bound for log argument)

The implementation in `math.compute_vec8(...)` matches:

```
denom      = max(A_i + ref_add_alpha, eps_ref)
y_linear_i = (I_i + eps_abs) / denom
log_arg    = y_linear_i + log2_offset_delta
y*_i       = log2(max(log_arg, eps_ratio))
```

The output columns are:

* `y00_star, y10_star, y01_star, y11_star`

All `y*_i` values are in **log2 space**.

---

### Output

Output is written by `reader.domains.logic.sfxi.writer.write_outputs(...)`
(typically via `reader.domains.logic.sfxi.run.run_sfxi(...)`).

#### Files

* `vec8.csv` (or configured `vec8_filename`)
* `sfxi_log.json` (or configured `log_filename`)

By default the output directory is:

* `out_dir / output_subdir` (default `output_subdir: "sfxi"`)

If `filename_prefix` is provided, both filenames are prefixed (e.g. `myrun_vec8.csv`).

#### Vec8 table: key fields and column conventions

`run._reorder_and_filter(...)` reorders columns to put the most-used fields first:

Preferred front matter (when present):

* `design_id`
* `sequence` (attached from tidy data if available; otherwise `NA`)
* `time_selected_h`
* `reference_design_id` (resolved reference label)
* `r_logic`
* `v00, v10, v01, v11`
* `y00_star, y10_star, y01_star, y11_star`
* `flat_logic`

Then all remaining diagnostics and identity columns are preserved (for example):

* `r_logic_min`, `r_logic_max`
* `logic_span_log2`
* `r_logic_corner_min`, `r_logic_corner_max`
* any additional `design_by` columns beyond the first (if configured)

#### Reference row handling

By default, the reference design_id rows are **excluded** from `vec8.csv`:

* `exclude_reference_from_output: true` (default)

This does **not** affect anchor computation: the reference design_id must still be present in the tidy data for anchors to be computed.

#### Log payload (`sfxi_log.json`)

When using `reader.domains.logic.sfxi.run.run_sfxi(...)` entry points, the JSON
log includes:

* resolved channels and config echo
* chosen time (global)
* out-of-tolerance messages (soft warnings)
* eps/alpha/delta parameters
* row counts at each stage
* summary stats for `r_logic`
* reference anchor values

> Note: the library (`run_sfxi`) writes `sfxi_log.json`. If you are using a higher-level transform wrapper, it may choose to surface the same information via console logging and/or pipeline metadata instead of writing a separate JSON file.

See `src/reader/contracts/builtins/` for the canonical contracts referenced by the pipeline (`plate_reader.annotated.v1`, `sfxi.vec8.v2`).

---

### Setpoint scatter plot

The protocol figure `sfxi_setpoint_scatter` consumes the typed `sfxi.vec8.v2`
record at `sfxi_vec8/vec8`, calls the public dnadesign scorer, and writes plot
files through reader's plot sink under `outputs/plots` by default.

Persisted score columns and plot axes keep the OPAL objective channel names:

* `logic_fidelity`
* `effect_scaled`
* `sfxi`

Reader does not persist compatibility aliases such as `f_logic`, `e_scaled`, or
`score` for this plot surface. If `dnadesign.opal.api.sfxi` is unavailable or
has an unsupported `SFXI_API_VERSION`, `reader validate` and
`reader plot --dry-run` report the missing optional dependency before plot
execution. Install or sync `reader[dnadesign]` for this figure.

When the scorer receives an aggregate vec8 table, it preserves source provenance
columns such as `source_id`, `source_path`, `table_path`, `source_kind`,
`source_row_index`, and `row_label` in the scored table. Those fields are
diagnostic metadata only; OPAL still owns the scalar objective math.

---

### Aggregate vec8 heatmap

The command `reader aggregate-sfxi-vec8` renders a reader-owned heatmap over
measured vec8 rows from one or more finished SFXI experiments:

```bash
uv run reader aggregate-sfxi-vec8 \
  experiments/2026/20260706_sfxi_sensor-panel-m9-glu-secg/config.yaml \
  experiments/2026/20260707_sfxi_sensor-panel-m9-glu-secg/config.yaml \
  --out-dir outputs/reviews/sfxi_vec8_aggregate
```

Accepted sources are experiment configs, experiment directories, outputs
directories, or direct `.csv` / `.parquet` / `.xlsx` vec8 tables. Experiment and
outputs-directory sources require the typed dataframe record `sfxi_vec8/vec8`.
Reader does not silently aggregate exported workbooks from experiment sources.
If you intentionally want to review an exported workbook, pass that workbook as
an explicit table source. Direct table sources under an experiment directory use
the experiment directory as `source_id`; otherwise the file stem is used, and
duplicate `source_id` values are rejected.

Aggregate inputs must include `time_selected_h`. The aggregate plot is a review
surface for measured vec8 records, so it fails on vec8 tables that do not carry
snapshot-time provenance.

The aggregate surface is intentionally decoupled from OPAL and `dnadesign`: it
does not score objectives, consume OPAL campaign ledgers, or import OPAL
plotting code. It only stacks measured reader vec8 rows and writes:

* `sfxi_vec8_heatmap.png`
* `sfxi_vec8_heatmap_tidy.csv`
* `sfxi_vec8_heatmap_manifest.json`

Existing aggregate bundles are not overwritten unless `--overwrite` is passed.

The tidy CSV has one row per `source_id` / `design_id` / vec8 channel and is the
stable downstream review table if a richer aggregate notebook or study-specific
plot layer is added later.

The PNG uses compact display labels for readability and annotates the two vec8
blocks separately: `v00` / `v10` / `v01` / `v11` are the measured logic-pattern
channels, and `y00*` / `y10*` / `y01*` / `y11*` are the transformed fluorescence
intensity channels. Rows are displayed with controls first, then natural
ascending design IDs, and each row label includes the selected snapshot time.
Colorbar labels use the same SFXI definitions as this document: logic shape is
reported as `v_i`, and fluorescence intensity is reported as log2
reference-normalized `y_i*`.
Full `source_id`, `design_id`, source path, table path, and the unsorted tidy
values stay in the tidy CSV and manifest. For record-backed sources, the
manifest also records the source record's contract id, content digest, config
digest, creation timestamp, and producer metadata.

The aggregate manifest records the unique `intensity_log2_offset_delta` values.
If more than one delta is present, `mixed_intensity_log2_offset_delta` is `true`;
the heatmap is still a measured vec8 review surface, but the intensity channels
should not be interpreted as sharing one linear-intensity inverse.

---

### Configuration entry point

In `reader/v7`, SFXI is normally configured through the bound protocol plus
semantic `protocol.inputs` / `protocol.analysis` / `protocol.outputs`
fields. A minimal example:

```yaml
protocol:
  id: logic/sfxi_screen
  inputs:
    ingest:
      mode: mixed
      channels: [OD600, CFP, YFP]
    fold_change:
      report_times: [14.0]
      use_global_baseline: true
      global_baseline_value: EtOH_0_percent_0nM_cipro
    response:
      logic_channel: YFP/CFP
      intensity_channel: YFP/OD600
    design_by: [design_id]
    logic_map_ref: induction_logic
    reference:
      design_id: REF
      stat: mean
  analysis:
    sfxi_objective:
      setpoints:
        and: [0.0, 0.0, 0.0, 1.0]
        or: [0.0, 1.0, 1.0, 1.0]
      scaling:
        percentile: 95
        min_n: 5
        eps: 1.0e-8
      exponents:
        logic_exponent_beta: 1.0
        intensity_exponent_gamma: 1.0
      intensity_log2_offset_delta: 0.0
  outputs:
    plots:
      include: [sfxi_setpoint_scatter]
    exports:
      include: [logic_summary_workbook]

annotations:
  logic_maps:
    induction_logic:
      column: treatment_alias
      corners:
        "00": EtOH 0%, 0 nM cipro
        "10": EtOH 3%, 0 nM cipro
        "01": EtOH 0%, 100 nM cipro
        "11": EtOH 3%, 100 nM cipro
      case_sensitive: true
```

Additional notes:

* `response.logic_channel` and `response.intensity_channel` are **required** and must match `channel` values in the tidy table exactly (string equality).
* `protocol.id` should bind the experiment to `logic/sfxi_screen` when SFXI is the primary analysis protocol.
* `annotations.logic_maps.<name>.corners` must contain exactly the four keys `00, 10, 01, 11`.
* `logic_map_ref` must point to a defined annotations logic map; reader fails fast on unknown refs.
* `reference.design_id` is required for intensity anchoring; missing reference data results in an error rather than a fallback.
* `time_column` must name a column present in the input table. The selector
  normalizes that configured column to `time` internally before choosing the
  snapshot.
* `analysis.sfxi_objective.intensity_log2_offset_delta` is compiled into both
  the vec8 transform (`log2_offset_delta`) and the setpoint-scatter scorer. The
  plot refuses to score vec8 rows whose `intensity_log2_offset_delta` provenance
  column does not match the scorer configuration.

---

### Usage demo

The following example uses the SFXI-capable experiment
`experiments/2025/20250915_sfxi_pSingle_ref/config.yaml`.

1) Run the pipeline to generate annotated plate-reader and vec8 outputs:

    ```bash
    uv run reader run experiments/2025/20250915_sfxi_pSingle_ref/config.yaml
    ```

    This writes:

    * dataframe records under `outputs/artifacts/`
    * a typed records catalog at `outputs/manifests/records.json`
    * an SFXI dataframe record at `sfxi_vec8/vec8`

2) Export the vec8 table via `reader export`. Bind the workbook export as a protocol artifact:

    ```yaml
    protocol:
      id: logic/sfxi_screen
      outputs:
        exports:
          include: [logic_summary_workbook]
          artifacts:
            logic_summary_workbook:
              path: sfxi/vec8.xlsx
              sheet_name: vec8
    ```

    Then run exports:

    ```bash
    uv run reader export experiments/2025/20250915_sfxi_pSingle_ref/config.yaml
    ```

    This writes:

    * `outputs/exports/sfxi/vec8.xlsx`

3) Render the SFXI setpoint scatter figure when configured:

    ```bash
    uv run reader plot experiments/2025/20250915_sfxi_pSingle_ref/config.yaml --only sfxi_setpoint_scatter
    ```

    This writes plot files under `outputs/plots/`, such as
    `outputs/plots/sfxi_setpoint_scatter.pdf`.

4) Launch the SFXI notebook template (interactive vec8 inspection + export panel):

    ```bash
    uv run reader notebook experiments/2025/20250915_sfxi_pSingle_ref/config.yaml --template notebook/sfxi_eda --mode edit
    ```

    Notes:

    * The notebook template is gated: it only scaffolds when the experiment has a valid
      `transform/sfxi` step or existing SFXI dataframe records.
    * You can repeat the same workflow with any of the other SFXI-capable experiments in `experiments/2025/`.

5) Render an aggregate heatmap over one or more completed SFXI experiments:

    ```bash
    uv run reader aggregate-sfxi-vec8 \
      experiments/2025/20250915_sfxi_pSingle_ref/config.yaml \
      --out-dir outputs/reviews/sfxi_vec8_aggregate
    ```

    This writes:

    * `sfxi_vec8_heatmap.png`
    * `sfxi_vec8_heatmap_tidy.csv`
    * `sfxi_vec8_heatmap_manifest.json`

    Pass exported `vec8.xlsx` files directly only when reviewing an explicit
    workbook snapshot rather than the latest experiment record.
