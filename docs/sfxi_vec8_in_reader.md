## Generating SFXI 8-vectors in `reader`

This document describes how **reader** processes **Setpoint Fidelity x Intensity** (SFXI) 8-vectors from experimental measurements. The objective/scalar spec is outside of reader (for more details on SFXI see [here](https://github.com/e-south/dnadesign/blob/main/src/dnadesign/opal/docs/setpoint_fidelity_x_intensity.md)). The process here involves collecting microplate reader data, selecting a timepoint, and then deriving an 8‑vector per *design_id* in the fixed state order **00, 10, 01, 11**.

8-vector definition:

```
[v00, v10, v01, v11, y00_star, y10_star, y01_star, y11_star]
```

* `v..` encodes the **logic shape** in `[0,1]` (derived from a **logic channel**, e.g. `YFP/CFP`).
* `y*_..` encodes **absolute intensity**, normalized to a reference design_id and stored in **log2** (derived from an **intensity channel**, e.g. `YFP/OD600`).

### Contents

1. [Reader hand-off to OPAL](#reader-hand-off-to-opal)
1. [Scope (vec8 vs objective)](#scope-vec8-vs-objective)
2. [Relevant modules](#relevant-modules)
3. [Input contract](#input-contract)
4. [Time selection](#time-selection)
5. [Corner mapping](#corner-mapping)
6. [Logic channel](#logic-channel)
7. [Intensity channel](#intensity-channel)
8. [Output](#output)
9. [Configuration entry point](#configuration-entry-point)
10. [Usage demo](#usage-demo)

---

### reader hand-off to OPAL

- **reader** is the source of truth for vec8 math. It computes:
  * `u = log2(ratio)` for logic, then per-design min–max into `v ∈ [0,1]`.
  * `y* = log2(y_linear + delta)` for intensity, where `y_linear` is reference-normalized.

- **OPAL is the source of truth for the scalar objective.** It expects `y*_..` columns already in log2 and converts back to linear using its `intensity_log2_offset_delta` parameter.
  * **Important:** The `delta` used in **reader** (`log2_offset_delta`) must match OPAL’s `intensity_log2_offset_delta`.
  * **Reader makes this explicit** by writing an `intensity_log2_offset_delta` column into every vec8 row. When `log2_offset_delta` is left at its default (`0.0`), this column will be all zeros.
  * **If OPAL uses a different delta, recovered linear intensities and downstream scores will be inconsistent.** Keep the values in sync (preferably by validating against the vec8 column at ingest time).

- The **reader** transform plugin (`src/reader/plugins/transform/sfxi.py`) delegates to `reader/lib/sfxi/*` and adds pipeline plumbing and logging.

---

### Relevant modules

Key modules in `src/reader/lib/sfxi/`:

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

---

### Input contract

SFXI consumes a **tidy+map** DataFrame (the typical source is the `validator/to_tidy_plus_map` artifact).

#### Required columns

The selector enforces the following columns (see `selection.REQUIRED_COLS` and `_enforce_columns`):

* `position` — well/position identifier (not used in vec8 math, but required for tidy contract consistency)
* `time` — measurement time (numeric; cast to float during selection)
* `channel` — channel name (matched by exact string equality)
* `value` — numeric measurement value
* `treatment` — treatment label (required even if you also provide an alias column)

#### Design identity

* `design_by` — one or more columns that define a design grouping (default: `["design_id"]`)

Notes:

* The **first** `design_by` column (i.e. `design_by[0]`) is treated as the primary design label for:

  * reference design_id lookup
  * sequence attachment
* Reader enforces `design_by[0] == "design_id"` for SFXI to keep outputs and logs consistent.

#### Optional columns

* `sequence` — if present, it is attached into the vec8 output (per `design_by[0]`)

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

`treatment_map` assigns your experimental treatment labels to the four logic corners:

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
    v_i = (u_i - u_min) / span
    flat_logic = False
```

Notes:

* The “flat” fallback of `0.25` is a neutral, symmetry-preserving choice: it does not imply a preferred corner when there is no measurable separation.
* `logic_span_log2` is the `span` value above (the log2-space separation across corners).

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

Output is written by `lib.sfxi.writer.write_outputs(...)` (typically via `run.run_sfxi(...)`).

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

When using `lib.sfxi.run` entry points, the JSON log includes:

* resolved channels and config echo
* chosen time (global)
* out-of-tolerance messages (soft warnings)
* eps/alpha/delta parameters
* row counts at each stage
* summary stats for `r_logic`
* reference anchor values

> Note: the library (`run_sfxi`) writes `sfxi_log.json`. If you are using a higher-level transform wrapper, it may choose to surface the same information via console logging and/or pipeline metadata instead of writing a separate JSON file.

See `src/reader/core/contracts.py` for the canonical contract referenced by the pipeline (`sfxi.vec8.v2`).

---

### Configuration entry point

In reader pipeline configs, SFXI runs as a transform step using `transform/sfxi`. A minimal example:

```yaml
- id: sfxi_vec8
  uses: transform/sfxi
  reads:
    df: promote_to_tidy_plus_map/df
  with:
    response:
      logic_channel: YFP/CFP
      intensity_channel: YFP/OD600

    design_by: [design_id]

    # treatment → corner mapping (state order: 00,10,01,11)
    treatment_case_sensitive: true
    treatment_map:
      "00": EtOH 0%, 0 nM cipro
      "10": EtOH 3%, 0 nM cipro
      "01": EtOH 0%, 100 nM cipro
      "11": EtOH 3%, 100 nM cipro

    # time selection
    target_time_h: 10.0            # null → latest available time
    time_mode: nearest             # nearest | last_before | first_after | exact
    time_tolerance_h: 0.25         # soft warning threshold

    # corner completeness
    require_all_corners_per_design: true

    # intensity reference anchor (required)
    reference:
      design_id: REF
      stat: mean                   # mean | median
      # on_missing: error          # currently only 'error' is supported

    # numerical guards / knobs
    eps_ratio: 1e-9
    eps_range: 1e-12
    eps_ref: 1e-9
    eps_abs: 0.0
    ref_add_alpha: 0.0
    log2_offset_delta: 0.0

    # output
    output_subdir: sfxi
    vec8_filename: vec8.csv
    log_filename: sfxi_log.json
```

Additional notes:

* `response.logic_channel` and `response.intensity_channel` are **required** and must match `channel` values in the tidy table exactly (string equality).
* `treatment_map` must contain exactly the four keys `00, 10, 01, 11`.
* `reference.design_id` is required for intensity anchoring; missing reference data results in an error rather than a fallback.
* Although a `time_column` config key exists in `SFXIConfig`, the current selector validation expects a literal `time` column in the tidy table. (If your upstream data uses a different name, rename it to `time` before SFXI.)

---

### Usage demo

The following example uses the SFXI-capable experiment
`experiments/2025/20250915_sfxi_pSingle_ref/config.yaml`.

1) Run the pipeline to generate tidy+map and vec8 outputs:

    ```bash
    uv run reader run experiments/2025/20250915_sfxi_pSingle_ref/config.yaml
    ```

    This writes:

    * `outputs/sfxi/vec8.csv`
    * `outputs/sfxi/sfxi_log.json`

2) Export the vec8 table via `reader export`. Add export specs to the experiment `config.yaml` (adjust the `reads` path to match your SFXI step id):

    ```yaml
    exports:
      specs:
        - id: export_vec8_xlsx
          uses: export/xlsx
          reads: { df: sfxi_vec8/vec8 }
          with: { path: "sfxi/vec8.xlsx", sheet_name: "vec8" }
    ```

    Then run exports:

    ```bash
    uv run reader export experiments/2025/20250915_sfxi_pSingle_ref/config.yaml
    ```

    This writes:

    * `outputs/exports/sfxi/vec8.xlsx`

3) Launch the SFXI notebook preset (interactive vec8 inspection + export panel):

    ```bash
    uv run reader notebook experiments/2025/20250915_sfxi_pSingle_ref/config.yaml --preset notebook/sfxi_eda --mode edit
    ```

    Notes:

    * The notebook preset is gated: it only scaffolds when the experiment has a valid
      `transform/sfxi` step or existing SFXI artifacts.
    * You can repeat the same workflow with any of the other SFXI-capable experiments in `experiments/2025/`.

4) (Optional) export vec8 from the notebook UI:

---

@e-south
