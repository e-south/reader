# SFXI vec8 (reader)

Reader computes the **vec8 label** used by OPAL. It does **not** score objectives.
Objective scoring lives in dnadesign (`src/dnadesign/opal/src/objectives/sfxi_v1.py`).

## Where it lives

- Plugin: `transform/sfxi`
- Input contract: `tidy+map.v2`
- Output contract: `sfxi.vec8.v2`

Artifacts are stored under `outputs/artifacts/<step_id>.sfxi/` (typically `vec8.parquet`).
Use `reader artifacts` to locate the latest path.

## Required inputs

`tidy+map.v2` is a tidy table with at least:

- `position`, `time`, `channel`, `value`
- `treatment`, `design_id` (and optionally `batch`)

`time` is in **hours** (this is what the Synergy H1 parser emits).

SFXI expects **logic** and **intensity** channels to already exist (e.g. `YFP/CFP` and `YFP/OD600`).
Compute them upstream with `transform/ratio`, then promote to `tidy+map.v2` via `validator/to_tidy_plus_map`.

## Config (minimal)

```yaml
steps:
  - id: "sfxi_vec8"
    uses: "transform/sfxi"
    reads: { df: "mapped/df" }
    with:
      response:
        logic_channel: "YFP/CFP"
        intensity_channel: "YFP/OD600"
      treatment_map: { "00": "...", "10": "...", "01": "...", "11": "..." }
      reference: { design_id: "REF", scope: "batch", stat: "mean" }
      batch_col: null          # optional; set to "batch" if present
      target_time_h: 10
      time_mode: "nearest"      # nearest | last_before | first_after | exact
      time_tolerance_h: 0.5
```

## Time selection

- `target_time_h` + `time_mode` control how the snapshot time is chosen.
- If `time_tolerance_h` is set, reader **logs a warning** when the chosen time is farther than the tolerance.
- Selection is **per batch** only when a `batch` column is present (or `batch_col` is set).  
  If no batch is provided, reader uses a single implicit batch (`0`).

## Math knobs (explicit)

These are the stabilizers used by `compute_vec8`:

- `eps_ratio`   (ratio/log guard)
- `eps_range`   (min‑max guard)
- `eps_ref`     (reference denominator guard)
- `eps_abs`     (intensity numerator guard)
- `ref_add_alpha` (α)
- `log2_offset_delta` (δ)

All values must be non‑negative floats and are logged at run time.

## Output columns

`sfxi.vec8.v2` includes (per `design_id × batch`; batch may be a single implicit `0`):

- `v00, v10, v01, v11` (logic in [0,1])
- `y00_star, y10_star, y01_star, y11_star` (log2 intensity)
- `r_logic`, `flat_logic`

Optional metadata columns (e.g., `sequence`, `id`) are carried through if configured.

## Strictness notes

- `reference.design_id` must resolve to a design in the data (hard error otherwise).
- `treatment_map` must map exactly the four states `00,10,01,11`.
- Logic/intensity channels must select the **same** time per batch.
- Flat logic (`v=0.25`) is **explicitly flagged** via `flat_logic` and logged.
