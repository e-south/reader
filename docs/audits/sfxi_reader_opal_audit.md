# SFXI Reader ↔ OPAL Audit (vec8 ↔ objective continuity)

Scope: align Reader vec8 generation, OPAL ingest-y transform, and OPAL objective with the SFXI spec (`dnadesign/src/dnadesign/opal/docs/setpoint_fidelity_x_intensity.md`). Reader remains canonical for vec8 math; OPAL remains canonical for objective scoring.

## What’s already aligned with the spec
- **State order [00,10,01,11]** is consistently enforced in Reader selection/pivot and in OPAL’s objective assumptions.
  - Reader: `reader/src/reader/lib/sfxi/selection.py::cornerize_and_aggregate`
  - OPAL: `dnadesign/src/dnadesign/opal/src/objectives/sfxi_v1.py::sfxi_v1`
- **Logic shape:** log2 → per‑design min–max → [0,1] matches spec intent.
  - Reader: `reader/src/reader/lib/sfxi/math.py::compute_vec8`
- **Intensity math:** reference‑normalized intensity stored as log2 with delta offset matches spec and OPAL inversion.
  - Reader: `reader/src/reader/lib/sfxi/math.py::compute_vec8`
  - OPAL: `dnadesign/src/dnadesign/opal/src/objectives/sfxi_v1.py::_recover_linear_intensity`
- **Objective:** logic fidelity (D‑normalized) + intensity scaling (round‑internal percentile) implements the spec.
  - OPAL: `dnadesign/src/dnadesign/opal/src/objectives/sfxi_v1.py::sfxi_v1`

## Mismatches found (and where fixes belong)
1) **All‑OFF setpoint should disable intensity (spec §4)**
   - **Mismatch:** OPAL `sfxi_v1` used E_scaled even when setpoint sums to 0, forcing score → 0.
   - **Fix location:** OPAL objective.
   - **File/function:** `dnadesign/src/dnadesign/opal/src/objectives/sfxi_v1.py::sfxi_v1`
   - **Status:** Fixed in this change. Intensity is disabled; E_raw=0, E_scaled=1, score=F_logic**beta, denom is set to 1.0 in RoundCtx.

2) **Vec8 ingest required sequence even when id is present**
   - **Mismatch:** OPAL transform always required sequence, blocking id‑only ingestion.
   - **Fix location:** OPAL transform_y (and schema).
   - **File/function:** `dnadesign/src/dnadesign/opal/src/transforms_y/sfxi_vec8_from_table_v1.py::sfxi_vec8_from_table_v1`
   - **Status:** Fixed in this change. Sequence is required only when id is absent.

3) **id_column too strict (must be literal “id”)**
   - **Mismatch:** OPAL transform refused `id_column: design_id` despite Reader exporting design_id.
   - **Fix location:** OPAL transform_y + schema.
   - **File/function:** 
     - `dnadesign/src/dnadesign/opal/src/transforms_y/sfxi_vec8_from_table_v1.py::sfxi_vec8_from_table_v1`
     - `dnadesign/src/dnadesign/opal/src/config/plugin_schemas.py::_Vec8TableParams`
   - **Status:** Fixed in this change. Any source column is allowed; output column is standardized to `id`.

4) **Flat‑logic warning missing**
   - **Mismatch:** Reader set `flat_logic=True` and `v=0.25` but emitted no warning and no run‑level summary in logs.
   - **Fix location:** Reader.
   - **File/function:** 
     - `reader/src/reader/lib/sfxi/run.py::build_vec8_from_tidy` (log stats)
     - `reader/src/reader/lib/sfxi/run.py::run_sfxi` (aggregated warning)
   - **Status:** Fixed in this change. One warning per run + log stats.

5) **Min–max stabilizer missing η**
   - **Mismatch:** Reader used v=(u−umin)/(umax−umin) without the spec’s +η stabilizer.
   - **Fix location:** Reader.
   - **File/function:** `reader/src/reader/lib/sfxi/math.py::_logic_minmax_from_four`
   - **Status:** Fixed in this change. Denom uses span+η and η is recorded in logs (reuses eps_range).

6) **Cross‑system risk: delta must match between Reader and OPAL**
   - **Mismatch:** There’s no enforcement that Reader’s `log2_offset_delta` matches OPAL’s `intensity_log2_offset_delta`.
   - **Fix location:** OPAL (ingest-time validation) or project‑level config conventions.
   - **File/function:** 
     - Reader config: `reader/src/reader/lib/sfxi/api.py::SFXIConfig`
     - OPAL objective: `dnadesign/src/dnadesign/opal/src/objectives/sfxi_v1.py::sfxi_v1`
   - **Status:** Not enforced; recommend validating against Reader’s `sfxi_log.json` when available.

