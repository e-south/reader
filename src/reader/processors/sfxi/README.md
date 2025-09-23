## setpoint_fidelity_x_intensity (SFXI)

**Goal.** Combine a sample’s observed **logic pattern** and **absolute fluorescent intensity** into a single 8-vector:

```
[v00, v10, v01, v11, y*00, y*10, y*01, y*11]
```

- `v` encodes the **shape** of the logic response in `[0,1]` (state order `00,10,01,11`).
- `y*` stores **absolute intensity** (anchor-normalized, in **log2** space).

---

### Inputs and notation

- Upstream tidy data provides: `position,time,value,channel,treatment,batch,genotype,id,sequence`.
- SFXI expects a **reference genotype** with constitutive YFP (keyed by `genotype`). It anchors absolute intensity across runs.
- **Channels (explicit):**
  - **Logic channel** (for `v`): usually `YFP/CFP`.
  - **Intensity channel** (for `y*`): typically `YFP/OD600` (compensates growth/size differences across conditions).
- Corners use state order `00,10,01,11`.

Let `L_i` be the logic channel value at corner `i` (e.g., `YFP/CFP`), and `I_i` be the intensity channel value at corner `i` (e.g., `YFP/OD600`).
Let `A_i` be the **reference** intensity value for corner `i` (from the reference genotype, per **scope**: batch or global, summarized by `stat` mean/median).

Stabilizers (recorded in `sfxi_log.json`):
`ε` (ratio guard), `η` (range guard), `α` (reference denom guard), `δ` (absolute add before log; here we keep `δ=0` and use `ε` guards).

---

### Computation

#### 1) Logic shape (unit interval, from logic channel)

1. Log2: \( u_i = \log_2\!\big(\max(L_i,\ \varepsilon)\big) \)
2. If flat: if \( \max(u) - \min(u) \le \eta \), set all \( v_i = 1/4 \) and **warn**.
3. Else min–max in log space:
   \[
   v_i = \frac{u_i - u_{\min}}{u_{\max} - u_{\min}} \in [0,1]
   \]

This preserves shape while discarding scale; hard 0/1 at extrema are expected. When flat, `0.25` is a neutral, symmetry-preserving fallback with no directional bias.

#### 2) Absolute intensity (anchor-normalized, stored in log2)

1. Reference anchor (per scope/stat): \( A_i = \text{stat}\{\text{reference } I_i\} \)
2. Unitless scale: \( y_i^{\text{linear}} = \dfrac{I_i + \alpha}{\max(A_i,\ \alpha)} \)
3. Store in log2: \( y_i^* = \log_2\!\big(\max(y_i^{\text{linear}},\ \varepsilon)\big) \)

Using **the same intensity channel** (`YFP/OD600`) for both the sample and the reference anchor keeps the effect size interpretable even when growth differs across conditions.

---

### Output files

- `sfxi/vec8.csv` — one row per (design × batch) with keys (`… design_by …, batch`), `v00..v11`, `y*00..y*11`, plus diagnostics (`r_logic`, `flat_logic`).
- `sfxi/sfxi_log.json` — echo of config, chosen snapshot times, dropped batches, epsilons, and row counts.

---

### Configuration

```yaml
transformations:
  - type: sfxi
    response:
      logic_channel: "YFP/CFP"     # for v00..v11
      intensity_channel: "YFP/OD600"  # for y*00..y*11
    design_by: ["genotype"]

    # time selection
    target_time_h: 10
    time_mode: nearest            # nearest | last_before | first_after | exact
    time_tolerance_h: 0.25
    time_per_batch: true
    on_missing_time: error        # error | skip-batch | drop-all

    # corners/treatments (state order: 00,10,01,11)
    treatment_case_sensitive: true
    treatment_map:
      "00": "EtOH_0_percent_0nM_cipro"
      "10": "EtOH_3_percent_0nM_cipro"
      "01": "EtOH_0_percent_100nM_cipro"
      "11": "EtOH_3_percent_100nM_cipro"

    # completeness
    require_all_corners_per_design: true

    # reference anchor for intensity
    reference:
      genotype: "REF"
      scope: batch                 # batch | global
      stat: mean                   # mean | median
      on_missing: error            # error | skip

    # numerical guards
    eps_ratio: 1e-9
    eps_range: 1e-12
    eps_ref:   1e-9
    eps_abs:   0.0

    # output
    output_subdir: "sfxi"
    vec8_filename: "vec8.csv"
    log_filename:  "sfxi_log.json"
```

@e-south
