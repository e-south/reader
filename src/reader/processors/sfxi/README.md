# setpoint\_fidelity\_x\_intensity (SFXI)

**Goal.** Combine a sample’s observed **logic pattern** and **absolute fluorescent intensity** into a single 8-vector:

```
[v00, v10, v01, v11, y*00, y*10, y*01, y*11]
```

* `v` encodes the **shape** of the logic response in `[0,1]` (state order `00,10,01,11`).
* `y*` stores **absolute intensity** (anchor-normalized, in **log2** space).

---

## Inputs and notation

* Upstream tidy data provides:
  `position,time,value,channel,treatment,batch,genotype,id,sequence`
* SFXI expects a **reference genotype** with constitutive YFP (keyed by `genotype`) to anchor absolute intensity across runs.
* **Channels (explicit):**

  * **Logic channel** (for `v`): usually `YFP/CFP`.
  * **Intensity channel** (for `y*`): typically `YFP/OD600` (compensates growth/size differences across conditions).
* Corners use state order `00,10,01,11`.

Let \$L\_i\$ be the logic-channel value at corner \$i\$ (e.g., `YFP/CFP`), and \$I\_i\$ be the intensity-channel value at corner \$i\$ (e.g., `YFP/OD600`).
Let \$A\_i\$ be the **reference** intensity value for corner \$i\$ (from the reference genotype, per **scope**: batch or global, summarized by a chosen statistic such as mean or median).

**Stabilizers** (recorded in `sfxi_log.json`):
\$\varepsilon\$ (ratio guard), \$\eta\$ (range guard), \$\alpha\$ (reference-denominator guard), \$\delta\$ (absolute add before log; we keep \$\delta=0\$ and rely on \$\varepsilon\$ guards).

---

## Computations

### Logic dynamic range

`r_logic` reports the **dynamic range** of the **logic channel** values *before* min–max scaling (how far apart the four corners are in linear space):

$$
r_{\text{logic}}
=
\frac{\max_i\,\max(L_i,\ \varepsilon)}
     {\min_i\,\max(L_i,\ \varepsilon)}
\;\ge\;1.
$$

* `r_logic ≈ 1` → corners are essentially indistinguishable on the logic channel (flat).
* `> 4×` → robust separation; the shape is driven by real differences.

### Logic shape (unit interval, from logic channel)

1. **Log2:**

   $$
   u_i = \log_2\big(\max(L_i,\ \varepsilon)\big)
   $$
2. **Flat check:** if \$\max(u)-\min(u)\le\eta\$, set all \$v\_i=\tfrac14\$ and flag `flat_logic=true`.
3. **Otherwise, min–max in log space:**

   $$
   v_i = \frac{u_i - u_{\min}}{u_{\max} - u_{\min}} \in [0,1].
   $$

This preserves shape while discarding scale; hard 0/1 at extrema are expected. When flat, `0.25` is a neutral, symmetry-preserving fallback with no directional bias.

### Absolute intensity (anchor-normalized, stored in log2)

1. **Reference anchor (per scope/stat):**

   $$
   A_i = \text{statistic over reference } I_i
   $$

   *(Use mean or median, per config.)*
2. **Unitless scale:**

   $$
   y_i^{\text{linear}} = \frac{I_i + \alpha}{\max(A_i,\ \alpha)}
   $$
3. **Store in log2:**

   $$
   y_i^* = \log_2\!\big(\max(y_i^{\text{linear}},\ \varepsilon)\big)
   $$

Using **the same intensity channel** (`YFP/OD600`) for both the sample and the reference anchor keeps the effect size interpretable even when growth differs across conditions.

---

## Output files

* `sfxi/vec8.csv` — one row per (design × batch) with keys (`… design_by …, batch`), `v00..v11`, `y*00..y*11`, plus diagnostics (`r_logic`, `flat_logic`).
  *By default, the reference genotype row(s) are excluded from this file.*
* `sfxi/sfxi_log.json` — echo of config, chosen snapshot times, dropped batches, epsilons, and row counts.

---

## Configuration

```yaml
transformations:
  - type: sfxi
    response:
      logic_channel: "YFP/CFP"        # for v00..v11
      intensity_channel: "YFP/OD600"  # for y*00..y*11
    design_by: ["genotype"]

    # time selection
    target_time_h: 10
    time_mode: nearest                 # nearest | last_before | first_after | exact
    time_tolerance_h: 0.25
    time_per_batch: true
    on_missing_time: error             # error | skip-batch | drop-all

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
      scope: batch                      # batch | global
      stat: mean                        # mean | median
      on_missing: error                 # error | skip

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

---

@e-south
