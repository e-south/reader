# Logic–symmetry space

A single-figure plotter that places each two-input promoter (per **design × batch**) at coordinates:

* **x = L (logic):** how the **A+B** response compares to the single-input responses
* **y = A (asymmetry):** whether **A-only** or **B-only** is larger
* Optional encodings for **size**, **hue**, **alpha**, and **shape** to summarize batch order, noise, categories, etc.

It plugs into the `reader` plots section and renders a logic-symmetry figure. It does not emit artifacts; if you need a table, derive it in the pipeline or export it explicitly.

---

## Conceptual background

[Cox *et al.* (2007)](https://www.embopress.org/doi/full/10.1038/msb4100187) summarized dual-input promoter behavior with three numbers: regulatory range (`r`), logic type (`l`), and asymmetry (`a`). In their analysis the four measured responses are **sorted** (`b1 ≤ b2 ≤ b3 ≤ b4`) and the method assumes **monotone** behavior. This arranges AND/OR/SIG/SLOPE within a triangular region—but it cannot represent **non-monotone** gates such as XOR, NAND, NOR, XNOR because sorting discards which input produced which single-input level.

### Our adaptation

We keep the same geometric intuition but **do not sort**. We keep the identity of each state by using **explicitly labeled conditions**:

* `00` = neither input
* `10` = A-only
* `01` = B-only
* `11` = A+B

This preserves the distinction between the two single-input states (`10` vs `01`) and allows both **monotone** and **non-monotone** patterns to be plotted on the same axes.

---

## Computation (step by step)

All quantities are computed from **replicate-aggregated means** at a **snapshot** time (replicates aggregated by `mean` or `median`; default `mean`).

### Inputs

```
b00 = response at 00 (neither input)
b10 = response at 10 (A-only)
b01 = response at 01 (B-only)
b11 = response at 11 (A+B)
```

All values should be positive after upstream blank correction.

### 1) Dynamic range and log normalization

**Why:** Different designs/reporters can differ by orders of magnitude. We want a **unit-free** comparison that depends only on **relative** differences across the four states. We therefore scale by the **span** and work in **log** units.

```
r   = max(b00, b10, b01, b11) / min(b00, b10, b01, b11)   # r ≥ 1 (range across the four states)
uij = (log(bij) - log(min(b))) / log(r)                   # uij ∈ [0,1] (log-normalized unit interval)
```

* `r` captures the **usable span** between the lowest and highest states.
* The `uij` transform removes units and any global multiplicative factor; it maps the smallest state to `0` and the largest to `1` in **log** space.

**Degenerate case:** if the four responses are essentially equal (`r ≈ 1`), set all `uij = 0.5` (center).

### 2) Coordinates

**Logic (L)** — “double vs singles” in log-normalized units:

```
L = u11 - 0.5*(u10 + u01)
```

* Interprets the **A+B** level relative to the **average of the single-input** levels.
* `L = +1` when only `11` is high (AND-like); `L = 0` when `11` matches the singles on average (OR/SLOPE-like); `L = −1` when `11` is lower than the singles (XOR-like).

**Asymmetry (A)** — “which single dominates”:

```
A = u10 - u01
```

* Positive `A` means **A-only > B-only**; negative `A` means **B-only > A-only**.

**Bounds:** Because each `u ∈ [0,1]`, both `L` and `A` lie in `[-1, 1]`.

### 3) Properties (what these choices buy us)

* **Scale-invariant:** multiply all `b` by any constant → `(L, A)` unchanged.
* **Swap symmetry:** swapping which input you call “A” and “B” flips the sign of `A` and leaves `L` unchanged.
* **Basal ambiguity separated:** gates that differ only at `00` share `(L, A)`. If you must differentiate them, color by basal (`u00`) via `hue`.

### 4) Optional noise summary (CV)

If you set `size_by: "cv"`, bubble area visualizes **replicate noise**:

```
CV_corner = SD(replicates at that corner) / Mean(replicates at that corner)
CV_point  = mean of CV_corner across {00,10,01,11}, ignoring corners with n<2
```

---

## Where ideal two-input boolean combinations land

To build intuition, plug in **idealized** corners with `u ∈ {0,1}` and evaluate `(L, A)`:

> Corner order is `(00, 10, 01, 11)` = (neither, A-only, B-only, A+B).

```
L = u11 - 0.5*(u10 + u01)
A = u10 - u01
```

| Gate        | (u00,u10,u01,u11) |    L |    A | Interpretation            |
| :---------- | :---------------- | ---: | ---: | :------------------------ |
| FALSE (0)   | (0,0,0,0)         |  0.0 |  0.0 | never ON                  |
| TRUE (1)    | (1,1,1,1)         |  0.0 |  0.0 | always ON                 |
| AND         | (0,0,0,1)         | +1.0 |  0.0 | needs both inputs         |
| NAND        | (1,1,1,0)         | −1.0 |  0.0 | NOT(AND)                  |
| OR          | (0,1,1,1)         |  0.0 |  0.0 | either input is enough    |
| NOR         | (1,0,0,0)         |  0.0 |  0.0 | NOT(OR)                   |
| XOR         | (0,1,1,0)         | −1.0 |  0.0 | exactly one input         |
| XNOR        | (1,0,0,1)         | +1.0 |  0.0 | equivalence (both same)   |
| A           | (0,1,0,1)         | +0.5 | +1.0 | passes A                  |
| NOT A       | (1,0,1,0)         | −0.5 | −1.0 | passes NOT A              |
| B           | (0,0,1,1)         | +0.5 | −1.0 | passes B                  |
| NOT B       | (1,1,0,0)         | −0.5 | +1.0 | passes NOT B              |
| A AND NOT B | (0,1,0,0)         | −0.5 | +1.0 | A only                    |
| NOT A AND B | (0,0,1,0)         | −0.5 | −1.0 | B only                    |
| A -> B      | (1,0,1,1)         | +0.5 | −1.0 | implication (if A then B) |
| B -> A      | (1,1,0,1)         | +0.5 | +1.0 | implication (if B then A) |

**Notes**

* These are the coordinates the plot will render for ideal gates. Real designs fall near these locations depending on their measured response.
* Several gates **collide** in `(L,A)` because they differ only at the basal corner `u00` (e.g., **AND** and **XNOR** both land at `(L=+1, A=0)`; **OR** and **NOR** both at `(0,0)`; **XOR** and **NAND** both at `(−1,0)`). If you need to distinguish such pairs, encode **basal** (`u00`) via color/hue.
* The “**SLOPE**” point (independent regulation) is **not** a Boolean gate; it is a physical baseline at `(L=+0.5, A=0)` corresponding to singles at 0.5 and double at 1.0 in the normalized scheme.

---

## What’s drawn

* One **point** per `(design × batch)` at `(L, A)`.
* **Size:** `log_r` (default), `cv`, or `fixed`.
* **Hue:** optional column (e.g., `batch` or any metadata).
* **Alpha:** optional ramp from a column (default `batch`: older→lighter, newer→opaque).
* **Shape:** optional marker by category (e.g., squares/triangles).
* Optional **ideals overlay**: faded markers for canonical gates.

---

## Computation order (precedence)

1. Filter to `response_channel`.
2. Map **exact** treatment labels → `00/10/01/11` (no sorting).
3. Enforce **snapshot** rule per `(design_by…, batch, treatment)` (fail-fast if violated).
4. Aggregate replicates per corner (mean/median), keep `n` and `SD`.
5. Compute `r`, `u`, `L`, `A`, and optional `CV`.
6. Build encodings (size/hue/alpha/shape).
7. Render the single scatter; optionally add ideals.
8. Write **PDF/PNG** and a **CSV** with metrics and encodings.

---

## YAML usage (minimal working example)

```yaml
plots:
    - name: logic_map
      module: logic_symmetry
      params:
        response_channel: "YFP/CFP"
        design_by: ["genotype"]
        batch_col: "batch"
        treatment_map:
          "00": "EtOH 3%, 0 nM cipro"
          "10": "EtOH 3%, 0 nM cipro"
          "01": "EtOH 0%,100 nM cipro"
          "11": "EtOH 3%, 100 nM cipro"
        treatment_case_sensitive: true

        # Optional pre-selection if data are time series
        prep:
          enable: true
          mode: "nearest"        # nearest|first|last|median|exact
          target_time: 10        # hours
          tolerance: 0.6
          align_corners: false   # set true to force one common time per (design,batch)

        aggregation:
          replicate_stat: "mean"     # mean|median
          uncertainty: "halo"        # none|errorbars|halo

        encodings:
          size_by: "log_r"           # log_r|cv|fixed
          size_fixed: 80
          hue: null
          alpha_by: "batch"
          alpha_min: 0.35
          alpha_max: 1.0
          shape_by: null
          shape_cycle: ["o","s","^","D","P","X","v","*"]
          shape_max_categories: null

        ideals_overlay:
          enable: false
          gate_set: "logic_family"   # core|logic_family|full16
          style:
            alpha: 0.25
            size: 40
            color: "#888888"
            show_labels: true

        visuals:
          color: "#6e6e6e"
          xlim: [-1.02, 1.02]
          ylim: [-1.02, 1.02]
          grid: true

        output:
          format: ["pdf"]
          dpi: 300
          figsize: [7, 6]
```

### Notes & options

* `design_by` can be a list, e.g. `["strain", "genotype"]`, to define what counts as a “design”.
* `batch_col` **must** exist and be numeric (0,1,2,…).

---

## Outputs

* `logic_symmetry_<name>.pdf|png` — one figure with all points
* **Table (in-memory):** one row per `(design_by…, batch)` when computed during plotting. Persist it explicitly via pipeline/export if needed.

```
<design_by...>, batch,
n00,n10,n01,n11,
b00,b10,b01,b11, sd00,sd10,sd01,sd11,
r, log_r, cv, u00,u10,u01,u11, L, A,
size_value, hue_value, alpha_value, shape_value
```

---

## Common errors (and how to fix)

* **Missing `batch` (or non-numeric):** add/rename your batch column or set `batch_col` accordingly; ensure it’s numeric 0,1,2,….
* **Snapshot violation:** more than one `time` for a `(design, batch, treatment)` → pre-filter upstream to one time per such group.
* **Incomplete corners:** one of `00/10/01/11` is absent for a `(design, batch)` → fix `treatment_map` or your plate map.

---

## TL;DR

* Map your **exact** treatment labels to `00/10/01/11`.
* Ensure **one time** per `(design, batch, treatment)`.
* Pick encodings (`size_by`, `hue`, `alpha_by`, `shape_by`).
* Run `reader` → get a clear `(L, A)` plot plus a CSV of supporting metrics.
