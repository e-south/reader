# Logic-symmetry (reader)

This doc has moved to `docs/logic_symmetry.md` (current pipeline usage and config).
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

* `design_by` can be a list, e.g. `["strain", "design_id"]`, to define what counts as a “design”.

---

## Outputs

* `logic_symmetry_<name>.pdf|png` — one figure with all points
* **Artifact:** `logic_symmetry.v1` (returned table) — one row per `design_by…`:

```
<design_by...>,
n00,n10,n01,n11,
b00,b10,b01,b11, sd00,sd10,sd01,sd11,
r, log_r, cv, u00,u10,u01,u11, L, A,
size_value, hue_value, alpha_value, shape_value
```

---

## Common errors (and how to fix)

* **Snapshot violation:** more than one `time` for a `(design, treatment)` → pre-filter upstream to one time per such group.
* **Incomplete corners:** one of `00/10/01/11` is absent for a `design` → fix `treatment_map` or your plate map.

---

## TL;DR

* Map your **exact** treatment labels to `00/10/01/11`.
* Ensure **one time** per `(design, treatment)`.
* Pick encodings (`size_by`, `hue`, `alpha_by`, `shape_by`).
* Run `reader` → get a clear `(L, A)` plot plus a CSV of supporting metrics.
