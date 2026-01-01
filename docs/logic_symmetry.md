# Logic-symmetry plot (reader)

Logic-symmetry places each two-input design at coordinates:

- **L (logic):** how the double-input response compares to the single inputs
- **A (asymmetry):** which single input dominates

This is a visualization tool; it also emits a typed table artifact for reuse.

## Where it lives

- Plugin: `plot/logic_symmetry`
- Input contract: `tidy+map.v2`
- Output artifact: `logic_symmetry.v1` (table) + plot files under `outputs/<plots_dir>/`

## Config (minimal)

```yaml
reports:
  - id: "logic_symmetry"
    uses: "plot/logic_symmetry"
    reads: { df: "mapped/df" }
    with:
      response_channel: "YFP/CFP"
      design_by: ["design_id"]
      treatment_map: { "00": "...", "10": "...", "01": "...", "11": "..." }
      aggregation: { replicate_stat: "mean", uncertainty: "halo" }
      output: { format: ["pdf"], dpi: 300 }
      fig: { figsize: [7, 6] }
```

## Notes

- The treatment map must be **exact** for the four corners `00,10,01,11`.
- If you need a snapshot from time series data, use `prep` in the plugin config.
- The table artifact can be found via `reader artifacts`.
