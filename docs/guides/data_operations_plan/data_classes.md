---
doc_id: reader-dop-data-classes
surface: operator-reference
owner: reader-maintainers
last_verified: 2026-07-28
summary: Reader data-class definitions and the evidence needed to choose one without guessing assay semantics.
---

# Data Classes

Use this page first during intake. Choose the first class that fits the
dataset, then use the matching protocol family or draft/template path.

Return to the [Data Operations Plan](../data_operations_plan.md) when you only
need the overview.

For machine-readable data-class and protocol-candidate output, use:

```bash
uv run reader dop classes --format json
```

| Data class | Use when | Preferred `reader` route | Minimum capture |
| --- | --- | --- | --- |
| Plate-reader screen | Raw input is a Synergy/plate-reader export with well-level measurements and assay metadata. | `plate_reader/dual_reporter_screen`, `plate_reader/single_reporter_screen`, or `plate_reader/retron_sponge_screen` | Raw workbook/export, sample map, channel semantics, treatment/control meaning, plate/well coverage |
| Flow-cytometry panel | Raw input is FCS or cytometry panel data. | `cytometry/flow_panel` | Raw FCS files and discovery policy, channel naming field, sample metadata, required metadata columns |
| Logic/SFXI analysis | Dataset is a logic-response or SFXI-style screen with explicit response/intensity channels and an ordered state space. | `logic/sfxi_screen` | Raw files, metadata map, response/intensity channel choices, reference design, ordered 00/10/01/11 states |
| Aggregate/review experiment | Inputs are prior `reader` records, plots, exports, or hand-authored review material rather than one raw assay run. | A dedicated `experiments/<year>/<experiment>/` unit using `workbench/generic` or a more specific aggregate protocol | Source experiment ids, record/export paths, review purpose, expected outputs or notebook template |
| Unsupported long-tail assay | The assay does not fit an existing protocol contract. | Start as a draft/template; add a protocol only after the metadata and execution contract are clear. | Raw source path, intended analysis, required metadata, missing protocol decision, owner for follow-up |

If a dataset fits multiple classes, prefer the class with the strictest
control and metadata contract. For example, a matched-control retron sponge
plate should use `plate_reader/retron_sponge_screen` instead of the more
general dual-reporter route.

## Decision Rules

- Prefer an existing protocol when its metadata and control assumptions match
  the run.
- Prefer a nearest-neighbor config only when it preserves real assay semantics,
  not just plot appearance.
- Use `draft` or `template` when the run is not yet executable or when the
  protocol contract is still being discovered.
- Do not make a long-tail assay look runnable by forcing it into an adjacent
  protocol with different control semantics.
- Treat context as part of the class decision. An instrument calibration,
  failed setup run, or review bundle can need a different route than an
  experiment even when the raw instrument family is the same.

After choosing a class, move to
[Metadata minimums](./metadata_minimums.md).
