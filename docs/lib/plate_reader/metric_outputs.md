---
doc_id: reader-plate-reader-metric-outputs
surface: library-router
owner: reader-maintainers
last_verified: 2026-07-29
summary: Route compatible measurement records to independent typed analysis outputs.
---

# Plate-reader metric outputs

Reader separates instrument ingestion from analysis-specific outputs. An
experiment name never selects analysis behavior; a protocol does.

## Shared measurement records

The common plate-reader path:

1. ingests a declared workbook format;
2. merges well metadata and authored labels;
3. applies blank and overflow policies; and
4. derives configured ratio channels.

These steps publish manifest-backed annotated records. They do not choose a
summary time, event, ordered-state ontology, classification, or objective.

## Output lanes

| Lane | Explicit selector | Time basis | Durable output |
| --- | --- | --- | --- |
| General plate-reader | `plate_reader/dual_reporter_screen` or `plate_reader/single_reporter_screen` | acquisition time and configured endpoints | annotated records, plots, and optional fold-change tables |
| SFXI vec8 | `logic/sfxi_screen` | one selected acquisition-time snapshot | `sfxi_vec8/vec8` under `sfxi.vec8.v3` |
| Response window | `plate_reader/response_window` | one declared event and event-relative windows | typed dataframe records plus registered plot and export artifacts |

Record-backed collection protocols use `resources` of kind `record`; they do
not copy source configs or publish custom manifests.

The lanes may read compatible source records, but they do not share reductions
or infer one another from filenames. Downstream interpretation remains a
consumer concern.

## Dual-reporter triptych

`dual_reporter_triptych` is an optional plot output of
`plate_reader/dual_reporter_screen`. It reads the persisted
`ratio_yfp_od600/df` record and writes one static growth, reporter-ratio, and
endpoint figure per design. It is not a notebook template or a separate
execution path.

The endpoint time is experiment policy, so Reader has no implicit hour value.
Declare it in the plot view:

```yaml
protocol:
  outputs:
    plots:
      profile: none
      include: [dual_reporter_triptych]
      views:
        dual_reporter_triptych:
          snapshot_time_h: 8.0
          snapshot_time_mode: nearest
          snapshot_time_tolerance_h: 0.25
          treatment_order_ref: conditions
          format: [png, pdf]
```

The plot summarizes observed reporter-ratio values at the selected acquisition
time; it does not assign study meaning to treatments, infer an intervention
from workbook boundaries, or compute a downstream objective.

## Continue by task

- [SFXI in Reader](../sfxi_vec8_in_reader.md)
- [Response-window analysis](response_window.md)
- [Notebook operation](../../guides/notebooks.md)
- [Plugin development](../../core/plugins.md)
