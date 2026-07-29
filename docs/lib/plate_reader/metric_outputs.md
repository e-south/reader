---
doc_id: reader-plate-reader-metric-outputs
surface: library-router
owner: reader-maintainers
last_verified: 2026-07-29
summary: Route compatible measurement records to independent typed analysis outputs.
---

# Plate-reader metric outputs

Reader separates instrument ingestion from analysis-specific outputs. An
experiment name never selects analysis behavior; a protocol or versioned
request does.

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
| Response window | `reader.response_window.request.v3` | one declared event and event-relative windows | verified `reader.response_window.bundle.v5` |

The lanes may read compatible source records, but they do not share reductions
or infer one another from filenames. Downstream interpretation remains a
consumer concern.

## Continue by task

- [SFXI in Reader](../sfxi_vec8_in_reader.md)
- [Response-window analysis](response_window.md)
- [Notebook operation](../../guides/notebooks.md)
- [Plugin development](../../core/plugins.md)
