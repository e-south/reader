---
doc_id: reader-cytometry-flow-panel
surface: library-contract
owner: reader-maintainers
last_verified: 2026-07-29
summary: Explicit cytometry ingestion, gating, QC, plot, and export lifecycle.
---

# Cytometry flow panels

`cytometry/flow_panel` turns declared FCS inputs into typed, persisted event and
summary records. Reader owns the executable measurement workflow; a consuming
study owns control meaning, comparisons, objectives, and interpretation.

## Lifecycle

The protocol compiles one path:

1. `ingest/flow_cytometer` parses FCS files into `ingest/df`;
2. `transform/sample_metadata` joins declared sample metadata into `merged/df`;
3. `transform/cytometry_gating` applies the authored gate, threshold, grouping,
   and QC policy;
4. `plot/cytometry_diagnostic` renders registered diagnostics from persisted
   inputs and outputs;
5. ordinary CSV exports project selected typed records;
6. `notebook/eda` reviews the resulting catalog without recomputation.

The gating transform persists:

| Record | Contract | Meaning |
| --- | --- | --- |
| `cytometry_gating/gate_definition` | `cytometry.gate_definition.v1` | Resolved channels, bounds, threshold, grouping, and QC policy |
| `cytometry_gating/gated_events` | `cytometry.gated_events.v1` | Events retained after the configured cells and singlet gates |
| `cytometry_gating/sample_stats` | `cytometry.sample_stats.v1` | Per-sample retention and fluorescence summaries |
| `cytometry_gating/group_stats` | `cytometry.group_stats.v1` | Optional summaries for one explicit metadata column |
| `cytometry_gating/qc` | `cytometry.qc.v1` | Per-sample threshold checks and pass/fail state |

## Required policy

Reader has no implicit cytometry channels, gate ranges, positive control, or
grouping. `protocol.inputs.gating` must state them. Manual thresholding rejects
control-estimation fields; control-quantile thresholding requires the exact
group column, control value, and quantile. An explicit null `group_column`
disables group summaries.

The singlet gate evaluates `singlet_y_channel / singlet_x_channel`. For QC,
`nonpositive_scope: gated_events` is fail-closed: a sample with no retained,
finite fluorescence events records `pct_nonpositive: 100.0`,
`passes_nonpositive: false`, and `qc_status: fail` instead of an undefined
value.

The diagnostic shows configured cells, singlets, fluorescence, and final
retention. It is descriptive evidence, not an objective function. Downstream
code should consume exact record revisions through `reader.api`, retain the
Reader-declared replicate identity, and add study semantics in the owning
study.

See [Configuring Reader v8](../core/pipeline.md#cytometry-example) for a complete
authoring example and [Running notebooks](../guides/notebooks.md) for the shared
review surface.
