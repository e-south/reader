---
doc_id: reader-plate-reader-metric-outputs
surface: library-router
owner: reader-maintainers
last_verified: 2026-07-30
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

The lanes may read compatible source records, but one output is never inferred
from filenames or relabeled as another record schema. They are separately
addressable and independently validatable. When a coordinate is derived from
the same verified source revision and selected rows with the same time basis or
event origin, temporal operator and support, ratio order, grouping,
within-experiment observation aggregation, reference operation, and numerical
tolerance, that shared reduced coordinate must be numerically identical.
Historical SFXI vec8 normally does
not meet that identity: it uses one acquisition-time snapshot, per-design logic
scaling, and corner-specific intensity normalization. Downstream interpretation
remains a consumer concern.

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

## Single-reporter diagnostic

`single_reporter_diagnostic` is an optional plot output of
`plate_reader/single_reporter_screen`. The protocol filters the persisted ratio
record to `type: SAMPLE`, validates required treatment and design metadata, and
publishes that projection as `sample_measurements/df` under
`plate_reader.annotated.v1`. Publication trims treatment and design identifiers,
rejects identifiers that are then blank, and requires finite time and value
columns. Unlabeled blank rows therefore cannot invalidate fully annotated
sample evidence. The diagnostic reads only that canonical
sample record and renders four approximately square
panels in one row: normalizer kinetics, reporter kinetics, their ratio
kinetics, and the ratio reduced by the declared condition. The reduction panel
keeps temporally reduced observation-unit values visible and shows the
normalizer on a labeled QC axis rather than treating it as an objective.

Temporal reduction is analysis policy, not plot configuration. The neutral
contract declares an absolute or event-relative time basis, an endpoint or
inclusive interval, a numerical method and output space, and explicit support,
gap, positivity, and censor handling. Single-reporter acquisition traces require
the absolute basis. Within-unit observation reduction and the displayed
across-unit center are configured separately:

```yaml
protocol:
  analysis:
    temporal_reduction:
      selection:
        kind: interval
        time_basis: absolute
        start_h: 8.0
        end_h: 12.0
        boundary: inclusive
      method: observed_median
      output_space: linear
      support:
        boundary_support: observed
        minimum_observations: 25
        maximum_interior_gap_h: 0.2
        positive_floor: null
        positive_value_scope: selected_support
        censored_values: reject
    observation_aggregation:
      within_unit_statistic: median
      across_unit_statistic: median
  outputs:
    plots:
      profile: none
      include: [single_reporter_diagnostic]
      views:
        single_reporter_diagnostic:
          partition:
            by: sample_id_alias
          condition_column: condition_alias
          condition_order_ref: condition_order
          format: [png, pdf]
```

When `evidence.replicate_identity_field` is declared, the plot reduces
observations within that explicitly named unit before comparing conditions. If
it is absent, the replicate-kind declaration applies to the experiment and each
well position remains a separate within-experiment plot unit. Thus a study in
which each physical plate is a biological replicate can declare
`replicate_kind: biological` on every plate experiment without inventing a
within-plate identity field. Neither a well position nor spatial proximity
establishes technical replication. Use `replicate_kind: unknown` when the kind
is not established, and omit the identity field when no grouping relationship
is established. The compiler owns the temporal and aggregation policies plus
the reporter, normalizer, ratio, and time-channel bindings; a plot view owns
only partitioning, condition presentation, and figure options.

This figure is descriptive. Its endpoint or interval is authored experiment
policy, not an inferred event, dose rule, control ontology, ranking, or study
objective. A downstream study may validate and select such a policy while the
Reader plot stays reusable across single-reporter assays. The same neutral
temporal contract also underlies response-window trace reduction, but the
response-window protocol separately owns event estimation, interpolation,
reference anchoring, log2 output, descriptive resampling, and event-time
sensitivity. Matching source data and nominal bounds imply matching values only
when every reduction and support
setting also matches.

## Continue by task

- [SFXI in Reader](../sfxi_vec8_in_reader.md)
- [Response-window analysis](response_window.md)
- [Notebook operation](../../guides/notebooks.md)
- [Plugin development](../../core/plugins.md)
