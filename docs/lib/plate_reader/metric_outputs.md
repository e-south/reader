---
doc_id: reader-plate-reader-metric-outputs
surface: library-router
owner: reader-maintainers
last_verified: 2026-08-01
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
| four-state vector | `logic/four_state_vector_screen` | one selected acquisition-time snapshot | `four_state_vector/vector` under `logic.four_state_vector.v1` |
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
Historical four-state vector normally does
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
          identity_scope:
            entity_columns: [sample_id_alias]
          condition_column: condition_alias
          condition_order_ref: condition_order
          observation_unit:
            role: observation_only
            column: position
          format: [png, pdf]
```

When `evidence.replicate_identity_field` is declared, the plot reduces
observations within that explicitly named identity before comparing conditions.
The identity is scoped by `identity_scope.entity_columns`, the declared
condition, and the replicate identity field; Reader does not impose a second
plate-level replicate tier. This semantic scope is independent of `partition`,
which selects presentation artifacts but may not redefine or pool replicate
populations. Each diagnostic partition must resolve to exactly one entity
tuple. Use a comparison figure with an explicit aggregation contract when
several subjects or genotypes belong in one visual. The single-reporter
compiler defaults the entity scope to canonical `design_id`; a view must
override it explicitly when another persisted subject or genotype column is
the correct owner. If one declared replicate contains several recorded
observations, the view must also name their column through the observation-only
contract shown above. Without that contract, each declared replicate identity
must resolve to one aligned trace. If the replicate identity field is absent,
grouping is unresolved even when `replicate_kind` is known. The plot then fails
unless its view explicitly declares an `observation_unit` with
`role: observation_only`. This opt-in keeps descriptive well or position traces
available while labeling their points as observations, not replicates.
Experiment, plate, sheet, well, and position fields otherwise remain
acquisition provenance and are never implicit replicate identities. Use
`replicate_kind: unknown` when the kind is not established. The compiler owns
the temporal and aggregation policies plus the reporter, normalizer, ratio, and
time-channel bindings; a plot view owns presentation partitioning, semantic
entity scope, any explicit observation-only unit, condition presentation, and
figure options.

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

- [four-state vector in Reader](../four_state_vector_in_reader.md)
- [Response-window analysis](response_window.md)
- [Notebook operation](../../guides/notebooks.md)
- [Plugin development](../../core/plugins.md)
