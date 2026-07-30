---
doc_id: reader-plate-response-window
surface: public-analysis-contract
owner: reader-maintainers
last_verified: 2026-07-29
summary: Event-relative response summaries produced by the canonical record-backed experiment lifecycle.
---

# Plate-reader response-window analysis

The `plate_reader/response_window` protocol converts declared dataframe records
into event-relative summaries. It is a specialized analysis over generic
Reader record resources, not a separate service or publication format.

## Lifecycle

```bash
uv run reader init experiments/response_summary --protocol plate_reader/response_window
# Edit config.yaml: bind response, magnitude, and trajectory record resources.
uv run reader inspect experiments/response_summary
uv run reader validate experiments/response_summary
uv run reader run experiments/response_summary
uv run reader records experiments/response_summary
uv run reader verify experiments/response_summary
uv run reader notebook experiments/response_summary --mode none
```

The aggregate is an experiment and owns its generated records, plots, exports,
notebook, and manifests under its own `outputs/`. Reader does not publish to a
repository-root output directory.

## Contract

The protocol declares three aligned record collections: response, magnitude,
and trajectory. Its analysis block owns channel bindings, the reference design,
the state-value mapping, the event definition, reductions, aggregation, and
quality thresholds. A normal run produces manifest-backed records for:

- resolved event timing and event-relative coordinates;
- well-level interpolation and reduction results;
- replicate aggregation, uncertainty, and censor bounds; and
- design summaries, traces, and event intervals.

The standard summary plot and CSV export plugins consume those records. An
optional focused diagnostic renders growth and response traces, magnitude
traces beside the same-source reference, and the eight persisted response and
anchored-magnitude components for one source design. It is a normal plot
artifact, so it appears in the same record catalog and canonical EDA notebook
as every other generated figure.

Enable the diagnostic explicitly and identify the record row to render:

```yaml
protocol:
  outputs:
    plots:
      include: [response_window_summary, response_window_diagnostic]
      views:
        response_window_diagnostic:
          source_experiment_id: trace-source
          design_id: design-a
```

The compiler supplies the reduction marked `role: primary`; plot configuration
cannot override that scientific contract. The diagnostic labels the reduction
method, response basis, replicate statistic, bootstrap confidence level, and
event-time uncertainty. It distinguishes bootstrap intervals from event-time
sensitivity, shows the pre-event window for `post_minus_pre`, and marks
non-exact, clipped, or overflow-affected values. Python callers discover the
manifest-backed record catalog with `reader.api.records()` and load dataframe
contents with `reader.api.read_dataframe()`.

Channel labels and state values are source metadata. The response-window
contract does not turn them, the selected design, or the eight-component record
into a downstream biological objective or campaign claim.

## Failure boundary

Before output mutation, Reader rejects empty collections, missing or changed
source records, incompatible dataframe contracts, and content-digest drift.
The domain analysis then rejects misaligned experiment order, ambiguous state
mappings, invalid event bounds or reductions, and insufficient trace support.
It does not infer identity or semantics from experiment names.

Open an experiment with `reader.open_experiment()` before using those public
`reader.api` operations.

## Related references

- [Plate-reader metric outputs](metric_outputs.md)
- [Record provenance](../../core/record_provenance.md)
- [Preflight, run, verify](../../guides/preflight_run_verify.md)
