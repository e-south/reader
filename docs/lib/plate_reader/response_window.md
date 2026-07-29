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

The standard plot and CSV export plugins consume those records. Python callers
discover the manifest-backed record catalog with `reader.api.records()` and
load dataframe contents with `reader.api.read_dataframe()`.

Channel labels are source metadata. The response-window contract does not turn
a named channel into a downstream biological or campaign claim.

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
