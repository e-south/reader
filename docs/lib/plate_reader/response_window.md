---
doc_id: reader-plate-response-window
surface: public-analysis-contract
owner: reader-maintainers
last_verified: 2026-07-29
summary: Event-relative response summaries and verified experiment-scoped review bundles.
---

# Plate-reader response-window analysis

`response-window` converts manifest-backed trajectories into event-relative
summaries. Only a `reader.response_window.request.v3` request selects source
records, event timing, reductions, and display examples.

## Lifecycle

```bash
uv run reader response-window preflight REQUEST.yaml --reader-root . --format json

uv run reader response-window build REQUEST.yaml \
  --reader-root . \
  --output-experiment experiments/2026/20260717_response_window_aggregate \
  --overwrite \
  --format json

uv run reader response-window verify \
  experiments/2026/20260717_response_window_aggregate/outputs/bundles/response-window \
  --format json

uv run reader response-window review \
  experiments/2026/20260717_response_window_aggregate/outputs/bundles/response-window \
  --mode run
```

The output experiment is required. An aggregate is a unit of work, so its
bundle lives below `experiments/<year>/<experiment>/outputs/bundles/`; Reader
does not publish to a repository-root output directory.

## Contract

Preflight verifies request syntax, source contracts, digests, event bounds,
reductions, aggregation, QC, and configured examples without publishing.
Build writes atomically and verifies before returning. Verify repeats the
public contract independently. Review verifies first and then opens the
generated notebook.

The bundle records:

- source experiment, record, contract, and content digests;
- resolved event timing and event-relative coordinates;
- well-level interpolation and reduction results;
- replicate aggregation, uncertainty, and censor bounds;
- display-ready tables and a generated review notebook.

Channel labels are source metadata. The response-window contract does not turn
a named channel into a downstream biological or campaign claim.

## Failure boundary

Reader rejects missing or changed records, ambiguous state mappings,
out-of-range events, invalid reductions, insufficient coverage, digest drift,
unsafe paths, incomplete displays, and incompatible schemas. It does not infer
missing identity from experiment names.

The Python facade is `reader.response_window`. Review-table helpers are
available from `reader.response_window_review`; consumers do not need package
internals.

## Related references

- [Plate-reader metric outputs](metric_outputs.md)
- [Record provenance](../../core/record_provenance.md)
- [Preflight, run, verify](../../guides/preflight_run_verify.md)
