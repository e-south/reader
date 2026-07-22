---
doc_id: reader-plate-reader-metric-outputs
surface: library-router
owner: reader-maintainers
last_verified: 2026-07-20
summary: Route compatible dual-reporter assay records to independent SFXI and response-window outputs.
---

# Plate-reader metric outputs

Reader separates assay ingestion from analysis-specific outputs. A compatible
dual-reporter experiment can support SFXI, response-window analysis, or both,
but each output requires its own explicit semantic contract. An experiment name
does not select either analysis.

## Shared assay records

The common plate-reader path performs these operations once:

1. ingest a declared workbook format such as Synergy H1;
2. merge the sample map and apply authored design and treatment labels;
3. apply blank-correction and overflow policy; and
4. derive configured reporter and growth-normalized ratios.

These steps publish manifest-backed `plate_reader.annotated.v1` records. They do
not choose a summary time, intervention event, vector ontology, target mask, or
objective.

## Choose an output lane

| Output lane | Explicit selector | Time basis | Durable Reader output | Review surface |
| --- | --- | --- | --- | --- |
| SFXI | `logic/sfxi_screen` protocol inputs and analysis settings | one acquisition-time snapshot | `sfxi_vec8/vec8` under `sfxi.vec8.v3` | per-experiment `notebook/sfxi_eda` and SFXI plot bundles |
| Response window | `reader.response_window.request.v3` | one declared intervention and event-relative windows | `reader.response_window.bundle.v5` with typed well, design, trace, event, bootstrap, and censor-bound records | bundle-generated `review.py` and optional promoter-evidence bundle |

The SFXI notebook may recompute a vector at an interactively selected time for
review, but it does not persist that table. The manifest-backed vec8 record is
the handoff. The response-window notebook reads a verified bundle and does not
calculate an OPAL objective.

## Candidate and sequence context

Candidate identity is supplied separately by a study-owned
`dnadesign.study.promoter_candidate_bindings.v1` artifact. Reader resolves only
the exact `reader.design_id` namespace and uses the public BaseRender sequence
panel API. It does not open a study candidate table, infer aliases, or treat
Reader sequence metadata as candidate authority.

Both output lanes may consume the same binding artifact without sharing
reductions, vector fields, notebooks, plots, or objective logic. Response-window
promoter evidence additionally requires the binding artifact and response
bundle to declare the same `study_id`.

The publication image in
`reader.response_window.promoter_evidence_bundle.v5` keeps trajectories, the
eight-value handoff, and the BaseRender sequence panel in one compact viewport.
Exact binding provenance, source digests, BaseRender diagnostics, QC claim
boundaries, and any screen-only objective overlay remain structured manifest
fields. A notebook can disclose those fields on demand without painting them
into the assay-evidence bitmap.

## Downstream boundary

- An SFXI vec8 remains in the SFXI y-space and is interpreted only by an
  explicitly configured SFXI objective.
- Response-window values remain assay summaries until the owning study applies
  its repeat-aggregation and label-promotion contract. OPAL then ingests that
  typed response y-space and applies only the objective named by the configured
  selection view.
- Candidate identity, model-feature readiness, label promotion, and objective
  evaluation are separate readiness checks.

## Continue by task

- [SFXI in Reader](../sfxi_vec8_in_reader.md)
- [Plate-reader response-window analysis](response_window.md)
- [Reader-to-OPAL handoff](opal_handoff.md)
- [Notebook operation](../../guides/notebooks.md)
- [Plugin development](../../core/plugins.md)
