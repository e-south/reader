---
doc_id: reader-opal-handoff
surface: ownership-reference
owner: reader-maintainers
last_verified: 2026-07-14
summary: Public Reader record boundary for study staging and OPAL use.
---

# Reader-to-OPAL handoff

The handoff is a verified record bundle, not a Python import across
repositories.

Reader publishes measured records and provenance. The study owns the study
identifier, treatment, intervention, and fluorescence-reference declarations in its
versioned request. Reader verifies that the declared source records and
segments exist, resolves the measured event interval, and publishes
event-relative records. The study then verifies the bundle, maps Reader
identities to candidate and sequence authority, aggregates repeated experiments
under one declared rule, and explicitly promotes one label contract. A separate
candidate-keyed contract connects study identity to an OPAL feature view. OPAL
validates labels and feature shape, fits a model, scores candidates, and records
selection decisions.

## Readiness

| State | Required evidence | Owner |
| --- | --- | --- |
| assay ready | Verified Reader bundle, request and study identity parity, source digests, event records, QC, and uncertainty | Reader |
| candidate ready | Exact typed Reader design-to-candidate/sequence binding | Study |
| label ready | One versioned reduction and repeat-aggregation rule | Study |
| model ready | Candidate identity joins exactly to the declared feature view | Study and OPAL |
| ingest ready | OPAL dry ingest validates identity, shape, and label contract | OPAL |
| applied | Explicit mutation intent and successful OPAL ingest | OPAL |

Reader `design_id`, aliases, and optional sequence columns are not candidate
authority. Reference rows anchor fluorescence magnitude and are not training
labels. Reader evidence shown in an OPAL notebook remains evidence, not a
second label source.

The response-window route accepts `reader.response_window.request.v3` and
publishes `reader.response_window.bundle.v5`. Its eight response and
reference-relative fluorescence values remain assay summaries until the study
applies its declared repeat-aggregation and label-promotion contracts. OPAL
ingests that typed response y-space; Reader does not calculate or publish a
downstream objective score.

For publication evidence, the study materializes
`dnadesign.study.promoter_candidate_bindings.v1`. This study-wide identity
artifact is available to any consumer that understands one of its typed alias
namespaces. Reader verifies the artifact and uses its exact `reader.design_id`
alias, candidate, sequence, and BaseRender projection without importing study
internals. The resulting
`reader.response_window.promoter_evidence_bundle.v3` is a display artifact for
downstream notebooks; OPAL does not reconstruct its trajectories or resolve its
candidate identity again.

The v2 evidence manifest requires response-bundle and candidate-binding
`study_id` parity. It preserves the selected binding's sequence authority,
source/design family, exact binding method, and adapter-specific DenseGen
provenance. Model features use a separate candidate-keyed projection. The
optional objective overlay is screen-only raw evidence. Objective scores,
normalization or calibration parameters, and promotion provenance belong to
study and OPAL contracts, not the Reader evidence bundle.

Start with [plate-reader metric outputs](metric_outputs.md) to choose the assay
route. The output contracts are documented in [SFXI vec8](../sfxi/vec8.md) and
[plate-reader response-window analysis](response_window.md).
