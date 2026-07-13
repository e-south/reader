---
doc_id: reader-opal-handoff
surface: ownership-reference
owner: reader-maintainers
last_verified: 2026-07-13
summary: Public Reader record boundary for study staging and OPAL use.
---

# Reader-to-OPAL handoff

The handoff is a verified record bundle, not a Python import across
repositories.

Reader publishes measured records and provenance. The study owns the treatment
and intervention declaration in its versioned request; Reader verifies that
the declared source segments exist, resolves the measured event interval, and
publishes event-relative records. A study then verifies the bundle,
maps Reader identities to candidate and sequence authority, aggregates repeated
experiments under one declared rule, and explicitly promotes one label
contract. A separate candidate-keyed contract connects study identity to an
OPAL feature view. OPAL validates labels and feature shape, fits a model, scores
candidates, and records selection decisions.

## Readiness

| State | Required evidence | Owner |
| --- | --- | --- |
| assay ready | Verified Reader bundle, source digests, event records, QC, and uncertainty | Reader |
| candidate ready | Exact typed Reader design-to-candidate/sequence binding | Study |
| label ready | One versioned reduction and repeat-aggregation rule | Study |
| model ready | Candidate identity joins exactly to the declared feature view | Study and OPAL |
| ingest ready | OPAL dry ingest validates identity, shape, and label contract | OPAL |
| applied | Explicit mutation intent and successful OPAL ingest | OPAL |

Reader `design_id`, aliases, and optional sequence columns are not candidate
authority. Reference rows anchor fluorescence magnitude and are not training
labels. Reader evidence shown in an OPAL notebook remains evidence, not a
second label source.

For publication evidence, the study materializes
`dnadesign.study.promoter_candidate_bindings.v1`. This study-wide identity
artifact is available to any consumer that understands one of its typed alias
namespaces. Reader verifies the artifact and uses its exact `reader.design_id`
alias, candidate, sequence, and BaseRender projection without importing study
internals. The resulting
`reader.response_window.promoter_evidence_bundle.v1` is a display artifact for
downstream notebooks; OPAL does not reconstruct its trajectories or resolve its
candidate identity again.

The v1 evidence manifest preserves the selected binding's sequence authority,
source/design family, exact binding method, and adapter-specific DenseGen
provenance. Model features use a separate candidate-keyed projection. The
optional objective overlay is screen-only raw evidence. Production RMF scores
and promotion provenance are outside this contract while the typed label
handoff is inactive.

For canonical SFXI vec8 use, see [SFXI vec8](../sfxi/vec8.md). For
event-relative response summaries, see
[plate-reader response-window analysis](response_window.md).
