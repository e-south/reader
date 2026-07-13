---
doc_id: reader-plate-response-window
surface: public-analysis-contract
owner: reader-maintainers
last_verified: 2026-07-13
summary: Reader contract for event-relative response and reference-relative fluorescence summaries.
---

# Plate-reader response-window analysis

`response-window` converts manifest-backed plate-reader trajectories into
event-relative state summaries. It is independent of SFXI. Experiment names do
not select the reduction contract; only the versioned request does.

## Public service

```bash
uv run reader response-window preflight REQUEST.yaml \
  --reader-root . \
  --format json

uv run reader response-window build REQUEST.yaml \
  --reader-root . \
  --out-dir outputs/reviews/response_window/latest \
  --overwrite \
  --format json

uv run reader response-window verify \
  outputs/reviews/response_window/latest \
  --format json

uv run reader response-window review \
  outputs/reviews/response_window/latest \
  --mode run

uv run reader response-window promoter-evidence \
  outputs/reviews/response_window/latest \
  PATH/TO/promoter_candidate_bindings \
  --out-dir outputs/reviews/promoter_evidence/design-id \
  --experiment-id EXPERIMENT_ID \
  --design-id READER_DESIGN_ID \
  --reduction-id event_logmean_6_12h_post \
  --format json

uv run reader response-window promoter-evidence-verify \
  outputs/reviews/promoter_evidence/design-id \
  --format json
```

The analysis Python facade is `reader.response_window`. Review and promoter
evidence publication use `reader.response_window_review`.
Both surfaces accept `reader.response_window.request.v2` and publish
`reader.response_window.bundle.v3`. No downstream repository imports Reader
internals.

`preflight` verifies request syntax, source contracts, artifact digests,
intervention bounds, and configured review examples without writing a bundle.
`build` publishes atomically and verifies before returning. `verify` repeats the
public bundle contract independently. `review` verifies first and then opens the
generated notebook. JSON output is available on the non-interactive commands for
agents and automation.

`promoter-evidence` first verifies the response-window bundle, then consumes a
separate study-owned candidate-binding artifact. It publishes one white-canvas
300-dpi PNG, one PDF, and a digest-bearing
`reader.response_window.promoter_evidence_bundle.v1` manifest. Its verifier
checks the claim boundary, source contract identities, artifact paths, byte
counts, and digests independently.
The manifest also records the selected row's sequence-authority digest,
source/design family, binding method, and adapter-specific DenseGen
plan/run/library provenance. DenseGen fields are explicitly `null` for
GenBank-backed candidates.

## Ownership

Reader owns source-record resolution and digests, intervention-event
resolution, well-level interpolation and reduction, replicate aggregation and
joint bootstrap draws, same-state reference subtraction, QC, plots, and the
review notebook.

A downstream study owns treatment targets, candidate identity,
repeat-measurement aggregation across experiments, and label promotion. OPAL
owns objective evaluation, model fitting, acquisition, and ledgers.

## Measured axes

Reader publishes an assay summary, not a campaign metric. For state `i` and a
declared window, the compact record fields are:

```text
r_i = replicate aggregate of reduced log2[(YFP / CFP)_design]
b_i = replicate aggregate of reduced log2[(YFP / OD600)_design]
      - replicate aggregate of reduced log2[(YFP / OD600)_reference]
```

The record retains `r00, r10, r01, r11, b00, b10, b01, b11`. These are assay
summaries, not an SFXI vec8, campaign score, or OPAL label. Channel names are
configured source fields; record semantics are response and
reference-relative fluorescence magnitude.

For the primary 6-12 hour post-event geometric time mean, each well is
reduced before wells are aggregated:

```text
response_well(i) = (1 / 6 h) integral[6 h, 12 h]
                   log2[(YFP / CFP)_design,i(t)] dt

fluorescence_well(i) = (1 / 6 h) integral[6 h, 12 h]
                       log2[(YFP / OD600)_design,i(t)] dt

r_i = median(response_well(i))
b_i = median(fluorescence_well_design(i))
      - median(fluorescence_well_pDual-10(i))
```

Thus `b_i = 0` means the design and pDual-10 have equal reduced fluorescence
in the same assay condition. Positive and negative values mean higher and
lower fluorescence than pDual-10, respectively. The reference subtraction is
state matched; no-stress reference values are not reused for stressed states.

The stress-study binding maps the fixed state order as follows:

| Field suffix | Assay condition |
| --- | --- |
| `00` | no stress |
| `10` | ethanol |
| `01` | ciprofloxacin |
| `11` | ethanol plus ciprofloxacin |

The request must declare `state_order: [00, 10, 01, 11]`, an explicit
`state_map_ref`, and a `reader.response_window.display.v1` vocabulary. The
display block names each condition, the intervention, the fluorescence anchor,
and the response examples used for review. Reader rejects missing examples,
reordered or implicit state ontologies, and a display anchor that disagrees
with the measurement reference. The
configured reference design is also checked against a named, contract-validated
authority record; mere presence on a plate is not reference authority.

## Time and intervention

Reader acquisition time is not automatically time since an intervention. The
request declares the pre- and post-intervention acquisition segments. The
The v2 event contract uses the segment-gap midpoint as the event estimate and half
the gap as symmetric timing uncertainty. Other estimate rules are rejected
until they have an explicit asymmetric uncertainty contract. Reader records the
interval, estimate, and event-relative time.

Sheet order is provenance, not treatment semantics. Missing, ambiguous,
nonchronological, or out-of-range event declarations stop materialization.

## Reductions

Every reduction declares its ID and role, event-relative window, method,
response basis, and replicate statistic. Reader has no implicit time window.
Geometric means integrate the declared log2 response (`YFP/CFP`) or
fluorescence (`YFP/OD600`) ratio. Normalized linear AUC integrates the
corresponding linear ratio and then takes log2. They are different estimands
and require different IDs. Extrapolation is forbidden, interior gaps are
bounded, and ratios at or below the declared positive floor fail.

## Bundle records

| Record | Contract | Grain |
| --- | --- | --- |
| `tables/wells.parquet` | `plate_reader.response_window.wells.v2` | experiment, design, condition, well, reduction |
| `tables/designs.parquet` | `plate_reader.response_window.designs.v2` | experiment, design, reduction |
| `tables/bootstrap_draws.parquet` | `plate_reader.response_window.bootstrap_draws.v2` | experiment, design, reduction, draw |
| `tables/traces.parquet` | `plate_reader.response_window.traces.v2` | event-relative trace observation |
| `tables/events.parquet` | `plate_reader.response_window.events.v2` | experiment event declaration |

`manifest.json` records contract versions, source-record digests, request
digest, row counts, and every artifact digest. The public verifier checks
root-bounded artifact paths, exact Parquet schemas, row counts, condition
coverage, cross-table identities, reduction semantics, reference identity,
bootstrap counts, review-plot metadata, and artifact digests before returning a
bundle. The normalized request and source config/catalog snapshots are bundled,
so provenance remains checkable after the bundle moves.

## Review evidence

Generated `review.py` uses Marimo's medium-width layout, one responsive control
row, and a dropdown for the review view. Only controls relevant to the selected
view remain in that row. The trajectory view shows replicate medians with a
central 90% replicate interval rather than individual-well lines, then places
the four response and four anchored-fluorescence handoff values directly below
the trajectories. Dedicated handoff, response-example, reduction-sensitivity,
and QC views remain available through the same viewport. Compact heatmaps use
square cells; row-dense static matrices may remain rectangular to avoid
unbounded page height.

Figure canvases remain white under either Marimo theme. State labels and axes
use the study vocabulary, while the numerical records remain
objective-agnostic within the explicit four-condition assay contract.

Static plots show event intervals, the primary handoff matrix, reduction
stability, repeated-design agreement, and uncertainty sources. Each plot
declares a premise, decision value, rationale, alt text, and non-claim boundary.

## Promoter evidence composite

The promoter-evidence composite is a new response-window surface, not an SFXI
triptych variant. One viewport contains:

1. an explicit header with experiment, reduction, candidate, and exact binding;
2. growth, `log2(YFP/CFP)`, and `log2(YFP/OD600)` trajectories, including the
   same-state pDual-10 fluorescence anchor;
3. separate `r_i` and `b_i` dot-and-whisker panels, with bootstrap SD and
   event-time sensitivity drawn and labeled separately;
4. an objective-neutral provenance and QC card; and
5. a full-width BaseRender sequence panel titled according to its actual
   `densegen_tfbs` or `usr_genbank_annotations_v1` adapter.

Reader calls only the versioned public `dnadesign.baserender` sequence-panel
API. It passes the verified binding row in memory. It does not import a study
package, open a candidate table or USR dataset, infer an alias, or compute RMF.

### Candidate-binding input

The required study artifact uses
`dnadesign.study.promoter_candidate_bindings.v1` at version `1` for
`stress_ethanol_cipro_growth`:

```text
manifest.json
bindings.parquet
```

The manifest pins the candidate selection, source artifacts, BaseRender
contract, Parquet row count, and content digest. Parquet metadata repeats the
schema, study, and record identities. Every typed `(alias_namespace, alias)`
pair occurs exactly once, while several aliases may identify one candidate.
Rows carry the canonical sequence and digest, candidate and sequence authority,
BaseRender adapter fields, DenseGen or GenBank annotations, and explicit
resolution status and method. Reader selects only the `reader.design_id`
namespace and resolves one exact alias for the requested design.

The SFXI triptych may consume the same binding artifact for candidate and
sequence context. That reuse shares study identity infrastructure only; it does
not share reductions, vector names, plot contracts, or objective math. Model
features and OPAL readiness are separate keyed contracts and are not fields in
the candidate-binding artifact.

Reader rejects missing or duplicated typed aliases, fuzzy resolution methods,
candidate/sequence digest disagreement, unsupported adapters, extra columns,
schema or Parquet-metadata drift, empty or non-contract annotation metadata,
invalid GenBank spans, non-POSIX or escaping artifact references, and source or
file digest drift.

### Optional objective display overlay

An optional `reader.response_window.objective_display_overlay.v1` JSON file may
display study-supplied raw objective components and must declare
`claim_status: screen_only`. Version 1 rejects calibrated scores, limiting
components, promotion fields, and `production` claim status entirely. A future
promoted display requires a new contract that verifies the referenced label and
calibration artifacts. Reader displays the supplied raw values; it never
derives, calibrates, or promotes them.

## Fail-fast boundary

Materialization rejects incomplete state support, missing or unauthorized references,
non-finite values, insufficient replicate or time coverage, duplicate reduction
identities, and source-record drift. It does not infer treatment masks,
candidate sequences, or an OPAL objective.

Promoter-evidence publication additionally rejects a response selection that
does not resolve to exactly one design row, a binding whose Reader alias differs
from that selection, an overlay whose experiment/design/reduction differs from
the selection, and BaseRender diagnostics that disagree with the binding.

## Stress-study binding

The study request lives in dnadesign at
`src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/config/reader_response_window.yaml`.
It names eight existing experiments and five reductions. The primary reduction
is the 6-12 hour post-event geometric log mean; adjacent windows, normalized
linear AUC, and pre-window delta remain sensitivity analyses.

See [Reader-to-OPAL handoff](opal_handoff.md) for the cross-repository boundary.
