---
doc_id: reader-plate-response-window
surface: public-analysis-contract
owner: reader-maintainers
last_verified: 2026-07-15
summary: Event-relative response summaries, verified review collections, and experiment-level evidence navigation.
---

# Plate-reader response-window analysis

`response-window` converts manifest-backed plate-reader trajectories into
event-relative state summaries. Only the versioned request selects the source
records, study identity, reference, event, and reductions. Experiment names and
other analysis outputs do not select this contract.

For the shared assay-record boundary and sibling analysis routes, see
[Plate-reader metric outputs](metric_outputs.md).

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
  --reduction-id REDUCTION_ID \
  --format json

uv run reader response-window promoter-evidence-verify \
  outputs/reviews/promoter_evidence/design-id \
  --format json
```

The analysis Python facade is `reader.response_window`. Review and promoter
evidence publication use `reader.response_window_review`.
Both surfaces accept `reader.response_window.request.v3` and publish
`reader.response_window.bundle.v5`. No downstream repository imports Reader
internals.

`preflight` verifies request syntax, source contracts, artifact digests,
intervention bounds, and configured review examples. It also runs every
declared reduction, aggregation, and QC check in memory without writing a
bundle, so `ready: true` means the same measured payload can be built.
`build` publishes atomically and verifies before returning. `verify` repeats the
public bundle contract independently. `review` verifies first and then opens the
generated notebook. JSON output is available on the non-interactive commands for
agents and automation.

`promoter-evidence` first verifies the response-window bundle, then consumes a
separate study-owned candidate-binding artifact. It publishes one white-canvas
300-dpi PNG, one PDF, and a digest-bearing
`reader.response_window.promoter_evidence_bundle.v3` manifest. Its verifier
checks the claim boundary, source contract identities, artifact paths, byte
counts, digests, and exact experiment/design/candidate/reduction selection
parity independently.
The manifest also records the selected row's sequence-authority digest,
source/design family, binding method, and adapter-specific DenseGen
plan/run/library provenance. DenseGen fields are explicitly `null` for
GenBank-backed candidates.

## Ownership

Reader owns source-record resolution and digests, intervention-event
resolution, well-level interpolation and reduction, replicate aggregation and
joint bootstrap draws, same-state reference subtraction, QC, plots, and the
review notebook.

A downstream study owns the request's study identity, treatment and event
declarations, fluorescence reference, treatment targets, candidate identity,
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

For any declared geometric time mean over the event-relative interval
`[a, b]`, each well is reduced before wells are aggregated:

```text
response_well(i) = (1 / (b - a)) integral[a, b]
                   log2[(YFP / CFP)_design,i(t)] dt

fluorescence_well(i) = (1 / (b - a)) integral[a, b]
                       log2[(YFP / OD600)_design,i(t)] dt

r_i = median(response_well(i))
b_i = median(fluorescence_well_design(i))
      - median(fluorescence_well_reference(i))
```

Thus `b_i = 0` means the design and declared reference have equal reduced
fluorescence in the same assay condition. Positive and negative values mean
higher and lower fluorescence than that reference, respectively. The reference
subtraction is state matched; no-stress reference values are not reused for
stressed states. The stress-study request names `pDual-10` as its reference.

The stress-study binding maps the fixed state order as follows:

| Field suffix | Assay condition |
| --- | --- |
| `00` | no stress |
| `10` | ethanol |
| `01` | ciprofloxacin |
| `11` | ethanol plus ciprofloxacin |

The request must declare `study_id`, `state_order: ["00", "10", "01", "11"]`, an
explicit `state_map_ref`, the exact `reference_design_id`, and a
`reader.response_window.display.v1` vocabulary. The display block names each
condition, the intervention, the fluorescence reference, and the response
examples used for review. The versioned request is the authority for that
reference. Reader rejects missing examples, reordered or implicit state
ontologies, a display reference that disagrees with the measurement reference,
or a reference design absent from a selected magnitude record.

## Time and intervention

Reader acquisition time is not automatically time since an intervention. The
request declares the pre- and post-intervention acquisition segments. The v2
event contract uses the segment-gap midpoint as the event estimate and half
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

Event-time sensitivity evaluates the reduction at both declared event bounds.
Because the reduction need not vary linearly with event time, the persisted
`*_event_half_range` is the half-width of a midpoint-centered envelope: the
larger absolute deviation between the midpoint estimate and either bound
estimate. It is not half the distance between the two bound estimates. This
conservative envelope is displayed separately from replicate-bootstrap
uncertainty.
Each response and fluorescence state also records separate policy-clipping and
instrument-overflow flags aggregated across the midpoint and both event-bound
windows. The central `*_bound_kind` describes only the midpoint estimate; it
does not qualify the gray event-sensitivity envelope.

## Clipping and censor bounds

Source records must declare `value_policy_clipped`,
`value_instrument_overflow`, and `value_bound_kind` for every observation.
Missing provenance stops the response-window build; an older capped record is
never inferred to be exact. Finite values changed by the configured cap are
kept distinct from explicit instrument overflow.
The generic ratio transform propagates this capability only when all three
fields are present. It does not upgrade a resumed pre-capability artifact by
inventing exact provenance.

Ratio construction carries both numerator and denominator provenance. An
affected numerator makes the stored ratio a lower bound, an affected
denominator makes it an upper bound, and opposing bounds make it
indeterminate. These labels describe the stored value; they do not impute an
uncensored measurement.

Trace records retain the observation-level flags and bound. Well records count
the policy-clipped and overflow observations that support each reduction and
declare the resulting response or magnitude bound. Design records expose the
same evidence for each `r` and `b` state. A `b` bound includes both the design
and its same-state reference because `b` subtracts the reference magnitude.
The verifier requires trace, well, and state provenance to be internally
consistent. Bootstrap intervals remain intervals around the stored values and
do not convert a bounded observation into an exact one.

## Bundle records

| Record | Contract | Grain |
| --- | --- | --- |
| `tables/wells.parquet` | `plate_reader.response_window.wells.v3` | experiment, design, condition, well, reduction, censor support |
| `tables/designs.parquet` | `plate_reader.response_window.designs.v3` | experiment, design, reduction, state-level censor bounds |
| `tables/bootstrap_draws.parquet` | `plate_reader.response_window.bootstrap_draws.v2` | experiment, design, reduction, draw |
| `tables/traces.parquet` | `plate_reader.response_window.traces.v3` | event-relative trace observation and value bound |
| `tables/events.parquet` | `plate_reader.response_window.events.v2` | experiment event declaration |

`manifest.json` records `study_id`, contract versions, source-record digests,
the request digest, row counts, and every artifact digest. The public verifier
requires manifest/request parity for study, request, experiment universe,
state, display, reference, event, every reduction, replicate statistic,
bootstrap count, confidence level, and persisted QC constraints. Selected
source record IDs, contracts, and content digests must
match each bundled Reader record catalog. It also checks root-bounded artifact
paths, exact Parquet schemas, row counts, condition coverage, cross-table
identities, bootstrap counts, review-plot metadata, and artifact digests before
returning a bundle. The normalized request and source config/catalog snapshots
are bundled, so provenance remains checkable after the bundle moves.

## Review evidence

Generated `review.py` uses Marimo's medium-width layout, one responsive control
row, and a dropdown for the review view. A review collection is the exact
experiment universe recorded in the verified bundle. Reader does not infer that
universe from protocol IDs, directory names, or shared slug fragments.

The notebook offers two identity-preserving navigation routes:

- **One experiment:** select an experiment, then an exact Reader `design_id`.
- **Across experiments:** keep one exact Reader `design_id` selected and show
  every experiment in the review collection that contains it.

Both routes share one focused-design memory. Changing experiments retains the
exact Reader design when the new experiment contains it. If it does not, the
local selector shows a deterministic fallback without erasing the remembered
design, so returning to a compatible experiment restores the intended focus.
Selecting a design in the across-experiment route updates the same focus. The
experiment, condition, and review-view controls have independent reactive
ownership and therefore do not reset when that focus changes.
The selected response-summary ID is remembered independently and is retained
across compatible experiment or design changes; an unavailable summary uses a
temporary deterministic fallback without overwriting that preference.

The second route includes only non-reference designs present in at least two
experiments under the primary reduction. This is navigation, not a declaration
that the experiments are biologically comparable. Reader labels the entity
"Reader design"; candidate, genotype, sequence, and repeat authority require a
separate study binding or study policy. Condition labels come from the verified
display contract while stable state codes remain visible.
When a bundle has no repeated non-reference design, Reader omits the
across-experiment view and preserves every experiment-local view. Within an
across-experiment selection, the response-summary selector offers only
definitions present identically in every displayed experiment; partial
reductions are omitted and semantic drift fails closed.

`experiment.title` is the presentation authority when a source config provides
it. Otherwise Reader applies a domain-neutral fallback to the stable
experiment ID. The fallback does not carry study- or metric-specific acronym
rules, so study-facing collections should author concise titles explicitly.

Only controls relevant to the selected view remain in the control row. View
contracts declare whether selection is experiment-local, multi-experiment, or
collection-wide; whether one reduction or all reductions are shown; and whether
one condition is selected. The primary figure stays visible, while exact rows,
coverage, interpretation, and bundle provenance are loaded in a lazy accordion.

The trajectory view shows replicate medians with a
central 90% replicate interval rather than individual-well lines, then places
the four response and four anchored-fluorescence handoff values directly below
the trajectories. The response handoff shows observed reduced well values as
hollow points and the published aggregate as a short line. Anchored
fluorescence compares independent design and reference aggregates, so Reader
does not fabricate per-well `b_i` points. Asymmetric bootstrap intervals and
event-time sensitivity remain separate. In the anchored-fluorescence
trajectory, solid lines and filled markers identify the selected design;
dashed lines, hollow markers, and lighter replicate intervals identify the
pDual-10 anchor. Dedicated handoff, response-example, reduction-sensitivity,
and QC views remain available through the same viewport. Compact heatmaps use
square cells; row-dense static matrices may remain rectangular to avoid
unbounded page height.

The across-experiment view keeps each experiment on its own sampling grid and
uses one line style and one endpoint row per experiment. It does not pool traces
or calculate an across-experiment mean. Response summaries show observed well
values as hollow points plus the published experiment aggregate. Anchored
fluorescence remains an independent design/reference comparison and therefore
has no fabricated paired-well points. Bootstrap intervals and event-time
sensitivity are displayed separately.
Trajectory lines and bands are pointwise replicate summaries. Endpoint panels
instead reduce each well over the declared window and then aggregate wells;
the figure states this distinction because those operations need not produce
the same numerical curve-derived value.

Assay-neutral navigation mechanics live in `reader.notebook_review`. That module
indexes exact review-collection memberships and has no plate-reader, SFXI,
response-window, study, or campaign semantics. The response-window package owns
its design-row adapter, condition selector, validation, and plots.

Figure canvases remain white under either Marimo theme. State labels and axes
use the study vocabulary, while the numerical records remain
objective-agnostic within the explicit four-condition assay contract.

Static plots show event intervals, the primary handoff matrix, reduction
stability, repeated-design agreement, and uncertainty sources. Each plot
declares a premise, decision value, rationale, alt text, and non-claim boundary.

## Promoter evidence composite

The response-window promoter-evidence composite contains:

1. a readable experiment and response-summary header, with exact identities in
   provenance;
2. growth, `log2(YFP/CFP)`, and `log2(YFP/OD600)` trajectories, including the
   same-state declared fluorescence reference;
3. one square, symbolic eight-value handoff with observed `r_i` wells, the
   published response and fluorescence aggregates, asymmetric bootstrap
   intervals, and event-time sensitivity drawn and labeled separately;
4. an objective-neutral provenance and QC card; and
5. a BaseRender sequence panel beside the handoff, titled according to its actual
   `densegen_tfbs` or `usr_genbank_annotations_v1` adapter.

Reader calls only the versioned public `dnadesign.baserender` sequence-panel
API. It passes the verified binding row in memory. It does not import a study
package, open a candidate table or USR dataset, infer an alias, or compute RMF.

### Candidate-binding input

The required study artifact uses
`dnadesign.study.promoter_candidate_bindings.v1` at version `1`:

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
namespace and resolves one exact alias for the requested design. The binding
manifest's `study_id` must equal the verified response-window bundle's
`study_id`.

Other assay outputs may consume the same binding artifact for candidate and
sequence context. This shares study identity infrastructure, not reductions,
vector names, plot contracts, or objective math. Model features and OPAL
readiness are separate keyed contracts and are not fields in the
candidate-binding artifact.

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

Materialization rejects incomplete state support, missing or unauthorized
references, non-finite values, insufficient replicate or time coverage,
duplicate reduction identities, and source-record drift. It does not infer
treatment masks, candidate sequences, or an OPAL objective.

Promoter-evidence publication additionally rejects a response selection that
does not resolve to exactly one design row, a binding whose Reader alias differs
from that selection, a binding issued for another study, an overlay whose
experiment/design/reduction differs from the selection, and BaseRender
diagnostics that disagree with the binding.

## Stress-study binding

The study request lives in dnadesign at
`src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/config/reader_response_window.yaml`.
It names eight existing experiments and seven reductions. The primary reduction
is the 4-8 hour post-event geometric log mean; the other declared windows,
normalized linear AUC, and pre-window delta remain sensitivity analyses. The request
declares `study_id: stress_ethanol_cipro_growth`; that identity must match the
bundle manifest and any candidate-binding artifact used for promoter evidence.

See [Reader-to-OPAL handoff](opal_handoff.md) for the cross-repository boundary.
