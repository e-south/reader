# SPOP Endpoint Scoring

**Owner:** reader-maintainers  
**Last verified:** 2026-05-26

SPOP (`sponging_percent_of_positive`) is a Reader-owned endpoint scalar for
pES-retron plus pBbS2c-RFP plate-reader assays. It measures nonzero-IPTG
derepression as a fraction of an aTc-positive anchor, then applies a one-sided
OD600 viability penalty.

```text
reader_spop_endpoint_dose_mean_v1
```

The metric id is literal: endpoint dose-ladder mean, not an AUC. The equation
does not integrate over time and does not weight doses by concentration spacing.

## Contract

- Source owner: Reader
- Public API: `reader.domains.plate_reader.analysis.spop.score_spop_endpoint`
- Metric id: `reader_spop_endpoint_dose_mean_v1`
- Metric family: `sponging_percent_of_positive`
- Numeric scope: `reader_experiment_normalized_tf_sponging`
- Primary scalar: `spop_score` / `normalized_value`
- Raw companion: `spop_score_raw` / `raw_value`
- Direction: maximize
- Reporter readout: `RFP/OD600`
- Viability readout: `OD600`
- Positive anchor: aTc-positive, zero-IPTG endpoint condition
- Scoring doses: nonzero-IPTG endpoint conditions only
- Default viability weight: `lambda = 0.5`

## Assay Shape

Each assay subject has three endpoint contexts:

1. zero-inducer baseline: `0 nM aTc; 0 uM IPTG`;
2. positive anchor: nonzero aTc, zero IPTG;
3. scoring ladder: one or more nonzero IPTG doses.

Reader resolves plate-reader records, treatment parsing, endpoint selection,
normalization, replicate aggregation, and assay QC. SPOP is an assay scalar,
not a sequence property. Construct-backed studies and OPAL campaigns can use it
after a bridge maps Reader assay subjects onto their own record identities.

Downstream records preserve Reader-owned `metric_id`, `numeric_scope`, dose
support vectors, normalization basis, and provenance. They do not redefine the
metric.

## Public API

```python
from reader.domains.plate_reader.analysis.spop import (
    SpopDoseValue,
    score_spop_endpoint,
)
```

Stable fields:

- `SPOP_METRIC_ID = "reader_spop_endpoint_dose_mean_v1"`
- `SPOP_NUMERIC_SCOPE = "reader_experiment_normalized_tf_sponging"`
- `SPOP_NORMALIZATION_BASIS = "rfp_od600_derepression_fraction_relative_to_atc_positive_control"`
- `SPOP_REPORTER_READOUT = "RFP/OD600"`
- `SPOP_VIABILITY_READOUT = "OD600"`
- `SPOP_DEFAULT_LAMBDA = 0.5`
- `score_spop_endpoint(...)`

The scorer consumes endpoint aggregates. Reader experiment bridges resolve
artifacts, choose the endpoint, and aggregate replicate well rows before calling
this API. The RT-lnRNA bridge reads Reader `outputs/manifests/records.json`,
selects the latest `ratio_reporter_normalizer/df` record, and aggregates
replicate endpoint rows by median per design, treatment, and channel.

Direct path scraping loses Reader record ids and content digests.

## Input Contract

`score_spop_endpoint(...)` requires:

- `baseline_rfp_over_od600`: endpoint `RFP/OD600` for `0 nM aTc; 0 uM IPTG`
- `baseline_od600`: endpoint `OD600` for `0 nM aTc; 0 uM IPTG`
- `positive_control_rfp_over_od600`: endpoint `RFP/OD600` for a nonzero-aTc,
  zero-IPTG positive control
- `dose_values`: one or more nonzero-IPTG `SpopDoseValue` rows with:
  - `iptg_uM`
  - `rfp_over_od600`
  - `od600`
  - `replicate_count`, a positive integer

The input values are already endpoint values. The scorer does not choose a
timepoint and does not average across time.

## Notation

- $D$: nonzero IPTG doses used for scoring
- $Z_0$: zero-inducer endpoint `RFP/OD600`
- $O_0$: zero-inducer endpoint `OD600`
- $Z_+$: aTc-positive, zero-IPTG endpoint `RFP/OD600`
- $Z_d$: endpoint `RFP/OD600` at IPTG dose $d$
- $O_d$: endpoint `OD600` at IPTG dose $d$
- $\lambda \in [0,1]$: viability weight

Reader rejects a subject when $Z_+ - Z_0 \le \varepsilon$. The positive anchor
must sit above the baseline.

## Scoring Math

### 1. Per-Dose Derepression

$$
y_d = \frac{Z_d - Z_0}{Z_+ - Z_0}
$$

$y_d = 0$ matches the zero-inducer baseline. $y_d = 1$ reaches the aTc-positive
anchor.

Reader allows observed values outside that interval and flags them:

- $y_d < 0$: `derepression_below_zero_inducer`
- $y_d > 1$: `derepression_exceeds_atc_positive`

### 2. Potency

$$
P = \frac{1}{|D|} \sum_{d \in D} \max(0, y_d)
$$

The raw companion keeps the unclipped mean:

$$
P_{\mathrm{raw}} = \frac{1}{|D|} \sum_{d \in D} y_d
$$

Negative derepression does not lower the primary potency below zero. The raw
companion preserves the sign for QC.

### 3. Viability

$$
g_d = \min\left(1,\frac{O_d}{O_0}\right)
$$

$$
V = \frac{1}{|D|} \sum_{d \in D} g_d
$$

Viability is one-sided: lower OD600 under IPTG can penalize the score, but OD600
above baseline does not add extra reward. Any $g_d < 0.8$ emits
`induction_growth_penalty`.

### 4. Final Scalar

$$
\boxed{
\mathrm{SPOP}
= P \cdot \left((1-\lambda) + \lambda V\right)
}
$$

$$
\mathrm{SPOP}_{\mathrm{raw}}
= P_{\mathrm{raw}} \cdot \left((1-\lambda) + \lambda V\right)
$$

At $\lambda=0.5$, viability and potency contribute equally to the penalty
factor. At $\lambda=0$, viability is ignored. At $\lambda=1$, potency is
multiplied by mean viability.

## Output Contract

`score_spop_endpoint(...)` returns `SpopEndpointScore`:

- `metric_id`
- `numeric_scope`
- `normalization_basis`
- `iptg_doses_uM`
- `y_derepression_by_dose`
- `viability_by_dose`
- `replicate_count_min`
- `spop_potency`
- `spop_viability`
- `spop_score`
- `spop_score_raw`
- `raw_value`
- `normalized_value`
- `qc_flags`

`raw_value` is `spop_score_raw`. `normalized_value` is `spop_score`.
`normalized_value` is normalized within the Reader experiment and positive
anchor context; it is not globally comparable across unrelated source families.

## Bridge Provenance Contract

`SpopEndpointScore` contains the assay scalar and support vectors. Bridges that
materialize SPOP outside Reader carry Reader artifact provenance next to the
scalar:

- `reader_artifact_ref`
- `reader_artifact_record_id`
- `reader_artifact_content_digest`
- `source_of_truth_doc`
- `source_of_truth_api`

These fields are bridge metadata, not alternate metric definitions. They let a
Construct study, OPAL campaign, or other downstream consumer prove which Reader
record supplied the endpoint aggregates.

## Metric Properties

- Maximize the score.
- The score is not guaranteed to stay in $[0,1]$. A dose can exceed the
  aTc-positive anchor, which can make $P > 1$; Reader flags this instead of
  clipping the upper tail.
- The primary score is monotonic in positive-clipped dose responses when
  viability is fixed.
- The viability factor is bounded in $[0,1]$.
- All scoring doses contribute equally.
- Dose spacing is not part of this versioned metric.
- Reader assay normalization is the numeric scope. Do not put SPOP on the same
  numeric scale as Khan or Crawford abundance priors.

## Example

```text
Z_0 = 100
Z_+ = 500
O_0 = 1.0
lambda = 0.5
```

| IPTG (uM) | Z_d = RFP/OD600 | O_d = OD600 |
| ---: | ---: | ---: |
| 5 | 160 | 1.0 |
| 50 | 300 | 1.0 |
| 500 | 460 | 1.0 |

$$
y_5 = \frac{160-100}{500-100}=0.15
$$

$$
y_{50} = \frac{300-100}{500-100}=0.50
$$

$$
y_{500} = \frac{460-100}{500-100}=0.90
$$

All OD600 values match baseline, so $V=1$.

$$
P = \frac{0.15 + 0.50 + 0.90}{3} = 0.5167
$$

$$
\mathrm{SPOP} = 0.5167 \cdot ((1-0.5) + 0.5 \cdot 1) = 0.5167
$$

## Edge Cases and Guards

Fail-fast inputs:

- `lambda_viability` is not finite or outside `[0, 1]`
- baseline OD600 is not positive
- positive-control `RFP/OD600` is not above baseline
- no nonzero-IPTG dose is provided
- a dose has `iptg_uM <= 0`
- a dose has negative OD600
- a dose has non-finite numeric values
- a dose has a non-positive or non-integer `replicate_count`

QC flags with a returned score:

- dose response below baseline
- dose response above the positive anchor
- OD600 under induction below the viability warning threshold
- only one nonzero-IPTG endpoint dose

## Versioning

Changing the equation, dose inclusion policy, viability rule, or normalization
anchor requires a new metric id. Do not silently reinterpret
`reader_spop_endpoint_dose_mean_v1`.
