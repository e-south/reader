---
doc_id: reader-sfxi-vec8-contract
surface: library-reference
owner: reader-maintainers
last_verified: 2026-07-17
summary: Reader-owned SFXI vec8 input, selection, calculation, output, and failure contract.
---

# SFXI vec8 contract

This reference describes the measured SFXI vec8 produced by Reader. It covers
the calculation and record contract only; downstream interpretation is outside
the package.

Return to the [SFXI entry point](../sfxi_vec8_in_reader.md) for operating,
plot, and handoff routes.

## Vector definition

Reader emits the fixed state order `00, 10, 01, 11`:

```text
[v00, v10, v01, v11, y00_star, y10_star, y01_star, y11_star]
```

- `v00..v11` describe the shape of the configured logic channel. Values are
  unit-interval values derived from a log2 transform and per-design scaling.
- `y00_star..y11_star` describe the configured intensity channel after
  corner-specific reference normalization. Values are stored in log2 space.
- `r_logic` is the guarded linear dynamic range of the four logic-channel
  corner means. It is a diagnostic, not a ninth vector component.

## Input contract

The `logic/sfxi_screen` protocol feeds the transform from
`promote_to_tidy_plus_map/df`, a `plate_reader.annotated.v1` dataframe record.
The calculation requires:

- `position`, the well identifier required by the annotated-table contract
- the configured time column, normally `time`
- `channel` and numeric `value`
- every configured `design_by` column; the first must be `design_id`
- a treatment column that matches the selected ordered state space

The protocol resolves `annotations.ordered_state_spaces.<name>` through
`protocol.inputs.state_map_ref`. The state space must declare exactly `00`, `10`,
`01`, and `11`, with one treatment label per corner. Duplicate labels are
rejected after the configured case-normalization rule.

If both `treatment` and `treatment_alias` are present, the configured state-space
column is preferred. Without an explicit choice, Reader selects the column
that matches more state values and uses `treatment` to break a tie.

An optional `sequence` column can be carried into vec8 output. It is experiment
metadata; it is not downstream sequence authority.

## Snapshot selection

`time_selected_h` is elapsed acquisition time. The vec8 contract does not
record a stress-induction event and must not be described as stress-relative.
Workbook segment transitions are acquisition provenance, not stress events.
Return to [plate-reader metric outputs](../plate_reader/metric_outputs.md) when
choosing an event-relative analysis contract.

Reader selects a snapshot independently for the logic and intensity channels:

1. Keep rows for the requested channel.
2. Keep rows whose treatment value belongs to the four-corner map.
3. Select one global time from that filtered set.
4. Require the logic and intensity selections to resolve to the same time.

`target_time_h: null` selects the latest available time. Otherwise `time_mode`
has one of four values:

- `exact`: require the target time.
- `nearest`: choose the closest time.
- `last_before`: choose the latest time at or before the target.
- `first_after`: choose the earliest time at or after the target.

`time_tolerance_h` is a warning threshold. It does not change the chosen
snapshot. If no time satisfies the selected mode, or if the two channels
select different times, Reader stops with an error.

## Corner aggregation

At the selected time, Reader groups observation rows by `design_by` and corner.
For each group it records:

- `y_mean`: mean of numeric values
- `y_sd`: observation dispersion as sample standard deviation, or `0.0` for
  one numeric observation
- `y_n`: numeric observation count

The wide intermediate table contains `b00..b11`, `sd00..sd11`, and
`n00..n11`. With `require_all_corners_per_design: true`, the default, any
missing design/corner pair is an error.

## Logic shape

Let `L_i` be a logic-channel corner mean and let `eps_ratio` be the configured
log guard.

```text
u_i = log2(max(L_i, eps_ratio))
r_logic = max(max(L_i, eps_ratio)) / min(max(L_i, eps_ratio))
span = max(u) - min(u)
```

When `span <= eps_range`, Reader sets all four `v_i` values to `0.25` and
marks `flat_logic = true`. Otherwise:

```text
v_i = (u_i - min(u)) / (span + eps_range)
```

Reader also keeps `logic_span_log2`, `r_logic_min`, `r_logic_max`, and the
corner names at the minimum and maximum as diagnostics.

## Reference-normalized intensity

`protocol.inputs.reference.design_id` is required. Reader resolves that value
in this order:

1. exact match in raw `design_id`
2. one unambiguous match in `design_id_alias`
3. error

There is no implicit reference and no partial alias matching. The resolved raw
design row provides one intensity anchor `A_i` per corner. Missing reference
rows or anchors are errors.

`protocol.inputs.reference.observation_stat` selects `mean` or `median` for
combining the available reference observations. It does not classify positions
as technical or biological replicates.

For sample intensity mean `I_i`:

```text
denominator = max(A_i + ref_add_alpha, eps_ref)
y_linear_i = (I_i + eps_abs) / denominator
y_i_star = log2(max(y_linear_i + log2_offset_delta, eps_ratio))
```

The selected `log2_offset_delta` is persisted in
`intensity_log2_offset_delta`. A downstream scorer must use the same value to
recover the intended linear intensity.

## Typed output

The main workbench output is the dataframe record:

- record ID: `sfxi_vec8/vec8`
- contract: `sfxi.vec8.v3`
- catalog: `outputs/manifests/records.json`
- dataframe artifact: under `outputs/artifacts/`

The primary columns are:

- `design_id`, optional `sequence`, and optional carried metadata
- `time_selected_h` and `reference_design_id`
- `intensity_log2_offset_delta`
- `r_logic`, `v00..v11`, and `y00_star..y11_star`
- `flat_logic` and the retained calculation diagnostics

The reference design is used for anchors and is excluded from the emitted
vec8 table by default. Use `reader records` to inspect the record and its
provenance; use `reader export` when a workbook is required.

## Fail-fast conditions

Reader rejects, rather than infers around:

- an ordered state space without exactly four unique corner labels
- missing required columns or an invalid first `design_by` column
- a configured channel or treatment map with no matching rows
- an unavailable snapshot under the selected time mode
- different selected times for logic and intensity
- incomplete corner sets when completeness is required
- a missing, ambiguous, or incomplete reference design

Unknown transform and nested response/reference settings are rejected.

## Source map

- [selection.py](../../../src/reader/domains/logic/sfxi/selection.py): treatment,
  time, and corner selection
- [math.py](../../../src/reader/domains/logic/sfxi/math.py): vec8 calculation
- [reference.py](../../../src/reader/domains/logic/sfxi/reference.py): reference
  identity and anchors
- [builder.py](../../../src/reader/domains/logic/sfxi/builder.py): orchestration and
  output assembly
- [logic.py](../../../src/reader/contracts/builtins/logic.py): dataframe
  contract
