---
doc_id: reader-ordered-state-spaces
surface: config-reference
owner: reader-maintainers
last_verified: 2026-08-01
summary: Metric-neutral experiment annotation that binds ordered state ids to exact metadata values.
---

# Ordered state spaces

An ordered state space tells Reader how experiment metadata values map to a
small, ordered set of state identifiers. It describes assay records, not a
metric or biological objective.

```yaml
annotations:
  ordered_state_spaces:
    four_state_conditions:
      column: treatment
      state_order: ["00", "10", "01", "11"]
      values:
        "00": baseline
        "10": condition A
        "01": condition B
        "11": conditions A and B
      case_sensitive: true
```

The fields have narrow meanings:

- `column` is the metadata column read from annotated Reader records.
- `state_order` is the ordered, non-empty list of analysis-facing state ids.
- `values` maps every declared state id to one exact source value in `column`.
- `case_sensitive` controls source-value matching only. State ids remain exact.

Reader rejects empty or duplicate state ids, missing or extra value mappings,
and source values that collide under the declared case rule.

## Analysis ownership

The annotation does not say which states are ON or OFF. It contains no target
mask, response formula, reduction window, calibration, or objective. Each
analysis resolves the state space and validates its own requirements.

Four-state vector and four-state event-window processing both require the exact order `00`, `10`,
`01`, `11` today. They enforce that rule independently and then perform
different reductions and publish different vector contracts. An experiment can
therefore share state identity without coupling either analysis to the other's
metric.

The four-state vector protocol field `protocol.inputs.state_map_ref` names an entry under
`annotations.ordered_state_spaces`. The four-state event-window protocol declares its
own source-value mapping because it combines records from multiple experiments.
Neither analysis infers state order from source labels.

## Code map

- Wire parsing: [`src/reader_workbench/workbench/config/`](../../src/reader_workbench/workbench/config/)
- Experiment semantics: [`src/reader_workbench/workbench/experiment/`](../../src/reader_workbench/workbench/experiment/)
- Four-state vector binding: [`src/reader_workbench/domains/logic/four_state_vector/treatment_semantics.py`](../../src/reader_workbench/domains/logic/four_state_vector/treatment_semantics.py)
- Four-state event-window analysis: [`src/reader_workbench/domains/plate_reader/analysis/four_state_event_window/`](../../src/reader_workbench/domains/plate_reader/analysis/four_state_event_window/)
- Record-collection binding: [`src/reader_workbench/workbench/records/sources.py`](../../src/reader_workbench/workbench/records/sources.py)
